"""SFT 训练 —— Qwen3-VL-2B QLoRA，在合成工程单据上微调。

关键点：
  1. import unsloth 必须在最前
  2. Windows 上禁用 dynamo（否则 InductorError 刷屏）
  3. 用 train_on_responses_only 只对 assistant 回复段算 loss
     —— 否则会对图像 token 与 prompt 一起算 loss，数值虚高且无意义
  4. 图片统一 resize 到 384px（对齐 32 像素网格）

用法：
    venv-gld/Scripts/python.exe src/train_sft.py --max-steps 60 --tag smoke
    venv-gld/Scripts/python.exe src/train_sft.py --max-steps 600 --tag v1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# Windows + WDDM 下 cudaMemGetInfo 会因显存换页而偶发返回接近 0 的空闲值，
# 于是 Unsloth 的 fused cross-entropy 分块器探测失败并抛
# "No or negligible GPU memory available for fused cross entropy"（随机在若干 step 后崩）。
# 固定分块预算即可绕开这次探测，代价仅是分块数略增。
os.environ.setdefault("UNSLOTH_CE_LOSS_TARGET_GB", "1")

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

MODEL_NAME = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
IMAGE_SIZE = 384
MAX_SEQ_LEN = 2048


def main() -> int:
    ap = argparse.ArgumentParser(description="SFT 微调 Qwen3-VL-2B")
    ap.add_argument("--max-steps", type=int, default=60,
                    help="训练步数（小样本验证用 60 即可）")
    ap.add_argument("--tag", default="smoke", help="输出目录后缀")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--save-steps", type=int, default=60)
    ap.add_argument("--no-vision-lora", action="store_true",
                    help="冻结视觉层（显存紧张时用）")
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE,
                    help="输入图边长（须为 32 的倍数）；消融实验用")
    ap.add_argument("--data-ratio", type=float, default=1.0,
                    help="训练数据比例 0-1，按源样本抽样（三档同进同出，防泄漏）")
    ap.add_argument("--no-save", action="store_true",
                    help="不保存 LoRA（仅测速度/显存时用）")
    args = ap.parse_args()

    # ---------------- 延迟导入，确保 unsloth 最先 ----------------
    import torch
    from datasets import load_dataset, Image as HFImage
    from unsloth import FastVisionModel, train_on_responses_only
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    from PIL import Image

    print("=" * 62)
    print("SFT 训练")
    print("=" * 62)
    print(f"  torch      : {torch.__version__}")
    print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  显存       : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print(f"  输入尺寸   : {args.image_size}px   数据比例: {args.data_ratio:.0%}")

    train_file = PROCESSED / "train.jsonl"
    if not train_file.exists():
        print(f"[FATAL] 缺少 {train_file}，请先运行 src/build_dataset.py")
        return 1

    # ---------------- 模型 ----------------
    print(f"\n[1/4] 加载模型 {MODEL_NAME}")
    model, processor = FastVisionModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        max_seq_length=MAX_SEQ_LEN,
        use_gradient_checkpointing="unsloth",
    )

    print(f"[2/4] 挂 LoRA（r={args.rank}，"
          f"{'仅语言层' if args.no_vision_lora else '视觉+语言全层'}）")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=not args.no_vision_lora,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=0.0,
        bias="none",
        random_state=3407,
        use_gradient_checkpointing="unsloth",
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"      可训练 {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    # ---------------- 数据 ----------------
    print(f"[3/4] 加载数据 {train_file.name}")
    ds = load_dataset("json", data_files=str(train_file), split="train")

    # 按源样本抽样（保证同一底图的三档同进同出，避免 split 内泄漏）
    if args.data_ratio < 1.0:
        import random as _random
        stems = sorted({r["stem"] for r in ds["meta"]})
        keep_n = max(1, int(round(len(stems) * args.data_ratio)))
        rng = _random.Random(3407)
        keep = set(rng.sample(stems, keep_n))
        # filter 的回调收到的是单条记录，stem 在 meta 子字典里
        ds = ds.filter(lambda m: m["meta"]["stem"] in keep)
        print(f"      抽样 {args.data_ratio:.0%} → {len(keep)}/{len(stems)} 源样本 "
              f"（{len(ds)} 条）")

    def prepare(example):
        """把 jsonl 里的图片路径读成 PIL，转成 Unsloth 期望的 messages 格式。"""
        msgs = example["messages"]
        img = Image.open(example["image"]).convert("RGB")
        img = img.resize((args.image_size, args.image_size), Image.BICUBIC)

        new_msgs = []
        for m in msgs:
            content = []
            for c in m["content"]:
                if c["type"] == "image":
                    content.append({"type": "image"})       # 占位，实际图在 images 字段
                else:
                    content.append({"type": "text", "text": c["text"]})
            new_msgs.append({"role": m["role"], "content": content})
        return {"messages": new_msgs, "images": [img]}

    ds = ds.map(prepare, remove_columns=ds.column_names)
    print(f"      样本数 {len(ds)}")
    print(f"      第一条 user 文本：{ds[0]['messages'][0]['content'][-1]['text'][:50]}...")

    # ---------------- 训练 ----------------
    out_dir = OUTPUTS / f"sft_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[4/4] 训练（max_steps={args.max_steps}）")
    trainer = SFTTrainer(
        model=model,
        tokenizer=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(model, processor),
        train_dataset=ds,
        args=SFTConfig(
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=5,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=str(out_dir),
            save_steps=args.save_steps,
            save_total_limit=2,
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=MAX_SEQ_LEN,
        ),
    )

    # 只对 assistant 回复段计 loss
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    stats = trainer.train()
    print()
    print(f"  训练完成  用时 {stats.metrics.get('train_runtime', 0):.0f}s")
    print(f"  最终 loss {stats.metrics.get('train_loss', float('nan')):.4f}")
    print(f"  峰值显存  {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    # ---------------- 保存 ----------------
    if args.no_save:
        print("  （--no-save，跳过保存）")
        return 0
    adapter_dir = out_dir / "lora"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    print(f"  LoRA 已保存: {adapter_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
