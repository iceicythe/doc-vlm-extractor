"""冒烟测试 —— 验证 Qwen3-VL-2B QLoRA 能否真正跑通。

用法：
    venv-gld/Scripts/python.exe smoke_test.py

依次执行：
    1. CUDA 环境检查
    2. 加载 Qwen3-VL-2B-Instruct (4bit)     ← 首次运行需下载约 4-5GB
    3. 挂 LoRA 适配器（vision + language 全层）
    4. 跑一次推理（验证模型可用）
    5. 跑一次前向 + 反向（验证能训练，打印峰值显存）

这是 W1 的验收标准：本脚本跑通 == 环境真正就绪。

注意：import unsloth 必须放在所有 torch/transformers 导入之前，
      否则 Unsloth 的优化补丁不会生效（会有 UserWarning 提示）。
"""

import os
import sys
import time
import traceback

# Windows 上 triton/inductor 对 Qwen3-VL 的部分算子编译会失败（InductorError），
# 表现为大量 WARNING 刷屏。这里直接关掉 dynamo 走 eager，输出更干净、运行更稳。
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# --- Unsloth 必须最先导入 ---
try:
    import unsloth  # noqa: F401
except Exception as exc:  # noqa: BLE001
    print(f"[FATAL] import unsloth 失败: {type(exc).__name__}: {exc}")
    print("        若为 NotImplementedError，检查 torch 是否为 CPU 版：")
    print("        python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"")
    sys.exit(1)

import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from unsloth import FastVisionModel  # noqa: E402

MODEL_NAME = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
MAX_SEQ_LEN = 2048
IMAGE_SIZE = 384  # 对齐 32 像素网格


def hr(title: str) -> None:
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def gpu_mem() -> str:
    if not torch.cuda.is_available():
        return "N/A"
    alloc = torch.cuda.memory_allocated() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return f"当前 {alloc:.2f}GB / 峰值 {peak:.2f}GB"


def make_test_image() -> Image.Image:
    """造一张工程单据风格的测试图。"""
    W, H = IMAGE_SIZE * 2, int(IMAGE_SIZE * 1.4)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    d.text((20, 16), "工程材料清单", fill="black")
    headers = ["序号", "名称", "单位", "数量", "单价(元)", "金额(元)"]
    rows = [
        ["1", "螺纹钢 HRB400", "吨", "12.5", "4200.00", "52500.00"],
        ["2", "C30 混凝土", "m3", "85.0", "480.00", "40800.00"],
        ["3", "模板", "m2", "260.0", "55.00", "14300.00"],
    ]

    cols = len(headers)
    cell_w = (W - 40) // cols
    top, row_h = 60, 46

    for i, h in enumerate(headers):
        x = 20 + i * cell_w
        d.rectangle([x, top, x + cell_w, top + row_h], outline="black")
        d.text((x + 8, top + 16), h, fill="black")

    for r, row in enumerate(rows, start=1):
        for i, val in enumerate(row):
            x = 20 + i * cell_w
            y = top + r * row_h
            d.rectangle([x, y, x + cell_w, y + row_h], outline="black")
            d.text((x + 8, y + 16), val, fill="black")

    return img


def main() -> int:
    hr("步骤 1/4 | CUDA 环境")
    print(f"  torch        : {torch.__version__}")
    print(f"  CUDA compiled: {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("  [FATAL] CUDA 不可用 —— torch 可能是 CPU 版")
        print("  修复：pip install --force-reinstall torch torchvision "
              "--index-url https://download.pytorch.org/whl/cu130")
        return 1
    print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    print(f"  显存         : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  镜像站       : {os.environ.get('HF_ENDPOINT', '(未设置)')}")

    # ---------------------------------------------------------------- 加载
    hr("步骤 2/4 | 加载 Qwen3-VL-2B (4bit)")
    print("  首次运行需下载约 4-5GB，请耐心等待...")
    t0 = time.time()
    try:
        model, processor = FastVisionModel.from_pretrained(
            MODEL_NAME,
            load_in_4bit=True,
            max_seq_length=MAX_SEQ_LEN,
            use_gradient_checkpointing="unsloth",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [FATAL] 模型加载失败: {type(exc).__name__}: {exc}")
        print("  排查：1) 网络/HF_ENDPOINT  2) transformers 是否含 qwen3_vl  3) 显存")
        return 1
    print(f"  [OK] 加载完成，耗时 {time.time() - t0:.1f}s | 显存 {gpu_mem()}")

    # ---------------------------------------------------------------- LoRA
    hr("步骤 3/4 | 挂 LoRA（vision + language 全层）")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        random_state=3407,
        use_gradient_checkpointing="unsloth",
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数 : {trainable:,} ({100 * trainable / total:.3f}%)")
    print(f"  总参数     : {total:,}")
    print(f"  显存       : {gpu_mem()}")

    # ---------------------------------------------------------------- 推理
    hr("步骤 4/4 | 推理 + 训练步")
    image = make_test_image()
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "把这张表提取成 JSON，只输出 JSON。"},
        ],
    }]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=[image], text=prompt, return_tensors="pt").to("cuda")

    torch.cuda.reset_peak_memory_stats()

    FastVisionModel.for_inference(model)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    gen_time = time.time() - t0
    text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  [推理 OK] 耗时 {gen_time:.1f}s")
    print(f"  输出前 200 字: {text[:200]!r}")

    # 训练步：前向 + 反向
    FastVisionModel.for_training(model)
    model.zero_grad()
    t0 = time.time()
    try:
        # 按标准对话模板构造训练样本（user + assistant 成对）
        target = '{"序号":"1","名称":"螺纹钢 HRB400","单位":"吨","数量":"12.5"}'
        train_msgs = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "把这张表提取成 JSON，只输出 JSON。"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": target}]},
        ]
        full = processor.apply_chat_template(
            train_msgs, tokenize=False, add_generation_prompt=False
        )
        tinputs = processor(images=[image], text=full, return_tensors="pt").to("cuda")
        tinputs["labels"] = tinputs["input_ids"].clone()
        loss = model(**tinputs).loss
        loss.backward()
        step_time = time.time() - t0
        print(f"  [训练步 OK] loss={loss.item():.4f} 耗时 {step_time:.1f}s")
    except Exception as exc:  # noqa: BLE001
        print(f"  [训练步 FAIL] {type(exc).__name__}: {exc}")
        print("  --- 完整 traceback ---")
        traceback.print_exc()
        if "out of memory" in str(exc).lower():
            print("  → 显存不足。对策：降 IMAGE_SIZE（384→320）、减 r（16→8）")
        return 1

    print(f"  峰值显存 (含训练步): {gpu_mem()}")
    print(f"  可用显存余量: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.max_memory_allocated()) / 1024**3:.2f} GB")

    hr("结论")
    print("  环境就绪 —— Qwen3-VL-2B QLoRA 可加载、可推理、可反向传播。")
    print("  下一步：进入 W1 数据管线（schema 设计 + HTML 模板）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
