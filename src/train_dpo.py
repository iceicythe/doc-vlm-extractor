"""DPO 后训练 —— 手写训练循环（Qwen3-VL-2B QLoRA，8GB 单卡）。

为什么不用 TRL 的 DPOTrainer
---------------------------
TRL 0.24 的 `DPOTrainer` **没有把 `image_grid_thw` 传给模型**。
`dpo_trainer.py` 里 vision 相关只有三处：

    model_kwargs["pixel_values"]        = concatenated_batch["pixel_values"]
    model_kwargs["pixel_attention_mask"] = ...
    model_kwargs["image_sizes"]         = ...

而 `Qwen3VLModel.forward` 必须有 `image_grid_thw` 才能把图像特征 scatter 回
视觉 token 位置（`Qwen3VLModel.get_rope_index` 也要它算 mRoPE）。
对照：TRL 的 GRPO / OnlineDPO 两个 trainer 都显式传了 `image_grid_thw`
（`grpo_trainer.py:722`、`online_dpo_trainer.py:1233`），只有离线 DPO 漏了。
所以这条路是死的，只能自己写循环 —— 好在 DPO 的数学本身很短。

参考模型怎么取（这是 8GB 上的关键）
---------------------------------
标准 DPO 的 ref 是**策略的初始快照**（这里就是 SFT 模型）。
两个错误做法：
  - `disable_adapter()` → 拿到的是 **base 模型**（SFT 之前），不是 SFT 快照。
    这样 β·log(π_θ/π_ref) 里混进了整个 SFT 阶段的增益，起点就不是 0 了。
  - 再加载一份模型当 ref → 4bit 的 2B 模型约 1.5GB + 激活，8GB 上必炸。

正确做法：**挂两个 adapter**。`default`（可训练，初值 = SFT 权重）、
`ref`（同目录再加载一份，`is_trainable=False` 冻结）。adapter 只有几十 MB，
`set_adapter()` 切换即可，显存代价接近 0。
于是 step 0 时 policy 与 ref 逐 token 完全一致 → **loss 必须 = ln2 = 0.6931**，
这正好当成一个免费的初始化自检（见 --check）。

用法
----
    # 自检：只跑一个 batch，验证 ref 对齐（loss 应 ≈ 0.6931）与显存
    venv-gld/Scripts/python.exe src/train_dpo.py --check

    # 正式训练（冷机跑）
    venv-gld/Scripts/python.exe src/train_dpo.py --epochs 3 --tag v1

    # 用 base 当 ref 跑一组对照（复现"错的做法"，用于报告里的 ablation）
    venv-gld/Scripts/python.exe src/train_dpo.py --ref-mode base --tag v1_refbase
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
# 同 train_sft.py：绕开 Windows/WDDM 下 fused-CE 的显存探测缺陷（详见技能库）
os.environ.setdefault("UNSLOTH_CE_LOSS_TARGET_GB", "1")

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

MODEL_NAME = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
SFT_ADAPTER = OUTPUTS / "sft_v1" / "lora"
IMAGE_SIZE = 384
MAX_SEQ_LEN = 2048
LN2 = math.log(2.0)


# --------------------------------------------------------------------------
# 纯函数部分（不依赖 unsloth，可在 CPU 上单测）
# --------------------------------------------------------------------------
def dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta=0.1):
    """DPO 损失。

    入参都是「completion 段 logprob 之和」的标量/向量（本脚本里是 batch=2 的两个元素）。

        loss = -logsigmoid( β·[(logπ_c - logπref_c) - (logπ_r - logπref_r)] )

    Returns: (loss, acc, reward_margin, rewards_chosen, rewards_rejected)
        acc            —— 隐式奖励偏好正确率（margin>0 即认为偏好对了）
        reward_margin  —— β·(logπ_c - logπ_r)，只看策略的"喜欢程度差"
    """
    import torch
    import torch.nn.functional as F

    rewards_chosen = beta * (pol_chosen - ref_chosen)
    rewards_rejected = beta * (pol_rejected - ref_rejected)
    margin = rewards_chosen - rewards_rejected
    loss = -F.logsigmoid(margin).mean()
    with torch.no_grad():
        acc = (margin > 0).float().mean()
        reward_margin = (beta * (pol_chosen - pol_rejected)).mean()
    return loss, acc, reward_margin, rewards_chosen.mean(), rewards_rejected.mean()


def find_lm_head(model):
    """定位 lm_head。

    Unsloth 的 PeftModel 包装层数不固定（PeftModel -> LoraModel -> 基座），
    所以逐层剥离地找，而不是硬编码属性路径。
    """
    seen: set[int] = set()
    queue = [model]
    while queue:
        m = queue.pop(0)
        if m is None or id(m) in seen:
            continue
        seen.add(id(m))
        try:
            head = getattr(m, "lm_head", None)
        except AttributeError:
            head = None
        if head is not None and hasattr(head, "weight"):
            return head
        for attr in ("base_model", "model"):
            queue.append(getattr(m, attr, None))
    raise RuntimeError("找不到 lm_head，模型结构与预期不符")


def extend_mm_token_type_ids(mm_ids, n_total):
    """把 `mm_token_type_ids` 从 prompt 长度补齐到「prompt + completion」全长。

    Qwen3-VL 的 `get_rope_index` 用 `mm_token_type_ids` 逐 token 标记
    文本(0) / 图像(1) / 视频(2)，用来算 mRoPE 的 3D position_ids。里面有一行

        input_token_type = input_token_type[attention_mask[batch_idx].bool()]

    它**要求该张量与 input_ids、attention_mask 严格等长**。而 processor 只对
    prompt 段产出它（长度 P），我们又在后面拼了 completion 段（长度 L）：

        attention_mask    (B, P+L)
        mm_token_type_ids (B, P)      ← 长度不符 → IndexError

    completion 段全是纯文本 token，补 0 即正确解（不引入任何图像占位）。
    """
    import torch

    if mm_ids is None:
        return None
    cur = int(mm_ids.shape[-1])
    if cur == n_total:
        return mm_ids
    if cur > n_total:
        raise ValueError(
            f"mm_token_type_ids 长度 {cur} 超过目标序列长度 {n_total}，无法补齐")
    pad = torch.zeros(*mm_ids.shape[:-1], n_total - cur,
                      dtype=mm_ids.dtype, device=mm_ids.device)
    return torch.cat([mm_ids, pad], dim=-1)


def ref_dispatch_probe(model, probe, scale=1.5):
    """验证「切到 ref adapter 后，前向真的走了 ref 的权重」。

    为什么必须单独验：ref adapter 是 policy 的逐位副本，初始 logp 天然相等 ——
    所以 `logp_ref == logp_policy` 这个现象**在 set_adapter 失效时同样成立**
    （那种情况下前向一直在用 default）。这个失效模式极隐蔽：loss 曲线看起来
    还挺正常，但 DPO 实际是在跟一个「跟着策略一起漂移」的参考模型对齐，
    整个实验作废。

    做法：把按 adapter 名索引的 LoRA ModuleDict 里 "ref" 那份权重放大 scale 倍，
    再跑一次 probe（probe 内部负责切 adapter）：
        结果变了 → 切换生效；结果没变 → 切换没生效
    最后精确还原（copy_ 原始张量，不做乘除往返，避免浮点残差）。

    Returns: 扰动后 probe() 的返回值；模型里找不到多 adapter 结构时返回 None。
    """
    import torch

    targets: list = []
    for _, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleDict):
            try:
                keys = set(mod.keys())
            except AttributeError:
                continue
            if {"ref", "default"} <= keys:
                targets.extend(list(mod["ref"].parameters()))

    if not targets:
        return None

    saved = [(t, t.detach().clone()) for t in targets]
    try:
        with torch.no_grad():
            for t, _ in saved:
                t.mul_(scale)
        return probe()
    finally:
        with torch.no_grad():
            for t, orig in saved:
                t.copy_(orig)


def seq_logprob_masked(model, input_ids, attention_mask, n_prompt, comp_mask,
                       model_kwargs=None, chunk=128):
    """求 completion 段每个序列的 logprob 之和，返回形状 (B,) 的张量。

    不用 `outputs.logits` 而是 `hidden_states[-1]` + lm_head 手算：
    logits 是 (B, T, 152k) 的 float32，T≈1100 时单份就 1.3GB，8GB 上放不下。
    只对 completion 位置算 lm_head，再把 vocab 维分块做 log_softmax，
    峰值显存从 GB 级降到百 MB 级。

    位置对齐：hidden[j] 预测的是 token[j+1]，所以 scoring token[n_prompt:] 要用
    hidden[n_prompt-1 : T-1]。
    """
    import torch
    import torch.nn.functional as F

    # 先做形状校验再跑前向：形状错的报错在 GPU 上很难看，且白跑一次前向
    t = input_ids.shape[1]
    if n_prompt < 1 or n_prompt >= t:
        raise ValueError(f"n_prompt={n_prompt} 与序列长度 {t} 不匹配")
    if comp_mask.shape != (input_ids.shape[0], t - n_prompt):
        raise ValueError(
            f"comp_mask 形状 {tuple(comp_mask.shape)} 与 completion 段长度 "
            f"{t - n_prompt}（序列 {t} − prompt {n_prompt}）不一致")
    mm = (model_kwargs or {}).get("mm_token_type_ids")
    if mm is not None and int(mm.shape[-1]) != t:
        raise ValueError(
            f"mm_token_type_ids 长度 {int(mm.shape[-1])} 与序列长度 {t} 不符 —— "
            f"Qwen3-VL 的 get_rope_index 会直接 IndexError。"
            f"请先用 extend_mm_token_type_ids() 补齐到 {t}。")

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        **(model_kwargs or {}),
    )
    if out.hidden_states is None:
        raise RuntimeError("模型没有返回 hidden_states —— 无法走低显存 logprob 路径")
    h = out.hidden_states[-1]                      # (B, T, H)

    hs = h[:, n_prompt - 1: t - 1, :]              # (B, Lc, H)，Lc = T - n_prompt
    targets = input_ids[:, n_prompt:]              # (B, Lc)

    head = find_lm_head(model)
    chunks = []
    for s in range(0, hs.shape[1], chunk):
        logits = head(hs[:, s:s + chunk, :]).float()          # (B, lc, V)
        lp = F.log_softmax(logits, dim=-1).gather(
            -1, targets[:, s:s + chunk].unsqueeze(-1)).squeeze(-1)
        chunks.append(lp)
    lp_all = torch.cat(chunks, dim=1)              # (B, Lc)
    # comp_mask: 1=真 token，0=padding。左乘即屏蔽 padding
    return (lp_all * comp_mask).sum(dim=1)         # (B,)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="DPO 后训练 Qwen3-VL-2B")
    ap.add_argument("--data", default="data/processed/dpo_train.jsonl")
    ap.add_argument("--sft-adapter", default=str(SFT_ADAPTER),
                    help="DPO 起点（SFT 产出的 LoRA）")
    ap.add_argument("--out", default="outputs/dpo_v1/lora")
    ap.add_argument("--tag", default="v1", help="日志目录后缀")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6,
                    help="DPO 的 lr 要比 SFT 小 1~2 个量级（SFT 用的是 2e-4）")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--grad-accum", type=int, default=8,
                    help="有效 batch（DPO 每条样本要两次前向，单步只放 1 条）")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=0, help=">0 时覆盖 epochs")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只用前 N 对（调试）")
    ap.add_argument("--ref-mode", choices=["snapshot", "base"], default="snapshot",
                    help="snapshot=冻结的 SFT adapter（正确做法）；"
                         "base=关掉 adapter 的基座模型（对照用）")
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--save-steps", type=int, default=30)
    ap.add_argument("--check", action="store_true",
                    help="只跑一个 batch 做自检（ref 对齐 / 显存 / 形状），不训练")
    args = ap.parse_args()

    # ---------------- 延迟导入 ----------------
    import torch
    from PIL import Image
    from unsloth import FastVisionModel

    torch.manual_seed(args.seed)

    print("=" * 62)
    print("DPO 后训练")
    print("=" * 62)
    print(f"  GPU        : {torch.cuda.get_device_name(0)}")
    print(f"  显存       : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print(f"  β={args.beta}  lr={args.lr}  ref-mode={args.ref_mode}")

    # ---------------- 数据 ----------------
    data_path = ROOT / args.data
    if not data_path.exists():
        print(f"[FATAL] 缺少 {data_path}，请先跑 src/build_pref_data.py")
        return 1
    pairs = [json.loads(l) for l in data_path.open(encoding="utf-8") if l.strip()]
    if not pairs:
        print(f"[FATAL] {data_path} 是空的")
        return 1
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"  偏好对     : {len(pairs)}  （{data_path.name}）")

    # ---------------- 模型 ----------------
    sft_adapter = Path(args.sft_adapter)
    if not sft_adapter.exists():
        print(f"[FATAL] 找不到 SFT adapter: {sft_adapter}")
        return 1
    print(f"  加载 SFT adapter: {sft_adapter}")
    model, processor = FastVisionModel.from_pretrained(
        str(sft_adapter), load_in_4bit=True, max_seq_length=MAX_SEQ_LEN,
        use_gradient_checkpointing="unsloth",
    )

    if args.ref_mode == "snapshot":
        # 再挂一份同样的 SFT 权重做冻结参考。adapter 很小，不额外占多少显存。
        try:
            model.load_adapter(str(sft_adapter), adapter_name="ref", is_trainable=False)
        except Exception as exc:
            print(f"[FATAL] 挂载 ref adapter 失败: {type(exc).__name__}: {exc}")
            print("        可用 --ref-mode base 退化为「关掉 adapter 的基座」当参考，")
            print("        但那不是策略的初始快照，报告里必须写明。")
            return 1
        print("  ref adapter 已挂载（冻结的 SFT 快照）")
    model.set_adapter("default")
    policy_adapter = "default"

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练参数 : {trainable:,}")

    lm_head = find_lm_head(model)
    print(f"  lm_head    : {type(lm_head).__name__} out={getattr(lm_head, 'out_features', '?')}")

    # ---------------- 单条样本 -> 张量 ----------------
    eos_id = processor.tokenizer.eos_token_id
    pad_id = processor.tokenizer.pad_token_id or eos_id
    stats = {"truncated": 0, "no_mm_type": False}

    def encode(rec):
        """返回 (input_ids, attention_mask, n_prompt, comp_mask, model_kwargs)。

        两次 completion（chosen/rejected）拼成 batch=2；prompt 完全相同（含图像），
        图像张量按 leading dim 复制一份。
        """
        img = Image.open(ROOT / rec["image"]).convert("RGB").resize(
            (args.image_size, args.image_size))
        msg = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": rec["prompt"]},
        ]}]
        prompt_text = processor.apply_chat_template(msg, add_generation_prompt=True)
        p = processor(images=[img], text=[prompt_text], return_tensors="pt")
        prompt_ids = p["input_ids"][0]                     # (P,)
        P = prompt_ids.shape[0]

        max_comp = MAX_SEQ_LEN - P
        comps = []
        for key in ("chosen", "rejected"):
            ids = processor.tokenizer(rec[key], add_special_tokens=False)["input_ids"]
            ids = ids + [eos_id]
            if len(ids) > max_comp:
                ids = ids[:max_comp]
                stats["truncated"] += 1
            comps.append(ids)

        L = max(len(c) for c in comps)
        input_ids, comp_mask = [], []
        for c in comps:
            padded = c + [pad_id] * (L - len(c))
            input_ids.append(torch.tensor(list(prompt_ids.tolist()) + padded))
            comp_mask.append(torch.tensor([1.0] * len(c) + [0.0] * (L - len(c))))
        input_ids = torch.stack(input_ids)                  # (2, P+L)
        comp_mask = torch.stack(comp_mask)                  # (2, L)

        model_kwargs = {}
        for k, v in p.items():
            if k in ("input_ids", "attention_mask") or not torch.is_tensor(v):
                continue
            # 两行共享同一张图：沿 leading dim 复制（pixel_values / image_grid_thw 都适用）
            model_kwargs[k] = torch.cat([v, v], dim=0)

        # 关键：mm_token_type_ids 只覆盖 prompt 段（长度 P），而序列已经拼到 P+L。
        # 不补齐的话 get_rope_index 在算 mRoPE position_ids 时会 IndexError。
        if "mm_token_type_ids" in model_kwargs:
            model_kwargs["mm_token_type_ids"] = extend_mm_token_type_ids(
                model_kwargs["mm_token_type_ids"], P + L)
        else:
            stats["no_mm_type"] = True

        return input_ids, torch.ones_like(input_ids), P, comp_mask, model_kwargs

    # ---------------- 自检 ----------------
    def run_ref(input_ids, attn, P, cmask, mkw):
        """参考模型的 completion logprob。切换 adapter，全程 no_grad。"""
        with torch.no_grad():
            if args.ref_mode == "base":
                with model.disable_adapter():
                    return seq_logprob_masked(model, input_ids, attn, P, cmask, mkw)
            model.set_adapter("ref", inference_mode=True)
            try:
                return seq_logprob_masked(model, input_ids, attn, P, cmask, mkw)
            finally:
                model.set_adapter(policy_adapter)

    print("\n[1/2] 自检第一个 batch")
    t0 = time.time()
    i_ids, a_mask, P, c_mask, mkw = encode(pairs[0])
    i_ids, a_mask, c_mask = i_ids.cuda(), a_mask.cuda(), c_mask.cuda()
    mkw = {k: v.cuda() for k, v in mkw.items()}
    T = i_ids.shape[1]
    print(f"  序列长度 prompt={P}  completion={T-P}  total={T}"
          f"  kwargs={ {k: tuple(v.shape) for k, v in mkw.items()} }")
    mmk = mkw.get("mm_token_type_ids")
    if mmk is None:
        print("  [注意] processor 未返回 mm_token_type_ids，"
              "模型可能自行推断（若报错请检查 transformers 版本）")
    else:
        ok = int(mmk.shape[-1]) == T
        print(f"  mm_token_type_ids 长度 {int(mmk.shape[-1])} vs 序列 {T}"
              f"  {'[OK]' if ok else '[FAIL] 长度不符'}")

    model.train()
    pol = seq_logprob_masked(model, i_ids, a_mask, P, c_mask, mkw)
    ref = run_ref(i_ids, a_mask, P, c_mask, mkw)
    loss0, acc0, rm0, rc0, rr0 = dpo_loss(pol[0], pol[1], ref[0], ref[1], args.beta)
    d0 = float((pol[0] - ref[0]).detach())
    print(f"  policy logp chosen/rejected = {pol[0].item():.2f} / {pol[1].item():.2f}")
    print(f"  ref    logp chosen/rejected = {ref[0].item():.2f} / {ref[1].item():.2f}")
    print(f"  step0 loss = {loss0.item():.4f}   (期望 {LN2:.4f})")

    if args.ref_mode == "snapshot":
        if abs(loss0.item() - LN2) > 0.05:
            print(f"  [警告] step0 loss 偏离 ln2 超过 0.05 —— ref 没对齐，"
                  f"检查 adapter 是否正确冻结/切换")
        else:
            print("  [OK] ref 与 policy 同源，起点对齐（DPO 的初始化前提成立）")
    else:
        print(f"  [说明] ref-mode=base：policy 与 ref 本就不同源，"
              f"step0 loss 不必等于 ln2（这次差 {d0:+.2f}）")

    # 决定性验证：ref 与 policy 权重初始逐位相同，所以「logp 相等」不足以
    # 证明 set_adapter 生效 —— 扰动 ref 权重，看结果跟不跟着变。
    if args.ref_mode == "snapshot":
        def _probe_ref():
            model.set_adapter("ref", inference_mode=True)
            try:
                with torch.no_grad():
                    return seq_logprob_masked(
                        model, i_ids, a_mask, P, c_mask, mkw)[0].item()
            finally:
                model.set_adapter(policy_adapter)

        shifted = ref_dispatch_probe(model, _probe_ref)
        if shifted is None:
            print("  [警告] 没找到按 adapter 名索引的 LoRA ModuleDict，"
                  "无法自动验证 ref 切换；请人工确认训练中 ref logp 与 policy 拉开")
        else:
            ok_switch = abs(shifted - ref[0].item()) > 1e-6
            print(f"  ref 切换验证：扰动 ref 权重后 logp {ref[0].item():.4f}"
                  f" → {shifted:.4f}  "
                  f"{'[OK] 前向确实走 ref 权重' if ok_switch else '[FAIL] 仍在用 policy 权重！'}")
            if not ok_switch:
                print("        这等于在一个「跟着策略一起漂移」的参考模型上对齐，"
                      "DPO 结论无效 —— 不要继续训练。")

    print(f"  峰值显存 {torch.cuda.max_memory_allocated()/1024**3:.2f} GB"
          f"  用时 {time.time()-t0:.1f}s")
    if stats["truncated"]:
        print(f"  [注意] 有 {stats['truncated']} 条 completion 被截断到 MAX_SEQ_LEN")

    if args.check:
        print("\n--check 结束，未训练")
        return 0

    # ---------------- 训练 ----------------
    accum_eff = args.grad_accum
    steps_per_epoch = max(1, math.ceil(len(pairs) / accum_eff))
    total_steps = args.max_steps or max(1, int(round(steps_per_epoch * args.epochs)))
    print(f"\n[2/2] 训练 {total_steps} 步"
          f"（{len(pairs)} 对 / 有效 batch {accum_eff} → {steps_per_epoch} 步每 epoch）")

    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)

    log_dir = OUTPUTS / f"dpo_{args.tag}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_f = (log_dir / "train_log.jsonl").open("w", encoding="utf-8")

    def lr_scale(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    rng = __import__("random").Random(args.seed)
    order = list(range(len(pairs)))
    opt_step, micro, t_start = 0, 0, time.time()
    running = 0.0
    order_idx = 0

    model.train()
    while opt_step < total_steps:
        if order_idx >= len(order):
            rng.shuffle(order)
            order_idx = 0
        rec = pairs[order[order_idx]]
        order_idx += 1

        i_ids, a_mask, P, c_mask, mkw = encode(rec)
        i_ids, a_mask, c_mask = i_ids.cuda(), a_mask.cuda(), c_mask.cuda()
        mkw = {k: v.cuda() for k, v in mkw.items()}

        pol = seq_logprob_masked(model, i_ids, a_mask, P, c_mask, mkw)
        ref = run_ref(i_ids, a_mask, P, c_mask, mkw)
        loss, acc, rmargin, rc, rr = dpo_loss(pol[0], pol[1], ref[0], ref[1], args.beta)

        (loss / args.grad_accum).backward()
        running += float(loss.item())
        micro += 1

        if micro % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            for g in optimizer.param_groups:
                g["lr"] = args.lr * lr_scale(opt_step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1

            el = time.time() - t_start
            row = {
                "step": opt_step,
                "loss": running / args.grad_accum,
                "acc": float(acc.item()),
                "reward_margin": float(rmargin.item()),
                "rewards_chosen": float(rc.item()),
                "rewards_rejected": float(rr.item()),
                "lr": args.lr * lr_scale(opt_step - 1),
                "s_per_step": el / opt_step,
            }
            running = 0.0
            log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log_f.flush()
            print(f"    step {opt_step}/{total_steps}"
                  f"  loss {row['loss']:.4f}  acc {row['acc']:.2f}"
                  f"  margin {row['reward_margin']:+.4f}"
                  f"  {row['s_per_step']:.1f}s/step", flush=True)

            if args.save_steps and opt_step % args.save_steps == 0 \
                    and opt_step < total_steps:
                ck = log_dir / f"step_{opt_step}"
                model.set_adapter(policy_adapter)
                model.save_pretrained(str(ck))
                processor.save_pretrained(str(ck))
                print(f"    已存断点 {ck}")

    log_f.close()

    # ---------------- 保存 ----------------
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    model.set_adapter(policy_adapter)
    try:
        model.save_pretrained(str(out_dir), selected_adapters=[policy_adapter])
    except TypeError:
        # 老版本 peft 没有 selected_adapters，退化为保存当前激活的 adapter
        model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))

    (log_dir / "run_meta.json").write_text(json.dumps({
        "data": str(data_path.relative_to(ROOT)).replace("\\", "/"),
        "n_pairs": len(pairs), "beta": args.beta, "lr": args.lr,
        "total_steps": total_steps, "ref_mode": args.ref_mode,
        "image_size": args.image_size, "seed": args.seed,
        "minutes": round((time.time() - t_start) / 60, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "truncated": stats["truncated"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"  训练完成  用时 {(time.time()-t_start)/60:.1f} 分钟")
    print(f"  峰值显存  {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print(f"  LoRA 已保存: {out_dir}")
    print(f"  训练日志  : {log_dir / 'train_log.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
