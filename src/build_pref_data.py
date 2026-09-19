"""构造 DPO 偏好数据 —— 采样多样输出、用 reward 打分、配成 chosen/rejected 对。

流程
----
    1. 从 train.jsonl 抽 N 个源样本（三档可指定比例）
    2. 用 SFT 模型对每张图做温度采样，得到 k 个不同输出
    3. 用 reward 函数打分（复用 evaluate.py 的评分逻辑）
    4. 最高分 → chosen，最低分 → rejected，分差太小的丢弃（减少噪声）
    5. 输出偏好对 jsonl，供 DPO 训练消费

reward 设计
-----------
    reward = F1 - 0.10 × 幻觉率     （JSON 不合法 → 0）

    - F1 本身已含幻觉惩罚（幻觉键计入 fp，拉低精确率）
    - 显式再加一项，是为了让"微调会放大幻觉"这个实测结论在训练目标里被强化
    - 这正是项目里"微调后跨域幻觉率 16%→28%"问题的对症下药

用法
----
    # 小规模验证（10 张图 × 4 采样，约 7 分钟）
    venv-gld/Scripts/python.exe src/build_pref_data.py --n 10 --out data/processed/dpo_probe.jsonl

    # 正式规模
    venv-gld/Scripts/python.exe src/build_pref_data.py --n 200 --k 4 --out data/processed/dpo_train.jsonl

    # 断点续跑（无人值守长任务建议加 --resume；崩了重跑同一条命令即可接上）
    venv-gld/Scripts/python.exe src/build_pref_data.py --resume --out data/processed/dpo_train.jsonl

产物
----
    <out>          偏好对 jsonl
    <out>.done     已处理源样本 journal（每张图落盘一次，--resume 靠它跳过）

调参备忘（2026-09-17 实测）
--------------------------
默认值是按 pilot 的失败结果改过的，别改回去：
  - temperature=0.9 → 1.3：temp<1 是**锐化**分布。模型 train loss=0.017 时，
    4 次采样输出文本完全相同的比例高达 12/20，等于白采样。
  - min-gap=0.05 → 0.02：每图约 33 字段，错 1 个字段的 F1 差只有 0.030，
    0.05 等于要求"至少错 2 个字段"，把有信号的对全筛掉了。
  - levels=medium → heavy：medium 上模型太稳（丢弃原因 60% 是"无差异"）。
  - max_new_tokens=640 → 768：与 evaluate.py 对齐，避免长单据截断。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

IMAGE_SIZE = 384
# 与 evaluate.py 对齐。曾用 640，长单据有被截断→JSON 非法→reward 记 0 的风险。
MAX_NEW_TOKENS = 768


def reward_for(gt: dict, raw: str) -> float:
    """reward = F1 − 0.10 × 幻觉率；JSON 不合法直接 0。"""
    from evaluate import score_one  # 复用同一套评分逻辑，避免"两张皮"

    s = score_one(gt, raw, mode="schema")
    if not s["json_valid"]:
        return 0.0
    tp, fp, fn = s["tp"], s["fp"], s["fn"]
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    n_pred = tp + fp
    hallu_rate = (len(s["hallucinated_keys"]) / n_pred) if n_pred else 0.0
    return f1 - 0.10 * hallu_rate


def main() -> int:
    ap = argparse.ArgumentParser(description="构造 DPO 偏好数据")
    ap.add_argument("--n", type=int, default=200, help="源样本数（按 stem）")
    ap.add_argument("--k", type=int, default=4, help="每张图采样几个输出")
    ap.add_argument("--adapter", default="outputs/sft_v1/lora")
    ap.add_argument("--out", default="data/processed/dpo_train.jsonl")
    ap.add_argument("--temperature", type=float, default=1.3,
                    help="必须 >1，否则是在锐化分布、采样不出多样性。"
                         "模型 train loss 已低到 0.017，temp=0.9 时 4 次采样会完全一致")
    ap.add_argument("--top-p", type=float, default=0.98)
    ap.add_argument("--min-gap", type=float, default=0.02,
                    help="chosen/rejected reward 分差下限。注意量级：每张图约 33 个字段，"
                         "错 1 个字段的 F1 差只有 0.030，阈值给 0.05 等于要求错 2 个以上")
    ap.add_argument("--levels", default="heavy",
                    help="只用哪些退化档。medium 上模型太稳、采样无差异，"
                         "heavy 才是错误集中且可区分的地方")
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--resume", action="store_true",
                    help="断点续跑：读 <out>.done journal，跳过已处理过的源样本。"
                         "长任务（2h）无人值守时建议加上")
    args = ap.parse_args()

    levels = {x.strip() for x in args.levels.split(",") if x.strip()}

    # ---------------- 数据 ----------------
    recs_all = [json.loads(l) for l in
                (PROCESSED / "train.jsonl").open(encoding="utf-8") if l.strip()]
    pool = [r for r in recs_all if r["meta"]["level"] in levels]
    by_stem: dict[str, list[dict]] = {}
    for r in pool:
        by_stem.setdefault(r["meta"]["stem"], []).append(r)

    stems = sorted(by_stem)
    rng = random.Random(args.seed)
    if len(stems) > args.n:
        stems = rng.sample(stems, args.n)
        stems.sort()
    print(f"  源样本 {len(stems)} 个（档位: {sorted(levels)}）"
          f"，每图采样 k={args.k}")

    # ---------------- 模型 ----------------
    import torch
    from PIL import Image
    from unsloth import FastVisionModel

    adapter = ROOT / args.adapter
    if not adapter.exists():
        print(f"[FATAL] LoRA 不存在: {adapter}")
        return 1

    print(f"  加载 {adapter} ...")
    model, processor = FastVisionModel.from_pretrained(
        str(adapter), load_in_4bit=True, max_seq_length=2048)
    FastVisionModel.for_inference(model)

    prompt = pool[0]["messages"][0]["content"][-1]["text"]

    # ---------------- 采样 ----------------
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_pairs, n_skip_gap, n_skip_dup = 0, 0, 0
    stats: list[dict] = []
    t0 = time.time()

    # 断点续跑：用 sidecar journal 记录「处理过的源样本」，而不是只记「产出过对的」。
    # 否则被丢弃的样本（无差异/分差小）下次会重跑一遍，白白浪费算子。
    journal = out_path.with_suffix(out_path.suffix + ".done")
    done_stems: set[str] = set()
    resume_mode = args.resume and journal.exists()
    if resume_mode:
        done_stems = {l.strip() for l in journal.open(encoding="utf-8") if l.strip()}
        print(f"  --resume：已处理 {len(done_stems)} 个源样本，跳过")

    with out_path.open("a" if resume_mode else "w", encoding="utf-8") as fout, \
            journal.open("a" if resume_mode else "w", encoding="utf-8") as fdone:
        for i, stem in enumerate(stems, 1):
            if stem in done_stems:
                continue
            recs = by_stem[stem]
            # 同一底图的三档取一张（优先 medium），避免偏好数据里重复
            rec = sorted(recs, key=lambda r: (r["meta"]["level"] != "medium"))[0]
            gt = rec["gt"]

            img = Image.open(rec["image"]).convert("RGB").resize(
                (args.image_size, args.image_size))
            msg = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ]}]
            text = processor.apply_chat_template(msg, add_generation_prompt=True)
            inputs = processor(images=[img], text=text, return_tensors="pt").to("cuda")
            n_in = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outs = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True, temperature=args.temperature,
                    top_p=args.top_p, num_return_sequences=args.k,
                )

            samples: list[tuple[float, str]] = []
            for o in outs:
                txt = processor.decode(o[n_in:], skip_special_tokens=True)
                samples.append((reward_for(gt, txt), txt))

            samples.sort(key=lambda x: x[0], reverse=True)
            best_r, best_t = samples[0]
            worst_r, worst_t = samples[-1]

            # 全部输出一模一样 → 无区分度，丢弃
            if best_r == worst_r and best_t == worst_t:
                n_skip_dup += 1
            elif best_r - worst_r < args.min_gap:
                n_skip_gap += 1
            else:
                fout.write(json.dumps({
                    "image": str(Path(rec["image"]).relative_to(ROOT)).replace("\\", "/"),
                    "prompt": prompt,
                    "chosen": best_t,
                    "rejected": worst_t,
                    "meta": {
                        "stem": stem,
                        "level": rec["meta"]["level"],
                        "reward_chosen": round(best_r, 4),
                        "reward_rejected": round(worst_r, 4),
                        "reward_mean": round(
                            sum(s[0] for s in samples) / len(samples), 4),
                        "n_samples": len(samples),
                    },
                }, ensure_ascii=False) + "\n")
                n_pairs += 1

            stats.append({"gap": best_r - worst_r, "best": best_r})

            # 立即落盘 + 记 journal：机器中途崩/被杀也能续，最多丢当前这一张
            fout.flush()
            fdone.write(stem + "\n")
            fdone.flush()

            if i % 10 == 0 or i == len(stems):
                el = time.time() - t0
                print(f"    {i}/{len(stems)}  已产出 {n_pairs} 对"
                      f"  {el:.0f}s ({el/i:.1f}s/图)", flush=True)

    # ---------------- 报告 ----------------
    print()
    print("=" * 62)
    print("偏好数据构造完成")
    print("=" * 62)
    print(f"  输出       : {out_path}")
    print(f"  偏好对     : {n_pairs}")
    print(f"  丢弃(无差异): {n_skip_dup}")
    print(f"  丢弃(分差小): {n_skip_gap}  (阈值 {args.min_gap})")
    if stats:
        gaps = sorted(s["gap"] for s in stats)
        cc = sum(1 for s in stats if s["best"] < 0.5)
        print(f"  平均分差   : {sum(gaps)/len(gaps):.4f}"
              f"  中位 {gaps[len(gaps)//2]:.4f}"
              f"  最大 {gaps[-1]:.4f}")
        print(f"  最优样本 reward < 0.5 的比例: {cc/len(stats):.1%}"
              f"   ← 越高说明模型在该图上越不靠谱")
    print(f"  用时       : {(time.time()-t0)/60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
