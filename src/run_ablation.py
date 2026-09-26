"""消融实验驱动 —— 串行训练 + 评测，汇总成对比表。

设计原则
--------
1. **固定评测子集**：所有实验都在同一批 102 条（三档各 34）上评测，
   否则不同实验的数字没法横向比。
2. **控制变量**：
   - 分辨率实验 → 固定 max_steps=600（相同计算预算）
   - 数据量实验 → max_steps 按比例缩放（保持 ≈2.3 epoch，才是真正的"数据效率曲线"）
   - LoRA 位置实验 → 固定 max_steps=600
3. **失败不中断**：某个实验 OOM/报错就记录原因继续跑下一个。

用法
----
    # 看实验清单
    venv-gld/Scripts/python.exe src/run_ablation.py --list

    # 跑全部（耗时数小时，建议睡前启动）
    venv-gld/Scripts/python.exe src/run_ablation.py

    # 只跑指定实验
    venv-gld/Scripts/python.exe src/run_ablation.py --only abl_r512 abl_d50

    # 只重建评测子集
    venv-gld/Scripts/python.exe src/run_ablation.py --build-subset
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
PY = str(ROOT / "venv-gld" / "Scripts" / "python.exe")

SUBSET_FILE = PROCESSED / "test_ablation.jsonl"
# 三档各取多少条。评测是整批实验的时间大头（每条 15-20 秒），
# 100 条量级已足够支撑对照结论（此前经验：样本量 <50 时数字不可信）。
SUBSET_PER_LEVEL = 34
EVAL_BATCH = 6          # 评测批大小；evaluate.py 内有 OOM 自动降级兜底
from scoring import SCORER_VERSION
SCORED = OUTPUTS / ("scoring-v" + SCORER_VERSION)
SUMMARY_FILE = SCORED / "ablation_summary.json"

# 基线：已有的 sft_v1（384px / 100% 数据 / 600 步 / 全层）
BASELINE = {
    "tag": "v1",
    "label": "基线 384px / 100% / 全层",
    "adapter": "outputs/sft_v1/lora",
    "image_size": 384,
}

# (tag, 中文标签, 训练参数, 评测用的 image_size)
# 顺序 = 执行顺序，按训练成本递增排。
# 最贵的 r768 放最后：万一时间不够、或 r512 已证明高分辨率无收益，可以中途叫停而不浪费。
EXPERIMENTS = [
    ("abl_d25", "数据量 25%",
     ["--data-ratio", "0.25", "--max-steps", "150"], 384),
    ("abl_d50", "数据量 50%",
     ["--data-ratio", "0.5", "--max-steps", "300"], 384),
    ("abl_r512", "分辨率 512px",
     ["--image-size", "512", "--max-steps", "600"], 512),
    ("abl_lang", "仅训语言层",
     ["--no-vision-lora", "--max-steps", "600"], 384),
    ("abl_r768", "分辨率 768px",
     ["--image-size", "768", "--max-steps", "600"], 768),
]


# ---------------------------------------------------------------- 评测子集
def build_subset() -> int:
    """从 test.jsonl 抽固定子集：三档各 N 条，seed 固定。"""
    src = PROCESSED / "test.jsonl"
    if not src.exists():
        print(f"[FATAL] 缺少 {src}")
        return 1

    recs = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
    by_level: dict[str, list[dict]] = {}
    for r in recs:
        by_level.setdefault(r["meta"]["level"], []).append(r)

    rng = random.Random(3407)
    picked: list[dict] = []
    for lv in ("clean", "medium", "heavy"):
        pool = sorted(by_level.get(lv, []), key=lambda r: r["meta"]["stem"])
        if len(pool) < SUBSET_PER_LEVEL:
            print(f"  [WARN] {lv} 只有 {len(pool)} 条，全部取用")
            picked.extend(pool)
        else:
            picked.extend(rng.sample(pool, SUBSET_PER_LEVEL))

    picked.sort(key=lambda r: (r["meta"]["level"], r["meta"]["stem"]))
    with SUBSET_FILE.open("w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[子集已建] {SUBSET_FILE.name}  {len(picked)} 条"
          f"（三档各 {SUBSET_PER_LEVEL}）")
    return 0


def ensure_subset() -> int:
    if SUBSET_FILE.exists():
        n = sum(1 for l in SUBSET_FILE.open(encoding="utf-8") if l.strip())
        print(f"[子集] 复用已有 {SUBSET_FILE.name}（{n} 条）")
        return 0
    return build_subset()


# ---------------------------------------------------------------- 执行
def run_train(tag: str, extra: list[str], log_dir: Path) -> tuple[bool, float]:
    log = log_dir / f"train_{tag}.log"
    cmd = [PY, "-u", str(ROOT / "src" / "train_sft.py"),
           "--tag", tag, *extra]
    print(f"    $ {' '.join(cmd[2:])}")
    t0 = time.time()
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    ok = proc.returncode == 0 and (OUTPUTS / f"sft_{tag}" / "lora").exists()
    print(f"    训练{'完成' if ok else '失败'}  用时 {dt/60:.1f} 分钟"
          f"  日志 {log.name}")
    return ok, dt


def run_eval(tag: str, adapter: str, image_size: int, log_dir: Path) -> dict | None:
    out_tag = f"abl_{tag}"
    log = log_dir / f"eval_{tag}.log"
    cmd = [PY, "-u", str(ROOT / "src" / "evaluate.py"),
           "--data", str(SUBSET_FILE), "--mode", "schema",
           "--limit", "0", "--batch", str(EVAL_BATCH),
           "--image-size", str(image_size),
           "--tag", out_tag]
    if adapter:
        cmd += ["--adapter", adapter]
    t0 = time.time()
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0

    res_file = SCORED / f"eval_{out_tag}.json"
    if proc.returncode != 0 or not res_file.exists():
        print(f"    评测失败  日志 {log.name}")
        return None
    res = json.loads(res_file.read_text(encoding="utf-8"))
    res["_eval_seconds"] = round(dt, 1)
    print(f"    评测完成  F1={res.get('f1', 0):.4f}"
          f"  额外字段={res.get('extra_field_sample_rate', 0):.2%}"
          f"  用时 {dt/60:.1f} 分钟")
    return res


def collect_levels(tag: str, log_dir: Path) -> dict:
    """从评测日志里补出分档 F1（evaluate.py 会逐档打印）。"""
    log = log_dir / f"eval_{tag}.log"
    out: dict[str, float] = {}
    if not log.exists():
        return out
    for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        for lv in ("clean", "medium", "heavy"):
            if line.startswith(lv) and "F1=" in line:
                try:
                    out[lv] = float(line.split("F1=")[1].split()[0])
                except (IndexError, ValueError):
                    pass
    return out


# ---------------------------------------------------------------- 汇总
def load_summary() -> dict:
    if SUMMARY_FILE.exists():
        return json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    return {}


def save_summary(data: dict) -> None:
    SCORED.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def print_table(data: dict) -> None:
    print()
    print("=" * 78)
    print("消融实验汇总")
    print("=" * 78)
    print(f"{'实验':<22}{'F1':>8}{'精确率':>9}{'召回率':>9}"
          f"{'额外字段率':>9}{'合法率':>8}{'耗时(分)':>10}")
    print("-" * 78)
    for tag, r in data.items():
        if r.get("failed"):
            print(f"{r['label']:<22}{'FAILED':>8}  {r.get('error', '')[:34]}")
            continue
        print(f"{r['label']:<22}{r['f1']:>8.4f}{r['precision']:>9.4f}"
              f"{r['recall']:>9.4f}{r['extra_field_sample_rate']:>9.2%}"
              f"{r['json_valid_rate']:>8.2%}"
              f"{r.get('train_minutes', 0):>10.1f}")
    print("=" * 78)
    print(f"评测子集：{SUBSET_FILE.name}（三档各 {SUBSET_PER_LEVEL} 条）")


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="消融实验驱动")
    ap.add_argument("--list", action="store_true", help="只列实验清单")
    ap.add_argument("--only", nargs="*", default=None, help="只跑指定 tag")
    ap.add_argument("--build-subset", action="store_true", help="只重建评测子集")
    ap.add_argument("--include-baseline", action="store_true",
                    help="同时重跑基线（默认复用已有 eval）")
    ap.add_argument("--table", action="store_true", help="只打印汇总表")
    args = ap.parse_args()

    if args.table:
        print_table(load_summary())
        return 0

    if args.build_subset:
        return build_subset()

    if args.list:
        print("评测子集：", SUBSET_FILE.name, "（三档各", SUBSET_PER_LEVEL, "条）")
        print()
        print(f"{'tag':<12}{'实验':<20}{'训练参数'}")
        print("-" * 62)
        for tag, label, extra, _ in EXPERIMENTS:
            print(f"{tag:<12}{label:<20}{' '.join(extra)}")
        return 0

    log_dir = OUTPUTS / "ablation_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if ensure_subset() != 0:
        return 1

    todo = EXPERIMENTS
    if args.only:
        todo = [e for e in EXPERIMENTS if e[0] in set(args.only)]
        if not todo:
            print(f"[FATAL] 没有匹配的实验：{args.only}")
            return 1

    summary = load_summary()
    t_start = time.time()

    # --- 基线 ---
    if args.include_baseline and "v1" not in summary:
        print("\n[基线] 384px / 100% / 全层（复用已训练好的 LoRA）")
        res = run_eval("base", BASELINE["adapter"], BASELINE["image_size"], log_dir)
        if res:
            summary["v1"] = {"label": BASELINE["label"], **{k: res[k] for k in
                             ("f1", "precision", "recall", "json_valid_rate",
                              "extra_field_sample_rate")},
                             "train_minutes": 58.7,
                             "by_level": collect_levels("base", log_dir)}
            save_summary(summary)

    for i, (tag, label, extra, img_size) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {label}  ({tag})")
        if tag in summary and not summary[tag].get("failed"):
            print("    已有结果，跳过（如需重跑请手动删除 ablation_summary.json 中的条目）")
            continue

        ok, train_min = run_train(tag, extra, log_dir)
        if not ok:
            tail = ""
            log = log_dir / f"train_{tag}.log"
            if log.exists():
                tail = log.read_text(encoding="utf-8", errors="ignore")[-400:]
                tail = tail.replace("\n", " ")[-180:]
            summary[tag] = {"label": label, "failed": True,
                            "error": tail or "训练未产出 LoRA"}
            save_summary(summary)
            continue

        adapter = f"outputs/sft_{tag}/lora"
        res = run_eval(tag, adapter, img_size, log_dir)
        if res is None:
            summary[tag] = {"label": label, "failed": True,
                            "error": "评测失败", "train_minutes": train_min / 60}
        else:
            summary[tag] = {
                "label": label,
                **{k: res[k] for k in ("f1", "precision", "recall",
                                       "json_valid_rate", "extra_field_sample_rate")},
                "train_minutes": round(train_min / 60, 1),
                "by_level": collect_levels(tag, log_dir),
            }
        save_summary(summary)

    print(f"\n总耗时 {(time.time() - t_start)/60:.1f} 分钟")
    print_table(summary)
    print(f"\n汇总文件：{SUMMARY_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
