"""临时验证：按 stem 抽样是否正常工作（不涉及模型加载）。"""
import random
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "check_sample.log"
lines: list[str] = []


def log(s: str) -> None:
    print(s, flush=True)
    lines.append(s)


ds = load_dataset("json", data_files=str(ROOT / "data/processed/train.jsonl"), split="train")
log(f"原始记录: {len(ds)}  列: {ds.column_names}")

stems = sorted({r["stem"] for r in ds["meta"]})
log(f"源样本数: {len(stems)}")

for ratio in (0.25, 0.5):
    keep_n = max(1, int(round(len(stems) * ratio)))
    rng = random.Random(3407)
    keep = set(rng.sample(stems, keep_n))
    sub = ds.filter(lambda m: m["meta"]["stem"] in keep)
    lv = Counter(r["level"] for r in sub["meta"])
    log(f"  ratio={ratio:.2f}: {len(keep)}/{len(stems)} 源样本 -> {len(sub)} 条  {dict(lv)}")

log("OK — 抽样逻辑正常")
OUT.write_text("\n".join(lines), encoding="utf-8")
sys.exit(0)
