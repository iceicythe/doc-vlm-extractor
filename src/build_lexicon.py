"""构建标准词表 —— 知识注入的「知识」来源。

词表 = train.jsonl 的 GT 值域（逐字段的闭集枚举）。

关键设计：归一化口径**复用 src/evaluate.py 的 normalize**，而不是另写一份。
因为「词表命中」必须和「打分时判等」用同一把尺子 —— 否则会出现
「命中了词表、分数却没变」这种自欺欺人的结果。

用法：
    python src/build_lexicon.py                          # → data/lexicon/v1.json
    python src/build_lexicon.py --data data/processed/train.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUTDIR = ROOT / "data" / "lexicon"

# 字段路径（与 evaluate.py 的摊平口径一致，便于下游对齐）
HEAD_FIELDS = ["项目名称", "供应商", "单据编号", "日期"]
ROW_FIELDS = ["名称", "规格型号", "单位"]


def load_evaluate():
    """按路径加载 evaluate.py（顶层不 import torch，代价很小）。"""
    spec = importlib.util.spec_from_file_location("_ev", ROOT / "src" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def harvest(records: list[dict], field: str, is_row: bool) -> Counter:
    """收集某字段在 GT 里出现过的所有原始写法的频次。"""
    c: Counter = Counter()
    for rec in records:
        gt = rec.get("gt") or {}
        if is_row:
            for item in (gt.get("明细") or []):
                if isinstance(item, dict):
                    v = item.get(field)
                    if v is not None and str(v).strip():
                        c[str(v)] += 1
        else:
            v = (gt.get("表头") or {}).get(field)
            if v is not None and str(v).strip():
                c[str(v)] += 1
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description="从训练集 GT 构建标准词表")
    ap.add_argument("--data", default=str(PROCESSED / "train.jsonl"))
    ap.add_argument("--out", default=str(OUTDIR / "v1.json"))
    args = ap.parse_args()

    ev = load_evaluate()
    normalize = ev.normalize

    src = Path(args.data)
    if not src.exists():
        print(f"[FATAL] 找不到 {src}")
        return 1
    records = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]

    fields: dict[str, dict] = {}
    for path, field, is_row in (
        [(f"表头.{f}", f, False) for f in HEAD_FIELDS]
        + [(f"明细.{f}", f, True) for f in ROW_FIELDS]
    ):
        raw_counter = harvest(records, field, is_row)
        if not raw_counter:
            continue

        # 归一化后聚簇：多种原始写法（"m3" / "M3"）落到同一个规范键上，
        # canonical 取该簇里出现次数最多的原始写法（人工读起来最自然）。
        clusters: dict[str, Counter] = {}
        for raw, n in raw_counter.items():
            key = normalize(raw)
            if key:
                clusters.setdefault(key, Counter())[raw] += n

        cluster_n = {k: sum(c.values()) for k, c in clusters.items()}
        fields[path] = {
            "n_raw": len(raw_counter),
            "n_norm": len(clusters),
            # 归一化值 → 规范原文（同簇中出现最多的写法）
            "values": {k: c.most_common(1)[0][0] for k, c in sorted(clusters.items())},
            # 出现频次最高的几个值，仅供人工核对词表是否合理
            "top3": [
                clusters[k].most_common(1)[0][0]
                for k, _ in sorted(cluster_n.items(), key=lambda kv: -kv[1])[:3]
            ],
        }

    out = {
        "version": Path(args.out).stem,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(src),
        "n_source": len(records),
        "fields": fields,
    }
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print("标准词表构建完成")
    print("=" * 70)
    print(f"  来源 : {src}  {len(records)} 条")
    print(f"  输出 : {dst}")
    print()
    print(f"  {'字段':<16}{'原始写法':>8}{'归一后':>8}   高频样例")
    print("  " + "-" * 62)
    for path, d in fields.items():
        sample = " / ".join(str(v)[:12] for v in d.get("top3", [])[:2])
        print(f"  {path:<16}{d['n_raw']:>8}{d['n_norm']:>8}   {sample}")
    print()
    print("  注：供应商 / 单据编号 逐条随机生成（几乎无重复），虽在词表中但")
    print("      测试集命中率 ≈0 —— 注入时应显式跳过，见 inject_knowledge.py。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
