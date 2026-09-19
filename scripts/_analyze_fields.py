"""按字段拆解两个（或多个）评测结果，用来做「同口径、同数据」的对照分析。

用法：
    python scripts/_analyze_fields.py base=outputs/eval_wr_sft_8f_preds.jsonl \
                                      strict=outputs/eval_wr_sft_8f_strict_preds.jsonl

口径与 evaluate.py 完全一致（复用 parse_json / flatten_flat / normalize），
避免「自己重写一遍打分」导致数字对不上。

输出两部分：
  1) 总体：tp/fp/fn、P/R/F1、平均每张预测键数 vs 真值键数
  2) 每字段：真值出现数 / 预测出现数 / tp / fp / fn / **无中生有数**（GT 里没有该键却预测了）
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import flatten_flat, parse_json  # noqa: E402

FIELDS = ["store_name", "store_addr", "tel", "date", "time", "subtotal", "tax", "total"]


def keyf(s: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(s).lower())


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def analyze(path: Path) -> dict:
    recs = load(path)
    tp = fp = fn = 0
    n_pred = n_gt = n_bad_json = 0
    per = {f: dict(tp=0, fp=0, fn=0, gt_n=0, pred_n=0, fab=0) for f in FIELDS}

    for r in recs:
        pred = parse_json(r.get("pred") or "")
        if pred is None:
            n_bad_json += 1
            pred = {}
        g, p = flatten_flat(r["gt"], pred)
        hit = sum(1 for k, v in g.items() if p.get(k) == v)
        tp += hit
        fn += len(g) - hit
        fp += sum(1 for k, v in p.items() if g.get(k) != v)
        n_pred += len(p)
        n_gt += len(g)

        for f in FIELDS:
            k = keyf(f)
            ing, inp = k in g, k in p
            if ing:
                per[f]["gt_n"] += 1
            if inp:
                per[f]["pred_n"] += 1
            if ing and inp and g[k] == p[k]:
                per[f]["tp"] += 1
            if ing and not (inp and g[k] == p[k]):
                per[f]["fn"] += 1
            if inp and not (ing and g[k] == p[k]):
                per[f]["fp"] += 1
            if inp and not ing:
                per[f]["fab"] += 1

    n = len(recs) or 1
    return dict(n=len(recs), tp=tp, fp=fp, fn=fn,
                p=tp / (tp + fp) if tp + fp else 0.0,
                r=tp / (tp + fn) if tp + fn else 0.0,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                avg_pred=n_pred / n, avg_gt=n_gt / n,
                bad_json=n_bad_json, per=per)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    runs = []
    for a in sys.argv[1:]:
        label, _, p = a.partition("=")
        path = ROOT / (p or label)
        if not path.exists():
            print(f"[!] 找不到 {path}")
            return 1
        runs.append((label, analyze(path)))

    print(f"{'组':<10}{'n':>4}{'tp':>6}{'fp':>6}{'fn':>6}{'P':>8}{'R':>8}{'F1':>8}"
          f"{'键/图':>8}{'真值键':>8}{'坏JSON':>8}")
    for label, x in runs:
        print(f"{label:<10}{x['n']:>4}{x['tp']:>6}{x['fp']:>6}{x['fn']:>6}"
              f"{x['p']:>8.4f}{x['r']:>8.4f}{x['f1']:>8.4f}"
              f"{x['avg_pred']:>8.2f}{x['avg_gt']:>8.2f}{x['bad_json']:>8}")

    for f in FIELDS:
        print(f"\n--- {f} ---")
        print(f"{'组':<10}{'真值有':>8}{'预测有':>8}{'tp':>6}{'fp':>6}{'fn':>6}{'无中生有':>10}{'占预测':>8}")
        for label, x in runs:
            d = x["per"][f]
            pct = d["fab"] / d["pred_n"] if d["pred_n"] else 0.0
            print(f"{label:<10}{d['gt_n']:>8}{d['pred_n']:>8}{d['tp']:>6}{d['fp']:>6}"
                  f"{d['fn']:>6}{d['fab']:>10}{pct:>8.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
