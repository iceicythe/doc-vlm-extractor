"""偏好数据质量体检：chosen 本身有多正确？

DPO 的有效前提是「chosen 是对的、rejected 是错的」。
但本项目的偏好对来自**同一个模型的多次采样排序** —— 如果 chosen 只是
「4 次采样里相对最好的那一次」，它本身仍可能带错，那 DPO 学到的就只是
「错法 A 优于错法 B」，而不是「对优于错」。

用法
----
    python src/check_pref_quality.py
    python src/check_pref_quality.py --pref data/processed/dpo_train.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def flatten(obj) -> dict:
    d: dict[str, object] = {}
    for k in ("单据类型", "表头"):
        v = obj.get(k)
        if isinstance(v, dict):
            for kk, vv in v.items():
                d[f"{k}.{kk}"] = vv
        else:
            d[k] = v
    v = obj.get("合计")
    if isinstance(v, dict):
        for kk, vv in v.items():
            d[f"合计.{kk}"] = vv
    for j, row in enumerate(obj.get("明细") or []):
        if isinstance(row, dict):
            for kk, vv in row.items():
                d[f"明细[{j}].{kk}"] = vv
    return d


def as_dict(v, fallback_obj=None):
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            o = json.loads(v)
            return o if isinstance(o, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def field_f1(pred: dict, gt: dict) -> tuple[float, int, int, int]:
    """字段级 micro-F1（与 evaluate.py 同口径）。"""
    fp_, fg = flatten(pred), flatten(gt)
    tp = sum(1 for k in set(fp_) & set(fg) if fp_[k] == fg[k])
    fp = len(fp_) - tp
    fn = len(fg) - tp
    denom = 2 * tp + fp + fn
    return (2 * tp / denom if denom else 1.0), tp, fp, fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pref", default="data/processed/dpo_train.jsonl")
    ap.add_argument("--train", default="data/processed/train.jsonl",
                    help="GT 来源（按 stem 匹配）")
    args = ap.parse_args()

    # GT: stem -> gt
    gt_by_stem: dict[str, dict] = {}
    for line in (ROOT / args.train).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        stem = (r.get("meta") or {}).get("stem")
        g = as_dict(r.get("gt"))
        if stem and g is not None:
            gt_by_stem.setdefault(stem, g)
    print(f"GT 索引：{len(gt_by_stem)} 个 stem\n")

    rows = []
    miss = 0
    for line in (ROOT / args.pref).open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        stem = (r.get("meta") or {}).get("stem")
        gt = gt_by_stem.get(stem)
        ch = as_dict(r.get("chosen"))
        rj = as_dict(r.get("rejected"))
        if gt is None or ch is None or rj is None:
            miss += 1
            continue
        rows.append((stem, gt, ch, rj, r.get("meta") or {}))
    print(f"可用偏好对：{len(rows)}（跳过 {miss}）\n")

    n = len(rows)
    ch_f1, rj_f1 = [], []
    ch_perfect = ch_wrong = 0
    both_wrong_same = 0
    diff_total = 0          # chosen/rejected 取值不同的字段总数
    diff_chosen_right = 0   # 其中 chosen 正确的（有效偏好信号）
    diff_chosen_wrong = 0   # 其中 chosen 也错的（噪声对）
    drop = Counter()

    for stem, gt, ch, rj, meta in rows:
        f1c, *_ = field_f1(ch, gt)
        f1r, *_ = field_f1(rj, gt)
        ch_f1.append(f1c)
        rj_f1.append(f1r)
        if f1c >= 0.99999:
            ch_perfect += 1
        else:
            ch_wrong += 1

        fc, fr, fg = flatten(ch), flatten(rj), flatten(gt)
        for k in set(fc) | set(fr):
            if fc.get(k) == fr.get(k):
                continue
            diff_total += 1
            ok_c = (k in fg and fc.get(k) == fg[k])
            ok_r = (k in fg and fr.get(k) == fg[k])
            if ok_c and not ok_r:
                diff_chosen_right += 1
            elif not ok_c:
                diff_chosen_wrong += 1
                drop[k.split(".")[-1]] += 1

        # chosen 与 rejected 在同一字段「都错且错法相同」
        for k in set(fc) & set(fr) & set(fg):
            if fc[k] != fg[k] and fr[k] != fg[k] and fc[k] == fr[k]:
                both_wrong_same += 1

    avg = lambda a: sum(a) / len(a) if a else 0.0

    print("=" * 64)
    print(f"chosen   F1 均值 {avg(ch_f1):.4f}   最低 {min(ch_f1):.4f}")
    print(f"rejected F1 均值 {avg(rj_f1):.4f}   最低 {min(rj_f1):.4f}")
    print(f"平均分差 {avg(ch_f1) - avg(rj_f1):.4f}")
    print()
    print(f"chosen 完全正确（F1=1）的对数 : {ch_perfect:>4} / {n}  ({ch_perfect/n*100:.1f}%)")
    print(f"chosen 自身带错的对数         : {ch_wrong:>4} / {n}  ({ch_wrong/n*100:.1f}%)")
    print()
    print(f"chosen/rejected 取值不同的字段总数 : {diff_total}")
    print(f"  · chosen 在此字段正确（有效信号）: {diff_chosen_right:>4} "
          f"({diff_chosen_right/diff_total*100:.1f}%)")
    print(f"  · chosen 在此字段也错（噪声对）  : {diff_chosen_wrong:>4} "
          f"({diff_chosen_wrong/diff_total*100:.1f}%)")
    print()
    print(f"chosen 与 rejected 同字段『都错且错法相同』: {both_wrong_same} 个字段")
    if drop:
        print("\n噪声对集中的字段：")
        for k, v in drop.most_common(8):
            print(f"  {k:<14} {v}")

    print("\n" + "=" * 64)
    print("判读：")
    if ch_wrong / n > 0.5:
        print(f"  ⚠ {ch_wrong/n*100:.0f}% 的 chosen 自身带错 —— DPO 在学"
              "「哪种错更轻」，而非「对优于错」，这解释了泛化收益接近于零。")
    if diff_chosen_right + diff_chosen_wrong:
        r = diff_chosen_right / (diff_chosen_right + diff_chosen_wrong)
        print(f"  有效偏好信号占比 {r*100:.1f}% —— 只有这部分梯度指向正确答案。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
