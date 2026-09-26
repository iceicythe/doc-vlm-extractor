"""偏好数据质量体检：chosen 本身有多正确？

偏好对要求 chosen 在固定标准下优于 rejected；两者都可能带错。
本脚本复用主评分器，分别检查相对排序和 chosen 的整单正确性。

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


from scoring import (flatten_object as flatten, parse_json, score_pred, score_one,
                     prf, SCORER_VERSION)


def as_dict(value, fallback_obj=None):
    return value if isinstance(value, dict) else parse_json(value)


def field_f1(pred, gt):
    result = score_pred(gt, pred)
    return prf(result["tp"], result["fp"], result["fn"])[2], result["tp"], result["fp"], result["fn"]


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
        if gt is None:
            miss += 1
            continue
        rows.append((stem, gt, ch, rj, r.get("meta") or {}, r.get("chosen"), r.get("rejected")))
    print(f"可用偏好对：{len(rows)}（跳过 {miss}）\n")

    n = len(rows)
    if not n:
        print("无可用 GT 匹配，未生成评分")
        return 1
    print(f"评分版本：{SCORER_VERSION}；无法解析的预测按失败保留")
    better = tied = worse = 0
    ch_f1, rj_f1 = [], []
    ch_perfect = ch_wrong = 0
    both_wrong_same = 0
    diff_total = 0          # chosen/rejected 取值不同的字段总数
    diff_chosen_right = 0   # 其中 chosen 正确的（有效偏好信号）
    diff_chosen_wrong = 0   # 其中 chosen 也错的（噪声对）
    drop = Counter()

    for stem, gt, ch, rj, meta, raw_ch, raw_rj in rows:
        f1c, *_ = field_f1(ch, gt)
        f1r, *_ = field_f1(rj, gt)
        ch_f1.append(f1c)
        rj_f1.append(f1r)
        better += f1c > f1r
        tied += f1c == f1r
        worse += f1c < f1r
        ch_score = score_one(gt, raw_ch) if isinstance(raw_ch, str) else score_pred(gt, ch)
        if ch_score["document_correct"]:
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
    print(f"chosen 整单正确的对数 : {ch_perfect:>4} / {n}  ({ch_perfect/n*100:.1f}%)")
    print(f"chosen 自身带错的对数         : {ch_wrong:>4} / {n}  ({ch_wrong/n*100:.1f}%)")
    print()
    print(f"chosen/rejected 取值不同的字段总数 : {diff_total}")
    print(f"  · chosen 在此字段正确（有效信号）: {diff_chosen_right:>4} "
          f"({diff_chosen_right/max(1,diff_total)*100:.1f}%)")
    print(f"  · chosen 在此字段也错（噪声对）  : {diff_chosen_wrong:>4} "
          f"({diff_chosen_wrong/max(1,diff_total)*100:.1f}%)")
    print()
    print(f"chosen 与 rejected 同字段『都错且错法相同』: {both_wrong_same} 个字段")
    if drop:
        print("\n噪声对集中的字段：")
        for k, v in drop.most_common(8):
            print(f"  {k:<14} {v}")

    print("\n" + "=" * 64)
    print(f"chosen 字段 F1 优于/持平/劣于 rejected：{better}/{tied}/{worse}，分母 {n}")
    print("相对排序与整单正确分开报告；仅凭这些统计不能归因 DPO 收益或梯度方向。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
