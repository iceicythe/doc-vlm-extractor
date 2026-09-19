"""错误一致性分析：区分「系统性错误」与「随机误差」

动机
----
消融知道残余错误是字符级误读，DPO 知道要对它下手 —— 但 DPO 跑完几乎没动。
一个关键问题没回答：**这些误读在多次独立推理之间稳不稳定？**

- 若同一张图、同一字段在多个配置下**总是错成同一个值** → 系统性错误，
  存在可被偏好学习捕捉的稳定模式（DPO 理论上能修）。
- 若同一字段在配置间**对/错交替、错值各不相同** → 属于决策边界上的
  随机翻转，不构成可泛化的行为倾向（DPO 学不到东西）。

用法
----
    # 默认比较四个同为 384px 的配置（唯一变量是训练方式，不含分辨率混淆）
    python src/analyze_error_consistency.py

    # 指定文件
    python src/analyze_error_consistency.py outputs/a_preds.jsonl outputs/b_preds.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# 默认比较组：同为 384px，排除分辨率这个已知主导因素
DEFAULT = [
    ("SFT 100%", "outputs/eval_abl_base_preds.jsonl"),
    ("SFT 50%", "outputs/eval_abl_abl_d50_preds.jsonl"),
    ("SFT 25%", "outputs/eval_abl_abl_d25_preds.jsonl"),
    ("仅语言层", "outputs/eval_abl_abl_lang_preds.jsonl"),
]


def flatten(obj) -> dict:
    """把嵌套 schema 摊平成 {字段路径: 值}。明细数组按索引对齐。"""
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


def _as_dict(v):
    """preds.jsonl 里 gt/pred 以 JSON 字符串存储（也可能是已解析的 dict）。"""
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            o = json.loads(v)
        except json.JSONDecodeError:
            return None
        return o if isinstance(o, dict) else None
    return None


def load(path: Path) -> dict[str, dict]:
    """→ {stem: {"gt": flat, "pred": flat, "level": str}}"""
    recs = {}
    skipped = 0
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("json_valid"):
            continue
        g, pr = _as_dict(r.get("gt")), _as_dict(r.get("pred"))
        if g is None or pr is None:
            skipped += 1
            continue
        recs[r["stem"]] = {
            "gt": flatten(g),
            "pred": flatten(pr),
            "level": r.get("level", "?"),
        }
    if skipped:
        print(f"  {path.name}: 跳过 {skipped} 条无法解析的记录")
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="preds.jsonl 路径；留空用默认四组")
    ap.add_argument("--out", default="outputs/report_error_consistency.md")
    args = ap.parse_args()

    if args.files:
        groups = [(Path(f).stem, f) for f in args.files]
    else:
        groups = DEFAULT

    data: list[tuple[str, dict[str, dict]]] = []
    for label, rel in groups:
        p = Path(rel)
        if not p.is_absolute():
            p = ROOT / rel
        if not p.exists():
            print(f"  跳过（不存在）：{p}")
            continue
        data.append((label, load(p)))
    if len(data) < 2:
        print("至少需要两个可读的预测文件")
        return 1

    labels = [lb for lb, _ in data]
    stems = sorted(set.intersection(*(set(d) for _, d in data)))
    print(f"比较 {len(data)} 个配置，共同样本 {len(stems)} 张\n")

    # ---- 每个 (stem, field) 在各配置下的对错 ----
    # wrong[key] = set(配置名)，wrong_val[key][配置名] = 预测值
    wrong: dict[tuple[str, str], set[str]] = {}
    wrong_val: dict[tuple[str, str], dict[str, object]] = {}
    n_fields = 0

    for stem in stems:
        gts = data[0][1][stem]["gt"]
        # 只统计在全部配置里都出现的字段（避免行数不一致导致的口径偏差）
        common = set(gts)
        for _, d in data[1:]:
            common &= set(d[stem]["gt"])
        for f in common:
            n_fields += 1
            key = (stem, f)
            for lb, d in data:
                pred = d[stem]["pred"].get(f)
                if pred != gts[f]:
                    wrong.setdefault(key, set()).add(lb)
                    wrong_val.setdefault(key, {})[lb] = pred

    n_wrong_any = len(wrong)
    print(f"字段总数（各配置共有）: {n_fields}")
    print(f"至少一个配置出错的字段: {n_wrong_any} "
          f"({n_wrong_any / n_fields * 100:.2f}%)\n")

    # ---- 1. 每配置错误数 + 两两重叠 ----
    per_cfg = {lb: sum(1 for s in wrong.values() if lb in s) for lb in labels}
    print("各配置错误字段数：")
    for lb in labels:
        print(f"  {lb:<10} {per_cfg[lb]}")

    print("\n两两错误集合重叠（交集 / 并集 = Jaccard）：")
    jac_rows = []
    for a, b in combinations(labels, 2):
        sa = {k for k, s in wrong.items() if a in s}
        sb = {k for k, s in wrong.items() if b in s}
        inter, union = len(sa & sb), len(sa | sb)
        j = inter / union if union else 0.0
        jac_rows.append((a, b, inter, union, j))
        print(f"  {a:<10} ∩ {b:<10} = {inter:>3} / {union:>3}   J={j:.3f}")

    # ---- 2. 按「在几个配置里出错」分层 ----
    layering = Counter(len(s) for s in wrong.values())
    print("\n按出错配置数分层：")
    for k in sorted(layering):
        tag = "稳定性" if k >= 3 else ("中等" if k == 2 else "仅单次")
        print(f"  在 {k} 个配置里出错: {layering[k]:>3} 个字段  ({tag})")

    stable = layering.get(len(data), 0)
    single = layering.get(1, 0)

    # ---- 3. 部分错字段：错值是否一致 ----
    print("\n错值一致性（在 ≥2 个配置里出错的字段）：")
    multi = {k: v for k, v in wrong_val.items() if len(wrong[k]) >= 2}
    same_val = 0
    for k, v in multi.items():
        vals = {repr(x) for x in v.values()}
        if len(vals) == 1:
            same_val += 1
    print(f"  多配置出错字段数: {len(multi)}")
    print(f"  其中错成同一个值  : {same_val} "
          f"({same_val / len(multi) * 100:.1f}%)" if multi else "  （无）")
    print(f"  其中各错各的      : {len(multi) - same_val}")

    # ---- 4. 错误模式：字符级混淆是否成对出现 ----
    print("\n错值特征（全部出错字段）：")
    feat = Counter()
    for (stem, f), cfgvals in wrong_val.items():
        for lb, pv in cfgvals.items():
            gv = data[0][1][stem]["gt"].get(f)
            ps, gs = str(pv), str(gv)
            if ps.isdigit() and gs.isdigit() and len(ps) == len(gs):
                diff = sum(1 for x, y in zip(ps, gs) if x != y)
                feat[f"数字·{diff}位不同"] += 1
            elif ps.replace(".", "").replace("-", "").isdigit() and \
                    gs.replace(".", "").replace("-", "").isdigit():
                feat["数字·位数/小数点不同"] += 1
            elif len(ps) == len(gs):
                diff = sum(1 for x, y in zip(ps, gs) if x != y)
                feat[f"文本·{diff}字不同"] += 1
            else:
                feat["文本·长度不同"] += 1
    for k, v in feat.most_common(10):
        print(f"  {k:<20} {v}")

    # ---- 结论 ----
    print("\n" + "=" * 62)
    print("判读：")
    print(f"  · 「仅单次出错」占比 {single / n_wrong_any * 100:.1f}% "
          f"→ 随配置切换翻转，属决策边界抖动")
    print(f"  · 「全配置都错」占比 {stable / n_wrong_any * 100:.1f}% "
          f"→ 稳定复现；能否消除取决于属感知层还是决策层")

    # ---- 写报告 ----
    md = ["# 错误一致性分析", ""]
    md.append(f"- 比较配置：{' / '.join(labels)}（均为 384px）")
    md.append(f"- 共同样本：{len(stems)} 张，共有字段 {n_fields} 个")
    md.append(f"- 至少一处出错：{n_wrong_any} 个字段（{n_wrong_any/n_fields*100:.2f}%）")
    md.append("")
    md.append("## 一、稳定性分层（核心）")
    md.append("")
    md.append("| 在几个配置里出错 | 字段数 | 占比 | 解读 |")
    md.append("|---:|---:|---:|---|")
    for k in sorted(layering):
        tag = ("**稳定错误**（可学）" if k == len(data)
               else "中等" if k >= 3 else ("边界翻转" if k == 1 else "中等"))
        md.append(f"| {k} | {layering[k]} | "
                  f"{layering[k]/n_wrong_any*100:.1f}% | {tag} |")
    md.append("")
    md.append("## 二、两两重叠")
    md.append("")
    md.append("| A | B | 交集 | 并集 | Jaccard |")
    md.append("|---|---|---:|---:|---:|")
    for a, b, i, u, j in jac_rows:
        md.append(f"| {a} | {b} | {i} | {u} | {j:.3f} |")
    md.append("")
    md.append("## 三、错值特征")
    md.append("")
    md.append("| 特征 | 次数 |")
    md.append("|---|---:|")
    for k, v in feat.most_common(12):
        md.append(f"| {k} | {v} |")
    md.append("")
    md.append("## 三、错值一致性")
    md.append("")
    if multi:
        n_same = sum(1 for k, v in multi.items() if len({repr(x) for x in v.values()}) == 1)
        md.append(f"在 ≥2 个配置里出错的字段共 {len(multi)} 个，其中错成**同一个值**的有 "
                  f"{n_same} 个（{n_same/len(multi)*100:.1f}%）。")
        md.append("")
        md.append("错值高度一致 → 不是决策抖动，而是同一输入的确定性误读。")
    md.append("")
    md.append("## 结论")
    md.append("")
    md.append(f"- 仅单次出错 **{single}** 个（{single/n_wrong_any*100:.1f}%）："
              "对/错随配置切换而翻转，属决策边界抖动。")
    md.append(f"- 全配置稳定出错 **{stable}** 个（{stable/n_wrong_any*100:.1f}%）："
              "稳定复现，不是随机噪声。")
    md.append("")
    md.append("**注意：稳定 ≠ 可被偏好学习消除。** 若稳定错误源于输入信息不足"
              "（感知层瓶颈），调 DPO 无效 —— 实测同一批错误提高分辨率减 78%、"
              "DPO 减 0%。判断顺序见技能库 §8.3。")
    md.append("")
    p = Path(args.out)
    if not p.is_absolute():
        p = ROOT / args.out
    p.write_text("\n".join(md), encoding="utf-8")
    print(f"\n报告：{p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
