"""失败分析 —— 把错例按「性质」分类，输出可指导下一步决策的报告。

为什么需要它
-----------
F1 只告诉你「有多少错」，不告诉你「错在哪、为什么错、该改什么」。
这个脚本把每个错误归到下面几类，并给出对应的行动建议：

    漏抽      GT 有、预测没有        → 召回问题（模型没看到 / 忽略了）
    多抽/幻觉  预测有、GT 没有        → 精确率问题（模型脑补）
    数值误读   数字字段值不同         → 视觉分辨率问题（最容易对症）
    文本误读   文本字段值不同         → 分「近似」(一字之差，视觉) / 「显著」(语义混淆)
    行列错位   未配对上的明细行       → 结构化对齐问题
    JSON 非法  输出无法解析           → 输出约束问题

对「误读」进一步按编辑距离分「近似」和「显著」是有意的：
    近似误读（广盛→广昌、883→881）⇒ 提分辨率有用
    显著误读（华建→中建）          ⇒ 提分辨率无用，是语言先验在瞎猜，要靠数据/训练

用法
----
    venv-gld/Scripts/python.exe src/analyze_errors.py outputs/eval_v1_full_preds.jsonl
    venv-gld/Scripts/python.exe src/analyze_errors.py outputs/eval_abl_r512_preds.jsonl --out outputs/report_r512.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import flatten, parse_json  # noqa: E402
from evaluate import HEAD_FIELDS, ROW_FIELDS  # noqa: E402

TOP_KEYS = {"单据类型", "表头", "明细", "合计"}

CATS = ["漏抽", "多抽/幻觉", "数值误读", "文本近似误读", "文本显著误读",
        "行列错位", "JSON 非法"]


def edit_distance(a: str, b: str, cap: int = 6) -> int:
    """带上限的 Levenshtein 距离（超过 cap 直接返回 cap+1，省算力）。"""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def is_numeric(s: str) -> bool:
    t = s.replace(",", "").replace(".", "").replace("-", "").strip()
    return bool(t) and t.isdigit()


def classify(gv: str, pv: str) -> tuple[str, int]:
    """返回 (类别, 编辑距离)。"""
    if is_numeric(gv) and is_numeric(pv):
        return "数值误读", edit_distance(gv, pv)
    d = edit_distance(gv, pv)
    return ("文本近似误读" if d <= 2 else "文本显著误读"), d


def group_of(key: str) -> str:
    if key.startswith("表头"):
        return "表头"
    if key.startswith("合计"):
        return "合计"
    if "unmatched" in key:
        return "明细(未配对)"
    return "明细"


def analyze(path: Path) -> dict:
    recs = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]

    cat_total: Counter = Counter()
    cat_by_level: dict[str, Counter] = defaultdict(Counter)
    cat_by_group: dict[str, Counter] = defaultdict(Counter)
    field_err: Counter = Counter()      # 字段 -> 错误数
    field_seen: Counter = Counter()     # 字段 -> 出现总数
    samples: dict[str, list[str]] = defaultdict(list)
    approx_dists: list[int] = []

    n_fields = 0
    n_valid = 0

    for rec in recs:
        lv = rec.get("level", "all")
        gt = rec["gt"]
        pred = parse_json(rec["pred"])

        if pred is None:
            cat_total["JSON 非法"] += 1
            cat_by_level[lv]["JSON 非法"] += 1
            samples["JSON 非法"].append(
                f"{rec['image']}  输出前 120 字: {rec['pred'][:120]!r}")
            continue
        n_valid += 1

        gout, pout = flatten(gt, pred)
        n_fields += len(gout)

        for k, gv in gout.items():
            field_seen[k.split(".", 1)[-1]] += 1
            pv = pout.get(k)
            if pv is None:
                cat = "漏抽"
                dist = -1
            elif pv == gv:
                continue
            else:
                cat, dist = classify(gv, pv)
                if cat in ("文本近似误读", "数值误读"):
                    approx_dists.append(dist)

            cat_total[cat] += 1
            cat_by_level[lv][cat] += 1
            cat_by_group[group_of(k)][cat] += 1
            field_err[k.split(".", 1)[-1]] += 1

            if len(samples[cat]) < 5:
                extra = f"  (编辑距离 {dist})" if dist >= 0 else ""
                samples[cat].append(
                    f"{rec['image']}  [{lv}]  {k}:  GT={gv!r}  →  PRED={pv!r}{extra}")

        for k in pout:
            if k not in gout:
                cat = "行列错位" if "unmatched" in k else "多抽/幻觉"
                cat_total[cat] += 1
                cat_by_level[lv][cat] += 1
                cat_by_group[group_of(k)][cat] += 1
                if len(samples[cat]) < 5:
                    samples[cat].append(
                        f"{rec['image']}  [{lv}]  {k} = {pout[k]!r}  (GT 中不存在)")

        # schema 之外的多余字段（flatten 只遍历预定义字段，这类要靠显式检查）
        # 这是「幻觉」在 schema 模式下的主要形态：模型凭空多编了字段
        extra: list[str] = []
        for k in pred:
            if k not in TOP_KEYS:
                extra.append(f"顶层.{k}")
        head = pred.get("表头")
        if isinstance(head, dict):
            extra += [f"表头.{k}" for k in head if k not in HEAD_FIELDS]
        rows = pred.get("明细")
        if isinstance(rows, list):
            for i, row in enumerate(rows):
                if isinstance(row, dict):
                    extra += [f"明细[{i}].{k}" for k in row if k not in ROW_FIELDS]
        for key in extra:
            cat_total["多抽/幻觉"] += 1
            cat_by_level[lv]["多抽/幻觉"] += 1
            cat_by_group["schema 外字段"]["多抽/幻觉"] += 1
            if len(samples["多抽/幻觉"]) < 5:
                samples["多抽/幻觉"].append(
                    f"{rec['image']}  [{lv}]  {key}  (不在 schema 中)")

    return {
        "path": path, "n_records": len(recs), "n_valid": n_valid,
        "n_fields": n_fields, "cat_total": cat_total,
        "cat_by_level": cat_by_level, "cat_by_group": cat_by_group,
        "field_err": field_err, "field_seen": field_seen,
        "samples": samples, "approx_dists": approx_dists,
    }


def level_f1(recs_path: Path) -> dict:
    """从同名 .json 汇总里取分档 F1（如果存在）。"""
    j = recs_path.with_name(recs_path.name.replace("_preds.jsonl", ".json"))
    if not j.exists():
        return {}
    try:
        d = json.loads(j.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for lv, b in (d.get("by_level") or {}).items():
        tp, fp, fn = b["tp"], b["fp"], b["fn"]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out[lv] = {"f1": 2 * p * r / (p + r) if p + r else 0.0,
                   "precision": p, "recall": r, "n": b["n"]}
    return out


def report(a: dict, md: bool) -> str:
    L: list[str] = []
    add = L.append
    nf = max(1, a["n_fields"])
    total_err = sum(a["cat_total"].values())

    if md:
        add(f"# 失败分析报告\n")
        add(f"- 输入：`{a['path'].name}`")
        add(f"- 样本：{a['n_records']} 条（JSON 合法 {a['n_valid']} 条）")
        add(f"- 字段总数：{a['n_fields']}")
        add(f"- 错误字段：{total_err}（{total_err/nf:.2%}）\n")
        add("## 一、错误类型分布\n")
        add("| 类型 | 数量 | 占错误 | 占全部字段 |")
        add("|---|---:|---:|---:|")
    else:
        add("=" * 74)
        add("失败分析报告")
        add("=" * 74)
        add(f"输入       : {a['path'].name}")
        add(f"样本       : {a['n_records']} 条（JSON 合法 {a['n_valid']}）")
        add(f"字段总数   : {a['n_fields']}")
        add(f"错误字段   : {total_err}  ({total_err/nf:.2%})")
        add("")
        add(f"{'类型':<14}{'数量':>7}{'占错误':>10}{'占全部字段':>12}")
        add("-" * 74)

    for cat in CATS:
        n = a["cat_total"].get(cat, 0)
        if n == 0:
            continue
        if md:
            add(f"| {cat} | {n} | {n/max(1,total_err):.1%} | {n/nf:.2%} |")
        else:
            add(f"{cat:<14}{n:>7}{n/max(1,total_err):>10.1%}{n/nf:>12.2%}")
    if md:
        add("")

    # ---- 分档 ----
    if a["cat_by_level"]:
        if md:
            add("## 二、按退化档分布\n")
            add("| 档位 | " + " | ".join(CATS) + " |")
            add("|---" * (len(CATS) + 1) + "|")
        else:
            add("")
            add("按退化档分布：")
            add(f"  {'档位':<9}" + "".join(f"{c:>13}" for c in CATS))
        for lv in ("clean", "medium", "heavy"):
            c = a["cat_by_level"].get(lv)
            if not c:
                continue
            if md:
                add(f"| {lv} | " + " | ".join(str(c.get(x, 0)) for x in CATS) + " |")
            else:
                add(f"  {lv:<9}" + "".join(f"{c.get(x,0):>13}" for x in CATS))
        if md:
            add("")

    # ---- 按结构分组 ----
    if md:
        add("## 三、按单据结构分布\n")
        add("| 结构 | " + " | ".join(CATS) + " |")
        add("|---" * (len(CATS) + 1) + "|")
        for g in ("表头", "明细", "明细(未配对)", "合计", "schema 外字段"):
            c = a["cat_by_group"].get(g)
            if not c:
                continue
            add(f"| {g} | " + " | ".join(str(c.get(x, 0)) for x in CATS) + " |")
        add("")

    # ---- 字段错误率 ----
    rows = []
    for f, seen in a["field_seen"].items():
        err = a["field_err"].get(f, 0)
        rows.append((err / max(1, seen), err, seen, f))
    rows.sort(reverse=True)
    if md:
        add("## 四、各字段错误率（降序）\n")
        add("| 字段 | 错误/总数 | 错误率 |")
        add("|---|---:|---:|")
    else:
        add("")
        add("各字段错误率：")
    for rate, err, seen, f in rows:
        if md:
            add(f"| {f} | {err}/{seen} | {rate:.2%} |")
        else:
            add(f"  {f:<12} {err:>5}/{seen:<6} {rate:>7.2%}")
    if md:
        add("")

    # ---- 典型样例 ----
    if md:
        add("## 五、典型错例\n")
    for cat in CATS:
        s = a["samples"].get(cat)
        if not s:
            continue
        add(("**" + cat + "**\n") if md else f"\n【{cat}】")
        for line in s:
            add(("    " + line) if not md else ("- `" + line + "`"))
        if md:
            add("")

    # ---- 结论 ----
    add("")
    if md:
        add("## 六、结论与下一步\n")
    else:
        add("=" * 74)
        add("结论与下一步")
        add("=" * 74)

    ct = a["cat_total"]
    hp = sum(a["cat_by_level"].get("heavy", Counter()).values())
    mp = sum(a["cat_by_level"].get("medium", Counter()).values())
    add(f"  错误集中在：")
    for cat, n in ct.most_common(3):
        add(f"    - {cat}: {n} ({n/max(1,total_err):.0%})")

    add("")
    approx = sum(ct.get(c, 0) for c in ("数值误读", "文本近似误读"))
    signif = ct.get("文本显著误读", 0)
    if approx + signif > 0:
        add(f"  误读细分：近似(一字之差) {approx}  vs  显著(语义混淆) {signif}")
        if approx > signif * 1.5:
            add("  ⇒ 近似误读为主 ⇒ **提分辨率**（384→512/768）是最直接的对策")
        elif signif > approx * 1.5:
            add("  ⇒ 显著误读为主 ⇒ 提分辨率作用有限，**要靠数据多样性/训练量**")
        else:
            add("  ⇒ 两类相当，建议两个方向都试")
    if ct.get("漏抽", 0) > ct.get("多抽/幻觉", 0) * 1.5:
        add("  ⇒ 漏抽多于幻觉 ⇒ 模型偏保守，可放宽生成 / 检查 prompt")
    elif ct.get("多抽/幻觉", 0) > ct.get("漏抽", 0) * 1.5:
        add("  ⇒ 幻觉多于漏抽 ⇒ 正是 DPO 用「幻觉惩罚」做 reward 的靶子")
    if ("heavy" in a["cat_by_level"] and "clean" in a["cat_by_level"]):
        ch = sum(a["cat_by_level"]["clean"].values())
        hh = sum(a["cat_by_level"]["heavy"].values())
        if ch and hh > ch * 3:
            add(f"  ⇒ heavy 档错误是 clean 的 {hh/max(1,ch):.1f} 倍 ⇒ "
                f"退化确实构成难度，可考虑针对性数据增广")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="失败分析")
    ap.add_argument("preds", help="eval_{tag}_preds.jsonl")
    ap.add_argument("--out", default="", help="写到文件（.md 结尾则输出 Markdown）")
    args = ap.parse_args()

    p = Path(args.preds)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        print(f"[FATAL] 找不到 {p}")
        print("提示：确认评测脚本已更新（会输出 *_preds.jsonl）")
        return 1

    a = analyze(p)
    md = args.out.endswith(".md")
    text = report(a, md)

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"报告已写入 {out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
