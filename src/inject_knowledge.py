"""知识注入 —— 用标准词表对模型预测做保守纠正，然后重打分。

为什么只做闭集字段（先读 src/check_lexicon_coverage.py 的结论）：
    零样本错误率最高的字段是 供应商 87.8% / 项目名称 80.6% / 单据编号 79.6%，
    但供应商与单据编号是**逐条随机生成**的（2100 条里有 692 / 698 个不同值），
    测试集的正确值根本不在词表里 —— 强行纠正只会把本来对的改错。
    **「错误率高」≠「可改进」。**
    真正可救的是真闭集：名称(20 值)、规格型号(37 值)、单位(5 值)、项目名称(231 值)。

三级规则（保守优先：宁可不改，不可改坏）：
    1. 归一化后精确命中词表 → 采纳词表规范写法
    2. 否则模糊匹配：top1 相似度 ≥ min_sim 且与次优差 ≥ min_gap → 替换
    3. 否则原样保留

**broke（改前对 → 改后错）在本方法里几乎恒为 0，而且是结构性保证**：
模型输出正确 ⇒ 该值等于 GT ⇒ GT 在词表内 ⇒ 精确命中 ⇒ 走规则 1、不改。
所以只有「GT 本身不在词表里」的样本才可能被改坏（本数据集 1260 个实例里只有 2 个）。

注意：规则 1 只统一写法、**不影响分数**（打分前本来就会归一化）。
真正改变分数的只有规则 2 的模糊替换 —— 报告里的 fixed/broke 都出自它。

日期默认**不做模糊匹配**：它不是「有标准写法的字段」，`2026-06-26` 与
`2026-06-25` 相似度约 0.91，模糊匹配会把相邻日期混淆。要用 `--include-date` 显式开。

本脚本纯 CPU：读已有的 *_preds.jsonl，不重跑推理、不加载模型、不花钱。

用法：
    python src/inject_knowledge.py --preds outputs/eval_zero_abl_preds.jsonl \
        --tag zero_abl_inj --report outputs/report_inject_zero_abl.md
    python src/inject_knowledge.py --preds ... --no-fuzzy      # 只精确命中（最保守）
    python src/inject_knowledge.py --preds ... --include-date  # 加上日期模糊匹配
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

# 默认只碰这四个真闭集字段（理由见模块 docstring）
DEFAULT_FIELDS = ["明细.名称", "明细.规格型号", "明细.单位", "表头.项目名称"]
SKIP_FIELDS = ["表头.供应商", "表头.单据编号"]

EV = None          # evaluate 模块
NORM = None        # evaluate.normalize


def load_evaluate():
    """按路径加载 evaluate.py —— 打分口径与本地各档完全一致，不另写一套。"""
    spec = importlib.util.spec_from_file_location("_ev", ROOT / "src" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def match_value(value, table, min_sim, min_gap, fuzzy):
    """返回 (新值, 方式)。(None, None) 表示不改。"""
    n = NORM(value)
    if not n:
        return None, None

    # ① 精确命中：只统一写法，不影响分数
    if n in table:
        return table[n], "exact"

    # ② 模糊匹配（词表只有 1 个值时禁用 —— 那等于无条件强制改写）
    if not fuzzy or len(table) < 2:
        return None, None

    best_r, best_key, second_r = -1.0, None, -1.0
    for key in table:
        r = difflib.SequenceMatcher(None, n, key).ratio()
        if r > best_r:
            second_r, best_r, best_key = best_r, r, key
        elif r > second_r:
            second_r = r
    if best_r >= min_sim and (best_r - second_r) >= min_gap:
        return table[best_key], "fuzzy"
    return None, None


def _rows(v) -> list:
    """把明细列表规整成 list[dict]，非 dict 行用 {} 占位（**保持索引**，
    因为行索引要和 align_rows 的配对结果对应）。"""
    if not isinstance(v, list):
        return []
    return [r if isinstance(r, dict) else {} for r in v]


def _one(path, field, holder, key, gt_value, table, min_sim, min_gap, fuzzy):
    """处理单个字段实例：需要时就地改写 holder[key]，并返回一条统计记录。"""
    before = holder[key]
    after, method = match_value(before, table, min_sim, min_gap, fuzzy)

    gt_n = NORM(gt_value)
    b_n = NORM(before)

    changed = False
    if after is not None and NORM(after) != b_n:
        holder[key] = after
        changed = True

    fmt_only = (after is not None and not changed and NORM(after) != str(before))
    final = holder[key]

    return {
        "path": path,
        "field": field,
        "before": before,
        "after": final,
        "gt": gt_value,
        "method": method,
        "changed": changed,
        "fmt_only": fmt_only,
        "before_ok": bool(gt_n) and b_n == gt_n,
        "after_ok": bool(gt_n) and NORM(final) == gt_n,
        "gt_empty": not gt_n,
        "gt_in_lex": gt_n in table,
    }


def correct_record(obj, gt, tables, min_sim, min_gap, fuzzy):
    """对一条已解析的预测做纠正。

    返回 (新对象, 实例记录列表)。实例记录覆盖**所有被纳入纠正范围的字段**
    （不只是被改动的），这样报告才能给出「改前错 / 改后错」的全量对照。
    """
    g_rows = _rows(gt.get("明细"))
    p_rows = _rows(obj.get("明细"))
    gt_row_of = {pi: gi for gi, pi in EV.align_rows(g_rows, p_rows)}

    new = dict(obj)
    inst: list[dict] = []

    # ---------------- 表头 ----------------
    head = obj.get("表头")
    if isinstance(head, dict):
        nh = dict(head)
        for path in [p for p in tables if p.startswith("表头.")]:
            field = path.split(".", 1)[1]
            if field not in nh:
                continue
            inst.append(_one(
                path, field, nh, field,
                (gt.get("表头") or {}).get(field, ""),
                tables[path], min_sim, min_gap, fuzzy,
            ))
        new["表头"] = nh

    # ---------------- 明细 ----------------
    if isinstance(obj.get("明细"), list):
        nr = [dict(r) if isinstance(r, dict) else r for r in obj["明细"]]
        for pi, row in enumerate(nr):
            if not isinstance(row, dict):
                continue
            gi = gt_row_of.get(pi)
            gt_row = g_rows[gi] if gi is not None else {}
            for path in [p for p in tables if p.startswith("明细.")]:
                field = path.split(".", 1)[1]
                if field not in row:
                    continue
                inst.append(_one(
                    f"明细[{pi}].{field}", field, row, field,
                    gt_row.get(field, ""),
                    tables[path], min_sim, min_gap, fuzzy,
                ))
        new["明细"] = nr

    return new, inst


def score_all(records, objs, tolerant):
    tp = fp = fn = jv = hal = 0
    for rec, obj in zip(records, objs):
        r = EV.score_pred(rec.get("gt") or {}, obj, "schema", tolerant)
        tp += r["tp"]; fp += r["fp"]; fn += r["fn"]
        jv += int(r["json_valid"])
        hal += int(bool(r["hallucinated_keys"]))
    p, rr, f1 = EV.prf(tp, fp, fn)
    n = max(1, len(records))
    return {"p": p, "r": rr, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "jv": jv / n, "hal": hal / n}


def main() -> int:
    ap = argparse.ArgumentParser(description="知识注入：词表纠正 + 重打分")
    ap.add_argument("--preds", required=True, help="evaluate.py 产出的 *_preds.jsonl")
    ap.add_argument("--lexicon", default=str(ROOT / "data" / "lexicon" / "v1.json"))
    ap.add_argument("--fields", default=",".join(DEFAULT_FIELDS),
                    help="逗号分隔的字段路径，如 明细.名称,表头.项目名称")
    ap.add_argument("--min-sim", type=float, default=0.50,
                    help="模糊匹配的最低相似度。默认 0.50 —— 阈值扫描的拐点："
                         "0.50→0.40 时新增改动的有效率从 92%% 掉到 43%%（见 report_inject_sweep.md）")
    ap.add_argument("--min-gap", type=float, default=0.05)
    ap.add_argument("--no-fuzzy", action="store_true", help="只做精确命中，完全禁用模糊匹配")
    ap.add_argument("--include-date", action="store_true",
                    help="把 表头.日期 也纳入（默认不含：相邻日期会被模糊匹配混淆）")
    ap.add_argument("--tag", default=None, help="产物 tag，默认在原 tag 后加 _inj")
    ap.add_argument("--report", default=None, help="markdown 报告路径")
    ap.add_argument("--no-dump", action="store_true", help="不写注入后的 preds.jsonl")
    args = ap.parse_args()

    global EV, NORM
    EV = load_evaluate()
    NORM = EV.normalize

    # ---------------- 词表 ----------------
    lex_path = Path(args.lexicon)
    if not lex_path.exists():
        print(f"[FATAL] 找不到词表 {lex_path}，先跑 src/build_lexicon.py")
        return 1
    lex = json.loads(lex_path.read_text(encoding="utf-8"))

    want = [f.strip() for f in args.fields.split(",") if f.strip()]
    if args.include_date and "表头.日期" not in want:
        want.append("表头.日期")
    missing = [f for f in want if f not in lex["fields"]]
    if missing:
        print(f"[FATAL] 词表里没有这些字段：{missing}")
        print(f"        可用：{sorted(lex['fields'])}")
        return 1
    tables = {f: lex["fields"][f]["values"] for f in want}

    # ---------------- 输入 ----------------
    src = Path(args.preds)
    if not src.exists():
        print(f"[FATAL] 找不到 {src}")
        return 1
    records = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]

    # 不同档位的 *_preds.jsonl 里 gt 有的是 dict、有的是 JSON 字符串（历史原因），
    # 统一规范化，避免下游 .get() 撞到 str。
    def _norm_gt(r: dict) -> dict:
        g = r.get("gt")
        if isinstance(g, str):
            try:
                g = json.loads(g)
            except json.JSONDecodeError:
                g = {}
        return g if isinstance(g, dict) else {}

    records = [{**r, "gt": _norm_gt(r)} for r in records]

    orig_tag = src.name.replace("eval_", "").replace("_preds.jsonl", "")
    tag = args.tag or f"{orig_tag}_inj"
    report_path = Path(args.report) if args.report else OUTPUTS / f"report_inject_{tag}.md"
    fuzzy = not args.no_fuzzy

    print("=" * 70)
    print("知识注入")
    print("=" * 70)
    print(f"  预测文件 : {src.name}  {len(records)} 条")
    print(f"  词表     : {lex_path.name} ({lex['version']}, 来自 {lex['n_source']} 条训练样本)")
    print(f"  纠正字段 : {', '.join(want)}")
    print(f"  跳过字段 : {', '.join(SKIP_FIELDS)}（逐条随机生成，纠正会把对的改错）")
    if fuzzy:
        print(f"  匹配策略 : 精确命中 + 模糊 (top1 ≥ {args.min_sim:.2f} 且与次优差 ≥ {args.min_gap:.2f})")
    else:
        print("  匹配策略 : 仅精确命中（注：这不改变分数，只统一写法）")
    print()

    # ---------------- 注入 ----------------
    inst_all: list[dict] = []
    new_objs: list[dict | None] = []
    for rec in records:
        obj = EV.parse_json(rec.get("pred") or "")
        if obj is None:
            new_objs.append(None)
            continue
        nobj, inst = correct_record(obj, rec.get("gt") or {}, tables,
                                    args.min_sim, args.min_gap, fuzzy)
        inst_all.extend(inst)
        new_objs.append(nobj)

    orig_objs = [EV.parse_json(r.get("pred") or "") for r in records]

    # ---------------- 指标 ----------------
    print("重打分中...")
    before_s = score_all(records, orig_objs, False)
    after_s = score_all(records, new_objs, False)
    before_t = score_all(records, orig_objs, True)
    after_t = score_all(records, new_objs, True)

    # ---------------- 统计（只看 GT 非空、且能被比对的实例） ----------------
    valid = [i for i in inst_all if not i["gt_empty"]]
    changed = [i for i in valid if i["changed"]]

    cm = Counter()
    for c in changed:
        if not c["before_ok"] and c["after_ok"]:
            cm["fixed"] += 1
        elif c["before_ok"] and not c["after_ok"]:
            cm["broke"] += 1
        else:
            cm["still_wrong"] += 1

    fmt_only = sum(1 for i in inst_all if i["fmt_only"])

    by_field: dict[str, Counter] = defaultdict(Counter)
    for i in valid:
        b = by_field[i["field"]]
        b["n"] += 1
        b["changed"] += int(i["changed"])
        b["wrong_before"] += int(not i["before_ok"])
        b["wrong_after"] += int(not i["after_ok"])

    by_lex: dict[str, Counter] = defaultdict(Counter)
    for i in valid:
        b = by_lex["in_lex" if i["gt_in_lex"] else "out_lex"]
        b["n"] += 1
        b["changed"] += int(i["changed"])
        b["wrong_before"] += int(not i["before_ok"])
        b["wrong_after"] += int(not i["after_ok"])
        if i["changed"] and not i["before_ok"] and i["after_ok"]:
            b["fixed"] += 1
        if i["changed"] and i["before_ok"] and not i["after_ok"]:
            b["broke"] += 1

    # ---------------- 落盘 ----------------
    dump_path = OUTPUTS / f"eval_{tag}_preds.jsonl"
    if not args.no_dump:
        with dump_path.open("w", encoding="utf-8") as f:
            for rec, nobj in zip(records, new_objs):
                out = dict(rec)
                if nobj is not None:
                    out["pred_orig"] = rec.get("pred")
                    out["pred"] = json.dumps(nobj, ensure_ascii=False)
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"  注入后预测 → {dump_path}")

    # ---------------- 报告 ----------------
    L: list[str] = []
    A = L.append
    A(f"# 知识注入报告 —— {tag}")
    A("")
    A(f"- 预测来源：`{src.name}`（{len(records)} 条）")
    A(f"- 词表：`{lex_path.name}`（{lex['version']}，{lex['n_source']} 条训练样本的 GT 值域）")
    A(f"- 纠正字段：{', '.join(want)}")
    A(f"- 跳过字段：{', '.join(SKIP_FIELDS)}")
    if fuzzy:
        A(f"- 匹配策略：精确命中 + 模糊匹配（top1 ≥ {args.min_sim:.2f}，与次优差 ≥ {args.min_gap:.2f}）")
    else:
        A("- 匹配策略：仅精确命中")
    A("")
    A("> 说明：**精确命中不改变分数**（打分前本来就会归一化），它只统一输出写法。")
    A("> 真正影响指标的是模糊替换，下面的 fixed / broke 都出自它。")
    A("")
    A("## 1. 整体指标")
    A("")
    A("| 口径 | 阶段 | P | R | F1 | JSON 合法率 | 幻觉率 | TP/FP/FN |")
    A("|---|---|---:|---:|---:|---:|---:|---|")
    for name, b, a in (("严格", before_s, after_s), ("宽容", before_t, after_t)):
        A(f"| {name} | 注入前 | {b['p']:.4f} | {b['r']:.4f} | {b['f1']:.4f} | {b['jv']:.1%} | {b['hal']:.1%} | {b['tp']}/{b['fp']}/{b['fn']} |")
        A(f"| {name} | **注入后** | {a['p']:.4f} | {a['r']:.4f} | **{a['f1']:.4f}** | {a['jv']:.1%} | {a['hal']:.1%} | {a['tp']}/{a['fp']}/{a['fn']} |")
        A(f"| {name} | 变化 | | | **{a['f1'] - b['f1']:+.4f}** | | | |")
    A("")
    A("## 2. 纠正动作的混淆矩阵")
    A("")
    A("| 动作 | 次数 | 含义 |")
    A("|---|---:|---|")
    A(f"| fixed | {cm['fixed']} | 改前错 → 改后对（净收益） |")
    A(f"| broke | {cm['broke']} | **改前对 → 改后错（净损失）** |")
    A(f"| still_wrong | {cm['still_wrong']} | 改了但仍然错（词表里没有正确值） |")
    A("")
    tot = cm["fixed"] + cm["broke"]
    if tot:
        A(f"**有效改动的净胜率 = {cm['fixed']}/{tot} = {cm['fixed'] / tot:.1%}**")
    else:
        A("本轮没有任何改变归一化值的改动。")
    if fmt_only:
        A("")
        A(f"另有 {fmt_only} 处只统一了写法（归一化值未变，不影响任何指标）。")
    A("")
    A("## 3. 按字段")
    A("")
    A("| 字段 | 实例数 | 改动数 | 改前错 | 改后错 | 修复 |")
    A("|---|---:|---:|---:|---:|---:|")
    for f in sorted(by_field):
        b = by_field[f]
        A(f"| {f} | {b['n']} | {b['changed']} | {b['wrong_before']} | {b['wrong_after']} | "
          f"{b['wrong_before'] - b['wrong_after']:+d} |")
    A("")
    A("## 4. 词表内 / 词表外（纠正的适用边界）")
    A("")
    A("分桶依据是 **GT 值是否在词表里** —— 词表外的那一桶就是纠正的禁区。")
    A("")
    A("| 桶 | 实例数 | 改动数 | 改前错 | 改后错 | fixed | broke |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for k, label in (("in_lex", "GT 在词表内（可救）"), ("out_lex", "GT 在词表外（不可救）")):
        b = by_lex[k]
        A(f"| {label} | {b['n']} | {b['changed']} | {b['wrong_before']} | {b['wrong_after']} | {b['fixed']} | {b['broke']} |")
    A("")

    fixed_ex = [i for i in changed if not i["before_ok"] and i["after_ok"]]
    broke_ex = [i for i in changed if i["before_ok"] and not i["after_ok"]]
    sw_ex = [i for i in changed if not i["before_ok"] and not i["after_ok"]]

    A("## 5. 改动样例")
    A("")
    for label, ex in (("纠正成功（fixed）", fixed_ex),
                      ("纠正改坏（broke）", broke_ex),
                      ("改了但仍然错（still_wrong）", sw_ex)):
        A(f"### {label} —— {len(ex)} 条，最多列 10")
        A("")
        if not ex:
            A("（无）")
        else:
            A("| 位置 | 改前 | 改后 | GT | 方式 |")
            A("|---|---|---|---|---|")
            for c in ex[:10]:
                A(f"| {c['path']} | `{str(c['before'])[:28]}` | `{str(c['after'])[:28]}` "
                  f"| `{str(c['gt'])[:28]}` | {c['method']} |")
        A("")

    leftovers = [i for i in valid if not i["after_ok"] and not i["changed"]]
    A(f"## 6. 未被纠正的错误（{len(leftovers)} 条，最多列 10）")
    A("")
    A("「模型错、且词表里找不到足够近的值」—— 这就是**知识注入的能力边界**。")
    A("")
    if leftovers:
        A("| 位置 | 模型输出 | GT |")
        A("|---|---|---|")
        for c in leftovers[:10]:
            A(f"| {c['path']} | `{str(c['before'])[:30]}` | `{str(c['gt'])[:30]}` |")
    else:
        A("（无）")
    A("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(L), encoding="utf-8")

    # ---------------- 控制台摘要 ----------------
    print()
    print("=" * 70)
    print("结果")
    print("=" * 70)
    print(f"  可比实例 : {len(valid)}   改动 {len(changed)}   (仅统一写法 {fmt_only})")
    print(f"  fixed / broke / still_wrong : {cm['fixed']} / {cm['broke']} / {cm['still_wrong']}")
    print()
    print(f"  {'口径':<6}{'F1 注入前':>12}{'F1 注入后':>12}{'变化':>12}")
    print(f"  {'严格':<6}{before_s['f1']:>12.4f}{after_s['f1']:>12.4f}{after_s['f1'] - before_s['f1']:>+12.4f}")
    print(f"  {'宽容':<6}{before_t['f1']:>12.4f}{after_t['f1']:>12.4f}{after_t['f1'] - before_t['f1']:>+12.4f}")
    print()
    print(f"  报告 → {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
