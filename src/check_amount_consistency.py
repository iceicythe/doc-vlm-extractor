"""统计输出里「金额 == 数量 × 单价」成立的比例 —— 内部一致性指标。

为什么单独看这个
----------------
金额是**派生量**（= 数量 × 单价），不是独立读出来的。所以它对模型的
要求比逐字段认字更高一层：账要平。

本项目的偏好数据里，143 对中有 139 对（97%）的差异涉及金额，是 top1 差异字段。
训练后这个比例如果提升，说明模型学到的是「结构约束」而不只是逐字段模仿。

一个必须同时看的风险
--------------------
`src/render.py` 里**故意让 15% 的样本金额 ≠ 数量 × 单价**（用来模拟真实单据
里的录入错误 / 折扣）。所以：

  - 如果 DPO 让模型学会「把账抹平」，它可能会去**修改那些本就该保留的错值** ——
    内部一致率上升，但对着 GT 的 F1 反而下降。

这就是脚本要把 GT 拆成「一致行 / 不一致行」两桶分别统计的原因：
只有两桶的分布一起看，才能判断模型是「读得更准了」还是「自作主张地修正了」。

用法
----
    python src/check_amount_consistency.py outputs/eval_abl_base_preds.jsonl \
                                          outputs/eval_dpo_v1_preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOL = 0.011          # 金额保留 2 位小数，容差取「半分钱」+ 浮点噪声


def to_num(v) -> float | None:
    """把 '1,234.50' / '¥1234.5' 这类字符串转成 float；失败返回 None。"""
    if v is None:
        return None
    s = str(v).replace(",", "").replace("，", "").replace("¥", "").replace("￥", "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def rows_of(obj) -> list[dict]:
    if not isinstance(obj, dict):
        return []
    det = obj.get("明细")
    if not isinstance(det, list):
        return []
    return [r for r in det if isinstance(r, dict)]


def row_consistent(row: dict) -> bool | None:
    """该明细行的金额是否等于 数量×单价。三个字段缺任一则返回 None（不可判定）。"""
    q, p, a = (to_num(row.get(k)) for k in ("数量", "单价", "金额"))
    if q is None or p is None or a is None:
        return None
    return abs(a - q * p) <= max(TOL, abs(q * p) * 1e-6)


def parse_json_field(v):
    if isinstance(v, (dict, list)):
        return v
    if not isinstance(v, str):
        return None
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return None


def analyse(path: Path, examples: int = 3) -> dict:
    st = {
        "path": path.name,
        "lines": 0, "bad_json": 0, "rows": 0, "undecidable": 0,
        # 按 GT 是否一致分桶统计 pred 的一致性
        "gt_ok_rows": 0, "gt_ok_pred_ok": 0,
        "gt_bad_rows": 0, "gt_bad_pred_ok": 0,
        # 「把本该保留的错值抹平」—— GT 不一致但 pred 一致，是可疑信号
        "gt_bad_pred_flat": 0,
        "ex_flat": [], "ex_break": [],
    }
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            st["lines"] += 1
            rec = json.loads(line)
            if not rec.get("json_valid"):
                st["bad_json"] += 1
                continue
            gt, pred = parse_json_field(rec.get("gt")), parse_json_field(rec.get("pred"))
            for gr, pr in zip(rows_of(gt), rows_of(pred)):
                g = row_consistent(gr)
                if g is None:
                    st["undecidable"] += 1
                    continue
                st["rows"] += 1
                p = row_consistent(pr)
                if g:
                    st["gt_ok_rows"] += 1
                    if p:
                        st["gt_ok_pred_ok"] += 1
                    elif len(st["ex_break"]) < examples:
                        st["ex_break"].append((rec.get("stem"), gr, pr))
                else:
                    st["gt_bad_rows"] += 1
                    if p:
                        st["gt_bad_pred_ok"] += 1
                        st["gt_bad_pred_flat"] += 1
                        if len(st["ex_flat"]) < examples:
                            st["ex_flat"].append((rec.get("stem"), gr, pr))
    return st


def pct(a: int, b: int) -> str:
    return f"{a / b * 100:.1f}%" if b else "—"


def main() -> int:
    ap = argparse.ArgumentParser(description="金额一致性（金额 == 数量 × 单价）统计")
    ap.add_argument("preds", nargs="+", help="一个或多个 *_preds.jsonl")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    stats = []
    for p in args.preds:
        path = Path(p)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            print(f"[跳过] 不存在: {p}")
            continue
        stats.append(analyse(path, args.examples))

    if not stats:
        print("没有可分析的预测文件")
        return 1

    print("=" * 78)
    print("金额一致性：金额 == 数量 × 单价")
    print("=" * 78)
    print(f"{'预测文件':<30} {'明细行':>6} {'pred一致':>9} {'GT一致行':>9} "
          f"{'一致行上pred':>12} {'不一致行上pred':>14}")
    for s in stats:
        print(f"{s['path']:<30} {s['rows']:>6} "
              f"{pct(s['gt_ok_pred_ok'] + s['gt_bad_pred_ok'], s['rows']):>9} "
              f"{pct(s['gt_ok_rows'], s['rows']):>9} "
              f"{pct(s['gt_ok_pred_ok'], s['gt_ok_rows']):>12} "
              f"{pct(s['gt_bad_pred_ok'], s['gt_bad_rows']):>14}")

    for s in stats:
        print()
        print("-" * 78)
        print(f"{s['path']}   （{s['lines']} 条，JSON 非法 {s['bad_json']}，"
              f"不可判定行 {s['undecidable']}）")
        print(f"  GT 不一致的明细行共 {s['gt_bad_rows']} 行（render 故意注入的等式错误），"
              f"其中 pred 把它抹平的有 {s['gt_bad_pred_flat']} 行")
        if s["ex_break"]:
            print("  样例 · GT 一致但 pred 破坏了一致性：")
            for stem, gr, pr in s["ex_break"]:
                print(f"    {stem}  GT {gr.get('数量')}×{gr.get('单价')}={gr.get('金额')}"
                      f"   → pred {pr.get('数量')}×{pr.get('单价')}={pr.get('金额')}")
        if s["ex_flat"]:
            print("  样例 · GT 本就不一致、pred 却抹平了（可能自作主张修正，需与 F1 一起看）：")
            for stem, gr, pr in s["ex_flat"]:
                print(f"    {stem}  GT {gr.get('数量')}×{gr.get('单价')}≠{gr.get('金额')}"
                      f"   → pred {pr.get('数量')}×{pr.get('单价')}={pr.get('金额')}")

    print()
    print("读法：")
    print("  · 「一致行上 pred」高 = 模型维持了内部一致性（读得准且账平）。")
    print("  · 「不一致行上 pred」高 ≠ 好事 —— 那意味着它把本该保留的错值抹平了，")
    print("    要同时看字段级 F1 有没有掉，才能区分「学得更准」与「自作主张修正」。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
