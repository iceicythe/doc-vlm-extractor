"""知识注入可行性前置检查：测试集的闭集字段值，有多少在训练集词表里？

如果覆盖率低（比如 <50%），那"用标准词表纠正"这条路就是死路，别浪费时间。
顺便统计错误率最高的字段是不是闭集（值域小的字段 = 闭集 = 知识注入对症）。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "processed"

FIELDS = ["供应商", "项目名称", "单据编号", "日期"]
ROW_FIELDS = ["名称", "规格型号", "单位"]


def load(name):
    return [json.loads(l) for l in (P / name).open(encoding="utf-8") if l.strip()]


train = load("train.jsonl")
test = load("test_ablation.jsonl")


def values(recs, field, row=False):
    out = []
    for r in recs:
        gt = r["gt"]
        if row:
            for it in (gt.get("明细") or []):
                if isinstance(it, dict):
                    v = str(it.get(field, "") or "").strip()
                    if v:
                        out.append(v)
        else:
            v = str((gt.get("表头") or {}).get(field, "") or "").strip()
            if v:
                out.append(v)
    return out


print("=" * 74)
print("词表覆盖率检查（词表 = train.jsonl 的 GT 值）")
print("=" * 74)
print(f"train {len(train)} 条 / test_ablation {len(test)} 条")
print()

for field in FIELDS + ROW_FIELDS:
    is_row = field in ROW_FIELDS
    tr = values(train, field, is_row)
    te = values(test, field, is_row)
    if not te:
        continue
    lex = set(tr)
    exact = sum(1 for v in te if v in lex)
    # 去重后的覆盖（更宽松：同一张单据重复出现只算一次）
    te_u = set(te)
    exact_u = sum(1 for v in te_u if v in lex)
    n_uniq_tr = len(set(tr))
    print(f"{field}")
    print(f"   训练集不同值 {n_uniq_tr:4d} 个 | 测试集 {len(te):4d} 次 / {len(te_u):3d} 个不同值")
    print(f"   **逐次命中 {exact/len(te):6.1%}**   逐值命中 {exact_u/len(te_u):6.1%}")
    print()

print("=" * 74)
print("闭集程度：不同值个数 / 出现次数 → 越小越闭集")
print("=" * 74)
for field in FIELDS + ROW_FIELDS:
    is_row = field in ROW_FIELDS
    tr = values(train, field, is_row)
    if not tr:
        continue
    c = Counter(tr)
    print(f"  {field:8s} 词表 {len(c):4d} 个 / {len(tr):5d} 次 = 占比 {len(c)/len(tr):6.2%}"
          f" | top3: {[v for v, _ in c.most_common(3)]}")
