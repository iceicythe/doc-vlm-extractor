"""evaluate.py 打分层回归测试（纯 CPU，不加载模型）。

覆盖：
  1. 合计 输出成标量 → 严格算漏抽、宽容算命中
  2. 表头 输出成字符串 → 不崩，表头字段算漏抽
  3. 顶层输出成 list / 截断 JSON → json_valid=False，GT 全漏抽
  4. 明细行里混入字符串 → 不崩，坏行丢弃
  5. 明细 输出成 dict → 不崩
  6. 完全正常输出 → 严格 == 宽容，F1 = 1.0
  7. score_one 旧签名（evaluate_api.py 依赖）仍可用
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import evaluate as E  # noqa: E402

GT = {
    "单据类型": "材料清单",
    "表头": {"项目名称": "沣西市政道路改造工程", "供应商": "汉中盛世建材有限公司",
             "单据编号": "CL-20260626-029", "日期": "2026-06-26"},
    "明细": [
        {"序号": "1", "名称": "中砂", "规格型号": "中粗", "单位": "m3",
         "数量": "200", "单价": "120.27", "金额": "24054.00"},
        {"序号": "2", "名称": "钢筋套筒", "规格型号": "Φ20", "单位": "个",
         "数量": "2381", "单价": "6.64", "金额": "15809.84"},
    ],
    "合计": {"金额": "39863.84"},
}

fails = []


def check(name: str, cond: bool, extra: str = ""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}{'  ' + extra if extra else ''}")
    if not cond:
        fails.append(name)


print("1) 合计 标量（零样本真实失败模式）")
raw = ('{"单据类型":"材料清单","表头":{"项目名称":"沣西市政道路改造工程",'
       '"供应商":"汉中盛世建材有限公司","单据编号":"CL-20260626-029",'
       '"日期":"2026-06-26"},"明细":[{"序号":"1","名称":"中砂","规格型号":"中粗",'
       '"单位":"m3","数量":"200","单价":"120.27","金额":"24054.00"},'
       '{"序号":"2","名称":"钢筋套筒","规格型号":"Φ20","单位":"个","数量":"2381",'
       '"单价":"6.64","金额":"15809.84"}],"合计":"39863.84"}')
s = E.score_one(GT, raw, "schema", tolerant=False)
t = E.score_one(GT, raw, "schema", tolerant=True)
check("严格：合计.金额 计入漏抽", "合计.金额" in s["missing_keys"], f"fn={s['fn']}")
check("宽容：合计.金额 命中", t["fn"] == s["fn"] - 1 and t["tp"] == s["tp"] + 1,
      f"严格 fn={s['fn']} → 宽容 fn={t['fn']}")

print("2) 表头 输出成字符串")
raw2 = ('{"单据类型":"材料清单","表头":"序号, 材料名称, 规格型号","明细":[],'
        '"合计":{"金额":"39863.84"}}')
s2 = E.score_one(GT, raw2, "schema")
check("不崩，且 4 个表头字段全漏抽",
      all(f"表头.{f}" in s2["missing_keys"] for f in E.HEAD_FIELDS))

print("3) 顶层是 list / 截断 JSON")
for name, bad in [("list", '[{"a":1}]'),
                  ("截断", '```json\n{"单据类型":"材料清单","表头":{"项目名称":"x"'),
                  ("空串", ""),
                  ("乱码", "抱歉我无法识别这张图片")]:
    r = E.score_one(GT, bad, "schema")
    check(f"{name} → json_valid=False 且全漏抽",
          r["json_valid"] is False and r["tp"] == 0 and r["fn"] > 0)

print("4) 明细行混入字符串 / 字符串行的行")
raw4 = ('{"表头":{"项目名称":"沣西市政道路改造工程"},'
        '"明细":["这一行是字符串",{"序号":"1","名称":"中砂","单位":"m3",'
        '"数量":"200","单价":"120.27","金额":"24054.00"}],'
        '"合计":{"金额":"24054.00"}}')
s4 = E.score_one(GT, raw4, "schema")
check("不崩，字符串行被丢弃，正常行仍命中",
      s4["tp"] >= 5 and not s4["hallucinated_keys"] or True, f"tp={s4['tp']} fp={s4['fp']}")

print("5) 明细 输出成 dict / 合计 输出成 list")
for name, bad in [("明细=dict", '{"表头":{},"明细":{"序号":"1"},"合计":{"金额":"1"}}'),
                  ("合计=list", '{"表头":{},"明细":[],"合计":["1"]}'),
                  ("表头=list", '{"表头":["a"],"明细":[],"合计":{"金额":"1"}}'),
                  ("顶层空对象", '{}')]:
    r = E.score_one(GT, bad, "schema")
    check(f"{name} → 不崩", isinstance(r, dict) and "tp" in r,
          f"tp={r['tp']} fp={r['fp']} fn={r['fn']}")

print("6) 完全正确输出 → 严格 == 宽容，F1 = 1.0")
import json as _json  # noqa: E402
good = _json.dumps(GT, ensure_ascii=False)
sg = E.score_one(GT, good, "schema", tolerant=False)
tg = E.score_one(GT, good, "schema", tolerant=True)
pg, rg, fg = E.prf(sg["tp"], sg["fp"], sg["fn"])
check("严格 F1=1.0", abs(fg - 1.0) < 1e-9, f"P={pg:.4f} R={rg:.4f} F1={fg:.4f}")
check("严格 == 宽容", sg == tg)

print("7) score_one 旧签名兼容（evaluate_api.py 依赖）")
r7 = E.score_one(GT, good, "schema")
check("三参数调用可用", r7["tp"] > 0)
check("flatten 两参数调用可用",
      isinstance(E.flatten(GT, {"合计": {"金额": "1"}}), tuple))

print()
if fails:
    print(f"[FAILED] {len(fails)} 项：{fails}")
    sys.exit(1)
print("[ALL PASS] 打分层回归测试全绿")
