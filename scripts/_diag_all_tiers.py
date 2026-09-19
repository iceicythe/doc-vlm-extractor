"""扫描所有 eval_*_preds.partial.jsonl，统计三类结构异常的出现率。

用途：确认「合计 被输出成标量」是零样本独有，还是各档位普遍存在 ——
如果是零样本独有，那么严格口径下 SFT 系列的 F1 不受影响，可比性成立。
"""
import json
import re
from collections import Counter
from pathlib import Path

OUT = Path(r"C:\MiniVLM\outputs")
CODE = re.compile(r"```(?:json)?\s*(.+?)```", re.S)
_lines = []


def print(*a, **kw):  # noqa: A001
    _lines.append(" ".join(str(x) for x in a))


def parse(text):
    if not text:
        return None
    s = text.strip()
    m = CODE.search(s)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(s[i:j + 1])
        except Exception:
            return None
    return None


for p in sorted(OUT.glob("eval_*_preds.partial.jsonl")):
    total = parse_fail = total_scalar = head_scalar = rows_scalar = 0
    for line in p.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("_meta"):
            continue
        total += 1
        obj = parse(r["pred"])
        if not isinstance(obj, dict):
            parse_fail += 1
            continue
        t = obj.get("合计")
        if t is not None and not isinstance(t, dict):
            total_scalar += 1
        h = obj.get("表头")
        if h is not None and not isinstance(h, dict):
            head_scalar += 1
        rows = obj.get("明细")
        if isinstance(rows, list) and any(not isinstance(x, dict) for x in rows):
            rows_scalar += 1

    print(f"{p.name:<42} n={total:4d}  解析失败={parse_fail:3d}  "
          f"合计标量={total_scalar:3d}  表头标量={head_scalar:3d}  明细含非dict={rows_scalar:3d}")

OUT.joinpath("_diag_all_tiers.txt").write_text("\n".join(_lines) + "\n", encoding="utf-8")
