"""Human-review records, arithmetic warnings and exports for materials-v1."""
from __future__ import annotations

import csv
import hashlib
import html
import json
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

import scoring as S

ROOT = Path(__file__).resolve().parent.parent
REVIEW_ROOT = ROOT / "outputs/reviews-v1"
ERROR_TYPES = ["文字识别错误", "行列对应错误", "漏字段/漏明细", "无依据生成", "结构或类型错误"]
CENT = Decimal("0.01")


def parse_document(text: str) -> dict:
    if not S.strict_json_valid(text):
        raise ValueError("编辑内容必须是完整、严格合法的 JSON")
    obj = S.parse_json(text)
    if not isinstance(obj, dict):
        raise ValueError("JSON 顶层必须是对象")
    return obj


def _decimal(value):
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip().replace(",", "").replace("，", "")
    if not text:
        return None
    try:
        value = Decimal(text)
        return value if value.is_finite() else None
    except InvalidOperation:
        return None


def validate_document(obj: dict | None) -> dict:
    schema = S.schema_errors(obj)
    warnings = []
    row_checks = []
    rows = obj.get("明细") if isinstance(obj, dict) else None
    rows = rows if isinstance(rows, list) else []
    row_amounts = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            row_checks.append({"row": i + 1, "status": "不可判定", "reason": "明细行不是对象"})
            continue
        q, p, a = (_decimal(row.get(k)) for k in ("数量", "单价", "金额"))
        if q is None or p is None or a is None:
            row_checks.append({"row": i + 1, "status": "不可判定", "reason": "数量、单价或金额不是有效数值"})
            continue
        expected = (q * p).quantize(CENT, rounding=ROUND_HALF_UP)
        actual = a.quantize(CENT, rounding=ROUND_HALF_UP)
        row_amounts.append(actual)
        if actual != expected:
            item = {"type": "行金额关系异常", "path": f"明细[{i}].金额",
                    "observed": str(a), "expected_by_formula": str(expected),
                    "message": f"第 {i + 1} 行：数量×单价={expected}，图上提取金额={a}；请人工确认，系统未改值"}
            warnings.append(item)
            row_checks.append({"row": i + 1, "status": "需确认", **item})
        else:
            row_checks.append({"row": i + 1, "status": "通过", "observed": str(a), "expected_by_formula": str(expected)})

    total_obj = obj.get("合计") if isinstance(obj, dict) else None
    reported = _decimal(total_obj.get("金额")) if isinstance(total_obj, dict) else None
    total_check = {"status": "不可判定", "reason": "合计金额或明细金额不完整"}
    if reported is not None and len(row_amounts) == len(rows):
        expected_total = sum(row_amounts, Decimal("0")).quantize(CENT, rounding=ROUND_HALF_UP)
        actual_total = reported.quantize(CENT, rounding=ROUND_HALF_UP)
        if actual_total != expected_total:
            item = {"type": "合计关系异常", "path": "合计.金额", "observed": str(reported),
                    "expected_by_formula": str(expected_total),
                    "message": f"明细金额之和={expected_total}，图上提取合计={reported}；请人工确认，系统未改值"}
            warnings.append(item)
            total_check = {"status": "需确认", **item}
        else:
            total_check = {"status": "通过", "observed": str(reported), "expected_by_formula": str(expected_total)}
    return {"schema_valid": not schema, "schema_errors": schema, "warnings": warnings,
            "row_checks": row_checks, "total_check": total_check,
            "policy": "忠实提取优先；算术校验只提示，不自动覆盖图上值"}


def validation_html(text: str) -> str:
    try:
        result = validate_document(parse_document(text))
    except ValueError as exc:
        return f"<div style='color:#b42318'><b>JSON 无法校验：</b>{html.escape(str(exc))}</div>"
    schema = ("Schema 合规" if result["schema_valid"] else
              f"Schema 问题 {len(result['schema_errors'])} 项：" + "；".join(result["schema_errors"][:4]))
    if result["warnings"]:
        items = "".join(f"<li>{html.escape(w['message'])}</li>" for w in result["warnings"])
        arithmetic = f"<b>算术提示 {len(result['warnings'])} 项</b><ul>{items}</ul>"
    else:
        arithmetic = "算术关系未发现异常"
    color = "#067647" if result["schema_valid"] and not result["warnings"] else "#b54708"
    return (f"<div style='padding:10px 12px;border:1px solid #d0d5dd;border-radius:8px;color:{color}'>"
            f"<b>{html.escape(schema)}</b><br>{arithmetic}<br>"
            "<small>忠实提取优先：提示不会自动修改任何字段。</small></div>")


def _leaves(value, path=""):
    out = {}
    if isinstance(value, dict):
        for key, child in value.items():
            out.update(_leaves(child, f"{path}.{key}" if path else key))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            out.update(_leaves(child, f"{path}[{i}]"))
    else:
        out[path] = value
    return out


def changed_fields(before: dict | None, after: dict) -> list[dict]:
    old, new = _leaves(before or {}), _leaves(after)
    return [{"path": path, "before": old.get(path), "after": new.get(path)}
            for path in sorted(set(old) | set(new)) if old.get(path) != new.get(path)]


def _export_rows(obj: dict):
    for key, value in (obj.get("表头") or {}).items():
        yield {"section": "表头", "row": "", "field": key, "value": value}
    for i, row in enumerate(obj.get("明细") or [], 1):
        if isinstance(row, dict):
            for key, value in row.items():
                yield {"section": "明细", "row": i, "field": key, "value": value}
    for key, value in (obj.get("合计") or {}).items():
        yield {"section": "合计", "row": "", "field": key, "value": value}


def save_review(state: dict, edited_text: str, error_types=None, notes="", root: Path = REVIEW_ROOT) -> dict:
    edited = parse_document(edited_text)
    error_types = list(error_types or [])
    unknown = sorted(set(error_types) - set(ERROR_TYPES))
    if unknown:
        raise ValueError("未知错误类型：" + "、".join(unknown))
    raw = str((state or {}).get("raw_output") or "")
    original = S.parse_json(raw)
    now = datetime.now().astimezone()
    base = now.strftime("%Y%m%dT%H%M%S%f")[:-3] + "_" + hashlib.sha256(edited_text.encode()).hexdigest()[:8]
    records, exports = root / "records", root / "exports"
    records.mkdir(parents=True, exist_ok=True); exports.mkdir(parents=True, exist_ok=True)
    record_id = base
    n = 1
    while (records / f"{record_id}.json").exists():
        n += 1; record_id = f"{base}_{n}"
    record = {
        "review_id": record_id, "annotation_version": "review-v1", "reviewed_at": now.isoformat(),
        "sample_id": (state or {}).get("sample_id"), "image_path": (state or {}).get("image_path"),
        "model_version": (state or {}).get("model_version"), "prompt_sha256": (state or {}).get("prompt_sha256"),
        "original_raw_output": raw, "original_parsed": original, "reviewed_document": edited,
        "modified_fields": changed_fields(original, edited), "error_types": error_types,
        "reviewer_notes": str(notes or ""), "validation": validate_document(edited),
    }
    record_path = records / f"{record_id}.json"
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = exports / f"{record_id}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["section", "row", "field", "value"])
        writer.writeheader(); writer.writerows(_export_rows(edited))
    return {"record": record, "record_path": str(record_path), "csv_path": str(csv_path)}


def list_reviews(root: Path = REVIEW_ROOT) -> list[str]:
    folder = root / "records"
    return [str(p) for p in sorted(folder.glob("*.json"), reverse=True)] if folder.exists() else []


def load_review(path: str, root: Path = REVIEW_ROOT) -> dict:
    target = Path(path).resolve()
    records = (root / "records").resolve()
    if records not in target.parents or target.suffix.lower() != ".json":
        raise ValueError("复核记录路径不合法")
    return json.loads(target.read_text(encoding="utf-8"))
