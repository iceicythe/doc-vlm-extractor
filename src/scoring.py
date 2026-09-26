"""CPU-only, versioned scoring shared by local/API evaluation and offline audits."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

SCORER_VERSION = "2.0.0"
SCHEMA_VERSION = "materials-v1"
ROW_ALIGNMENT = "reading-order"
HEAD_FIELDS = ["项目名称", "供应商", "单据编号", "日期"]
ROW_FIELDS = ["序号", "名称", "规格型号", "单位", "数量", "单价", "金额"]
FLAT_FIELDS = {"store_name", "store_addr", "tel", "date", "time", "subtotal", "tax", "total"}
INVALID = "\x00invalid:"


def field_kind(path: str) -> str:
    if path in {"合计.金额", "subtotal", "tax", "total"} or re.fullmatch(
        r"明细\[\d+\]\.(数量|单价|金额)", path
    ):
        return "number"
    if path in {"表头.日期", "date"}:
        return "date"
    return "text"


def normalize(v: object, path: str = "") -> str:
    """Missing/null/empty mean no extracted value; only typed numeric paths use Decimal.

    Text keeps internal spaces, punctuation, units and leading zeros. Wrong leaf
    types remain errors, even if their textual rendering resembles a GT string.
    """
    if v is None:
        return ""
    if not isinstance(v, str):
        return INVALID + type(v).__name__ + ":" + json.dumps(v, ensure_ascii=False, sort_keys=True)
    s = unicodedata.normalize("NFKC", v).strip()
    if not s:
        return ""
    kind = field_kind(path)
    if kind == "number":
        # Commas only in groups of three; no currencies, units, exponents or guessing.
        if not re.fullmatch(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?", s):
            return INVALID + s
        try:
            value = Decimal(s.replace(",", ""))
            if not value.is_finite():
                return INVALID + s
            if value == 0:
                return "0"
            # normalize() on Decimal can round using the ambient context; don't use it.
            out = format(value, "f")
            return out.rstrip("0").rstrip(".") if "." in out else out
        except InvalidOperation:
            return INVALID + s
    if kind == "date":
        m = re.fullmatch(r"([0-9]{4})([-/.])([0-9]{1,2})\2([0-9]{1,2})", s)
        cn = re.fullmatch(r"([0-9]{4})年([0-9]{1,2})月([0-9]{1,2})日?", s)
        parts = (m[1], m[3], m[4]) if m else cn.groups() if cn else None
        if parts:
            try:
                return date(*(int(x) for x in parts)).isoformat()
            except ValueError:
                pass
        return INVALID + s
    return s


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate JSON key: {key}")
        obj[key] = value
    return obj


def _loads(text):
    return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_object)


def strict_json_valid(text: str) -> bool:
    if not isinstance(text, str):
        return False
    try:
        _loads(text)
        return True
    except (ValueError, TypeError, RecursionError):
        return False


def parse_json(text: str) -> dict | None:
    """Loose object extraction; a valid JSON array/scalar is never salvaged as an object."""
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.strip()
    try:
        obj = _loads(t)
        return obj if isinstance(obj, dict) else None
    except (ValueError, RecursionError):
        pass
    # If the entire answer is JSON syntax with duplicate keys/nonfinite values,
    # reject it instead of extracting an inner object.
    if t.startswith(("{", "[")):
        try:
            json.loads(t)
            return None
        except (ValueError, RecursionError):
            pass
    block = re.search(r"```(?:json)?\s*(.+?)```", t, re.S)
    candidate = block.group(1).strip() if block else t[t.find("{"):t.rfind("}") + 1]
    try:
        obj = _loads(candidate)
        return obj if isinstance(obj, dict) else None
    except (ValueError, TypeError, RecursionError):
        return None


def align_rows(gt_rows: list, pred_rows: list) -> list[tuple[int, int]]:
    """Primary metric uses reading order, regardless of duplicate/missing serial numbers."""
    return [(i, i) for i in range(min(len(gt_rows), len(pred_rows)))]


def _expected(path: str, mode: str) -> str | None:
    if mode == "flat":
        return "leaf" if path in FLAT_FIELDS else None
    if path in {"表头", "合计"} or re.fullmatch(r"明细\[\d+\]", path):
        return "object"
    if path == "明细":
        return "array"
    if path == "单据类型" or path == "合计.金额" or path in {f"表头.{x}" for x in HEAD_FIELDS}:
        return "leaf"
    m = re.fullmatch(r"明细\[\d+\]\.(.+)", path)
    return "leaf" if m and m[1] in ROW_FIELDS else None


def _key(key: str) -> str:
    # Literal punctuation in extra keys must not impersonate a nested field path.
    return key.replace("\\", "\\\\").replace(".", "\\.").replace("[", "\\[").replace("]", "\\]")


def extra_fields(obj: dict | None, mode: str = "schema") -> list[str]:
    found = []

    def visit(value, path):
        if path and _expected(path, mode) is None:
            found.append(path)
            return
        if isinstance(value, dict):
            for k, v in value.items():
                visit(v, f"{path}.{_key(k)}" if path else _key(k))
        elif isinstance(value, list):
            for i, v in enumerate(value):
                visit(v, f"{path}[{i}]")

    if isinstance(obj, dict):
        visit(obj, "")
    return found


def flatten_object(obj: dict | None, mode: str = "schema") -> dict:
    """Preserve extra keys and malformed containers as FP, including empty extras."""
    out = {}

    def visit(value, path):
        expected = _expected(path, mode)
        if expected is None:
            out[path] = INVALID + "extra:" + json.dumps(value, ensure_ascii=False, sort_keys=True)
        elif expected == "object" and not isinstance(value, dict):
            out[path] = INVALID + "object"
        elif expected == "array" and not isinstance(value, list):
            out[path] = INVALID + "array"
        elif expected == "leaf":
            norm = normalize(value, path)
            if norm:
                out[path] = norm
        elif isinstance(value, dict):
            for k, v in value.items():
                visit(v, f"{path}.{_key(k)}")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                visit(v, f"{path}[{i}]")

    if isinstance(obj, dict):
        for key, value in obj.items():
            visit(value, _key(key))
    return out


def flatten(gt: dict, pred: dict, tolerant: bool = False) -> tuple[dict, dict]:
    if tolerant and isinstance(pred, dict) and isinstance(pred.get("合计"), str):
        norm = normalize(pred["合计"], "合计.金额")
        if norm and not norm.startswith(INVALID):
            pred = {**pred, "合计": {"金额": pred["合计"]}}
    return flatten_object(gt), flatten_object(pred)


def flatten_flat(gt: dict, pred: dict, tolerant: bool = False) -> tuple[dict, dict]:
    return flatten_object(gt, "flat"), flatten_object(pred, "flat")


def schema_errors(obj: dict | None, mode: str = "schema") -> list[str]:
    errors = []
    if not isinstance(obj, dict):
        return ["$: expected object"]

    def object_fields(value, path, allowed, required):
        if not isinstance(value, dict):
            errors.append(f"{path or '$'}: expected object")
            return
        for key in sorted(required - value.keys()):
            errors.append(f"{path + '.' if path else ''}{key}: required")
        for key in value:
            p = f"{path}.{key}" if path else key
            if key not in allowed:
                errors.append(f"{p}: extra field")
            elif not isinstance(value[key], str):
                errors.append(f"{p}: expected string")
            elif key in required and not normalize(value[key]):
                errors.append(f"{p}: empty required value")
            elif field_kind(p) != "text" and normalize(value[key], p).startswith(INVALID):
                errors.append(f"{p}: invalid {field_kind(p)}")

    if mode == "flat":
        object_fields(obj, "", FLAT_FIELDS, set())
        return errors
    tops = {"单据类型", "表头", "明细", "合计"}
    for key in sorted(tops - obj.keys()):
        errors.append(f"{key}: required")
    for key in sorted(obj.keys() - tops):
        errors.append(f"{key}: extra field")
    if obj.get("单据类型") != "材料清单":
        errors.append("单据类型: expected 材料清单")
    object_fields(obj.get("表头"), "表头", set(HEAD_FIELDS), set(HEAD_FIELDS))
    object_fields(obj.get("合计"), "合计", {"金额"}, {"金额"})
    rows = obj.get("明细")
    if not isinstance(rows, list):
        errors.append("明细: expected array")
    else:
        if not 1 <= len(rows) <= 8:
            errors.append("明细: expected 1..8 rows")
        for i, row in enumerate(rows):
            object_fields(row, f"明细[{i}]", set(ROW_FIELDS), set(ROW_FIELDS) - {"规格型号"})
    return errors


def score_pred(gt: dict, pred: dict | None, mode: str = "schema", tolerant: bool = False) -> dict:
    if mode not in {"schema", "flat"}:
        raise ValueError(f"Unknown mode: {mode}")
    g, p = (flatten_flat if mode == "flat" else flatten)(gt, pred, tolerant)
    matches = {k for k, v in g.items() if p.get(k) == v and not v.startswith(INVALID)}
    amounts = [k for k in g if k == "合计.金额" or re.fullmatch(r"明细\[\d+\]\.金额", k)
               or mode == "flat" and k in {"subtotal", "tax", "total"}]
    errors = schema_errors(pred, mode)
    tp, fp, fn = len(matches), len(p) - len(matches), len(g) - len(matches)
    return {
        "scorer_version": SCORER_VERSION,
        "json_valid": isinstance(pred, dict),  # compatibility alias: loose object parse
        "loose_parse_success": isinstance(pred, dict),
        "strict_json_valid": None,  # unknowable without raw generation text
        "schema_valid": not errors, "schema_errors": errors,
        "tp": tp, "fp": fp, "fn": fn,
        "hallucinated_keys": [k for k in p if k not in g],  # legacy alias, not hallucination evidence
        "missing_keys": [k for k in g if k not in p],
        "extra_fields": extra_fields(pred, mode),
        "amount_correct": sum(k in matches for k in amounts), "amount_total": len(amounts),
        "document_correct": not errors and not fp and not fn,
        "field_errors": [{"path": k, "gt": g.get(k), "pred": p.get(k)}
                         for k in sorted(g.keys() | p.keys()) if k not in matches],
    }


def score_one(gt: dict, raw_text: str, mode: str = "schema", tolerant: bool = False) -> dict:
    result = score_pred(gt, parse_json(raw_text), mode, tolerant)
    result["strict_json_valid"] = strict_json_valid(raw_text)
    result["document_correct"] &= result["strict_json_valid"]
    return result


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def summarize(scores: list[dict]) -> dict:
    """All sample rates include parse/inference failures; fields use micro counts."""
    n = len(scores)
    counts = {k: sum(s[k] for s in scores) for k in ("tp", "fp", "fn", "amount_correct", "amount_total")}
    p, r, f = prf(counts["tp"], counts["fp"], counts["fn"])
    result = {"scorer_version": SCORER_VERSION, "schema_version": SCHEMA_VERSION,
              "row_alignment": ROW_ALIGNMENT, "n": n, **counts,
              "precision": p, "recall": r, "f1": f,
              "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for name in ("strict_json_valid", "loose_parse_success", "schema_valid", "document_correct"):
        known = sum(s.get(name) is not None for s in scores)
        count = sum(bool(s.get(name)) for s in scores)
        result[name + "_count"] = count
        result[name + "_denominator"] = n
        result[name + "_observed"] = known
        result[name + "_rate"] = count / n if n and known == n else None
    extra = sum(bool(s["extra_fields"]) for s in scores)
    result.update(extra_field_sample_count=extra, extra_field_sample_denominator=n,
                  extra_field_sample_rate=extra / n if n else None,
                  parse_failure_count=sum(not s["loose_parse_success"] for s in scores),
                  inference_failure_count=(sum(bool(s.get("inference_error")) for s in scores)
                                           if all("inference_error" in s for s in scores) else None),
                  inference_status_observed=sum("inference_error" in s for s in scores),
                  amount_accuracy=counts["amount_correct"] / counts["amount_total"] if counts["amount_total"] else None)
    return result
