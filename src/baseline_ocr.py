"""RapidOCR + geometric/header rules. Extraction never receives GT or sample IDs.

Develop on val only; freeze this file/config before a held-out run. CPU only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import time
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import scoring as S

ROOT = Path(__file__).resolve().parents[1]
RULE_VERSION = "ocr-rules-1.0.0"
ENGINE_CONFIG = {"intra_op_num_threads": 4, "inter_op_num_threads": 1,
                 "text_score": 0.5, "det_use_cuda": False, "rec_use_cuda": False,
                 "cls_use_cuda": False, "det_use_dml": False, "rec_use_dml": False,
                 "cls_use_dml": False, "det_limit_side_len": 736, "det_limit_type": "min"}
ALIASES = {"序号": {"序号"}, "名称": {"材料名称", "名称", "品名"},
           "规格型号": {"规格型号", "规格"}, "单位": {"单位"}, "数量": {"数量"},
           "单价": {"单价", "单价(元)"}, "金额": {"金额", "金额(元)"}}
HEAD_LABELS = {"项目名称": ["项目名称"], "供应商": ["供应商"],
               "单据编号": ["单据编号", "编号"], "日期": ["日期"]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def compact(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def heading(text):
    text = compact(text)
    # Currency suffix punctuation may be separated/missing in OCR; it is a label,
    # never a numeric value. Do not apply this cleanup to extracted amounts.
    label = re.sub(r"[()]", "", text)
    if label in {"单价元", "金额元"}:
        return label[:-1]
    return next((key for key, aliases in ALIASES.items() if text in aliases), None)


def clean_value(text, field):
    text = unicodedata.normalize("NFKC", text).strip()
    if field == "日期":
        # Explicit output mapping for the source renderer's unambiguous YYYYMMDD.
        m = re.fullmatch(r"([0-9]{4})([0-9]{2})([0-9]{2})", text)
        if m:
            try:
                return date(*(int(v) for v in m.groups())).isoformat()
            except ValueError:
                pass
        normal = S.normalize(text, "表头.日期")
        if normal and not normal.startswith(S.INVALID):
            return normal
    return text


def blocks_from_result(result):
    return [{"box": [[float(x), float(y)] for x, y in item[0]],
             "text": item[1], "confidence": float(item[2])} for item in result or []]


def geometry(blocks):
    angles = []
    for b in blocks:
        p, q = b["box"][0], b["box"][1]
        if q[0] - p[0] > 15:
            slope = (q[1] - p[1]) / (q[0] - p[0])
            if abs(slope) < .2:
                angles.append(slope)
    slope = statistics.median(angles) if angles else 0.
    tokens = []
    for i, block in enumerate(blocks):
        xs, ys = zip(*block["box"])
        x, y = sum(xs) / 4, sum(ys) / 4
        height = (math.dist(block["box"][0], block["box"][3]) +
                  math.dist(block["box"][1], block["box"][2])) / 2
        tokens.append({**block, "id": i, "x": x, "y": y, "v": y - slope * x,
                       "height": height, "heading": heading(block["text"])})
    return tokens, slope


def group_lines(tokens, tolerance):
    groups = []
    for token in sorted(tokens, key=lambda b: (b["v"], b["x"])):
        if groups and abs(token["v"] - statistics.median(t["v"] for t in groups[-1])) <= tolerance:
            groups[-1].append(token)
        else:
            groups.append([token])
    return [sorted(group, key=lambda b: b["x"]) for group in groups]


def vertical_lines(image, header_y):
    """Image-derived grid lines; no renderer coordinates or known column widths."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    h, w = gray.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, threshold=25,
                            minLineLength=max(35, h * .12), maxLineGap=12)
    candidates = []
    for x1, y1, x2, y2 in lines[:, 0] if lines is not None else []:
        dx, dy = float(x2 - x1), float(y2 - y1)
        if abs(dy) < 35 or abs(dx / dy) > .2:
            continue
        if min(y1, y2) > header_y + h * .2 or max(y1, y2) < header_y:
            continue
        lean = dx / dy
        x = float(x1) + (header_y - float(y1)) * lean
        candidates.append({"x": x, "lean": lean, "length": abs(dy), "at_y": header_y})
    # Retain the longest segment per near-identical x intercept (two Canny edges).
    kept = []
    for line in sorted(candidates, key=lambda l: -l["length"]):
        if all(abs(line["x"] - k["x"]) > 5 for k in kept):
            kept.append(line)
    return sorted(kept, key=lambda l: l["x"])


def extract(image, blocks):
    """Return schema-shaped JSON + trace. No arithmetic correction or GT lookup."""
    tokens, slope = geometry(blocks)
    height = statistics.median(t["height"] for t in tokens) if tokens else 16.
    lines = group_lines(tokens, max(5., height * .7))
    pred = {"单据类型": "", "表头": {k: "" for k in S.HEAD_FIELDS}, "明细": [], "合计": {"金额": ""}}
    trace = {"slope": slope, "median_text_height": height, "warnings": [], "fields": {}, "rows": []}
    best = max(lines, key=lambda g: len({t["heading"] for t in g if t["heading"]}), default=[])
    headings = {t["heading"]: t for t in best if t["heading"]}
    header_v = statistics.median(t["v"] for t in best) if len(headings) >= 4 else float("inf")
    for token in tokens:
        text = compact(token["text"])
        if text in {"工程材料清单", "材料清单"} and token["v"] < header_v:
            pred["单据类型"] = "材料清单"
            trace["fields"]["单据类型"] = [token["id"]]
        if token["v"] >= header_v:
            continue
        for field, labels in HEAD_LABELS.items():
            for label in labels:
                # Whole label at the start of an OCR box; no supplier/name dictionaries.
                m = re.match(r"^" + re.escape(label) + r"\s*[:：]\s*(.*)$", token["text"].strip())
                if not m:
                    continue
                value = m[1].strip()
                sources = [token["id"]]
                if not value:
                    right = [t for t in tokens if t["x"] > token["x"] and abs(t["v"] - token["v"]) < height * .6]
                    if right:
                        nearest = min(right, key=lambda t: t["x"])
                        value = nearest["text"]
                        sources.append(nearest["id"])
                if value and not pred["表头"][field]:
                    pred["表头"][field] = clean_value(value, field)
                    trace["fields"]["表头." + field] = sources
    totals = [t for t in tokens if compact(t["text"]).startswith(("合计金额", "合计")) and t["v"] > header_v]
    total = min(totals, key=lambda t: t["v"]) if totals else None
    if total:
        candidates = [t for t in tokens if t["x"] > total["x"] and abs(t["v"] - total["v"]) < height * .9
                      and re.fullmatch(r"[+-]?[0-9][0-9,.]*", compact(t["text"]))]
        if candidates:
            token = min(candidates, key=lambda t: t["x"])
            pred["合计"]["金额"] = clean_value(token["text"], "金额")
            trace["fields"]["合计.金额"] = [token["id"]]
    if len(headings) < 4:
        trace["warnings"].append("fewer_than_four_column_headings")
        return pred, trace
    ordered = sorted(headings.items(), key=lambda item: item[1]["x"])
    header_y = statistics.median(t["y"] for t in best)
    grid = vertical_lines(image, header_y)
    grid_columns = {}
    for field, token in ordered:
        left = [line for line in grid if line["x"] < token["x"]]
        right = [line for line in grid if line["x"] > token["x"]]
        if left and right:
            grid_columns[field] = (left[-1], right[0])
    boundaries = []
    for (_, left), (_, right) in zip(ordered, ordered[1:]):
        inside = [line for line in grid if left["x"] + 4 < line["x"] < right["x"] - 4]
        if inside:
            edge = max(inside, key=lambda l: l["length"])
            boundaries.append({**edge, "source": "image_grid"})
        else:
            boundaries.append({"x": (left["x"] + right["x"]) / 2, "lean": 0., "at_y": header_y,
                               "source": "heading_midpoint"})
    trace["column_headings"] = {key: token["id"] for key, token in ordered}
    trace["boundaries"] = boundaries
    trace["grid_columns"] = grid_columns
    footers = [t["v"] for t in tokens if compact(t["text"]).startswith(("制单人", "审核人", "备注")) and t["v"] > header_v]
    end_v = total["v"] - height * .55 if total else min(footers, default=float("inf")) - height
    body = [t for t in tokens if header_v + height * .7 < t["v"] < end_v]
    for group in group_lines(body, max(5., height * .7)):
        cells = {}
        for token in group:
            if len(grid_columns) == len(ordered):
                # A missing heading leaves an unmapped interval. Never pour that
                # entire column into the next recognized field.
                matches = [field for field, (left, right) in grid_columns.items()
                           if left["x"] + left["lean"] * (token["y"]-header_y) < token["x"]
                           < right["x"] + right["lean"] * (token["y"]-header_y)]
                if len(matches) != 1:
                    trace["warnings"].append("token_in_unmapped_column")
                    continue
                field = matches[0]
            else:
                column = sum(token["x"] > b["x"] + b["lean"] * (token["y"] - b["at_y"]) for b in boundaries)
                field = ordered[column][0]
            cells.setdefault(field, []).append(token)
        values = {field: clean_value(" ".join(t["text"] for t in values), field) for field, values in cells.items()}
        numeric = sum(bool(re.fullmatch(r"[+-]?[0-9][0-9,.]*", compact(values.get(k, "")))) for k in ("数量", "单价", "金额"))
        has_serial = bool(re.fullmatch(r"[0-9]+", values.get("序号", "")))
        sources = {field: [t["id"] for t in ts] for field, ts in cells.items()}
        if has_serial or numeric >= 2:
            row = {field: values.get(field, "") for field in S.ROW_FIELDS}
            pred["明细"].append(row)
            trace["rows"].append({"sources": sources, "v": statistics.median(t["v"] for t in group)})
        elif pred["明细"] and set(values) <= {"名称", "规格型号"}:
            distance = statistics.median(t["v"] for t in group) - trace["rows"][-1]["v"]
            if 0 < distance < height * 1.8:
                for field, value in values.items():
                    pred["明细"][-1][field] += " " + value
                    trace["rows"][-1]["sources"].setdefault(field, []).extend(sources[field])
            else:
                trace["warnings"].append("unassigned_text_line")
        else:
            trace["warnings"].append("unassigned_text_line")
    for i, row in enumerate(trace["rows"]):
        for field, ids in row["sources"].items():
            trace["fields"][f"明细[{i}].{field}"] = ids
    return pred, trace


def environment():
    import rapidocr_onnxruntime
    root = Path(rapidocr_onnxruntime.__file__).parent
    return {"engine_config": ENGINE_CONFIG,
            "packages": {p: importlib.metadata.version(p) for p in ["rapidocr-onnxruntime", "onnxruntime", "numpy", "opencv-python", "pillow"]},
            "models": {p.name: sha(p) for p in sorted((root / "models").glob("*.onnx"))},
            "package_config_sha256": sha(root / "config.yaml"),
            "rules_version": RULE_VERSION, "rules_sha256": sha(__file__),
            "scorer_version": S.SCORER_VERSION, "scorer_sha256": sha(S.__file__)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--frozen-config", help="Required for test runs; rejects changes after development")
    ap.add_argument("--ocr-cache", help="Replay saved OCR on development images only; timing is not inference")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    if out.exists():
        raise SystemExit("Output exists; choose a new directory")
    config = environment()
    if args.frozen_config:
        frozen = json.loads(Path(args.frozen_config).read_text(encoding="utf-8"))
        if frozen["configuration"] != config:
            raise SystemExit("Frozen rule/engine/scorer configuration changed")
    elif "test" in data.name.lower():
        raise SystemExit("Test evaluation requires --frozen-config")
    records = read_rows(data)
    if args.limit:
        records = records[:args.limit]
    if not records or len({Path(r["image"]).name for r in records}) != len(records):
        raise SystemExit("Empty manifest or duplicate image IDs")
    if args.frozen_config:
        dev_sources = set(frozen["development_source_ids"])
        if dev_sources & {r.get("meta", {}).get("stem") for r in records}:
            raise SystemExit("Development/test source groups overlap")
    cached = {r["image"]: r for r in read_rows(args.ocr_cache)} if args.ocr_cache else {}
    from rapidocr_onnxruntime import RapidOCR
    start = time.perf_counter()
    engine = RapidOCR(**ENGINE_CONFIG) if not cached else None
    load_seconds = time.perf_counter() - start
    # One fixed warmup image, no GT or test data. Excluded from per-image latency.
    if engine:
        warmup = np.full((400, 760, 3), 255, np.uint8)
        cv2.putText(warmup, "OCR warmup 123.45", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        engine(warmup)
    out.mkdir(parents=True)
    all_scores = []
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as pf, (out / "ocr.jsonl").open("w", encoding="utf-8") as of:
        for i, record in enumerate(records):
            image_path = Path(record["image"])
            error = None
            blocks, trace = [], {}
            started = time.perf_counter()
            raw_bytes = image_path.read_bytes()  # Missing source data aborts, never silently changes the test set.
            image_hash = hashlib.sha256(raw_bytes).hexdigest()
            image = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot decode {image_path}")
            try:
                if cached:
                    item = cached[image_path.name]
                    if item["image_sha256"] != image_hash:
                        raise ValueError("OCR cache image hash mismatch")
                    blocks = item["blocks"]
                else:
                    result, _ = engine(image)
                    blocks = blocks_from_result(result)
                pred, trace = extract(image, blocks)
                raw = json.dumps(pred, ensure_ascii=False)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                pred, raw = None, ""
            elapsed = time.perf_counter() - started
            score = S.score_one(record["gt"], raw)
            score["inference_error"] = error
            item = {"image": image_path.name, "image_sha256": image_hash, "image_size": [image.shape[1], image.shape[0]],
                    "blocks": blocks, "trace": trace, "end_to_end_seconds": elapsed if not cached else None,
                    "inference_error": error}
            of.write(json.dumps(item, ensure_ascii=False) + "\n"); of.flush()
            full = {"image": image_path.name, "sample_id": image_path.name, "gt": record["gt"], "pred": raw,
                    "level": record.get("meta", {}).get("level"), "stem": record.get("meta", {}).get("stem"),
                    **score, "end_to_end_seconds": item["end_to_end_seconds"]}
            pf.write(json.dumps(full, ensure_ascii=False) + "\n"); pf.flush()
            all_scores.append(full)
            if (i + 1) % 6 == 0 or i + 1 == len(records):
                print(f"{i+1}/{len(records)} images complete", flush=True)
    summary = S.summarize(all_scores)
    timings = [r["end_to_end_seconds"] for r in all_scores if r["end_to_end_seconds"] is not None]
    summary.update(configuration=config, manifest=str(data), manifest_sha256=sha(data),
                   sample_gt_sha256=S.stable_hash([{"image": r["image"], "gt": r["gt"]} for r in all_scores]),
                   frozen_config_sha256=sha(args.frozen_config) if args.frozen_config else None,
                   image_content_list_sha256=S.stable_hash([{ "image": r["image"], "sha256": r["image_sha256"]} for r in read_rows(out / "ocr.jsonl")]),
                   created_at=datetime.now(timezone.utc).isoformat(), backend="CPUExecutionProvider",
                   platform=platform.platform(), processor=platform.processor(), logical_cpus=os.cpu_count(),
                   model_load_seconds=load_seconds if not cached else None,
                   warmup="one synthetic text image excluded", timing_scope="disk read + decode + OCR + rules + JSON; excludes scoring and result-file writes",
                   mean_seconds=statistics.mean(timings) if timings else None,
                   median_seconds=statistics.median(timings) if timings else None,
                   p95_seconds=float(np.percentile(timings, 95)) if timings else None,
                   by_level={level: S.summarize([r for r in all_scores if r["level"] == level]) for level in sorted({r["level"] for r in all_scores})},
                   ocr_cache_replayed=bool(cached), output_kind="program-serialized JSON; strict JSON validity is not model instruction-following")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n", "f1", "schema_valid_rate", "document_correct_rate", "amount_accuracy", "mean_seconds")}, indent=2))


if __name__ == "__main__":
    main()
