"""Inventory historical assets and rescore saved raw generations without inference.

Run from repo root: python scripts/audit_evaluation.py
Existing result directories are never overwritten; pass a fresh --out to repeat.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import scoring as S


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rel(path):
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def inventory():
    summaries = {p.stem: p for p in (ROOT / "outputs").glob("eval_*.json")}
    predictions = {p.name.removesuffix("_preds.jsonl"): p for p in (ROOT / "outputs").glob("eval_*_preds.jsonl")}
    assets = []
    for tag in sorted(summaries.keys() | predictions.keys()):
        summary_path = summaries.get(tag)
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path else {}
        pred_path = predictions.get(tag)
        partial = ROOT / "outputs" / (tag + "_preds.partial.jsonl")
        cached = read_rows(partial) if partial.exists() else []
        meta = next((r for r in cached if r.get("_meta")), {})
        rows = read_rows(pred_path) if pred_path else []
        usable = bool(rows) and all(isinstance(r.get("gt"), dict) and isinstance(r.get("pred"), str)
                                    and r.get("image") for r in rows)
        ids = [r.get("image") for r in rows]
        unique = len(ids) == len(set(ids))
        adapter = meta.get("adapter", summary.get("adapter"))
        adapter_path = ROOT / adapter if adapter else None
        adapter_files = list(adapter_path.glob("*.safetensors")) if adapter_path else []
        adapter_cfg = adapter_path / "adapter_config.json" if adapter_path else None
        cfg = json.loads(adapter_cfg.read_text(encoding="utf-8")) if adapter_cfg and adapter_cfg.exists() else {}
        data_name = meta.get("data")
        data_path = ROOT / "data" / "processed" / Path(data_name).name if data_name else None
        data_exists = bool(data_path and data_path.exists())
        manifest = read_rows(data_path) if data_exists else []
        manifest_index = {}
        for r in manifest:
            manifest_index.setdefault(Path(r["image"]).name, []).append(r)
        matched = sum(len(manifest_index.get(Path(r["image"]).name, [])) == 1
                      and manifest_index[Path(r["image"]).name][0].get("gt") == r.get("gt") for r in rows)
        images_present = sum(Path(r["image"]).exists() for r in manifest)
        if usable and unique:
            status = "可直接重算"
        elif cached and data_exists:
            status = "需关联 GT 后核验覆盖"
        elif data_exists and (not adapter or adapter_files):
            status = "需要补推理（尚需核验基座与配置）"
        else:
            status = "缺少条件暂不能复现"
        assets.append({"experiment": tag, "status": status, "n": len(rows) or summary.get("n"),
                       "prediction_file": rel(pred_path) if pred_path else None,
                       "prediction_sha256": digest(pred_path) if pred_path else None,
                       "summary_file": rel(summary_path) if summary_path else None,
                       "summary_sha256": digest(summary_path) if summary_path else None,
                       "partial_file": rel(partial) if partial.exists() else None,
                       "partial_sha256": digest(partial) if partial.exists() else None,
                       "saved_metadata": meta, "summary_model": summary.get("model"),
                       "raw_text_count": sum(isinstance(r.get("pred"), str) for r in rows),
                       "gt_count": sum(isinstance(r.get("gt"), dict) for r in rows),
                       "unique_sample_ids": unique, "adapter": adapter,
                       "adapter_config": cfg,
                       "adapter_files": [{"name": p.name, "bytes": p.stat().st_size} for p in adapter_files],
                       "data_file": rel(data_path) if data_exists else None,
                       "data_sha256": digest(data_path) if data_exists else None,
                       "manifest_sample_count": len(manifest), "manifest_gt_matched": matched,
                       "manifest_images_present": images_present,
                       "prompt_identity_verified": False,
                       "notes": "哈希仅固定本次观察到的文件；旧 prompt_hash 不可跨进程验证。权重存在不等于已验证可加载。"})
    return assets


def legacy_scorer(revision):
    commit = subprocess.check_output(["git", "rev-parse", revision], cwd=ROOT, text=True).strip()
    source = subprocess.check_output(["git", "show", f"{commit}:src/evaluate.py"], cwd=ROOT).decode("utf-8")
    legacy = types.ModuleType("legacy_evaluate")
    legacy.__file__ = str(ROOT / "src" / "evaluate.py")
    exec(compile(source, legacy.__file__, "exec"), legacy.__dict__)
    if getattr(legacy, "SCORER_VERSION", None):
        raise ValueError("Selected revision is not the legacy scorer; use the original report's legacy commit")
    return legacy, commit, hashlib.sha256(source.encode("utf-8")).hexdigest()


def rescore(path, out, legacy, commit, legacy_hash):
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"Empty predictions: {path}")
    ids = [r["image"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs; resolve before scoring")
    mode = "schema" if "表头" in rows[0]["gt"] else "flat"
    for row in rows:
        if not isinstance(row.get("gt"), dict) or not isinstance(row.get("pred"), str):
            raise ValueError("Raw generation and GT are required; parsed objects cannot recover strict JSON validity")
        errors = S.schema_errors(row["gt"], mode)
        if errors:
            raise ValueError(f"GT needs review: {row['image']}: {errors}")
    tag = path.name.removesuffix("_preds.jsonl")
    full, diffs = [], []
    old_scores, precision_scores = [], []
    original_norm = legacy.normalize

    def exact_legacy(value):
        # Diagnostic only: keep legacy type/date/row policies, fix only six-digit rounding.
        text = legacy.unicodedata.normalize("NFKC", str(value).strip())
        text = re.sub(r"\s+", "", text).replace(",", "").replace("，", "")
        if re.fullmatch(r"-?\d+(\.\d+)?", text):
            return S.normalize(text, "合计.金额")
        return original_norm(value)

    for row in rows:
        old = legacy.score_one(row["gt"], row["pred"], mode)
        legacy.normalize = exact_legacy
        try:
            precision = legacy.score_one(row["gt"], row["pred"], mode)
        finally:
            legacy.normalize = original_norm
        new = S.score_one(row["gt"], row["pred"], mode)
        old_scores.append(old); precision_scores.append(precision)
        sample = {"sample_id": row["image"], "image": row["image"], "level": row.get("level"),
                  "stem": row.get("stem"), "gt": row["gt"], "pred": row["pred"],
                  "parsed_pred": S.parse_json(row["pred"]), **new}
        full.append(sample)
        old_counts = {k: old[k] for k in ("tp", "fp", "fn")}
        new_counts = {k: new[k] for k in ("tp", "fp", "fn")}
        if old_counts != new_counts or not new["document_correct"]:
            diffs.append({"sample_id": row["image"], "old": old_counts,
                          "precision_only": {k: precision[k] for k in ("tp", "fp", "fn")},
                          "new": new_counts, "old_new_f1_delta": S.prf(**new_counts)[2] - S.prf(**old_counts)[2],
                          "schema_errors": new["schema_errors"], "field_errors": new["field_errors"]})
    summary = S.summarize(full)
    old_total = {k: sum(r[k] for r in old_scores) for k in ("tp", "fp", "fn")}
    exact_total = {k: sum(r[k] for r in precision_scores) for k in ("tp", "fp", "fn")}
    old_summary_path = path.with_name(tag + ".json")
    published = json.loads(old_summary_path.read_text(encoding="utf-8")) if old_summary_path.exists() else {}
    summary.update(mode=mode, schema_version="materials-v1" if mode == "schema" else "wildreceipt-flat-v1",
                   source_prediction=rel(path), source_prediction_sha256=digest(path),
                   sample_gt_sha256=S.stable_hash([{"image": r["image"], "gt": r["gt"]} for r in rows]),
                   legacy_commit=commit, legacy_scorer_sha256=legacy_hash,
                   legacy_recomputed={**old_total, "f1": S.prf(**old_total)[2]},
                   legacy_precision_only={**exact_total, "f1": S.prf(**exact_total)[2]},
                   historical_reported_f1=published.get("f1"),
                   precision_affected_sample_count=sum(any(a[k] != b[k] for k in ("tp", "fp", "fn"))
                                                       for a, b in zip(old_scores, precision_scores)),
                   inference_performed=False, infer_seconds=None,
                   inference_config_status="See inventory; incomplete historical metadata, no inferred defaults")
    diffs.sort(key=lambda r: -abs(r["old_new_f1_delta"]))
    for suffix, value in [(".json", summary), ("_diffs.json", diffs)]:
        (out / (tag + suffix)).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / (tag + "_preds.jsonl")).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in full), encoding="utf-8")
    return tag, summary, diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/evaluation-v2-final")
    ap.add_argument("--legacy-commit", default="0adab0c6481ad4d5fcb912e9a673f0c8ab4198ad")
    ap.add_argument("--preds", nargs="+", default=["outputs/eval_zero_abl_preds.jsonl", "outputs/eval_abl_base_preds.jsonl"])
    args = ap.parse_args()
    out = (ROOT / args.out).resolve()
    if out.exists():
        raise SystemExit("Output exists; choose a fresh --out directory. Historical files are not overwritten.")
    assets = inventory()
    legacy, commit, legacy_hash = legacy_scorer(args.legacy_commit)
    out.mkdir(parents=True)
    (out / "inventory.json").write_text(json.dumps(assets, ensure_ascii=False, indent=2), encoding="utf-8")
    results = [rescore((ROOT / p).resolve(), out, legacy, commit, legacy_hash) for p in args.preds]
    if len({s["sample_gt_sha256"] for _, s, _ in results}) != 1:
        raise ValueError("Selected experiments have different ordered samples or GT; don't present a paired comparison")
    lines = ["# P0 历史评测资产盘点", "", f"工作目录：`{ROOT}`。仅检查本地资产，未加载模型、未调用 API。",
             "", "状态‘可直接重算’只表示预测和 GT 足够，不等于推理可完整复现。完整配置、文件哈希及缺失项见",
             f"[`inventory.json`](../{rel(out)}/inventory.json)。", "",
             "| 实验 | 样本数 | 原始文本/GT | 数据清单 GT 匹配 | 适配器权重 | 状态 |",
             "|---|---:|---|---|---|---|"]
    for a in assets:
        lines.append(f"| {a['experiment']} | {a['n']} | {a['raw_text_count']}/{a['gt_count']} | {a['manifest_gt_matched']}/{a['n']} | {'存在' if a['adapter_files'] else '未定位/不适用'} | {a['status']} |")
    lines += ["", "## 已知缺口与恢复方式", "",
              "- 原始文本保存在 `pred` 字符串中；不能因字段不叫 raw_text 就判定缺失。",
              "- 旧缓存部分只记录不稳定的 prompt_hash，无法据此核验提示词；当前源代码默认参数不能冒充历史配置。",
              "- 适配器只检查配置和文件存在性；基座完整缓存、驱动、量化库兼容性与模型加载尚未验证。",
              "- 仅有汇总的实验需找回预测，或恢复数据、提示词与模型后补推理；不覆盖或重建历史数据清单。",
              "- 仅有后处理预测、缺来源配置的实验可核验输出，但推理与后处理溯源仍需补齐。",
              "- 推理失败状态未在历史记录中明确保存，报告记为 null（未知），不把它当成 0。",
              "- 本次仅重算下方报告中的两组；其他实验的旧成绩仍未经新口径核验。"]
    (ROOT / "docs/evaluation-inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = ["# P0 评测修复与历史重算", "", f"评分器：`{S.SCORER_VERSION}`；旧代码提交：`{commit}`。",
             "同一固定图片列表和 GT；直接使用历史生成文本，无训练或新推理。", "",
             "| 方法 | 历史 F1 | 旧规则重算 | 仅修数值精度（诊断） | v2 主 F1 | 金额正确/总数 | 整单正确/总数 |",
             "|---|---:|---:|---:|---:|---|---|"]
    for tag, s, _ in results:
        lines.append(f"| {tag} | {s['historical_reported_f1']:.6f} | {s['legacy_recomputed']['f1']:.6f} | {s['legacy_precision_only']['f1']:.6f} | {s['f1']:.6f} | {s['amount_correct']}/{s['amount_total']} | {s['document_correct_count']}/{s['n']} |")
    lines += ["", "## 解释边界", "", "v2 同时改变字段归一化、明细按阅读顺序对齐、单据类型计分、错误结构/额外字段计分。",
              "因此 v2 与旧分数之差不全是金额精度造成，更不能解释为模型提升或退步。",
              "‘仅修数值精度’保留旧类型处理及行匹配，只去掉六位有效数字截断，是定位精度影响的辅助诊断，不是新主指标。",
              "金额准确率只统计明细金额和合计金额；数量、单价计入字段 F1。所有解析失败保留在分母。",
              "Schema 合规基于宽松解析出的对象，整单正确还要求原始输出严格 JSON 合法。", ""]
    for tag, s, diffs in results:
        lines += [f"## {tag}", "", f"- 严格 JSON：{s['strict_json_valid_count']}/{s['n']}；宽松解析：{s['loose_parse_success_count']}/{s['n']}；Schema：{s['schema_valid_count']}/{s['n']}。",
                  f"- 额外字段样本：{s['extra_field_sample_count']}/{s['n']}；解析失败：{s['parse_failure_count']}；推理失败：未知。",
                  f"- 仅修精度改变了 {s['precision_affected_sample_count']} 条样本的字段计数。",
                  f"- [完整汇总](../{rel(out)}/{tag}.json) · [逐样本结果](../{rel(out)}/{tag}_preds.jsonl) · [差异明细](../{rel(out)}/{tag}_diffs.json)", "",
                  "变化最大的样本（字段 F1 差值；可凭 sample_id 在逐样本文件回查 GT 和原始输出）：", ""]
        for d in diffs[:5]:
            lines.append(f"- `{d['sample_id']}`：ΔF1={d['old_new_f1_delta']:+.6f}；旧 TP/FP/FN={d['old']}；新={d['new']}。")
            if d['schema_errors']:
                lines.append("  Schema：" + "；".join(d['schema_errors'][:2]))
            for e in d['field_errors'][:2]:
                lines.append(f"  `{e['path']}`：GT={e['gt']!r}，预测={e['pred']!r}。")
    lines += ["", "## 复算", "", "```powershell", ".\\venv-gld\\Scripts\\python.exe scripts\\audit_evaluation.py --out outputs/evaluation-v2-repeat", "```",
              "", "输出目录必须尚不存在；脚本不覆盖旧预测、旧汇总和旧数据。旧规则对照通过 --legacy-commit 固定历史提交，后续提交不会改变对照代码。",
              "", "下一阶段：OCR＋规则同集对比、陌生版式与真实单据留出评测；本报告不宣称这些已完成。"]
    (ROOT / "docs/evaluation-v2-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({tag: {k: s[k] for k in ("n", "f1", "schema_valid_rate", "document_correct_rate", "amount_accuracy", "precision_affected_sample_count")} for tag, s, _ in results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
