"""Compare frozen OCR test predictions and existing VLM generations, without inference."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import scoring as S


def load(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def file_hash(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_rows(rows):
    return [{"image": r["image"], "gt": r["gt"], "pred": r["pred"],
             "level": r["level"], "stem": r.get("stem"), **S.score_one(r["gt"], r["pred"])} for r in rows]


def main():
    out = ROOT / "outputs/baseline-comparison-v1"
    if out.exists():
        raise SystemExit("Comparison already exists; do not overwrite the frozen test report")
    paths = {"ocr_rules": ROOT / "outputs/ocr-test-v1/predictions.jsonl",
             "zero_shot": ROOT / "outputs/eval_zero_abl_preds.jsonl",
             "sft": ROOT / "outputs/eval_abl_base_preds.jsonl",
             "sft_lexicon": ROOT / "outputs/eval_abl_base_inj_preds.jsonl"}
    raw = {key: load(p) for key, p in paths.items()}
    identity = lambda rows: [{"image": Path(r["image"]).name, "gt": r["gt"]} for r in rows]
    expected = identity(raw["ocr_rules"])
    for key, rows in raw.items():
        if identity(rows) != expected:
            raise ValueError(f"Image order or GT mismatch: {key}")
    for before, after in zip(raw["sft"], raw["sft_lexicon"]):
        if after.get("pred_orig") != before["pred"]:
            raise ValueError("Postprocessing lineage does not match SFT raw generation")
    ocr_summary = json.loads((ROOT / "outputs/ocr-test-v1/summary.json").read_text(encoding="utf-8"))
    if ocr_summary["scorer_sha256"] != file_hash(ROOT / "src/scoring.py"):
        raise ValueError("Scorer changed since held-out evaluation")
    rows = {key: score_rows(value) for key, value in raw.items()}
    summaries = {key: S.summarize(value) for key, value in rows.items()}
    summaries["ocr_rules"] = ocr_summary
    lex_changes = []
    for before, after in zip(rows["sft"], rows["sft_lexicon"]):
        gb, pb = S.flatten(before["gt"], S.parse_json(before["pred"]))
        _, pa = S.flatten(after["gt"], S.parse_json(after["pred"]))
        for path in sorted(pb.keys() | pa.keys()):
            if pb.get(path) != pa.get(path):
                lex_changes.append({"image": before["image"], "path": path, "gt": gb.get(path),
                                    "before": pb.get(path), "after": pa.get(path),
                                    "fixed": pb.get(path) != gb.get(path) and pa.get(path) == gb.get(path),
                                    "broke": pb.get(path) == gb.get(path) and pa.get(path) != gb.get(path)})
    blocks = {r["image"]: r for r in load(ROOT / "outputs/ocr-test-v1/ocr.jsonl")}
    candidates = []
    for level in ("clean", "medium", "heavy"):
        failed = [r for r in rows["ocr_rules"] if r["level"] == level and not r["document_correct"]]
        candidates.extend(sorted(failed, key=lambda r: S.prf(r["tp"], r["fp"], r["fn"])[2])[:2])
    errors = []
    for r in candidates:
        source = blocks[r["image"]]
        evidence = []
        for err in r["field_errors"][:12]:
            ids = source["trace"].get("fields", {}).get(err["path"], [])
            evidence.append({**err, "assigned_ocr_blocks": [source["blocks"][i] for i in ids]})
        errors.append({"image": r["image"], "level": r["level"], "gt": r["gt"],
                       "prediction": S.parse_json(r["pred"]), "schema_errors": r["schema_errors"],
                       "field_errors_with_evidence": evidence, "warnings": source["trace"].get("warnings", []),
                       "note": "Automatic trace, not an independently image-audited causal label"})
    out.mkdir()
    comparison = {"scorer_version": S.SCORER_VERSION, "scorer_sha256": file_hash(ROOT / "src/scoring.py"),
                  "sample_gt_sha256": S.stable_hash(expected), "source_groups": len({r["stem"] for r in rows["ocr_rules"]}),
                  "source_files": {k: {"path": str(p.relative_to(ROOT)), "sha256": file_hash(p)} for k, p in paths.items()},
                  "methods": summaries, "lexicon_changes": lex_changes,
                  "lexicon_file_sha256": file_hash(ROOT / "data/lexicon/v1.json"),
                  "limitations": ["VLM reused historical generation; no new GPU inference or controlled latency comparison",
                                  "OCR and lexicon outputs are program-serialized JSON; strict JSON does not measure VLM compliance there",
                                  "Current image bytes hashed for OCR; historical VLM files lack image-content hashes",
                                  "Legacy lexicon threshold-selection provenance is incomplete; this is retrospective output comparison",
                                  "Synthetic known-layout test only, not evidence of real-document or unseen-layout generalization"]}
    (out / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "sft_lexicon_preds.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows["sft_lexicon"]), encoding="utf-8")
    (out / "failure_cases.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    dev = [json.loads((ROOT / f"outputs/ocr-dev-v{i}/summary.json").read_text(encoding="utf-8")) for i in (1, 2)]
    (out / "development-summary.json").write_text(json.dumps(dev, ensure_ascii=False, indent=2), encoding="utf-8")
    labels = {"ocr_rules": "OCR＋规则（CPU，本轮新运行）", "zero_shot": "原始 VLM（历史预测重算）",
              "sft": "微调 VLM（历史预测重算）", "sft_lexicon": "微调 VLM＋词表（历史后处理重算）"}
    lines = ["# OCR＋规则基线与四方案比较", "", "本轮没有训练。OCR 新运行 102 张图片；其余方案复用历史预测并按同一评分器重算。",
             f"评分器 `{S.SCORER_VERSION}`，样本/GT 哈希 `{S.stable_hash(expected)}`。",
             "", "## 数据边界", "", "开发集：原 val 的前 8 个源单据组，包含 clean/medium/heavy，共 24 张。",
             "规则在开发集上修订一次：避免标题漏识别时把缺失列并入相邻列，并容许列标题货币后缀括号缺失。",
             f"开发集 F1：{dev[0]['f1']:.4f} → {dev[1]['f1']:.4f}；第二次复用相同 OCR 缓存，只调整规则。",
             "正式测试前冻结源码、评分器、依赖、OCR 模型哈希和配置；测试后未修改规则。",
             f"测试集：原 test_ablation 的 102 张，来自 {comparison['source_groups']} 个源单据；clean/medium/heavy 各 34 张，并非同一批 34 个源单据各取三档。",
             "开发源单据与测试源单据无重叠；测试中部分源单据重复出现，其退化变体不能当作独立来源。",
             "四方案按顺序图片 ID 和 GT 精确核验一致。历史 VLM 缺输入图片字节哈希，不能补造历史图像指纹。",
             "", "## 同集结果", "", "| 方案 | 字段 F1 | 金额正确/总数 | Schema 合规/总数 | 整单正确/总数 |",
             "|---|---:|---|---|---|"]
    for key, s in summaries.items():
        lines.append(f"| {labels[key]} | {s['f1']:.4f} | {s['amount_correct']}/{s['amount_total']} | {s['schema_valid_count']}/{s['n']} | {s['document_correct_count']}/{s['n']} |")
    lines += ["", "## 分退化档位", "", "| 方案 | clean F1 | medium F1 | heavy F1 |", "|---|---:|---:|---:|"]
    for key, value in rows.items():
        scores = [S.summarize([r for r in value if r["level"] == level])["f1"] for level in ("clean", "medium", "heavy")]
        lines.append(f"| {labels[key]} | " + " | ".join(f"{v:.4f}" for v in scores) + " |")
    s = ocr_summary
    lines += ["", "## 推理配置与时间", "", "OCR：RapidOCR ONNX Runtime 1.4.4；PP-OCRv4 检测/识别、v2 方向分类；CPU 4 个算子内线程、1 个算子间线程。",
              "读原始尺寸图片，OCR 检测保持长宽比并按最短边 736 处理。参数依据 [RapidOCR 官方接口文档](https://rapidai.github.io/RapidOCRDocs/v1.4.4/install_usage/api/RapidOCR/)，实际模型与配置指纹见产物。",
              f"OCR 端到端平均 **{s['mean_seconds']:.3f} 秒/张**，中位数 {s['median_seconds']:.3f} 秒，P95 {s['p95_seconds']:.3f} 秒。",
              f"包含读图、解码、OCR、规则提取和 JSON 序列化；排除模型加载（{s['model_load_seconds']:.3f} 秒）、一次预热、评分和结果文件写入。",
              "OCR 未使用 GPU。未采集进程峰值内存；本轮未重新测量 VLM 时延或显存，因此不作速度或资源优劣结论。",
              "VLM 历史缓存记录输入 384px，提示词身份等配置缺口沿用 [资产盘点](evaluation-inventory.md)，不把当前默认值冒充历史参数。",
              "", "## 结构指标和失败分母", "",
              f"OCR 推理/规则异常 {s['inference_failure_count']}/{s['n']}；对象解析失败 {s['parse_failure_count']}/{s['n']}。",
              "OCR 与词表输出由程序序列化，严格 JSON 合法率不代表模型遵循指令能力；主表因此不拿该指标作四方案能力排名。",
              "空字段、识别错误、结构不合规均保留在分母。金额保持图上 OCR 值，不按数量×单价自动重算。",
              "", "## 词表收益与误改", "",
              f"按当前评分归一化值，历史微调后处理共有 {len(lex_changes)} 处变化：修复 {sum(c['fixed'] for c in lex_changes)}，误改 {sum(c['broke'] for c in lex_changes)}。",
              "已逐条验证后处理产物的 pred_orig 与原始微调文本一致；后处理原始实验的阈值选择过程仍未完整核验。"]
    for c in lex_changes:
        lines.append(f"- `{c['image']}` / `{c['path']}`：{c['before']!r} → {c['after']!r}，GT={c['gt']!r}。")
    counts = Counter(e["path"].split(".")[-1] for r in rows["ocr_rules"] for e in r["field_errors"])
    lines += ["", "## 可追溯失败案例", "", "以下按每档最低字段 F1 选取两例，不只展示成功图。错误字段计数是路径事件，可一图多错，不是图片错误率。",
              "OCR 识字错误与行列关联错误可借文字框、字段来源和原图复核；自动统计不把字符不匹配直接等同于视觉原因，也不把多字段直接称为幻觉。",
              "", "错误路径事件数（按字段汇总）：" + "；".join(f"{k} {v}" for k, v in counts.most_common()) + "。", ""]
    for e in errors:
        lines.append(f"- `{e['image']}`（{e['level']}）：Schema 问题 {len(e['schema_errors'])} 项。")
        for issue in e["field_errors_with_evidence"][:2]:
            texts = [b["text"] for b in issue["assigned_ocr_blocks"]]
            lines.append(f"  `{issue['path']}`：GT={issue['gt']!r}，提取={issue['pred']!r}；分配的 OCR 文本={texts!r}。")
    lines += ["", "## 产物与复现", "", "- [四方案汇总](../outputs/baseline-comparison-v1/comparison.json)",
              "- [OCR 汇总](../outputs/ocr-test-v1/summary.json) · [逐样本预测](../outputs/ocr-test-v1/predictions.jsonl) · [OCR 文字框和规则中间结果](../outputs/ocr-test-v1/ocr.jsonl)",
              "- [失败案例与字段证据](../outputs/baseline-comparison-v1/failure_cases.json) · [开发集两版结果](../outputs/baseline-comparison-v1/development-summary.json)",
              "- [冻结配置](../data/benchmarks/ocr_v1/frozen_config.json) · [依赖锁定](../requirements-ocr-lock.txt)",
              "", "```powershell", ".\\venv-gld\\Scripts\\python.exe -m venv .venv-ocr", ".\\.venv-ocr\\Scripts\\python.exe -m pip install -r requirements-ocr-lock.txt",
              "$env:PYTHONUTF8 = \"1\"", ".\\.venv-ocr\\Scripts\\python.exe scripts\\_test_baseline_ocr.py",
              ".\\.venv-ocr\\Scripts\\python.exe src\\baseline_ocr.py --data data/processed/test_ablation.jsonl --out outputs/ocr-test-repeat --frozen-config data/benchmarks/ocr_v1/frozen_config.json", "```",
              "", "输出目录必须不存在；原数据清单含本机绝对路径，迁移机器前需制作单独的路径映射清单，不覆盖历史清单。",
              "", "## 未完成边界", "", "这只是已有合成模板上的比较，不能推出真实中文单据或陌生版式的泛化效果。",
              "四方案准确率比较已完成；同机 VLM 时延/显存重测、真实单据、陌生版式和人工复核流程仍未完成。",
              "测试结果已观察；如后续据此改 OCR 规则，必须将这批数据作为开发参考，并另设最终留出测试集。"]
    (ROOT / "docs/ocr-baseline-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: {k: s[k] for k in ("f1", "schema_valid_count", "document_correct_count", "amount_correct", "amount_total")} for key, s in summaries.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
