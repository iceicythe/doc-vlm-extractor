"""Compare the frozen pre/post layout-training predictions and write the final report."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/layout-training-v1"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transitions(old_rows, new_rows, key="image", template=None):
    old = {r[key]: r for r in old_rows if template is None or r.get("template_id") == template}
    new = {r[key]: r for r in new_rows if template is None or r.get("template_id") == template}
    if set(old) != set(new):
        raise ValueError("Prediction sample sets differ")
    counts = {"both_correct": 0, "old_only_correct": 0, "new_only_correct": 0, "both_incorrect": 0}
    changed = []
    for sample in sorted(old):
        a, b = old[sample], new[sample]
        if a["gt"] != b["gt"]:
            raise ValueError(f"GT differs: {sample}")
        ca, cb = a["document_correct"], b["document_correct"]
        bucket = "both_correct" if ca and cb else "old_only_correct" if ca else "new_only_correct" if cb else "both_incorrect"
        counts[bucket] += 1
        old_n, new_n = len(a["field_errors"]), len(b["field_errors"])
        if old_n != new_n or ca != cb:
            changed.append({"image": sample, "old_error_count": old_n, "new_error_count": new_n,
                            "old_document_correct": ca, "new_document_correct": cb,
                            "old_errors": a["field_errors"], "new_errors": b["field_errors"]})
    return counts, changed


def main():
    old_reg_summary = read_json(ROOT / "outputs/evaluation-v2-final/eval_abl_base.json")
    new_reg_summary = read_json(ROOT / "outputs/scoring-v2.0.0/eval_layout_v1_regression.json")
    old_reg = read_jsonl(ROOT / "outputs/evaluation-v2-final/eval_abl_base_preds.jsonl")
    new_reg = read_jsonl(ROOT / "outputs/scoring-v2.0.0/eval_layout_v1_regression_preds.jsonl")
    old_layout_summary = read_json(ROOT / "outputs/layout-pilot-v1/summary.json")
    new_layout_summary = read_json(ROOT / "outputs/layout-pilot-layout-v1/summary.json")
    old_layout = read_jsonl(ROOT / "outputs/layout-pilot-v1/predictions.jsonl")
    new_layout = read_jsonl(ROOT / "outputs/layout-pilot-layout-v1/predictions.jsonl")
    train_protocol = read_json(ROOT / "data/benchmarks/layout_train_v1/protocol.json")
    run_config = read_json(ROOT / "outputs/sft_layout_v1/run_config.json")
    train_result = read_json(ROOT / "outputs/sft_layout_v1/training_result.json")

    if old_reg_summary["sample_gt_sha256"] != new_reg_summary["sample_gt_sha256"]:
        raise ValueError("Regression GT hash differs")
    if old_reg_summary["scorer_sha256"] != new_reg_summary["scorer_sha256"]:
        raise ValueError("Regression scorer differs")
    if run_config["data_sha256"] != sha(ROOT / "data/benchmarks/layout_train_v1/train.jsonl"):
        raise ValueError("Training data changed")
    if train_result["adapter_model_sha256"] != sha(ROOT / "outputs/sft_layout_v1/lora/adapter_model.safetensors"):
        raise ValueError("Trained adapter changed")

    reg_trans, reg_changed = transitions(old_reg, new_reg)
    layout_trans = {}
    layout_changed = []
    for template in ("A", "D"):
        layout_trans[template], changed = transitions(old_layout, new_layout, template=template)
        layout_changed.extend({"template": template, **row} for row in changed)

    def compact(summary):
        return {k: summary[k] for k in ("n", "f1", "amount_correct", "amount_total",
                                        "strict_json_valid_count", "schema_valid_count", "document_correct_count")}

    comparison = {
        "training": {"protocol": train_protocol, "run_config": run_config, "result": train_result},
        "original_test": {
            "old": compact(old_reg_summary), "new": compact(new_reg_summary),
            "f1_delta": new_reg_summary["f1"] - old_reg_summary["f1"],
            "document_correct_delta": new_reg_summary["document_correct_count"] - old_reg_summary["document_correct_count"],
            "transitions": reg_trans,
        },
        "layout_holdout": {
            template: {"old": compact(old_layout_summary["by_template"][template]),
                       "new": compact(new_layout_summary["by_template"][template]),
                       "f1_delta": new_layout_summary["by_template"][template]["f1"] - old_layout_summary["by_template"][template]["f1"],
                       "document_correct_delta": new_layout_summary["by_template"][template]["document_correct_count"] - old_layout_summary["by_template"][template]["document_correct_count"],
                       "transitions": layout_trans[template]}
            for template in ("A", "D")
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "sample-transitions.json").write_text(
        json.dumps({"original_test": reg_changed, "layout_holdout": layout_changed}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    o, n = comparison["original_test"]["old"], comparison["original_test"]["new"]
    la, ld = comparison["layout_holdout"]["A"], comparison["layout_holdout"]["D"]
    lines = [
        "# 版式增强继续微调报告", "",
        "在已有 SFT LoRA 上继续训练 120 步。训练只使用预先定义的 B/C 新版式与 A 原版式回放；冻结的 D 留出图、预测和错误均未进入训练数据构建。", "",
        "## 训练配置", "",
        f"- 数据：{train_protocol['row_count']} 张（A 回放 {train_protocol['template_counts']['A_replay']}、B {train_protocol['template_counts']['B']}、C {train_protocol['template_counts']['C']}），与原 val/test 和 D 留出内容重合为 0。",
        f"- 继续训练：{run_config['max_steps']} 步，batch={run_config['batch']}，梯度累积 {run_config['gradient_accumulation_steps']}，学习率 {run_config['learning_rate']:.0e}，384×384。",
        f"- 实际：{train_result['epoch']:.1f} epoch，{train_result['train_runtime_seconds']:.1f} 秒，train loss {train_result['train_loss']:.6f}，峰值 allocated {train_result.get('peak_memory_allocated_gib_reported', train_result.get('peak_memory_allocated_bytes', 0)/1024**3):.2f} GiB。", "",
        "## 冻结评测结果", "",
        "| 集合 | 训练前 F1 | 训练后 F1 | ΔF1 | 训练前整单 | 训练后整单 | 金额正确 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| 原 102 张测试 | {o['f1']:.4f} | {n['f1']:.4f} | {n['f1']-o['f1']:+.4f} | {o['document_correct_count']}/102 | {n['document_correct_count']}/102 | {o['amount_correct']}→{n['amount_correct']}/{n['amount_total']} |",
        f"| A 原版式留出 | {la['old']['f1']:.4f} | {la['new']['f1']:.4f} | {la['f1_delta']:+.4f} | {la['old']['document_correct_count']}/12 | {la['new']['document_correct_count']}/12 | {la['old']['amount_correct']}→{la['new']['amount_correct']}/{la['new']['amount_total']} |",
        f"| D 陌生版式留出 | {ld['old']['f1']:.4f} | {ld['new']['f1']:.4f} | {ld['f1_delta']:+.4f} | {ld['old']['document_correct_count']}/12 | {ld['new']['document_correct_count']}/12 | {ld['old']['amount_correct']}→{ld['new']['amount_correct']}/{ld['new']['amount_total']} |", "",
        "原测试集严格 JSON 与 Schema 合规均保持 102/102。整单转移：训练前后都正确 56，仅训练前正确 2，仅训练后正确 5，两者都不正确 39。",
        "D 留出整单转移：前后都正确 4，仅训练前正确 1，仅训练后正确 5，两者都不正确 2。", "",
        "## 结论与边界", "",
        "版式增强在 D 留出上带来明显收益，同时原 102 张测试没有出现总体遗忘，因此 120 步版本可作为候选适配器。A 留出字段 F1 小幅下降 0.0020，说明单个小留出集仍有波动，不能宣称所有原版式样本都提升。",
        "D 只有 12 个合成源单据，而且一次训练后复测已经观察了该集合；后续若继续调参，应把 D 转为开发参考并新建未观察的最终模板留出集。真实中文单据尚未验证。", "",
        "## 产物", "",
        "- [训练数据协议](../data/benchmarks/layout_train_v1/protocol.json) · [训练配置](../outputs/sft_layout_v1/run_config.json) · [训练结果](../outputs/sft_layout_v1/training_result.json)",
        "- [训练前布局报告](layout-holdout-report.md) · [训练后布局汇总](../outputs/layout-pilot-layout-v1/summary.json)",
        "- [总对比](../outputs/layout-training-v1/comparison.json) · [逐样本得失](../outputs/layout-training-v1/sample-transitions.json)",
        "- [原 102 张新模型汇总](../outputs/scoring-v2.0.0/eval_layout_v1_regression.json)",
    ]
    (ROOT / "docs/layout-training-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"original_test": comparison["original_test"], "layout_D": comparison["layout_holdout"]["D"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
