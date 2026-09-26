"""Offline integrity checks for the layout-augmentation training experiment."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import scoring as S


def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    train_dir = ROOT / "data/benchmarks/layout_train_v1"
    protocol = json.loads((train_dir / "protocol.json").read_text(encoding="utf-8"))
    train = rows(train_dir / "train.jsonl")
    assert len(train) == 600
    assert digest(train_dir / "train.jsonl") == protocol["data_sha256"]
    counts = {t: sum(r["meta"]["template_id"] == t for r in train) for t in ("A_replay", "B", "C")}
    assert counts == protocol["template_counts"] == {"A_replay": 200, "B": 200, "C": 200}

    forbidden = set()
    for name in ("val", "test"):
        forbidden.update(S.stable_hash(r["gt"]) for r in rows(ROOT / f"data/processed/{name}.jsonl"))
    forbidden.update(S.stable_hash(r["gt"]) for r in rows(ROOT / "data/benchmarks/layout_v1/manifest.jsonl"))
    augmented = [r for r in train if r["meta"]["template_id"] in {"B", "C"}]
    assert not ({S.stable_hash(r["gt"]) for r in augmented} & forbidden)
    paired = {}
    for r in augmented:
        paired.setdefault(r["meta"]["source_id"], {})[r["meta"]["template_id"]] = r["gt"]
    assert len(paired) == 200
    assert all(set(v) == {"B", "C"} and v["B"] == v["C"] for v in paired.values())

    adapter = ROOT / "outputs/sft_layout_v1/lora/adapter_model.safetensors"
    result = json.loads((ROOT / "outputs/sft_layout_v1/training_result.json").read_text(encoding="utf-8"))
    assert digest(adapter) == result["adapter_model_sha256"]

    reg_rows = rows(ROOT / "outputs/scoring-v2.0.0/eval_layout_v1_regression_preds.jsonl")
    reg_summary = json.loads((ROOT / "outputs/scoring-v2.0.0/eval_layout_v1_regression.json").read_text(encoding="utf-8"))
    recounted = S.summarize(reg_rows)
    for key in ("n", "tp", "fp", "fn", "f1", "amount_correct", "amount_total",
                "strict_json_valid_count", "schema_valid_count", "document_correct_count"):
        assert recounted[key] == reg_summary[key], key

    manifest = rows(ROOT / "data/benchmarks/layout_v1/manifest.jsonl")
    post = rows(ROOT / "outputs/layout-pilot-layout-v1/predictions.jsonl")
    assert len(manifest) == len(post) == 24
    assert [Path(r["image"]).name for r in manifest] == [r["image"] for r in post]
    assert all(digest(r["image"]) == r["image_sha256"] for r in manifest)
    assert all(a["image_sha256"] == b["image_sha256"] and a["gt"] == b["gt"] for a, b in zip(manifest, post))
    layout_summary = json.loads((ROOT / "outputs/layout-pilot-layout-v1/summary.json").read_text(encoding="utf-8"))
    for template in ("A", "D"):
        recounted = S.summarize([r for r in post if r["template_id"] == template])
        for key in ("n", "tp", "fp", "fn", "f1", "amount_correct", "amount_total",
                    "strict_json_valid_count", "schema_valid_count", "document_correct_count"):
            assert recounted[key] == layout_summary["by_template"][template][key], (template, key)

    print(json.dumps({"status": "ok", "training_rows": len(train), "new_sources": len(paired),
                      "regression_samples": len(reg_rows), "layout_samples": len(post)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
