"""Inference-only paired layout pilot with existing SFT weights. No optimizer/training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / "outputs/vlm-runtime-cache/triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT / "outputs/vlm-runtime-cache/inductor"))
sys.path.insert(0, str(ROOT / "src"))
import scoring as S

DATA = ROOT / "data/benchmarks/layout_v1"
MODEL = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default="outputs/sft_v1/lora")
    ap.add_argument("--out", default="outputs/layout-pilot-v1")
    args = ap.parse_args()
    adapter = Path(args.adapter)
    out = Path(args.out)
    if not adapter.is_absolute():
        adapter = ROOT / adapter
    if not out.is_absolute():
        out = ROOT / out
    if not (adapter / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Adapter not found: {adapter}")

    from unsloth import FastVisionModel
    import torch
    from peft import PeftModel
    from PIL import Image, ImageDraw
    from huggingface_hub import snapshot_download

    protocol = json.loads((DATA / "protocol.json").read_text(encoding="utf-8"))
    if protocol["manifest_sha256"] != sha(DATA / "manifest.jsonl"):
        raise ValueError("Frozen holdout manifest changed")
    prompt = (DATA / "prompt.txt").read_text(encoding="utf-8")
    if sha(DATA / "prompt.txt") != protocol["prompt_sha256"]:
        raise ValueError("Frozen prompt changed")
    records = [json.loads(l) for l in (DATA / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in records:
        if sha(r["image"]) != r["image_sha256"]:
            raise ValueError("Frozen image changed")
    snapshot = Path(snapshot_download(MODEL, local_files_only=True))
    adapter_label = str(adapter.relative_to(ROOT)) if adapter.is_relative_to(ROOT) else str(adapter)
    config = {"model":MODEL,"snapshot_revision":snapshot.name,
              "adapter":adapter_label, "adapter_weights_sha256":sha(adapter / "adapter_model.safetensors"),
              "adapter_config_sha256":sha(adapter / "adapter_config.json"),
              "image_size":384,"max_new_tokens":768,"max_seq_length":2048,"do_sample":False,"batch":1,
              "prompt_sha256":sha(DATA / "prompt.txt"),"manifest_sha256":sha(DATA / "manifest.jsonl"),
              "scorer_version":S.SCORER_VERSION,"scorer_sha256":sha(S.__file__),"runner_sha256":sha(__file__),
              "torch":torch.__version__,"gpu":torch.cuda.get_device_name(0),"inference_only":True}
    out.mkdir(parents=True,exist_ok=True)
    cfg_path = out / "configuration.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Existing output configuration differs; refusing mixed results")
    cfg_path.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    preds = out / "predictions.jsonl"
    existing = [json.loads(l) for l in preds.read_text(encoding="utf-8").splitlines() if l.strip()] if preds.exists() else []
    done = {r["image"]:r for r in existing}
    if len(done) != len(existing):
        raise ValueError("Duplicate cached image IDs")
    print(f"Loading cached base model and adapter {adapter_label} (inference only)",flush=True)
    t0 = time.perf_counter()
    model, processor = FastVisionModel.from_pretrained(MODEL, load_in_4bit=True, max_seq_length=2048)
    model = PeftModel.from_pretrained(model, str(adapter))
    FastVisionModel.for_inference(model)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    load_seconds = time.perf_counter()-t0
    print(f"Model loaded in {load_seconds:.1f}s",flush=True)

    def generate(image, max_tokens=768):
        messages=[{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":prompt}]}]
        text=processor.apply_chat_template(messages,add_generation_prompt=True)
        inputs=processor(images=[image],text=[text],return_tensors="pt",padding=True).to("cuda")
        with torch.inference_mode():
            output=model.generate(**inputs,max_new_tokens=max_tokens,do_sample=False)
        return processor.decode(output[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True)

    warmup=Image.new("RGB",(384,384),"white")
    ImageDraw.Draw(warmup).text((20,20),"warmup 123",fill="black")
    generate(warmup,32)
    torch.cuda.synchronize()
    peak_allocated, peak_reserved = 0,0
    with preds.open("a",encoding="utf-8") as output:
        for record in records:
            image_id=Path(record["image"]).name
            if image_id in done:
                continue
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started=time.perf_counter()
            error=None
            try:
                with Image.open(record["image"]) as original:
                    img=original.convert("RGB").resize((384,384))
                raw=generate(img)
            except Exception as exc:
                error=f"{type(exc).__name__}: {exc}"
                raw=""
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-started
            alloc=torch.cuda.max_memory_allocated(); reserved=torch.cuda.max_memory_reserved()
            peak_allocated=max(peak_allocated,alloc); peak_reserved=max(peak_reserved,reserved)
            score=S.score_one(record["gt"],raw); score["inference_error"]=error
            row={"image":image_id,"image_sha256":record["image_sha256"],"gt":record["gt"],"pred":raw,
                 "source_id":record["meta"]["source_id"],"template_id":record["meta"]["template_id"],
                 "end_to_end_seconds":elapsed,"peak_allocated_bytes":alloc,"peak_reserved_bytes":reserved,**score}
            output.write(json.dumps(row,ensure_ascii=False)+"\n");output.flush();done[image_id]=row
            print(f"{len(done)}/{len(records)} {image_id} {elapsed:.1f}s" + (f" ERROR {error}" if error else ""),flush=True)
    ordered=[done[Path(r["image"]).name] for r in records]
    summary={"configuration":config,"model_load_seconds":load_seconds,
             "timing_scope":"read image, resize, processor encode, generate, decode; CUDA synchronized; excludes model load, synthetic 32-token warmup, scoring and output writes",
             "peak_memory_scope":"PyTorch allocated/reserved CUDA bytes, including resident model; not total device usage",
             "by_template":{template:S.summarize([r for r in ordered if r["template_id"]==template]) for template in ["A","D"]},
             "mean_seconds":{template:sum(r["end_to_end_seconds"] for r in ordered if r["template_id"]==template)/12 for template in ["A","D"]},
             "peak_allocated_bytes":max(r["peak_allocated_bytes"] for r in ordered),
             "peak_reserved_bytes":max(r["peak_reserved_bytes"] for r in ordered)}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:{m:v[m] for m in ['n','f1','document_correct_count','amount_accuracy','inference_failure_count']} for k,v in summary['by_template'].items()},indent=2))


if __name__ == "__main__":
    main()
