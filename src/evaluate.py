"""评测 —— 字段级 F1 / 严格 JSON / Schema / 整单正确率。

设计依据：docs/schema_v1.md 第 4 节（评估口径）

用法：
    # 零样本（不挂 LoRA）
    venv-gld/Scripts/python.exe src/evaluate.py --limit 30 --tag zeroshot

    # 微调后
    venv-gld/Scripts/python.exe src/evaluate.py --adapter outputs/sft_smoke/lora --limit 30 --tag sft

产出：
    outputs/scoring-v2.0.0/eval_{tag}.json    指标汇总
    outputs/scoring-v2.0.0/eval_{tag}_cases.txt   错误样例（便于人工分析）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

MODEL_NAME = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
IMAGE_SIZE = 384
MAX_NEW_TOKENS = 768

HEAD_FIELDS = ["项目名称", "供应商", "单据编号", "日期"]
ROW_FIELDS = ["序号", "名称", "规格型号", "单位", "数量", "单价", "金额"]

# 不同任务模式的默认 prompt
PROMPTS = {
    "schema": None,      # 从 train.jsonl 首条取
    "flat": (
        "请从这张收据图片中提取以下字段并输出 JSON："
        "store_name（店铺名）、date（日期）、total（总额）。"
        "字段名用英文，值保留图中原文。找不到的字段可省略。"
        "只输出 JSON，不要任何解释或代码块标记。"
    ),
}


sys.path.insert(0, str(ROOT / "src"))

# Public compatibility imports: all callers use the same versioned CPU scorer.
from scoring import (SCORER_VERSION, ROW_ALIGNMENT, normalize, parse_json, align_rows,
                     flatten, flatten_flat, score_pred, score_one, prf, summarize,
                     stable_hash, schema_errors)


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_rows(value):
    # Keep malformed rows in their original positions for display/lexicon callers.
    return [_as_dict(row) for row in value] if isinstance(value, list) else []


# ================================================================ 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="评测")
    ap.add_argument("--data", default="", help="指定 jsonl（默认 data/processed/{split}.jsonl）")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部")
    ap.add_argument("--adapter", default="", help="LoRA 目录；留空 = 零样本")
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--batch", type=int, default=4,
                    help="推理批大小（显存不够时降到 1）")
    ap.add_argument("--mode", choices=["schema", "flat"], default="schema",
                    help="schema=工程单据（分层）；flat=通用键值对（真实收据）")
    ap.add_argument("--prompt", default=None, help="自定义 prompt（默认取 PROMPTS[mode]）")
    ap.add_argument("--prompt-file", default=None,
                    help="从文件读 prompt（长 prompt 别走命令行，换行/引号会被 shell 吃掉）")
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE,
                    help="输入图边长（须与训练一致）")
    ap.add_argument("--no-resume", action="store_true",
                    help="忽略续跑缓存，从头推理")
    ap.add_argument("--score-only", action="store_true",
                    help="只对已落盘的 *.partial.jsonl 重新打分，不加载模型、不推理")
    args = ap.parse_args()

    if not args.score_only:
        import torch
        from unsloth import FastVisionModel
        from PIL import Image

    data_file = Path(args.data) if args.data else PROCESSED / f"{args.split}.jsonl"
    if not data_file.exists():
        print(f"[FATAL] 缺少 {data_file}")
        return 1

    records = []
    with open(data_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if args.limit:
        records = records[: args.limit]
    if not records:
        print("[FATAL] 数据为空")
        return 1

    if args.prompt is not None and args.prompt_file is not None:
        print("[FATAL] --prompt 与 --prompt-file 只能给一个")
        return 1
    prompt_src = "内置"
    if args.prompt_file:
        pf = Path(args.prompt_file)
        if not pf.exists():
            print(f"[FATAL] prompt 文件不存在：{pf}")
            return 1
        prompt = pf.read_text(encoding="utf-8").strip()
        prompt_src = f"文件 {pf.name}（{len(prompt)} 字符）"
    else:
        prompt = args.prompt
        if prompt is not None:
            prompt_src = "命令行"
    if not prompt:
        prompt = PROMPTS.get(args.mode) or records[0]["messages"][0]["content"][-1]["text"]

    print("=" * 62)
    print("评测")
    print("=" * 62)
    print(f"  数据      : {data_file.name}  {len(records)} 条")
    print(f"  模式      : {'历史缓存重算（模型身份见缓存配置）' if args.score_only else '微调后 ' + args.adapter if args.adapter else '零样本'}")
    print(f"  prompt    : {args.mode} · 来源={prompt_src}")

    # ---------------- 模型 ----------------
    model = processor = None
    if not args.score_only:
        model, processor = FastVisionModel.from_pretrained(
            MODEL_NAME, load_in_4bit=True, max_seq_length=2048,
        )
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter)
        FastVisionModel.for_inference(model)

    # ---------------- 推理（批量，支持断点续跑） ----------------
    # 长任务（数百条）动辄 1-2 小时，中途断掉不该全废：
    # 每批推理完立刻落盘，重启后自动跳过已完成的条目。
    bsz = max(1, args.batch)
    partial = OUTPUTS / f"eval_{args.tag}_preds.partial.jsonl"

    # 指纹必须跨进程稳定：早期版本用内置 hash(str)，而 CPython 默认对 str
    # 随机化哈希种子（PYTHONHASHSEED），导致每次重跑都判定「配置不一致」
    # 并把缓存删掉重建 —— 长任务的续跑形同虚设。改用 md5。
    fingerprint = {
        "adapter": args.adapter or "",
        "data": data_file.name,
        "image_size": args.image_size,
        "prompt_md5": hashlib.md5(prompt.encode("utf-8")).hexdigest()[:16],
    }

    def img_key(rec: dict) -> str:
        return Path(rec["image"]).name

    done_map: dict[str, str] = {}
    meta_seen: dict | None = None
    legacy_meta = False
    if partial.exists() and not args.no_resume:
        stale_keys: list[str] = []
        bad_line = False
        for line in partial.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad_line = True
                break
            if r.get("_meta"):
                meta_seen = r
                if not args.score_only:
                    for k, v in fingerprint.items():
                        stored = r.get(k)
                        # 旧版 meta 只有不稳定的 prompt_hash：无法校验，只校对其余项
                        if stored is None and k == "prompt_md5" and "prompt_hash" in r:
                            legacy_meta = True
                            continue
                        if stored != v:
                            stale_keys.append(k)
                continue
            done_map[r["image"]] = r["pred"]

        if args.score_only:
            # 只打分不生成：不做指纹门禁（真正的一致性由下面「图片名覆盖检查」保证），
            # 但把缓存里记录的配置打出来，便于人工核对。
            m = meta_seen or {}
            print("  [仅打分] 缓存配置: data=%s adapter=%r image_size=%s"
                  % (m.get("data", "?"), m.get("adapter", "?"), m.get("image_size", "?")))
        elif bad_line or stale_keys:
            why = "末行损坏" if bad_line else f"字段不一致 {stale_keys}"
            print(f"  [续跑] 缓存不可用（{why}），旧文件改名保留，本次从头推理")
            done_map = {}
            bak = partial.parent / (partial.stem + ".bak.jsonl")
            partial.replace(bak)
            print(f"  [续跑] 旧缓存 → {bak.name}")
        elif done_map:
            print(f"  [续跑] 已完成 {len(done_map)} 条，跳过"
                  + ("（旧版 meta，prompt 指纹无法校验）" if legacy_meta else ""))

    if args.score_only and not done_map:
        print(f"[FATAL] --score-only：{partial.name} 无可用预测")
        return 1

    if args.score_only:
        missing = [r for r in records if img_key(r) not in done_map]
        if missing:
            print(f"[FATAL] --score-only：缓存缺 {len(missing)} 条"
                  f"（首条缺失 {img_key(missing[0])}），无法只打分")
            return 1
        todo = []
        print(f"  [仅打分] 复用已落盘预测 {len(done_map)}/{len(records)} 条，不加载模型")
    else:
        todo = [r for r in records if img_key(r) not in done_map]
        print(f"  开始推理（待跑 {len(todo)}/{len(records)} 条，batch={bsz}）...")
    t0 = time.time()
    n_done = 0
    mode = "a" if (done_map or args.score_only) else "w"
    with partial.open(mode, encoding="utf-8") as fout:
        if mode == "w":
            fout.write(json.dumps({"_meta": True, **fingerprint},
                                  ensure_ascii=False) + "\n")
        i = 0
        while i < len(todo):
            batch = todo[i:i + bsz]
            imgs, texts = [], []
            for rec in batch:
                img = Image.open(rec["image"]).convert("RGB").resize(
                    (args.image_size, args.image_size))
                imgs.append(img)
                msg = [{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ]}]
                texts.append(processor.apply_chat_template(msg, add_generation_prompt=True))

            try:
                inputs = processor(images=imgs, text=texts, return_tensors="pt",
                                   padding=True).to("cuda")
                with torch.no_grad():
                    outs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                          do_sample=False)
            except torch.cuda.OutOfMemoryError:
                # 自动降 batch 重试，而不是整批失败
                torch.cuda.empty_cache()
                if bsz == 1:
                    raise
                bsz = max(1, bsz // 2)
                print(f"    [OOM] 显存不足，batch 降为 {bsz} 重试", flush=True)
                continue

            n_in = inputs["input_ids"].shape[1]
            for rec, out in zip(batch, outs):
                raw = processor.decode(out[n_in:], skip_special_tokens=True)
                fout.write(json.dumps({"image": img_key(rec), "pred": raw},
                                      ensure_ascii=False) + "\n")
                done_map[img_key(rec)] = raw
            fout.flush()          # 每批强制落盘，断电最多丢一批

            i += len(batch)
            n_done += len(batch)
            if n_done % (bsz * 3) == 0 or i >= len(todo):
                el = time.time() - t0
                print(f"    {i}/{len(todo)}  {el:.0f}s  ({el/max(1,n_done):.1f}s/条)",
                      flush=True)
    infer_time = time.time() - t0

    # 按 records 原始顺序组装（续跑时顺序才对得上）
    preds = [done_map.get(img_key(r), "") for r in records]

    # ---------------- 打分 ----------------
    # 两套口径同时算：
    #   主字段指标 —— 不修复结构；Schema 合规另行统计
    #   宽容   —— 把 合计 标量这类「形状不对但语义同一」的输出救回来
    # v2 与历史口径不同；宽容字段指标只用于诊断。
    total = {"tp": 0, "fp": 0, "fn": 0, "json_valid": 0, "hallucinated": 0}
    tol = {"tp": 0, "fp": 0, "fn": 0, "coerced": 0}
    by_level: dict[str, dict] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "n": 0, "valid": 0,
                 "tp_t": 0, "fp_t": 0, "fn_t": 0})
    cases = []
    full = []          # 全量记录，供失败分析用（cases 只留错例且截断）
    scoring_errors = []

    for rec, raw in zip(records, preds):
        try:
            pred = parse_json(raw)
            r = score_one(rec["gt"], raw, args.mode, tolerant=False)
            r_t = (r if args.mode == "flat"
                   else score_one(rec["gt"], raw, args.mode, tolerant=True))
        except Exception as e:       # 打分崩了不能连累已落盘的推理结果
            scoring_errors.append({"image": Path(rec["image"]).name,
                                   "error": f"{type(e).__name__}: {e}"})
            print(f"    [打分异常] {Path(rec['image']).name}: {type(e).__name__}: {e}",
                  flush=True)
            raise RuntimeError("Scoring failed; no partial metric report written") from e

        if pred is not None:
            t = pred.get("合计") if isinstance(pred, dict) else None
            if t is not None and not isinstance(t, dict):
                tol["coerced"] += 1

        total["tp"] += r["tp"]; total["fp"] += r["fp"]; total["fn"] += r["fn"]
        total["json_valid"] += int(r["json_valid"])
        total["hallucinated"] += int(bool(r["hallucinated_keys"]))
        tol["tp"] += r_t["tp"]; tol["fp"] += r_t["fp"]; tol["fn"] += r_t["fn"]

        lv = (rec.get("meta") or {}).get("level", "all")
        b = by_level[lv]
        b["tp"] += r["tp"]; b["fp"] += r["fp"]; b["fn"] += r["fn"]
        b["n"] += 1; b["valid"] += int(r["json_valid"])
        b["tp_t"] += r_t["tp"]; b["fp_t"] += r_t["fp"]; b["fn_t"] += r_t["fn"]

        # 全量落盘（含正确样本），失败分析需要完整 GT/PRED
        full.append({
            **r,
            "image": Path(rec["image"]).name,
            "level": lv,
            "stem": (rec.get("meta") or {}).get("stem", ""),
            "gt": rec["gt"],
            "pred": raw,
            "json_valid": r["json_valid"],
            "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
            "hallucinated_keys": r["hallucinated_keys"],
            "missing_keys": r["missing_keys"],
        })

        if r["fp"] or r["fn"] or not r["json_valid"]:
            cases.append({
                "image": Path(rec["image"]).name,
                "level": lv,
                "gt": json.dumps(rec["gt"], ensure_ascii=False)[:400],
                "pred": raw[:400],
                "json_valid": r["json_valid"],
                "hallucinated_keys": r["hallucinated_keys"][:8],
                "missing_keys": r["missing_keys"][:8],
            })

    p, r_, f1 = prf(total["tp"], total["fp"], total["fn"])
    p_t, r_t_, f1_t = prf(tol["tp"], tol["fp"], tol["fn"])
    n = len(records)

    print()
    print("=" * 62)
    print("结果")
    print("=" * 62)
    print(f"  字段级 P/R/F1 : {p:.4f} / {r_:.4f} / {f1:.4f}   [严格]")
    print(f"  字段级 P/R/F1 : {p_t:.4f} / {r_t_:.4f} / {f1_t:.4f}   [宽容]"
          f"（合计标量修复 {tol['coerced']}/{n} 条）")
    print(f"  宽松解析成功率 : {total['json_valid']}/{n} = {total['json_valid']/n:.1%}")
    print(f"  旧多出路径比例 : {total['hallucinated']}/{n} = {total['hallucinated']/n:.1%}")
    print(f"  TP/FP/FN      : {total['tp']}/{total['fp']}/{total['fn']}")
    print(f"  本次生成耗时  : {infer_time:.0f}s（实际新生成 {n_done} 条；仅评分时不能作为推理耗时）")
    print()
    print("  分档位：")
    for lv in sorted(by_level):
        b = by_level[lv]
        _, _, lf = prf(b["tp"], b["fp"], b["fn"])
        _, _, lf_t = prf(b["tp_t"], b["fp_t"], b["fn_t"])
        print(f"    {lv:<7} F1={lf:.4f} (宽容 {lf_t:.4f})  "
              f"宽松解析={b['valid']}/{b['n']}")

    summary = {
        "tag": args.tag, "split": args.split, "n": n,
        "adapter": args.adapter or None,
        # 顶层键 = 当前版本主字段指标
        "precision": p, "recall": r_, "f1": f1,
        "json_valid_rate": total["json_valid"] / n,
        "tp": total["tp"], "fp": total["fp"], "fn": total["fn"],
        "strict": {"precision": p, "recall": r_, "f1": f1,
                   "tp": total["tp"], "fp": total["fp"], "fn": total["fn"]},
        "tolerant": {"precision": p_t, "recall": r_t_, "f1": f1_t,
                     "tp": tol["tp"], "fp": tol["fp"], "fn": tol["fn"],
                     "coerced_total_scalar": tol["coerced"]},
        "infer_seconds": infer_time,
        "by_level": {k: dict(v) for k, v in by_level.items()},
        "scoring_errors": scoring_errors,
    }
    summary.update(summarize(full))
    summary["mode"] = args.mode
    summary["schema_version"] = "materials-v1" if args.mode == "schema" else "wildreceipt-flat-v1"
    summary["sample_gt_sha256"] = stable_hash([{ "image": x["image"], "gt": x["gt"] } for x in full])
    summary["inference_config"] = {"model": MODEL_NAME, **fingerprint,
                                    "max_new_tokens": MAX_NEW_TOKENS, "do_sample": False}
    summary["prompt"] = prompt
    summary["json_valid_rate"] = summary["loose_parse_success_rate"]  # compatibility alias
    if args.score_only or not n_done:
        summary["infer_seconds"] = None
        summary["inference_config"] = {"saved_metadata": meta_seen,
                                        "historical_configuration_verified": False}
        summary["prompt"] = None
        summary["adapter"] = (meta_seen or {}).get("adapter")
    print(f"  新评分版本    : {SCORER_VERSION} / {ROW_ALIGNMENT}")
    print(f"  严格 JSON     : {summary['strict_json_valid_count']}/{n}")
    print(f"  Schema 合规   : {summary['schema_valid_count']}/{n}")
    print(f"  整单正确      : {summary['document_correct_count']}/{n}")
    version_dir = OUTPUTS / ("scoring-v" + SCORER_VERSION)
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / f"eval_{args.tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    case_file = version_dir / f"eval_{args.tag}_cases.txt"
    with open(case_file, "w", encoding="utf-8") as f:
        for c in cases[:40]:
            f.write(f"[{c['level']}] {c['image']}  json_valid={c['json_valid']}\n")
            f.write(f"  GT  : {c['gt']}\n")
            f.write(f"  PRED: {c['pred']}\n")
            if c["hallucinated_keys"]:
                f.write(f"  多出目标路径: {c['hallucinated_keys']}\n")
            if c["missing_keys"]:
                f.write(f"  漏抽字段: {c['missing_keys']}\n")
            f.write("\n")
    # 全量预测（失败分析用；cases 只有错例且截断到 400 字符）
    preds_file = version_dir / f"eval_{args.tag}_preds.jsonl"
    with open(preds_file, "w", encoding="utf-8") as f:
        for row in full:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print(f"  汇总  : {version_dir / f'eval_{args.tag}.json'}")
    print(f"  错例  : {case_file}  （{len(cases)} 条有问题）")
    print(f"  全量  : {preds_file}  （{len(full)} 条，供失败分析）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
