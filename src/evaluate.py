"""评测 —— 字段级 F1 / JSON 合法率 / 幻觉率。

设计依据：docs/schema_v1.md 第 4 节（评估口径）

用法：
    # 零样本（不挂 LoRA）
    venv-gld/Scripts/python.exe src/evaluate.py --limit 30 --tag zeroshot

    # 微调后
    venv-gld/Scripts/python.exe src/evaluate.py --adapter outputs/sft_smoke/lora --limit 30 --tag sft

产出：
    outputs/eval_{tag}.json    指标汇总
    outputs/eval_{tag}_cases.txt   错误样例（便于人工分析）
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


# ================================================================ 文本归一化
def normalize(v: object) -> str:
    """比对前的文本归一化：去空白、全角转半角、去千分位、统一小数、统一日期。"""
    s = str(v).strip()
    s = unicodedata.normalize("NFKC", s)          # 全角 → 半角
    s = re.sub(r"\s+", "", s)                     # 去所有空白
    s = s.replace(",", "").replace("，", "")       # 去千分位

    # 日期统一 YYYY-MM-DD
    m = re.search(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})", s)
    if m:
        y, mo, d = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"

    # 纯数字统一小数位（去尾零）
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        try:
            f = float(s)
            return f"{f:g}"
        except ValueError:
            pass
    return s


# ================================================================ 输出解析
def parse_json(text: str) -> dict | None:
    """从模型输出里抠出 JSON。容忍 ```json 代码块与前后噪声。"""
    if not text:
        return None
    t = text.strip()

    # 去 markdown 代码块
    m = re.search(r"```(?:json)?\s*(.+?)```", t, re.S)
    if m:
        t = m.group(1).strip()

    # 直接试
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    # 截取第一个 { 到最后一个 }
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            obj = json.loads(t[i:j + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


# ================================================================ 指标
def align_rows(gt_rows: list, pred_rows: list) -> list[tuple[int, int]]:
    """明细行对齐：先按序号，再按内容相似度贪心。返回 (gt_idx, pred_idx) 配对。"""
    pairs: list[tuple[int, int]] = []
    used_g, used_p = set(), set()

    # ① 按序号
    for gi, g in enumerate(gt_rows):
        gs = normalize(g.get("序号", ""))
        if not gs:
            continue
        for pi, p in enumerate(pred_rows):
            if pi in used_p:
                continue
            if normalize(p.get("序号", "")) == gs:
                pairs.append((gi, pi))
                used_g.add(gi)
                used_p.add(pi)
                break

    # ② 剩余按字段重合度贪心
    rest_g = [i for i in range(len(gt_rows)) if i not in used_g]
    rest_p = [i for i in range(len(pred_rows)) if i not in used_p]
    scored = []
    for gi in rest_g:
        for pi in rest_p:
            hit = sum(
                1 for f in ROW_FIELDS
                if normalize(gt_rows[gi].get(f, "")) == normalize(pred_rows[pi].get(f, ""))
                and normalize(gt_rows[gi].get(f, "")) != ""
            )
            if hit:
                scored.append((hit, gi, pi))
    scored.sort(reverse=True)
    for _, gi, pi in scored:
        if gi in used_g or pi in used_p:
            continue
        pairs.append((gi, pi))
        used_g.add(gi)
        used_p.add(pi)

    return pairs


def _as_dict(v: object) -> dict:
    """只接受 dict，其余（None / str / list）一律当空 dict，避免下游 .get() 崩。"""
    return v if isinstance(v, dict) else {}


def _as_rows(v: object) -> list[dict]:
    """只保留是 dict 的明细行；非 dict 的行无法提供字段，直接丢弃。"""
    if not isinstance(v, list):
        return []
    return [r for r in v if isinstance(r, dict)]


def _salvage_total(pred: dict, tolerant: bool) -> dict:
    """把「合计」的值规整成 dict。

    零样本模型经常把合计直接输出成标量（"合计": "51330.24"），而不是
    schema 要求的 {"金额": "51330.24"}。严格口径下这算没抽到（计入 FN）；
    tolerant=True 时提升成 {"金额": v}，用于区分「格式不适配」与「内容抽错」。
    """
    t = pred.get("合计")
    if isinstance(t, dict):
        return t
    if tolerant and isinstance(t, (str, int, float)):
        s = normalize(t)
        if re.fullmatch(r"-?\d+(\.\d+)?", s):
            return {"金额": t}
    return {}


def flatten(gt: dict, pred: dict, tolerant: bool = False) -> tuple[dict, dict]:
    """把 GT / 预测摊平成 {路径: 归一化值}。

    全程类型防御：模型输出任何怪形状都不应让整轮评测崩掉
    （推理成本 30 分钟起，打分崩了不该连累推理结果）。
    """
    gt, pred = _as_dict(gt), _as_dict(pred)
    gout, pout = {}, {}

    g_head, p_head = _as_dict(gt.get("表头")), _as_dict(pred.get("表头"))
    for f in HEAD_FIELDS:
        gv = normalize(g_head.get(f, ""))
        if gv:
            gout[f"表头.{f}"] = gv
        pv = normalize(p_head.get(f, ""))
        if pv:
            pout[f"表头.{f}"] = pv

    g_rows = _as_rows(gt.get("明细"))
    p_rows = _as_rows(pred.get("明细"))
    pairs = align_rows(g_rows, p_rows)

    for n, (gi, pi) in enumerate(pairs):
        for f in ROW_FIELDS:
            gv = normalize(g_rows[gi].get(f, ""))
            if gv:
                gout[f"明细[{n}].{f}"] = gv
            pv = normalize(p_rows[pi].get(f, ""))
            if pv:
                pout[f"明细[{n}].{f}"] = pv

    # 未配对上的 GT 行 → 全部计入漏抽；未配对的预测行 → 全部计入多抽/幻觉
    for k, gi in enumerate(i for i in range(len(g_rows)) if i not in {p[0] for p in pairs}):
        for f in ROW_FIELDS:
            gv = normalize(g_rows[gi].get(f, ""))
            if gv:
                gout[f"明细[unmatched_g{k}].{f}"] = gv
    for k, pi in enumerate(i for i in range(len(p_rows)) if i not in {p[1] for p in pairs}):
        for f in ROW_FIELDS:
            pv = normalize(p_rows[pi].get(f, ""))
            if pv:
                pout[f"明细[unmatched_p{k}].{f}"] = pv

    gv = normalize(_as_dict(gt.get("合计")).get("金额", ""))
    if gv:
        gout["合计.金额"] = gv
    pv = normalize(_salvage_total(pred, tolerant).get("金额", ""))
    if pv:
        pout["合计.金额"] = pv

    return gout, pout


def flatten_flat(gt: dict, pred: dict, tolerant: bool = False) -> tuple[dict, dict]:
    """扁平键值对模式（用于 WildReceipt 等真实收据）。

    键名宽松匹配：忽略大小写、下划线、连字符、空格。
    tolerant 参数只为与 flatten 保持同一签名，此处无额外含义。
    """
    def key(s: str) -> str:
        return re.sub(r"[\s_\-]+", "", str(s).lower())

    gout = {}
    for k, v in _as_dict(gt).items():
        nv = normalize(v)
        if nv:
            gout[key(k)] = nv

    pout = {}
    for k, v in _as_dict(pred).items():
        nv = normalize(v)
        if nv:
            pout[key(k)] = nv
    return gout, pout


def score_pred(gt: dict, pred: dict | None, mode: str = "schema",
               tolerant: bool = False) -> dict:
    """对**已解析**的预测打分。pred=None 表示 JSON 解析失败（GT 全计入漏抽）。"""
    flatten_fn = flatten_flat if mode == "flat" else flatten
    res = {
        "json_valid": pred is not None,
        "tp": 0, "fp": 0, "fn": 0,
        "hallucinated_keys": [],
        "missing_keys": [],
    }
    if pred is None:
        g, _ = flatten_fn(gt, {}, tolerant)
        res["fn"] = len(g)
        return res

    g, p = flatten_fn(gt, pred, tolerant)
    tp = sum(1 for k, v in g.items() if p.get(k) == v)
    fn = len(g) - tp
    fp = sum(1 for k, v in p.items() if g.get(k) != v)

    res["tp"], res["fn"], res["fp"] = tp, fn, fp
    res["hallucinated_keys"] = [k for k in p if k not in g]
    res["missing_keys"] = [k for k in g if k not in p]
    return res


def score_one(gt: dict, raw_text: str, mode: str = "schema",
              tolerant: bool = False) -> dict:
    """单条样本打分。mode: schema（工程单据）/ flat（通用键值对）。"""
    return score_pred(gt, parse_json(raw_text), mode, tolerant)


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


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
    print(f"  模式      : {'微调后 ' + args.adapter if args.adapter else '零样本'}"
          + ("  [仅打分]" if args.score_only else ""))
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
    #   严格   —— 必须完全符合 schema（合计 是标量 = 没抽到）
    #   宽容   —— 把 合计 标量这类「形状不对但语义同一」的输出救回来
    # 严格口径与历史所有档位口径一致，保证跨档位可比；宽容口径用于
    # 把「格式不适配」与「内容抽错」分开看。
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
            r = score_pred(rec["gt"], pred, args.mode, tolerant=False)
            r_t = (r if args.mode == "flat"
                   else score_pred(rec["gt"], pred, args.mode, tolerant=True))
        except Exception as e:       # 打分崩了不能连累已落盘的推理结果
            scoring_errors.append({"image": Path(rec["image"]).name,
                                   "error": f"{type(e).__name__}: {e}"})
            print(f"    [打分异常] {Path(rec['image']).name}: {type(e).__name__}: {e}",
                  flush=True)
            continue

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
    print(f"  JSON 合法率   : {total['json_valid']}/{n} = {total['json_valid']/n:.1%}")
    print(f"  幻觉率        : {total['hallucinated']}/{n} = {total['hallucinated']/n:.1%}")
    print(f"  TP/FP/FN      : {total['tp']}/{total['fp']}/{total['fn']}")
    print(f"  推理耗时      : {infer_time:.0f}s（{infer_time/n:.1f}s/条）")
    if scoring_errors:
        print(f"  [警告] {len(scoring_errors)} 条打分异常，已跳过（见 summary.json）")
    print()
    print("  分档位：")
    for lv in sorted(by_level):
        b = by_level[lv]
        _, _, lf = prf(b["tp"], b["fp"], b["fn"])
        _, _, lf_t = prf(b["tp_t"], b["fp_t"], b["fn_t"])
        print(f"    {lv:<7} F1={lf:.4f} (宽容 {lf_t:.4f})  "
              f"JSON合法={b['valid']}/{b['n']}")

    summary = {
        "tag": args.tag, "split": args.split, "n": n,
        "adapter": args.adapter or None,
        # 顶层键 = 严格口径（与历史档位口径一致，勿改语义）
        "precision": p, "recall": r_, "f1": f1,
        "json_valid_rate": total["json_valid"] / n,
        "hallucination_rate": total["hallucinated"] / n,
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
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / f"eval_{args.tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    case_file = OUTPUTS / f"eval_{args.tag}_cases.txt"
    with open(case_file, "w", encoding="utf-8") as f:
        for c in cases[:40]:
            f.write(f"[{c['level']}] {c['image']}  json_valid={c['json_valid']}\n")
            f.write(f"  GT  : {c['gt']}\n")
            f.write(f"  PRED: {c['pred']}\n")
            if c["hallucinated_keys"]:
                f.write(f"  幻觉字段: {c['hallucinated_keys']}\n")
            if c["missing_keys"]:
                f.write(f"  漏抽字段: {c['missing_keys']}\n")
            f.write("\n")
    # 全量预测（失败分析用；cases 只有错例且截断到 400 字符）
    preds_file = OUTPUTS / f"eval_{args.tag}_preds.jsonl"
    with open(preds_file, "w", encoding="utf-8") as f:
        for row in full:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print(f"  汇总  : {OUTPUTS / f'eval_{args.tag}.json'}")
    print(f"  错例  : {case_file}  （{len(cases)} 条有问题）")
    print(f"  全量  : {preds_file}  （{len(full)} 条，供失败分析）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
