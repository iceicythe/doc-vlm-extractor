"""API 上界评测 —— 与本地两档口径完全一致，可直接并列进五档对比表。

设计要点
--------
1. **打分口径复用 `evaluate.py`**（同一份 `score_one` / `prf` / `flatten`），
   不另写一套，否则和 SFT 那两档没法并列比较。
2. **默认送原图**（`--image-size 0`）。本地最好的一档是 768px，
   "上界对照"的含义就是让 API 拿到它想要的分辨率。
   想做分辨率对齐的严格对照就显式传 `--image-size 384` 或 `768`。
3. 产出与 `evaluate.py` 同名同构：`eval_{tag}.json` / `_cases.txt` / `_preds.jsonl`，
   所以 `analyze_errors.py`、`check_amount_consistency.py` 可以直接吃。

用法
----
    # 自检：发 1 张图，打印原始返回 + token 用量 + 预估花费（几秒钟）
    venv-gld/Scripts/python.exe src/evaluate_api.py --provider deepseek --check

    # 同域 102 子集（对比 SFT 基线 F1 0.9782）
    venv-gld/Scripts/python.exe src/evaluate_api.py --provider deepseek --tag api_ds_abl \
        --data data/processed/test_ablation.jsonl

    # 跨域 WildReceipt n=100（对比 SFT 的 F1 0.2842 / 幻觉 28%）
    venv-gld/Scripts/python.exe src/evaluate_api.py --provider deepseek --tag api_ds_wr \
        --mode flat --data data/processed/wildreceipt_test.jsonl --limit 100

    # 换成百炼的 Qwen3-VL-Plus（需要 DASHSCOPE_API_KEY）
    venv-gld/Scripts/python.exe src/evaluate_api.py --provider qwen --tag api_qw_abl \
        --data data/processed/test_ablation.jsonl

密钥
----
从环境变量读，不写在代码里、不进 git：

    $env:DEEPSEEK_API_KEY="sk-..."       # DeepSeek
    $env:DASHSCOPE_API_KEY="sk-..."      # 阿里云百炼
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import mimetypes
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"


def load_dotenv(path: Path | None = None) -> None:
    """读项目根的 .env（若存在）补进环境变量，省得每次开终端重贴 key。

    格式极简：KEY=VALUE，忽略空行与 # 注释，已存在的环境变量优先。
    .env 已在 .gitignore 里，不会进版本库。

    `path` 仅供单测注入临时文件用，正常调用留空即可。
    """
    env_file = Path(path) if path is not None else ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and not os.environ.get(k):
            os.environ[k] = v

# 复用 evaluate.py 的打分实现（用 importlib 按路径加载，避免和 HF 的 evaluate 包撞名）
_spec = importlib.util.spec_from_file_location(
    "_minivlm_evaluate", Path(__file__).resolve().parent / "evaluate.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

MAX_NEW_TOKENS = 768          # 与 evaluate.py 对齐
TIMEOUT = 120.0

# 每百万 token 的价格（人民币，仅供估算；实际以控制台账单为准）
PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-flash",
        "key_env": "DEEPSEEK_API_KEY",
        "extra": {},
        "price_in": 1.8,
        "price_out": 3.6,
        "note": "deepseek-v4-pro 不支持图像输入（传图 400），必须用 flash；"
                "非高峰时段半价",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3-vl-plus",
        "key_env": "DASHSCOPE_API_KEY",
        # 关闭思考模式，让输出直接是 JSON（与本地 do_sample=False 的行为对齐）
        "extra": {"enable_thinking": False},
        "price_in": 1.76,
        "price_out": 14.09,
        "note": "阿里云百炼 OpenAI 兼容端点；max_pixels 等原生参数见 --max-pixels",
    },
}


def build_extra(provider: str, want_thinking: bool, no_thinking: bool) -> dict:
    """按端点方言组装「思考模式」参数。

    ⚠️ 两家端点的参数名不一样，且**默认值相反**：
        DashScope: enable_thinking=False 默认已关 → 只有显式 True 才开
        DeepSeek : 不传 = 开（默认思考）；只有 thinking={"type":"disabled"} 才关

    2026-09-19 踩到的坑：早期版本只做 pop("enable_thinking")，
    对 DeepSeek 是彻头彻尾的空操作 —— 加了 --no-thinking 照样按思考模式跑，
    每条白烧 ~2500 reasoning token（实测同域 F1 还更低 0.9546 vs 0.9681）。
    所以这里按 provider 分方言写，并留回归测试钉住。
    """
    extra = dict(PROVIDERS[provider]["extra"])
    if want_thinking:
        if provider == "deepseek":
            extra.pop("thinking", None)          # 不传 = 端点默认（开）
        else:
            extra["enable_thinking"] = True
    if no_thinking:
        extra.pop("enable_thinking", None)
        if provider == "deepseek":
            extra["thinking"] = {"type": "disabled"}
    return extra


# ================================================================ 图片编码
def encode_image(path: str, image_size: int = 0) -> str:
    """转成 data URI。

    image_size=0 → 直接用原始文件字节（不重编码、不损失质量）。
    否则用 PIL 缩放后重新编码成 PNG。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"图片不存在：{p}")

    if image_size:
        import io
        from PIL import Image
        im = Image.open(p).convert("RGB").resize((image_size, image_size))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        blob, mime = buf.getvalue(), "image/png"
    else:
        blob = p.read_bytes()
        mime = mimetypes.guess_type(p.name)[0] or "image/png"
        if mime == "image/jpg":
            mime = "image/jpeg"

    return f"data:{mime};base64," + base64.b64encode(blob).decode("ascii")


# ================================================================ 单次调用
class Caller:
    """封装一次带重试的 API 调用。额外 body 参数不被接受时自动降级。"""

    def __init__(self, client, model: str, prompt: str, max_tokens: int,
                 extra_body: dict | None, max_pixels: int = 0):
        self.client = client
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.extra_body = dict(extra_body) if extra_body else None
        self.max_pixels = max_pixels
        self.usage = {"in": 0, "out": 0}
        self._lock = threading.Lock()

    def _messages(self, data_uri: str) -> list:
        img: dict = {"url": data_uri}
        if self.max_pixels:
            img["max_pixels"] = self.max_pixels      # DashScope 原生参数
        return [{"role": "user", "content": [
            {"type": "image_url", "image_url": img},
            {"type": "text", "text": self.prompt},
        ]}]

    def __call__(self, data_uri: str, retries: int = 5) -> str:
        last_err: Exception | None = None
        for attempt in range(retries):
            kwargs: dict = {}
            with self._lock:
                if self.extra_body:
                    kwargs["extra_body"] = dict(self.extra_body)
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=self._messages(data_uri),
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                    timeout=TIMEOUT,
                    **kwargs,
                )
            except Exception as e:                      # noqa: BLE001
                last_err = e
                msg = str(e)

                # 端点不认 extra_body 里的字段（换实现/换区域时会发生）→ 摘掉重试
                if kwargs.get("extra_body"):
                    names = list(kwargs["extra_body"])
                    if any(n in msg for n in names) or "invalid" in msg.lower() \
                            or "unsupported" in msg.lower():
                        with self._lock:
                            self.extra_body = None
                        print(f"    [降级] 端点不接受 {names}，已去掉该参数重试",
                              flush=True)
                        continue

                # 限流 / 服务端错误 / 超时 → 退避重试
                if any(t in msg for t in ("429", "500", "502", "503", "504",
                                          "timeout", "Timeout", "overloaded")):
                    time.sleep(min(2 ** attempt + 0.3 * (attempt + 1), 30))
                    continue
                raise

            u = getattr(resp, "usage", None)
            if u is not None:
                with self._lock:
                    self.usage["in"] += getattr(u, "prompt_tokens", 0) or 0
                    self.usage["out"] += getattr(u, "completion_tokens", 0) or 0

            choice = resp.choices[0]
            content = choice.message.content or ""
            finish = getattr(choice, "finish_reason", None)

            # 空响应 / 半截响应都是瞬时故障的典型表现，不能静默接受。
            # 2026-09-19 实测：102 条里 16 条栽在这里（9 条空串 + 7 条断在
            # JSON 值中间），存下去就是 json_valid=False → 整条按「全没抽到」
            # 计分 → 把 API 档的 F1 悄悄压低 15 个百分点，且不报任何错。
            if not content.strip() or finish == "length":
                last_err = RuntimeError(
                    f"空响应或截断（finish_reason={finish}）")
                if attempt < retries - 1:
                    time.sleep(min(2 ** attempt + 0.3 * (attempt + 1), 30))
                    continue
                print(f"    [警告] 重试 {retries} 次仍空/截断，原样保留",
                      flush=True)
            return content

        raise RuntimeError(f"重试 {retries} 次仍失败：{last_err}")


# ================================================================ 主流程
def main() -> int:
    # ⚠️ 必须在读 key 之前调用：否则 .env 里填的 key 永远读不到，
    #    现场表现是「明明填了 key 却报未设置」。（2026-09-19 实际踩到）
    load_dotenv()

    ap = argparse.ArgumentParser(description="API 上界评测（口径同 evaluate.py）")
    ap.add_argument("--provider", choices=list(PROVIDERS), default="deepseek")
    ap.add_argument("--model", default=None, help="覆盖默认模型名")
    ap.add_argument("--data", default="", help="指定 jsonl（默认 data/processed/{split}.jsonl）")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部")
    ap.add_argument("--tag", default="api")
    ap.add_argument("--mode", choices=["schema", "flat"], default="schema")
    ap.add_argument("--prompt", default=None, help="自定义 prompt（默认取数据首条的）")
    ap.add_argument("--prompt-file", default=None,
                    help="从文件读 prompt（长 prompt 不适合走命令行；"
                         "优化器产出的指令用这个喂进来）")
    ap.add_argument("--image-size", type=int, default=0,
                    help="0 = 送原图（默认，上界对照）；传 384/768 做分辨率对齐")
    ap.add_argument("--max-pixels", type=int, default=0,
                    help="DashScope 原生参数，仅 qwen 有效（默认 0 = 不传，用服务端默认）")
    ap.add_argument("--workers", type=int, default=4, help="并发数")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--no-resume", action="store_true", help="忽略续跑缓存")
    ap.add_argument("--no-thinking", action="store_true",
                    help="强制关闭思考模式（qwen 默认已关）")
    ap.add_argument("--thinking", action="store_true",
                    help="开启思考模式（会显著变慢，且输出非纯 JSON）")
    ap.add_argument("--check", action="store_true",
                    help="只跑第一条：打印原始返回 / 用量 / 预估花费，不写产物")
    ap.add_argument("--list-models", action="store_true",
                    help="列出端点当前可用的模型名（排查 404 model not found）")
    args = ap.parse_args()

    cfg = dict(PROVIDERS[args.provider])
    model = args.model or cfg["model"]

    extra = build_extra(args.provider, args.thinking, args.no_thinking)

    key = os.environ.get(cfg["key_env"], "")
    if not key:
        print(f"[FATAL] 环境变量 {cfg['key_env']} 未设置。")
        print(f"        在 PowerShell 里先设：$env:{cfg['key_env']}=\"sk-...\"")
        print(f"        或写进 {ROOT / '.env'}（该文件已在 .gitignore 里）")
        return 1

    # 模型名写错是最常见的翻车点（换区域/换实现时模型 ID 会变），先给个自检口
    if args.list_models:
        from openai import OpenAI
        cli = OpenAI(api_key=key, base_url=cfg["base_url"])
        print("=" * 62)
        print(f"端点可用模型：{cfg['base_url']}")
        print("=" * 62)
        try:
            ids = [m.id for m in cli.models.list().data]
        except Exception as e:                              # noqa: BLE001
            print(f"[FAIL] {type(e).__name__}: {e}")
            return 1
        for mid in sorted(ids):
            print(f"  {mid}" + ("   ← 当前默认" if mid == model else ""))
        if model not in ids:
            print(f"\n  [警告] 默认模型 {model!r} 不在列表里，"
                  f"请用 --model 指定，否则会 404")
        return 0

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

    prompt = args.prompt
    prompt_src = "--prompt" if prompt is not None else ""
    if prompt is None and args.prompt_file:
        pf = Path(args.prompt_file)
        if not pf.is_absolute():
            pf = ROOT / pf
        if not pf.exists():
            print(f"[FATAL] 找不到 prompt 文件：{pf}")
            return 1
        prompt = pf.read_text(encoding="utf-8").strip()
        prompt_src = f"--prompt-file {pf.name}"
    if prompt is None:
        prompt = ev.PROMPTS.get(args.mode) or \
            records[0]["messages"][0]["content"][-1]["text"]
        prompt_src = f"内置 PROMPTS[{args.mode}]"

    res_note = "原图（不缩放）" if args.image_size == 0 else f"{args.image_size}px"
    print("=" * 62)
    print("API 上界评测")
    print("=" * 62)
    print(f"  端点      : {args.provider} / {model}")
    print(f"  数据      : {data_file.name}  {len(records)} 条")
    print(f"  模式      : {args.mode}")
    print(f"  输入分辨率: {res_note}")
    print(f"  并发      : {args.workers}    max_tokens={args.max_new_tokens}")
    print(f"  prompt 来源: {prompt_src or '（未知）'}  [{len(prompt)} 字符]")
    if args.thinking:
        print("  [警告] 已开启思考模式：输出会含推理文本，JSON 解析率大概率下降")
    print(f"  备注      : {cfg['note']}")

    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=cfg["base_url"])
    caller = Caller(client, model, prompt, args.max_new_tokens,
                    extra or None, args.max_pixels)

    # ---------------- 自检 ----------------
    if args.check:
        rec = records[0]
        print()
        print("自检：发 1 张图 ...")
        t0 = time.time()
        data_uri = encode_image(rec["image"], args.image_size)
        print(f"  base64 长度 {len(data_uri) / 1024:.0f} KB")
        try:
            raw = caller(data_uri)
        except Exception as e:                          # noqa: BLE001
            print(f"[FAIL] {type(e).__name__}: {e}")
            return 1
        dt = time.time() - t0
        r = ev.score_one(rec["gt"], raw, args.mode)
        cost = (caller.usage["in"] * cfg["price_in"]
                + caller.usage["out"] * cfg["price_out"]) / 1e6
        print(f"  用时      : {dt:.1f}s")
        print(f"  用量      : 入 {caller.usage['in']} tok / 出 {caller.usage['out']} tok"
              f"  → 约 ¥{cost:.4f}")
        print(f"  打分      : json_valid={r['json_valid']}  tp={r['tp']} "
              f"fp={r['fp']} fn={r['fn']}")
        print()
        print("  原始返回（前 600 字符）：")
        print("  " + raw[:600].replace("\n", "\n  "))
        est = cost * len(records)
        print()
        print(f"  按此推算全部 {len(records)} 条约 ¥{est:.2f}")
        return 0

    # ---------------- 断点续跑 ----------------
    # 指纹必须跨进程稳定：内置 hash(str) 受 PYTHONHASHSEED 随机化影响，
    # 每次重跑都会判「配置不一致」→ 删缓存 → 整批 API 重发（= 重复付费）。
    # 改用 md5，且不匹配时改名保留而非删除。
    partial = OUTPUTS / f"eval_{args.tag}_preds.partial.jsonl"
    fingerprint = {
        "provider": args.provider,
        "model": model,
        "data": data_file.name,
        "mode": args.mode,
        "image_size": args.image_size,
        "max_pixels": args.max_pixels,
        "prompt_md5": hashlib.md5(prompt.encode("utf-8")).hexdigest()[:16],
    }
    done: dict[str, str] = {}
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
                for k, v in fingerprint.items():
                    stored = r.get(k)
                    # 旧版 meta 只有不稳定的 prompt_hash：无法校验，跳过该项
                    if stored is None and k == "prompt_md5" and "prompt_hash" in r:
                        continue
                    if stored != v:
                        stale_keys.append(k)
                continue
            # 空响应不是有效结果，不入缓存 —— 否则瞬时故障会被「续跑」
            # 永久固化，之后再跑一百遍也补不回来（2026-09-19 实测踩到）。
            if r.get("pred", "").strip():
                done[r["image"]] = r["pred"]
        if bad_line or stale_keys:
            why = "末行损坏" if bad_line else f"字段不一致 {stale_keys}"
            print(f"  [续跑] 缓存不可用（{why}），旧文件改名保留，本次从头请求")
            done = {}
            bak = partial.parent / (partial.stem + ".bak.jsonl")
            partial.replace(bak)
            print(f"  [续跑] 旧缓存 → {bak.name}")
        elif done:
            print(f"  [续跑] 已完成 {len(done)} 条，跳过")

    todo = [r for r in records if r["image"] not in done]
    print(f"  开始请求（待跑 {len(todo)}/{len(records)} 条）...")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lock = threading.Lock()
    n_done = 0

    with partial.open("a" if done else "w", encoding="utf-8") as fout:
        if not done:
            fout.write(json.dumps({"_meta": True, **fingerprint},
                                  ensure_ascii=False) + "\n")
            fout.flush()

        def job(rec: dict) -> tuple[str, str]:
            uri = encode_image(rec["image"], args.image_size)
            return rec["image"], caller(uri)

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futs = {pool.submit(job, r): r for r in todo}
            for fut in as_completed(futs):
                rec = futs[fut]
                try:
                    img, raw = fut.result()
                except Exception as e:                  # noqa: BLE001
                    print(f"    [单条失败] {Path(rec['image']).name}: "
                          f"{type(e).__name__}: {e}", flush=True)
                    raise
                with lock:
                    done[img] = raw
                    fout.write(json.dumps({"image": img, "pred": raw},
                                          ensure_ascii=False) + "\n")
                    fout.flush()
                    n_done += 1
                    if n_done % 10 == 0 or n_done == len(todo):
                        el = time.time() - t0
                        print(f"    {n_done}/{len(todo)}  {el:.0f}s "
                              f"({el / n_done:.1f}s/条)", flush=True)
    infer_time = time.time() - t0

    preds = [done.get(r["image"], "") for r in records]

    # ---------------- 打分（与 evaluate.py 完全同一套） ----------------
    total = {"tp": 0, "fp": 0, "fn": 0, "json_valid": 0, "hallucinated": 0}
    by_level: dict[str, dict] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "n": 0, "valid": 0})
    cases, full = [], []

    for rec, raw in zip(records, preds):
        r = ev.score_one(rec["gt"], raw, args.mode)
        total["tp"] += r["tp"]; total["fp"] += r["fp"]; total["fn"] += r["fn"]
        total["json_valid"] += int(r["json_valid"])
        total["hallucinated"] += int(bool(r["hallucinated_keys"]))

        lv = (rec.get("meta") or {}).get("level", "all")
        b = by_level[lv]
        b["tp"] += r["tp"]; b["fp"] += r["fp"]; b["fn"] += r["fn"]
        b["n"] += 1; b["valid"] += int(r["json_valid"])

        full.append({
            "image": Path(rec["image"]).name, "level": lv,
            "stem": (rec.get("meta") or {}).get("stem", ""),
            "gt": rec["gt"], "pred": raw, "json_valid": r["json_valid"],
            "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
            "hallucinated_keys": r["hallucinated_keys"],
            "missing_keys": r["missing_keys"],
        })
        if r["fp"] or r["fn"] or not r["json_valid"]:
            cases.append({
                "image": Path(rec["image"]).name, "level": lv,
                "gt": json.dumps(rec["gt"], ensure_ascii=False)[:400],
                "pred": raw[:400], "json_valid": r["json_valid"],
                "hallucinated_keys": r["hallucinated_keys"][:8],
                "missing_keys": r["missing_keys"][:8],
            })

    p, r_, f1 = ev.prf(total["tp"], total["fp"], total["fn"])
    n = len(records)
    cost = (caller.usage["in"] * cfg["price_in"]
            + caller.usage["out"] * cfg["price_out"]) / 1e6

    print()
    print("=" * 62)
    print("结果")
    print("=" * 62)
    print(f"  字段级 P/R/F1 : {p:.4f} / {r_:.4f} / {f1:.4f}")
    print(f"  JSON 合法率   : {total['json_valid']}/{n} = {total['json_valid'] / n:.1%}")
    print(f"  幻觉率        : {total['hallucinated']}/{n} = {total['hallucinated'] / n:.1%}")
    print(f"  TP/FP/FN      : {total['tp']}/{total['fp']}/{total['fn']}")
    print(f"  耗时          : {infer_time:.0f}s（{infer_time / n:.1f}s/条）")
    print(f"  用量          : 入 {caller.usage['in']} tok / 出 {caller.usage['out']} tok"
          f"  → 约 ¥{cost:.2f}")
    print()
    print("  分档位：")
    for lv in sorted(by_level):
        b = by_level[lv]
        _, _, lf = ev.prf(b["tp"], b["fp"], b["fn"])
        print(f"    {lv:<7} F1={lf:.4f}  JSON合法={b['valid']}/{b['n']}")

    summary = {
        "tag": args.tag, "split": args.split, "n": n,
        "provider": args.provider, "model": model,
        "adapter": None,
        "image_size": args.image_size,          # 0 = 原图
        "max_pixels": args.max_pixels,
        "precision": p, "recall": r_, "f1": f1,
        "json_valid_rate": total["json_valid"] / n,
        "hallucination_rate": total["hallucinated"] / n,
        "tp": total["tp"], "fp": total["fp"], "fn": total["fn"],
        "infer_seconds": infer_time,
        "tokens_in": caller.usage["in"], "tokens_out": caller.usage["out"],
        "cost_cny_estimate": round(cost, 4),
        "by_level": {k: dict(v) for k, v in by_level.items()},
    }
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
