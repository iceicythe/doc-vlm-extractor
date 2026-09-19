"""DSPy GEPA 指令优化 —— 跨域（WildReceipt 英文收据）flat 模式。

回答的问题：**不改权重、纯靠自动优化 prompt，能走到哪一步？**
这是「DSPy 值不值得做」的直接检验，而不是硬加一个模块。

---------------------------------------------------------------- 口径设计
1. 评测集 = wildreceipt_test.jsonl **前 100 条**（与 ⑤ API 档完全相同 → 数字可直接比）。
2. trainset / valset 从**第 101 条起**切 —— 优化过程从未见过评测集，
   最终对比无偏（避免「在测试集上优化」这种致命瑕疵）。
3. 优化在 DSPy 内进行：GEPA（反射式指令优化器）**只改 signature 的 instruction**，
   不塞带图 demo —— VLM 场景把带图 demo 塞进 prompt 会让 token 爆炸。
4. 最终对比统一走 `evaluate_api.py`（同一 JSON 解析 + 同一打分函数）。
   原因：dspy 发给 LM 的 prompt ≠ instruction 原文（它还会包上字段名与格式要求），
   两边分数不可混谈。dspy 内的分数只用于「优化过程」；正式数字一律来自评测管道。

---------------------------------------------------------------- 成本
关思考下单次调用 ≈ 1.2k in + 0.2k out token。`--max-metric-calls 160` 约 ¥0.5。
"""
# ⚠️ 不要加 `from __future__ import annotations`：它会把 `dspy.Image` 这类注解
#    变成字符串（ForwardRef），dspy 解析 Signature 时直接抛
#    "Field types must be types, but received: ForwardRef('dspy.Image')"。
import argparse
import json
import os
import sys
import time
from pathlib import Path

# litellm 启动时会去 GitHub 拉 model cost map，国内网络必然超时并重试 3 次
# （实测白等约 15s）。置真让它直接用随包分发的本地备份。
# 必须在 import litellm / dspy 之前设置。
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# 复用评测管道：打分函数与图片编码都从既有模块取，保证口径唯一
import evaluate as ev                                   # noqa: E402
import evaluate_api as eapi                             # noqa: E402

OUT = ROOT / "outputs"
DATA = ROOT / "data/processed/wildreceipt_test.jsonl"

# 评测集占用前 100 条，优化用样本必须从这里之后取
EVAL_RESERVED = 100

# 手写基线 prompt（与 ⑤ 档跑的完全同一个）
BASELINE_INSTRUCTION = ev.PROMPTS["flat"]

# 成本估算按「调用次数 × 实测单价」，**不按 token 折算** ——
# dspy/litellm 的 usage 把图片 base64 也按字符折进 prompt_tokens
# （实测每张图报到 ~21k "token"，而 DeepSeek 的视觉 token 只有约 1k），
# 拿它算钱会高估一个数量级。
# 单价来自 ⑤ 档实测：100 条跨域关思考共 ¥0.14 ⇒ ¥0.0014/次。
UNIT_COST_CNY = 0.0014


# ================================================================ 数据
def load_rows() -> list[dict]:
    return [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]


def to_example(rec: dict, dspy):
    """转成 dspy.Example。图片预先编码成 data URI（放主线程做，别在线程池里读盘）。"""
    return dspy.Example(
        image=dspy.Image(eapi.encode_image(rec["image"])),
        gt=rec["gt"],
    ).with_inputs("image")


# ================================================================ 打分与反馈
def make_metric():
    """GEPA 用的反馈式 metric。

    GEPA 的反思器靠 feedback 文本决定「下一版指令怎么改」，所以反馈必须
    **具体到字段**：漏了哪些字段、哪些值错了、错成什么。只回一个 F1 数字，
    反思器只能瞎猜。
    """
    import dspy

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        text = getattr(pred, "json_out", "") or ""
        obj = ev.parse_json(text)
        r = ev.score_one(gold.gt, text, mode="flat")
        _, _, f1 = ev.prf(r["tp"], r["fp"], r["fn"])

        lines: list[str] = []
        if not text.strip():
            lines.append("输出为空。")
        if obj is None:
            lines.append("输出不是合法 JSON 对象。")
        else:
            g, p = ev.flatten_flat(gold.gt, obj)
            missing = [k for k in g if k not in p]
            wrong = [k for k in g if k in p and p[k] != g[k]]
            extra = [k for k in p if k not in g]
            if missing:
                lines.append(f"漏抽 {len(missing)} 个字段：{missing}")
            if wrong:
                detail = "; ".join(
                    f"{k} 应为 {g[k]!r} 实为 {p[k]!r}" for k in wrong[:6]
                )
                lines.append(f"值错误 {len(wrong)} 处：{detail}")
            if extra:
                lines.append(f"多抽了不存在的字段：{extra}")
        if not lines:
            lines.append("全部字段正确。")

        return dspy.Prediction(score=f1, feedback=" ".join(lines))

    return metric


def eval_module(mod, dataset, metric) -> tuple[float, list[float]]:
    """在给定数据集上顺序评测，返回 (平均分, 逐条分)。"""
    scores: list[float] = []
    for ex in dataset:
        try:
            pred = mod(image=ex.image)
            scores.append(float(metric(ex, pred).score))
        except Exception as e:                          # noqa: BLE001
            check_fatal(e)                              # 余额/鉴权问题立刻停
            print(f"    [warn] 单条失败 {type(e).__name__}: {str(e)[:80]}", flush=True)
            scores.append(0.0)
    n = len(scores) or 1
    return sum(scores) / n, scores


# ================================================================ 致命错误防护
# 这一族「静默失败」在本项目里已经踩到第三次了，逐条记下来：
#   1. evaluate_api.py: 空响应被 `content or ""` 当成有效结果（F1 被悄悄压低）
#   2. evaluate_api.py: load_dotenv() 定义了但没接线（跑不起来还算好的）
#   3. 这里: 余额耗尽后 GEPA 照样空转完 200 轮（396 次调用全失败），
#      最后把 **baseline 原文**当优化结果写盘，退出码 0、Δ=0、不报任何错。
#      若只看最后那三行输出，结论会是「DSPy 优化无效」—— 而实际是根本没跑成。
# 共同点：**失败被降级成「一个看起来正常的结果」**。所以下面加两道闸。
FATAL_MARKERS = (
    "insufficient balance", "insufficient_quota", "exceeded your current quota",
    "invalid_api_key", "incorrect api key", "unauthorized",
)


class FatalAPIError(RuntimeError):
    """余额耗尽 / 鉴权失败 —— 重试无意义，必须立刻停。"""


def check_fatal(exc: Exception) -> None:
    msg = str(exc).lower()
    if any(m in msg for m in FATAL_MARKERS):
        raise FatalAPIError(str(exc)[:300]) from exc


def preflight(lm) -> bool:
    """开跑前花几厘钱确认端点可用，别跑 20 分钟才发现余额不足。"""
    try:
        lm("ping")
    except Exception as e:                              # noqa: BLE001
        try:
            check_fatal(e)
        except FatalAPIError:
            print(f"\n[FATAL] 端点拒绝请求：{str(e)[:200]}", flush=True)
            print("        多为余额耗尽或 key 失效。充值/换 key 后重跑。",
                  flush=True)
            print("        （优化尚未开始，没有浪费算力）", flush=True)
            return False
        print(f"[warn] preflight 遇到非致命错误，继续：{str(e)[:120]}", flush=True)
    return True


# ================================================================ token 统计
def usage_of(lm) -> tuple[int, int, int]:
    """从 dspy.LM 的 history 累计 (in, out, calls)。"""
    tin = tout = calls = 0
    for h in getattr(lm, "history", []) or []:
        u = h.get("usage") or {}
        tin += u.get("prompt_tokens") or 0
        tout += u.get("completion_tokens") or 0
        calls += 1
    return tin, tout, calls


# ================================================================ 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="DSPy GEPA 指令优化（跨域 flat）")
    ap.add_argument("--n-train", type=int, default=24, help="trainset 条数")
    ap.add_argument("--n-val", type=int, default=40, help="valset 条数")
    ap.add_argument("--max-metric-calls", type=int, default=160,
                    help="GEPA 评估预算（每个 = 1 次 VLM 调用），控制成本")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--out", default="outputs/dspy_optimized_flat.txt")
    ap.add_argument("--report", default="outputs/report_dspy_flat.md")
    ap.add_argument("--smoke", action="store_true", help="小规模跑通链路（约 ¥0.05）")
    args = ap.parse_args()

    eapi.load_dotenv(ROOT / ".env")
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        print("[FATAL] DEEPSEEK_API_KEY 未设置（或 .env 里为空）", flush=True)
        return 1

    if args.smoke:
        args.n_train, args.n_val = 6, 8
        args.max_metric_calls = 24
        print("[smoke] 小规模模式：train=6 val=8 budget=24", flush=True)

    import dspy

    base_url = "https://api.deepseek.com/v1"
    # extra_body 是**唯一生效**的写法：litellm 对 openai/ 前缀会拦下 thinking 这个顶层
    # 参数（UnsupportedParamsError），而 extra_body 能原样透传。
    no_think = {"thinking": {"type": "disabled"}}
    lm = dspy.LM(
        "openai/deepseek-flash", api_base=base_url, api_key=key,
        max_tokens=args.max_tokens, temperature=0.0,
        extra_body=no_think, num_retries=4,
    )
    reflection_lm = dspy.LM(
        "openai/deepseek-flash", api_base=base_url, api_key=key,
        max_tokens=4096, temperature=1.0,
        extra_body=no_think, num_retries=4,
    )
    dspy.configure(lm=lm)

    # 闸门 1：开跑前确认端点真的能用（余额/鉴权）
    if not preflight(lm):
        return 1

    # ---------------------------------------------------------- 数据切分
    rows = load_rows()
    lo, hi = EVAL_RESERVED, EVAL_RESERVED + args.n_train + args.n_val
    if hi > len(rows):
        print(f"[FATAL] 样本不够：需要 {hi} 条，只有 {len(rows)} 条", flush=True)
        return 1

    train_rows = rows[lo:lo + args.n_train]
    val_rows = rows[lo + args.n_train:hi]
    trainset = [to_example(r, dspy) for r in train_rows]
    valset = [to_example(r, dspy) for r in val_rows]

    print(f"评测集（保留不碰）= 前 {EVAL_RESERVED} 条"
          f" | trainset = 第 {lo+1}~{lo+args.n_train} 条"
          f" | valset = 第 {lo+args.n_train+1}~{hi} 条", flush=True)
    print(f"基线 prompt（手写）：{BASELINE_INSTRUCTION}", flush=True)

    class ReceiptExtract(dspy.Signature):
        __doc__ = BASELINE_INSTRUCTION

        image: dspy.Image = dspy.InputField(desc="收据 / 小票的照片")
        json_out: str = dspy.OutputField(
            desc="一个 JSON 对象字面量，不含解释文字与代码块标记"
        )

    metric = make_metric()
    student = dspy.Predict(ReceiptExtract)

    # ---------------------------------------------------------- baseline
    print("\n[1/3] 手写 prompt 在 valset 上的分数 …", flush=True)
    t0 = time.time()
    base_score, base_each = eval_module(student, valset, metric)
    print(f"      baseline val F1 = {base_score:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---------------------------------------------------------- GEPA
    print(f"\n[2/3] GEPA 优化（budget={args.max_metric_calls}, "
          f"threads={args.threads}）…", flush=True)
    # ⚠️ log_dir 必须每次独立：GEPA 见到已有 run dir 会**静默加载旧状态续跑**，
    #    若两次运行的 valset 规模不同（如 smoke 8 条 → 全量 40 条），
    #    状态错位会让它在 _evaluate_programs_on_valset 里抛 IndexError。
    #    2026-09-19 实际踩到：全量跑栽在这里，报错点离真因隔了好几层。
    stamp = time.strftime("%m%d_%H%M%S")
    log_dir = OUT / "gepa_logs" / \
        f"flat_t{args.n_train}_v{args.n_val}_b{args.max_metric_calls}_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"      GEPA 状态目录: {log_dir.relative_to(ROOT)}", flush=True)
    gepa = dspy.GEPA(
        metric=metric,
        auto=None,
        max_metric_calls=args.max_metric_calls,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=3,
        num_threads=args.threads,
        track_stats=True,
        log_dir=str(log_dir),
        seed=0,
    )
    t0 = time.time()
    optimized = gepa.compile(student, trainset=trainset, valset=valset)
    opt_secs = time.time() - t0

    inst = getattr(getattr(optimized, "signature", None), "instructions", None) \
        or getattr(optimized, "instructions", "")

    # ---------------------------------------------------------- 优化后在 val 上评
    print("\n[3/3] 优化后 prompt 在 valset 上的分数 …", flush=True)
    opt_score, opt_each = eval_module(optimized, valset, metric)

    tin, tout, calls = usage_of(lm)
    rin, rout, rcalls = usage_of(reflection_lm)
    cost = (calls + rcalls) * UNIT_COST_CNY

    # ---------------------------------------------------------- 落盘
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(inst.strip() + "\n", encoding="utf-8")

    # 指令与基线一字不差 ⇒ 本轮没产出候选（见闸门 2 的说明）
    unchanged = inst.strip() == BASELINE_INSTRUCTION.strip()

    summary = {
        "mode": "flat", "task": "wildreceipt 跨域",
        "optimization_effective": not unchanged,
        "eval_reserved": EVAL_RESERVED,
        "trainset_span": [lo, lo + args.n_train - 1],
        "valset_span": [lo + args.n_train, hi - 1],
        "n_train": len(trainset), "n_val": len(valset),
        "max_metric_calls": args.max_metric_calls,
        "baseline_instruction": BASELINE_INSTRUCTION,
        "optimized_instruction": inst.strip(),
        "baseline_val_f1": round(base_score, 4),
        "optimized_val_f1": round(opt_score, 4),
        "delta_val_f1": round(opt_score - base_score, 4),
        "baseline_each": [round(x, 4) for x in base_each],
        "optimized_each": [round(x, 4) for x in opt_each],
        "optimize_seconds": round(opt_secs, 1),
        "tokens": {"vlm_in": tin, "vlm_out": tout, "vlm_calls": calls,
                   "reflect_in": rin, "reflect_out": rout, "reflect_calls": rcalls},
        "cost_cny_estimate": round(cost, 3),
        "smoke": bool(args.smoke),
    }
    (OUT / "dspy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    print(f"baseline  val F1 = {base_score:.4f}")
    print(f"optimized val F1 = {opt_score:.4f}   (Δ {opt_score-base_score:+.4f})")
    print(f"VLM 调用 {calls} 次 / 反思 {rcalls} 次 | "
          f"token in={tin+rin} out={tout+rout} | 估算 ¥{cost:.2f} | {opt_secs:.0f}s")
    print(f"最优 instruction 已写入 {outp.relative_to(ROOT)}")
    print("=" * 66)
    print("\n---- 优化后的 instruction ----")
    print(inst.strip())

    # 闸门 2：优化没生效就别假装成功。
    # 指令与基线一字不差 ⇒ 这一轮根本没产出候选（余额耗尽/限流/反思输出解析失败），
    # 此时 ΔF1=0 的含义是「没跑成」，不是「优化无效」。
    if unchanged:
        print("\n" + "!" * 66)
        print("[WARNING] 优化后的指令与手写基线完全相同 —— 本轮**没有产出任何有效候选**。")
        print("          常见原因：余额耗尽 / 限流 / 反思器输出无法解析。")
        print("          ΔF1=0 此时意味着「没跑成」，不能据此下「优化无效」的结论。")
        print("          请在上方日志里搜 Insufficient Balance / 429 / ERROR。")
        print("!" * 66)
    return 2 if unchanged else 0


if __name__ == "__main__":
    raise SystemExit(main())
