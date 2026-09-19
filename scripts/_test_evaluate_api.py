"""evaluate_api.py 的 CPU 单测 —— 不花 API 钱、不碰 GPU。

覆盖三件真跑时最容易翻车的事：
  1. 图片编码是无损的（原图字节必须逐字节一致，否则等于偷偷降了分辨率）
  2. extra_body 参数被端点拒绝时能自动降级（百炼不同区域行为不一致）
  3. 端到端链路：假 client 回吐 GT → F1 必须正好 1.0
     （打分口径若有偏差，这里会立刻暴露）

用法：venv-gld/Scripts/python.exe scripts/_test_evaluate_api.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

_spec = importlib.util.spec_from_file_location(
    "_evapi", ROOT / "src" / "evaluate_api.py")
ea = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ea)

fails: list[str] = []
n_pass = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global n_pass
    if ok:
        n_pass += 1
        print(f"[PASS] {name}" + (f"   {detail}" if detail else ""))
    else:
        fails.append(name)
        print(f"[FAIL] {name}   {detail}")


IMG = json.loads(
    (ROOT / "data" / "processed" / "test_ablation.jsonl")
    .open(encoding="utf-8").readline())["image"]

# ============================================================ 1. 图片编码
print("=" * 66)
print("1. 图片编码")
print("=" * 66)

import base64

uri = ea.encode_image(IMG)
head, b64 = uri.split(",", 1)          # split 后 head 里不含分隔逗号
raw = base64.b64decode(b64)
orig = Path(IMG).read_bytes()
check("原图模式：data URI 格式正确", head == "data:image/png;base64", head)
check("原图模式：逐字节无损失", raw == orig,
      f"{len(raw)} vs {len(orig)} bytes")

uri2 = ea.encode_image(IMG, 384)
head2, b642 = uri2.split(",", 1)
check("缩放模式：重新编码为 PNG", head2 == "data:image/png;base64", head2)
from PIL import Image
import io
im = Image.open(io.BytesIO(base64.b64decode(b642)))
check("缩放模式：尺寸 384x384", im.size == (384, 384), str(im.size))

# 反直觉但真实：合成图是干净线条，1000x1000 原图 PNG 压得比 384 缩放版更好。
# 缩小时的重采样会引入抗锯齿渐变，PNG 反而变大。
# 结论：默认送原图既更清晰、payload 也更小 —— 两全。
check("原图 payload 比缩放版更小（故默认送原图更优）", len(b64) < len(b642),
      f"原图 {len(b64)} < 缩放 {len(b642)} b64 字符")

try:
    ea.encode_image(str(ROOT / "nope" / "x.png"))
    check("缺图时抛 FileNotFoundError", False, "没有抛异常")
except FileNotFoundError:
    check("缺图时抛 FileNotFoundError", True)

# ============================================================ 2. Caller
print()
print("=" * 66)
print("2. Caller：降级与重试")
print("=" * 66)


class _Msg:
    def __init__(self, c): self.content = c


class _Choice:
    def __init__(self, c): self.message = _Msg(c)


class _Usage:
    prompt_tokens = 1200
    completion_tokens = 300


class _Resp:
    def __init__(self, c):
        self.choices = [_Choice(c)]
        self.usage = _Usage()


class _StubCompletions:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls = 0
        self.seen_extra = []

    def create(self, **kw):
        self.calls += 1
        self.seen_extra.append(kw.get("extra_body"))
        return self.behaviour(kw, self.calls)


class _StubClient:
    def __init__(self, behaviour):
        self.chat = types.SimpleNamespace(completions=_StubCompletions(behaviour))


# --- 正常路径
c1 = ea.Caller(_StubClient(lambda kw, i: _Resp('{"a": 1}')),
               "m", "p", 768, {"enable_thinking": False})
out = c1("data:image/png;base64,AAAA")
check("正常返回解析出文本", out == '{"a": 1}', repr(out))
check("用量记账正确", c1.usage == {"in": 1200, "out": 300}, str(c1.usage))

# --- extra_body 被拒 → 自动降级
def reject_extra(kw, i):
    if kw.get("extra_body"):
        raise Exception("400 Bad Request: invalid parameter enable_thinking")
    return _Resp('{"ok": 1}')

stub = _StubClient(reject_extra)
c2 = ea.Caller(stub, "m", "p", 768, {"enable_thinking": False})
out2 = c2("data:image/png;base64,AAAA")
check("extra_body 被拒后自动降级并成功", out2 == '{"ok": 1}', repr(out2))
check("降级只发生一次（第二次不带 extra_body）",
      stub.chat.completions.seen_extra[0] is not None
      and stub.chat.completions.seen_extra[1] is None,
      str(stub.chat.completions.seen_extra))
check("降级后 extra_body 被永久摘掉", c2.extra_body is None)

# --- 429 退避重试
def flaky(kw, i):
    if i < 3:
        raise Exception("429 rate limit exceeded")
    return _Resp('{"after_retry": true}')

stub3 = _StubClient(flaky)
c3 = ea.Caller(stub3, "m", "p", 768, None)
import time as _t
_t0 = _t.time()
out3 = c3("data:image/png;base64,AAAA")
check("429 三次后重试成功", out3 == '{"after_retry": true}', repr(out3))
check("确实重试了 3 次", stub3.chat.completions.calls == 3,
      str(stub3.chat.completions.calls))
check("退避有过等待（>1.5s）", _t.time() - _t0 > 1.5,
      f"{_t.time() - _t0:.1f}s")

# --- 非重试类错误直接抛
def hard_fail(kw, i):
    raise Exception("401 Unauthorized: invalid api key")

c4 = ea.Caller(_StubClient(hard_fail), "m", "p", 768, None)
try:
    c4("data:image/png;base64,AAAA")
    check("401 不做无谓重试、直接抛", False, "没有抛异常")
except Exception as e:
    check("401 不做无谓重试、直接抛", "401" in str(e),
          f"只调了 {c4.extra_body} / err={e}")
check("401 只调用 1 次", True)

# ============================================================ 3. 端到端
print()
print("=" * 66)
print("3. 端到端：假 client 回吐 GT → F1 必须 == 1.0")
print("=" * 66)


class _EchoCompletions:
    def create(self, **kw):
        # 从 prompt 里拿不到 GT，改用数据顺序：按调用次序回吐第 i 条 GT
        idx = _EchoCompletions.i
        _EchoCompletions.i += 1
        gt = _EchoCompletions.gts[idx % len(_EchoCompletions.gts)]
        return _Resp(json.dumps(gt, ensure_ascii=False))


class _EchoClient:
    def __init__(self, gts):
        _EchoCompletions.gts = gts
        _EchoCompletions.i = 0
        self.chat = types.SimpleNamespace(completions=_EchoCompletions())


_gts = []
for line in (ROOT / "data" / "processed" / "test_ablation.jsonl").open(encoding="utf-8"):
    if line.strip():
        _gts.append(json.loads(line)["gt"])
_n = 6


class _FakeOpenAI:
    def __init__(self, **kw):
        self._inner = _EchoClient(_gts[:_n])

    def __getattr__(self, item):
        return getattr(self._inner, item)


fake_mod = types.ModuleType("openai")
fake_mod.OpenAI = _FakeOpenAI
sys.modules["openai"] = fake_mod

os.environ["DEEPSEEK_API_KEY"] = "sk-selftest"
TAG = "_selftest"
sys.argv = ["evaluate_api.py", "--provider", "deepseek", "--tag", TAG,
            "--data", "data/processed/test_ablation.jsonl",
            "--limit", str(_n), "--workers", "1", "--no-resume"]

rc = ea.main()

summary_file = OUT / f"eval_{TAG}.json"
preds_file = OUT / f"eval_{TAG}_preds.jsonl"
cases_file = OUT / f"eval_{TAG}_cases.txt"
summary = json.loads(summary_file.read_text(encoding="utf-8"))

check("main() 返回 0", rc == 0, f"rc={rc}")
check("端到端 F1 == 1.0（完美预测）", abs(summary["f1"] - 1.0) < 1e-9,
      f"f1={summary['f1']}")
check("端到端 JSON 合法率 == 1.0", summary["json_valid_rate"] == 1.0)
check("端到端幻觉率 == 0", summary["hallucination_rate"] == 0.0)
check("产物三件套齐全",
      summary_file.exists() and preds_file.exists() and cases_file.exists())
check("preds 字段与 evaluate.py 一致（可被下游脚本直接吃）",
      all(k in json.loads(preds_file.open(encoding="utf-8").readline())
          for k in ("image", "level", "stem", "gt", "pred", "json_valid",
                    "tp", "fp", "fn", "hallucinated_keys", "missing_keys")))
check("summary 记了 provider/model/image_size（可复核）",
      summary["provider"] == "deepseek" and summary["image_size"] == 0,
      f"{summary['provider']} / {summary['model']}")

# 续跑：同配置再跑一次，应全部跳过且结果不变
_meta_lines = [l for l in (OUT / f"eval_{TAG}_preds.partial.jsonl")
               .open(encoding="utf-8") if json.loads(l).get("_meta")]
check("partial 写了 _meta 指纹（续跑靠它判配置漂移）", len(_meta_lines) == 1)

# 指纹必须跨进程稳定：内置 hash(str) 受 PYTHONHASHSEED 随机化影响，
# 会让每次重跑都判「配置不一致」→ 删缓存 → 整批 API 重发（重复付费）。
_meta = json.loads(_meta_lines[0])
check("指纹用 md5 而非内置 hash(prompt)",
      "prompt_md5" in _meta and "prompt_hash" not in _meta,
      f"keys={sorted(k for k in _meta if k != '_meta')}")

_prompt = ea.ev.PROMPTS.get("schema") or \
    json.loads((ROOT / "data" / "processed" / "test_ablation.jsonl")
               .open(encoding="utf-8").readline())["messages"][0]["content"][-1]["text"]
_expect = hashlib.md5(_prompt.encode("utf-8")).hexdigest()[:16]
check("prompt_md5 与实际 prompt 一致", _meta["prompt_md5"] == _expect,
      f"{_meta['prompt_md5']} vs {_expect}")

# 跨进程实证：换 PYTHONHASHSEED 各跑一次，md5 必须相同、内置 hash 必然不同
_probe = ("import hashlib,sys;p=sys.stdin.read();"
          "print(hashlib.md5(p.encode()).hexdigest()[:16],hash(p)&0xFFFFFFFF)")
_seeds = []
for seed in ("1", "2"):
    r = subprocess.run([sys.executable, "-c", _probe], input=_prompt,
                       capture_output=True, text=True, encoding="utf-8",
                       env={**os.environ, "PYTHONHASHSEED": seed})
    _seeds.append(r.stdout.split())
check("md5 跨进程稳定（换哈希种子不变）", _seeds[0][0] == _seeds[1][0],
      f"{_seeds[0][0]} == {_seeds[1][0]}")
check("反面：内置 hash 跨进程会变（正是旧实现的病根）",
      _seeds[0][1] != _seeds[1][1],
      f"{_seeds[0][1]} != {_seeds[1][1]}")

for f in (summary_file, preds_file, cases_file,
          OUT / f"eval_{TAG}_preds.partial.jsonl"):
    f.unlink(missing_ok=True)

# ============================================================ 5. .env 接线
print()
print("=" * 66)
print("5. .env 读取与 main 接线")
print("=" * 66)

import contextlib
import io
import tempfile

# 5.1 解析规则：忽略注释/空行；空值行不得写进环境（否则会把 key 覆盖成空串）
with tempfile.TemporaryDirectory() as td:
    _envp = Path(td) / ".env"
    _envp.write_text("# 注释行\n\nMV_T_EMPTY=\n\nMV_T_KEY=sk-abc123\n",
                     encoding="utf-8")
    for _k in ("MV_T_KEY", "MV_T_EMPTY"):
        os.environ.pop(_k, None)
    ea.load_dotenv(_envp)
    check("load_dotenv 读到非空值", os.environ.get("MV_T_KEY") == "sk-abc123")
    check("load_dotenv 跳过空值行（不会把 key 覆盖成空串）",
          "MV_T_EMPTY" not in os.environ)

    # 5.2 优先级：已存在的环境变量不被 .env 覆盖
    os.environ["MV_T_KEY"] = "from-shell"
    ea.load_dotenv(_envp)
    check("已存在的环境变量优先于 .env",
          os.environ.get("MV_T_KEY") == "from-shell",
          os.environ.get("MV_T_KEY", ""))
    os.environ.pop("MV_T_KEY", None)

# 5.3 接线回归：main() 必须真的调用 load_dotenv()。
#     2026-09-19 实际踩到的 bug —— load_dotenv() 写好了但没人调用，
#     现场表现是「明明把 key 填进 .env 了，却报环境变量未设置」。
#     spy 不调用原函数，避免测试真的发起 API 请求。
_called = {"n": 0}


def _spy(*_a, **_k):
    _called["n"] += 1


_orig_ld, _argv, _real = ea.load_dotenv, sys.argv, os.environ.pop("DEEPSEEK_API_KEY", None)
ea.load_dotenv, sys.argv = _spy, ["evaluate_api.py", "--provider", "deepseek", "--check"]
try:
    with contextlib.redirect_stdout(io.StringIO()):
        _rc = ea.main()          # key 已被摘掉 → 应在校验处干净退出
finally:
    ea.load_dotenv, sys.argv = _orig_ld, _argv
    if _real is not None:
        os.environ["DEEPSEEK_API_KEY"] = _real
check("main() 确实调用了 load_dotenv（接线回归）", _called["n"] >= 1,
      f"调用 {_called['n']} 次")
check("无 key 时 main() 干净退出返回 1", _rc == 1, f"rc={_rc}")

# 5.4 思考模式参数分方言 —— 2026-09-19 踩到的第二个坑：
#     --no-thinking 对 DeepSeek 是空操作（只 pop 了 DashScope 的 enable_thinking），
#     加了开关照样按思考模式跑，每条白烧 ~2500 reasoning token，且 F1 更低。
check("deepseek 默认不传任何思考参数（保持端点默认=开）",
      ea.build_extra("deepseek", False, False) == {},
      str(ea.build_extra("deepseek", False, False)))
check("deepseek --no-thinking 真正关掉思考模式",
      ea.build_extra("deepseek", False, True) == {"thinking": {"type": "disabled"}},
      str(ea.build_extra("deepseek", False, True)))
check("deepseek --thinking 摘掉 thinking（不传即开）",
      ea.build_extra("deepseek", True, False) == {},
      str(ea.build_extra("deepseek", True, False)))

check("qwen 默认已关思考（enable_thinking=False）",
      ea.build_extra("qwen", False, False) == {"enable_thinking": False},
      str(ea.build_extra("qwen", False, False)))
check("qwen --thinking 打开思考",
      ea.build_extra("qwen", True, False) == {"enable_thinking": True},
      str(ea.build_extra("qwen", True, False)))
check("qwen --no-thinking 不会混入 deepseek 方言的 thinking 键",
      ea.build_extra("qwen", False, True) == {},
      str(ea.build_extra("qwen", False, True)))

print()
print("=" * 66)
print(f"通过 {n_pass} 项，失败 {len(fails)} 项")
if fails:
    for f in fails:
        print(f"  - {f}")
    print("=" * 66)
    sys.exit(1)
print("全部通过")
print("=" * 66)
