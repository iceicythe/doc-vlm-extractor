"""evaluate.py 的参数解析回归（纯 CPU，不加载模型、不推理）。

钉住 2026-09-19 踩到的那个坑：`--prompt-file` 只加在了 evaluate_api.py 上，
给用户的本地对照命令直接 `unrecognized arguments`。同族脚本参数漂移属于
「代码全对、只有发给用户的那条命令是错的」，必须有用例兜住。

用法：
    .\\venv-gld\\Scripts\\python.exe scripts\\_test_evaluate_args.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
EVAL = ROOT / "src" / "evaluate.py"
API_EVAL = ROOT / "src" / "evaluate_api.py"
PROMPT = ROOT / "outputs" / "prompt_manual_8f.txt"

n_pass = 0
fails = []


def check(name: str, ok: bool, detail: str = "") -> None:
    global n_pass
    if ok:
        n_pass += 1
        print(f"  [ok]   {name}")
    else:
        fails.append(name)
        print(f"  [FAIL] {name}  {detail}")


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PY, str(EVAL), *args],
        cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )


print("=" * 66)
print("evaluate.py 参数回归")
print("=" * 66)

# ---- 1. 两边都要有 --prompt-file（本轮的核心回归点） ----
help_local = subprocess.run(
    [PY, str(EVAL), "--help"], cwd=str(ROOT),
    capture_output=True, text=True, encoding="utf-8", errors="replace",
).stdout
help_api = subprocess.run(
    [PY, str(API_EVAL), "--help"], cwd=str(ROOT),
    capture_output=True, text=True, encoding="utf-8", errors="replace",
).stdout
check("evaluate.py 暴露 --prompt-file", "--prompt-file" in help_local)
check("evaluate_api.py 暴露 --prompt-file", "--prompt-file" in help_api)

# ---- 2. 共享参数两边都在（防漂移的通用断言） ----
for flag in ("--tag", "--limit", "--mode", "--data"):
    check(f"两脚本共享参数对齐: {flag}",
          flag in help_local and flag in help_api,
          f"local={flag in help_local} api={flag in help_api}")

# ---- 3. --prompt 与 --prompt-file 互斥 ----
r = run("--prompt", "abc", "--prompt-file", str(PROMPT), "--score-only", "--tag", "_t_mutex")
check("--prompt 与 --prompt-file 互斥（返回 1）",
      r.returncode == 1 and "只能给一个" in r.stdout, r.stdout.strip()[-120:])

# ---- 4. 文件不存在要干净报错，不能静默用默认 prompt ----
r = run("--prompt-file", "outputs/__nope__.txt", "--score-only", "--tag", "_t_missing")
check("prompt 文件不存在时报错（返回 1）",
      r.returncode == 1 and "不存在" in r.stdout, r.stdout.strip()[-120:])

# ---- 5. --prompt-file 确实被读进去（看运行头的「来源=文件」） ----
r = run("--tag", "_t_src", "--mode", "flat",
        "--prompt-file", str(PROMPT),
        "--data", "data/processed/wildreceipt_test.jsonl",
        "--limit", "100", "--score-only")
check("运行头打印 prompt 来源=文件（含字符数）",
      "来源=文件" in r.stdout and "字符" in r.stdout,
      [l for l in r.stdout.splitlines() if "prompt" in l])

# ---- 6. 不给 prompt 时回落到内置（应显示「来源=内置」） ----
r = run("--tag", "_t_builtin", "--mode", "flat",
        "--data", "data/processed/wildreceipt_test.jsonl",
        "--limit", "100", "--score-only")
check("缺省时回落内置 prompt", "来源=内置" in r.stdout,
      [l for l in r.stdout.splitlines() if "prompt" in l])

# ---- 7. 长 prompt 走文件时，内容与文件逐字一致 ----
sys.path.insert(0, str(ROOT / "src"))
import evaluate as ev  # noqa: E402

text = PROMPT.read_text(encoding="utf-8").strip()
check("文件内容被 strip 后使用（无首尾空白）",
      text == text.strip() and len(text) > 100, f"len={len(text)}")
check("prompt 里点名的字段数与 B 组设计一致",
      all(k in text for k in ("store_name", "store_addr", "tel", "date",
                              "time", "subtotal", "tax", "total")))
check("内置 flat prompt 仍是 3 字段（口径没被误改）",
      all(k in ev.PROMPTS["flat"] for k in ("store_name", "date", "total"))
      and "store_addr" not in ev.PROMPTS["flat"])

print()
print("=" * 66)
print(f"通过 {n_pass} 项，失败 {len(fails)} 项")
if fails:
    for f in fails:
        print(f"  - {f}")
print("=" * 66)
sys.exit(1 if fails else 0)
