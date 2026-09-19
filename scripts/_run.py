"""通用子进程运行器：跑一条命令，把 stdout/stderr 原样落到 utf-8 文件。

为什么需要它：在 Windows 上通过 PowerShell 调 python 时，stdout 经常抓不到
（管道/编码问题），屏幕上什么都看不到。落文件再看最稳。

用法：
    python scripts/_run.py outputs/_run_xxx.txt src/evaluate.py --tag foo --score-only
    python scripts/_run.py outputs/_run_test.txt scripts/_test_evaluate_scoring.py
"""
import os
import subprocess
import sys
from pathlib import Path

PY = r"C:\MiniVLM\venv-gld\Scripts\python.exe"
ROOT = Path(__file__).resolve().parent.parent

# 子进程默认按 Windows ANSI 代码页（GBK）写 stdout，父进程按 utf-8 解码会全乱码。
# 强制子进程用 utf-8，两端就对齐了。
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(2)

outname = sys.argv[1]
target = sys.argv[2]
rest = sys.argv[3:]

if target.endswith(".py") and (ROOT / target).exists():
    argv = [PY, str(ROOT / target), *rest]
else:                       # 脚本名，默认去 src/ 找
    argv = [PY, str(ROOT / "src" / target), *rest]

r = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                   errors="replace", cwd=str(ROOT), env=ENV)
out = ROOT / "outputs" / outname
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text((r.stdout or "") + ("\n[stderr]\n" + r.stderr if r.stderr else "")
               + f"\n[exit={r.returncode}]\n", encoding="utf-8")
print(f"exit={r.returncode} -> {out}")
