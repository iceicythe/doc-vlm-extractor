"""查看日志尾部（因为当前 bash 环境的 PATH 失效，用 python 兜底）。

用法:
    python scripts/_tail.py <文件路径> [行数]
输出写到项目根目录的 _tail.out（utf-8），再用编辑器查看。
"""
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("用法: python scripts/_tail.py <文件> [行数]")
    sys.exit(1)

p = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
if not p.exists():
    print(f"不存在: {p}")
    sys.exit(1)

raw = p.read_bytes()
text = raw.decode("utf-8", errors="replace").replace("\r", "\n")
lines = [l.rstrip() for l in text.splitlines() if l.strip()]

size_kb = p.stat().st_size / 1024
import time
mtime = time.strftime("%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))

header = f"== {p}  |  {size_kb:.0f} KB  |  共 {len(lines)} 行  |  最后修改 {mtime} =="
out = Path(__file__).resolve().parent.parent / "_tail.out"
out.write_text(header + "\n\n" + "\n".join(lines[-n:]), encoding="utf-8")
print(f"已写入 {out}")
