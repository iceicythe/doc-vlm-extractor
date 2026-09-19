"""环境/进程快照（bash 环境 PATH 失效时的兜底诊断工具）。

输出写到项目根目录 _ps.out（utf-8）。
"""
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "_ps.out"
lines: list[str] = []


def run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                           encoding="utf-8", errors="replace")
        return (r.stdout or "") + (r.stderr or "")
    except Exception as exc:  # noqa: BLE001
        return f"<失败 {type(exc).__name__}: {exc}>"


lines.append("=== GPU 整体 ===")
lines.append(run(["nvidia-smi",
                  "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                  "--format=csv,noheader"]))

lines.append("=== GPU 上的计算进程 ===")
lines.append(run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                  "--format=csv,noheader"]))

lines.append("=== python 进程（按内存降序）===")
out = run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"])
rows = [l for l in out.splitlines() if l.strip().startswith('"')]
parsed = []
for r in rows:
    parts = [p.strip('"') for p in r.split('","')]
    if len(parts) >= 5:
        parsed.append((parts[0], parts[1], parts[4]))   # 名称, PID, 内存
for name, pid, mem in parsed:
    lines.append(f"  PID {pid:<8} {mem}")
lines.append(f"  （共 {len(parsed)} 个 python 进程）")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"已写入 {OUT}")
