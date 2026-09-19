"""验证离线模式：模型已缓存时，HF_HUB_OFFLINE=1 能否正常加载。

背景：r768 训练卡在 hf-mirror 的 HEAD 请求重试上（16:05 后 2.5 小时无输出，进程僵死）。
模型早已下载到本地缓存，每次启动却还要联网检查配置文件 —— 网络一抖动就卡住。
"""
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

OUT = Path(__file__).resolve().parent.parent / "_offline_check.out"
MODEL = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
lines: list[str] = []


def log(s: str) -> None:
    lines.append(s)
    print(s, flush=True)


try:
    t0 = time.time()
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL)
    log(f"[OK] AutoProcessor 加载成功  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL)
    log(f"[OK] AutoConfig 加载成功  ({time.time()-t0:.1f}s)  架构={cfg.model_type}")

    t0 = time.time()
    from unsloth import FastVisionModel
    log(f"[OK] unsloth 导入成功  ({time.time()-t0:.1f}s)")
    log("")
    log("结论：可以安全使用 HF_HUB_OFFLINE=1")
except Exception as exc:  # noqa: BLE001
    import traceback
    log(f"[FAIL] {type(exc).__name__}: {exc}")
    log(traceback.format_exc()[-1500:])
    log("")
    log("结论：离线模式不可用，需排查缓存完整性")

OUT.write_text("\n".join(lines), encoding="utf-8")
sys.exit(0)
