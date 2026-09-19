"""环境自检 —— 确认 Qwen3-VL QLoRA 训练环境是否就绪。

用法：
    venv-gld/Scripts/python.exe check_env.py

检查项：
    1. Python / torch / CUDA / 显存
    2. 关键依赖包版本（训练栈 + 推理栈）
    3. transformers 是否原生支持 qwen3_vl
    4. unsloth 能否导入

本脚本不下载模型、不占显存，秒级返回。
"""

from __future__ import annotations

import importlib
import sys

# ---------------------------------------------------------------- 关键依赖
# (导入名, 显示名, 用途, 是否必需)
DEPS = [
    ("torch",        "PyTorch",       "训练/推理引擎",      True),
    ("torchvision",  "torchvision",   "图像变换",           True),
    ("transformers", "transformers",  "模型与处理器",       True),
    ("accelerate",   "accelerate",    "设备分发",           True),
    ("peft",         "peft",          "LoRA 适配器",        True),
    ("trl",          "trl",           "SFT / DPO Trainer",  True),
    ("bitsandbytes", "bitsandbytes",  "4bit 量化 (QLoRA)",  True),
    ("datasets",     "datasets",      "数据集加载",         True),
    ("unsloth",      "unsloth",       "显存优化 + 加速",    True),
    ("PIL",          "pillow",        "图像处理",           True),
    ("qwen_vl_utils", "qwen-vl-utils", "Qwen-VL 输入处理",  True),
    ("dspy",         "dspy",          "prompt 自动优化",    False),
]

OK = "[ OK ]"
NG = "[FAIL]"
WARN = "[WARN]"

problems: list[str] = []
warnings: list[str] = []


def line() -> None:
    print("-" * 62)


def check_python() -> None:
    print("\n[1/4] 运行环境")
    line()
    v = sys.version_info
    print(f"  Python        : {sys.version.split()[0]}")
    print(f"  解释器        : {sys.executable}")
    if v < (3, 10):
        problems.append(f"Python {v.major}.{v.minor} 过低，Unsloth 需要 >= 3.10")
    print(f"  {'OK' if v >= (3, 10) else 'FAIL':>4}")


def check_torch() -> None:
    print("\n[2/4] torch 与 GPU")
    line()
    try:
        import torch
    except ImportError:
        problems.append("torch 未安装")
        print("  torch 未安装")
        return

    print(f"  torch         : {torch.__version__}")
    try:
        import torchvision
        tv_ver = torchvision.__version__
    except Exception as exc:  # noqa: BLE001
        tv_ver = f"导入失败 ({type(exc).__name__})"
        warnings.append(f"torchvision 导入失败 —— 常见于依赖被卸载到一半: {exc}")
    print(f"  torchvision   : {tv_ver}")
    print(f"  CUDA compiled : {torch.version.cuda}")

    avail = torch.cuda.is_available()
    print(f"  CUDA available: {avail}")
    if not avail:
        problems.append("torch.cuda.is_available() == False，GPU 不可用")
        return

    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 1024**3
    print(f"  GPU           : {name}")
    print(f"  显存          : {total:.1f} GB")
    print(f"  算力          : sm_{props.major}{props.minor}")
    print(f"  bf16 支持     : {torch.cuda.is_bf16_supported()}")

    if total < 7.5:
        warnings.append(f"显存仅 {total:.1f} GB，低于 8GB 预期，训练参数需相应收紧")

    # 最小前向/反向冒烟，确认内核真的能跑
    try:
        a = torch.ones((64, 64), device="cuda", dtype=torch.bfloat16)
        b = (a @ a).float().mean()
        del a
        torch.cuda.empty_cache()
        print(f"  矩阵乘法冒烟  : {OK}  (mean={b.item():.3f})")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"GPU 矩阵运算失败: {exc}")
        print(f"  矩阵乘法冒烟  : {NG}  {exc}")


def check_deps() -> None:
    print("\n[3/4] 依赖包")
    line()
    for mod, disp, usage, required in DEPS:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  {OK} {disp:<16} {str(ver):<14} {usage}")
        except Exception as exc:  # noqa: BLE001
            tag = NG if required else WARN
            print(f"  {tag} {disp:<16} {'--':<14} {usage}  ({type(exc).__name__})")
            if required:
                problems.append(f"{disp} 缺失（用于 {usage}）")
            else:
                warnings.append(f"{disp} 未安装（可选：{usage}）")


def check_transformers_arch() -> None:
    print("\n[4/4] 架构支持与关键配置")
    line()
    try:
        import transformers
        import os
    except ImportError:
        problems.append("transformers 未安装，无法检查架构支持")
        return

    print(f"  transformers  : {transformers.__version__}")

    models_dir = os.path.join(os.path.dirname(transformers.__file__), "models")
    for arch in ("qwen3_vl", "qwen2_5_vl"):
        present = os.path.isdir(os.path.join(models_dir, arch))
        print(f"  {OK if present else WARN} 原生支持 {arch:<12} {'是' if present else '否'}")
        if arch == "qwen3_vl" and not present:
            problems.append("transformers 不含 qwen3_vl，需升级到 >= 4.57")

    # transformers 大版本提示
    major = int(transformers.__version__.split(".")[0])
    if major >= 5:
        warnings.append(
            "transformers 为 5.x。Unsloth 官方验证组合为 4.57.x；若装 unsloth 时被降级属正常。"
        )

    # HF 缓存与镜像
    hf_endpoint = os.environ.get("HF_ENDPOINT")
    print(f"  HF_ENDPOINT   : {hf_endpoint or '(未设置，走 huggingface.co 官方源)'}")
    if not hf_endpoint:
        warnings.append("未设置 HF_ENDPOINT；若下载模型缓慢，可设为 https://hf-mirror.com")


def main() -> int:
    print("=" * 62)
    print("Qwen3-VL QLoRA 训练环境自检")
    print("=" * 62)

    check_python()
    check_torch()
    check_deps()
    check_transformers_arch()

    print("\n" + "=" * 62)
    if problems:
        print(f"结论：环境未就绪，{len(problems)} 项待解决")
        line()
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
    else:
        print("结论：环境就绪，可以进行冒烟测试（加载 Qwen3-VL-2B）")
    if warnings:
        print(f"\n提示（{len(warnings)} 项，不影响运行）：")
        line()
        for i, w in enumerate(warnings, 1):
            print(f"  {i}. {w}")
    print("=" * 62)

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
