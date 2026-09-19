"""下载公开数据集到 data/public/。

数据集：
    xfund        XFUND-zh    中文表单，199 张     ← 中文真实数据（关键）
    wildreceipt  WildReceipt 英文收据，1765 张
    cord         CORD        英文小票，1000 张
    funsd        FUNSD       英文表单，199 张（仅做零样本测试，不参与训练）

用法：
    venv-gld/Scripts/python.exe scripts/download_datasets.py
    venv-gld/Scripts/python.exe scripts/download_datasets.py --only xfund wildreceipt
    venv-gld/Scripts/python.exe scripts/download_datasets.py --list

说明：
    - tar 类数据集走 PaddleOCR 直链（国内快），已存在则跳过
    - HuggingFace 类会自动使用 HF_ENDPOINT 镜像（若已设置）
    - 所有内容落在 data/public/<name>/
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "public"

# ------------------------------------------------------------------ 数据集定义
TAR_DATASETS = {
    "xfund": {
        "url": "https://paddleocr.bj.bcebos.com/dataset/XFUND.tar",
        "desc": "XFUND-zh 中文表单 199 张",
    },
    "wildreceipt": {
        "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.1/kie/wildreceipt.tar",
        "desc": "WildReceipt 英文收据 1765 张",
    },
}

HF_DATASETS = {
    "cord": {
        "id": "naver-clova-ix/cord-v2",
        "desc": "CORD 英文小票 1000 张",
    },
    "funsd": {
        "id": "nielsr/funsd",
        "desc": "FUNSD 英文表单 199 张（仅测试）",
    },
}


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def download(url: str, dest: Path) -> bool:
    """带进度显示与断点续传的下载。"""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    done = tmp.stat().st_size if tmp.exists() else 0

    headers = {"Range": f"bytes={done}-"} if done else {}
    try:
        r = requests.get(url, stream=True, timeout=30, headers=headers)
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"    [FAIL] 请求失败: {type(exc).__name__}: {exc}")
        return False

    total = int(r.headers.get("Content-Length", 0)) + done
    mode = "ab" if done else "wb"
    t0 = time.time()
    with open(tmp, mode) as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 / total
                speed = done / max(time.time() - t0, 0.1) / 1024**2
                print(f"\r    {pct:5.1f}%  {human(done)}/{human(total)}  {speed:.1f}MB/s",
                      end="", flush=True)
    print()
    tmp.replace(dest)
    return True


def fetch_tar(name: str, spec: dict) -> bool:
    target = DATA_DIR / name
    if target.exists() and any(target.iterdir()):
        print(f"  [跳过] {name} 已存在")
        return True

    print(f"  [下载] {spec['desc']}")
    tar_path = DATA_DIR / f"{name}.tar"
    if not download(spec["url"], tar_path):
        return False

    print(f"  [解压] {tar_path.name}")
    try:
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path) as tf:
            tf.extractall(DATA_DIR)
    except Exception as exc:  # noqa: BLE001
        print(f"    [FAIL] 解压失败: {type(exc).__name__}: {exc}")
        return False

    # 清理 tar 包：删除失败只提示，不影响数据可用性
    try:
        tar_path.unlink()
    except Exception as exc:  # noqa: BLE001
        print(f"    [提示] tar 包未能自动删除（{tar_path.name}: {type(exc).__name__}），可手动清理")

    return True


def fetch_hf(name: str, spec: dict) -> bool:
    target = DATA_DIR / name
    if target.exists() and any(target.iterdir()):
        print(f"  [跳过] {name} 已存在")
        return True

    print(f"  [下载] {spec['desc']}")
    try:
        from datasets import load_dataset
        ds = load_dataset(spec["id"])
        target.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(target))
    except Exception as exc:  # noqa: BLE001
        print(f"    [FAIL] {type(exc).__name__}: {exc}")
        print("    提示：可设 HF_ENDPOINT=https://hf-mirror.com 后重试")
        return False
    return True


def summarize() -> None:
    print("\n" + "=" * 62)
    print("data/public/ 内容概览")
    print("=" * 62)
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir():
            continue
        files = [f for f in d.rglob("*") if f.is_file()]
        imgs = [f for f in files if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}]
        size = sum(f.stat().st_size for f in files)
        print(f"  {d.name:<16} 图片 {len(imgs):>5} 张 | 文件 {len(files):>5} | {human(size)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="下载公开数据集")
    ap.add_argument("--only", nargs="*", metavar="NAME",
                    help="只下载指定数据集（xfund/wildreceipt/cord/funsd）")
    ap.add_argument("--list", action="store_true", help="列出可用数据集")
    args = ap.parse_args()

    if args.list:
        print("tar 直链数据集：")
        for k, v in TAR_DATASETS.items():
            print(f"  {k:<14} {v['desc']}")
        print("HuggingFace 数据集：")
        for k, v in HF_DATASETS.items():
            print(f"  {k:<14} {v['desc']}")
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("下载公开数据集")
    print("=" * 62)
    if not os.environ.get("HF_ENDPOINT"):
        print("提示：HF_ENDPOINT 未设置，HuggingFace 下载可能较慢")
        print("      建议先执行: setx HF_ENDPOINT https://hf-mirror.com\n")

    want = set(args.only) if args.only else None
    ok, fail = [], []

    for name, spec in TAR_DATASETS.items():
        if want and name not in want:
            continue
        (ok if fetch_tar(name, spec) else fail).append(name)

    for name, spec in HF_DATASETS.items():
        if want and name not in want:
            continue
        (ok if fetch_hf(name, spec) else fail).append(name)

    summarize()

    print()
    if fail:
        print(f"完成：成功 {len(ok)} 个，失败 {len(fail)} 个 -> {fail}")
        return 1
    print(f"完成：{len(ok)} 个数据集全部就绪")
    return 0


if __name__ == "__main__":
    sys.exit(main())
