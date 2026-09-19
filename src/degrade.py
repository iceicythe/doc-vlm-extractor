"""图像退化管线 —— 让合成图逼近真实拍摄/扫描的工程单据。

设计参考：公开同构项目（qwen3-vl-8b-constat-amiable-lora）的退化参数清单，
并结合工程单据的特点（印章、折痕、装订孔）。

退化只改像素，**不改 Ground Truth** —— 文字被旋转/模糊后仍然要能被读出，
这正是我们希望模型学会的能力。

用法：
    venv-gld/Scripts/python.exe src/degrade.py --n 200
    venv-gld/Scripts/python.exe src/degrade.py --n 200 --level heavy
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SYN_DIR = ROOT / "data" / "synthetic"
SRC_IMG = SYN_DIR / "images"
OUT_IMG = SYN_DIR / "images_degraded"
OUT_GT = SYN_DIR / "gt_degraded.jsonl"


# ================================================================ 配置
@dataclass
class DegradeConfig:
    """各退化的触发概率与强度区间。"""

    name: str = "medium"

    # 几何
    rotate_range: tuple[float, float] = (-3.0, 3.0)
    rotate_p: float = 0.85
    perspective_p: float = 0.35
    perspective_str: tuple[float, float] = (0.004, 0.018)

    # 光学
    blur_p: float = 0.45
    blur_radius: tuple[float, float] = (0.4, 1.2)
    motion_p: float = 0.20
    brightness_p: float = 0.60
    brightness_range: tuple[float, float] = (0.82, 1.12)
    contrast_p: float = 0.55
    contrast_range: tuple[float, float] = (0.78, 1.15)

    # 噪声
    noise_p: float = 0.55
    noise_sigma: tuple[float, float] = (2.0, 9.0)

    # 物理
    crease_p: float = 0.30
    stain_p: float = 0.22
    shadow_p: float = 0.35
    stamp_p: float = 0.50

    # 压缩
    jpeg_p: float = 0.80
    jpeg_quality: tuple[int, int] = (62, 92)


# 三档难度
LEVELS = {
    "light": DegradeConfig(
        name="light",
        rotate_p=0.6, rotate_range=(-1.5, 1.5),
        perspective_p=0.15, perspective_str=(0.002, 0.008),
        blur_p=0.25, blur_radius=(0.3, 0.7),
        motion_p=0.05, noise_p=0.30, noise_sigma=(1.5, 5.0),
        crease_p=0.10, stain_p=0.08, shadow_p=0.15, stamp_p=0.35,
        jpeg_p=0.6, jpeg_quality=(80, 95),
    ),
    "medium": DegradeConfig(),
    "heavy": DegradeConfig(
        name="heavy",
        rotate_p=0.95, rotate_range=(-6.0, 6.0),
        perspective_p=0.60, perspective_str=(0.008, 0.030),
        blur_p=0.65, blur_radius=(0.8, 2.2),
        motion_p=0.35, noise_p=0.80, noise_sigma=(4.0, 16.0),
        crease_p=0.50, stain_p=0.40, shadow_p=0.60, stamp_p=0.70,
        jpeg_p=0.95, jpeg_quality=(45, 80),
    ),
}


# ================================================================ 单项退化
def rotate(img: Image.Image, rng: random.Random, cfg: DegradeConfig) -> Image.Image:
    if rng.random() > cfg.rotate_p:
        return img
    angle = rng.uniform(*cfg.rotate_range)
    # expand=True 会把画布撑大，底色用白模拟纸张
    return img.rotate(angle, resample=Image.BICUBIC, expand=True,
                      fillcolor=(255, 255, 255))


def perspective(img: Image.Image, rng: random.Random, cfg: DegradeConfig) -> Image.Image:
    """四点透视变换，模拟拍摄角度倾斜。"""
    if rng.random() > cfg.perspective_p:
        return img
    w, h = img.size
    s = rng.uniform(*cfg.perspective_str)

    def jitter(v: float, span: float) -> float:
        return v + rng.uniform(-span, span)

    dx, dy = w * s, h * s
    # 源四角（左上、左下、右下、右上）
    src = [(0, 0), (0, h), (w, h), (w, 0)]
    dst = [
        (jitter(0, dx), jitter(0, dy)),
        (jitter(0, dx), jitter(h, dy)),
        (jitter(w, dx), jitter(h, dy)),
        (jitter(w, dx), jitter(0, dy)),
    ]
    coeffs = _find_coeffs(dst, src)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs,
                         resample=Image.BICUBIC, fillcolor=(255, 255, 255))


def _find_coeffs(pa: list, pb: list) -> tuple:
    """解出 PIL PERSPECTIVE 所需的 8 个系数。"""
    matrix = []
    for (x, y), (X, Y) in zip(pa, pb):
        matrix.append([X, Y, 1, 0, 0, 0, -x * X, -x * Y])
        matrix.append([0, 0, 0, X, Y, 1, -y * X, -y * Y])
    A = np.array(matrix, dtype=np.float64)
    B = np.array(pa, dtype=np.float64).reshape(8)
    res = np.linalg.solve(A, B)
    return tuple(res.tolist())


def blur(img: Image.Image, rng: random.Random, cfg: DegradeConfig) -> Image.Image:
    if rng.random() > cfg.blur_p:
        return img
    r = rng.uniform(*cfg.blur_radius)
    return img.filter(ImageFilter.GaussianBlur(radius=r))


def motion_blur(img: Image.Image, rng: random.Random, cfg: DegradeConfig) -> Image.Image:
    """方向性模糊，模拟手抖。"""
    if rng.random() > cfg.motion_p:
        return img
    size = rng.choice([3, 5])          # PIL Kernel 仅支持 3x3 / 5x5
    angle = rng.uniform(0, math.pi)
    kernel = [0.0] * (size * size)
    c = size // 2
    for i in range(size):
        x = int(round(c + (i - c) * math.cos(angle)))
        y = int(round(c + (i - c) * math.sin(angle)))
        if 0 <= x < size and 0 <= y < size:
            kernel[y * size + x] = 1.0
    total = sum(kernel)
    if total == 0:
        return img
    kernel = [k / total for k in kernel]   # 归一化，避免整体变暗
    return img.filter(ImageFilter.Kernel((size, size), kernel, scale=1.0))


def brightness_contrast(img: Image.Image, rng: random.Random,
                        cfg: DegradeConfig) -> Image.Image:
    if rng.random() < cfg.brightness_p:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(*cfg.brightness_range))
    if rng.random() < cfg.contrast_p:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(*cfg.contrast_range))
    return img


def add_noise(img: Image.Image, rng: random.Random, cfg: DegradeConfig) -> Image.Image:
    """传感器噪声（高斯）。"""
    if rng.random() > cfg.noise_p:
        return img
    sigma = rng.uniform(*cfg.noise_sigma)
    arr = np.asarray(img, dtype=np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def shadow(img: Image.Image, rng: random.Random, cfg: DegradeConfig) -> Image.Image:
    """边缘阴影，模拟拍摄时的光照不均。"""
    if rng.random() > cfg.shadow_p:
        return img
    w, h = img.size
    arr = np.asarray(img, dtype=np.float32)

    side = rng.choice(["top", "bottom", "left", "right", "corner"])
    strength = rng.uniform(0.10, 0.32)
    if side in ("top", "bottom"):
        n = max(2, h // 3)
        ramp = np.linspace(1.0, 1.0 - strength, n)
        if side == "bottom":
            ramp = ramp[::-1]
        arr[:n] *= ramp[:, None, None]
    elif side in ("left", "right"):
        n = max(2, w // 3)
        ramp = np.linspace(1.0, 1.0 - strength, n)
        if side == "right":
            ramp = ramp[::-1]
        arr[:, :n] *= ramp[None, :, None]
    else:
        yy, xx = np.mgrid[0:h, 0:w]
        d = np.sqrt((xx / w) ** 2 + (yy / h) ** 2)
        d = d / d.max()
        arr *= (1.0 - strength * d)[:, :, None]

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def crease(img: Image.Image, rng: random.Random, cfg: DegradeConfig) -> Image.Image:
    """折痕：一条带柔化的浅灰线。"""
    if rng.random() > cfg.crease_p:
        return img
    w, h = img.size
    layer = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(layer)
    horizontal = rng.random() < 0.5
    width = rng.randint(1, 3)
    shade = rng.randint(70, 150)
    if horizontal:
        y = rng.randint(int(h * 0.1), int(h * 0.9))
        pts = [(0, y + rng.randint(-8, 8)), (w // 2, y + rng.randint(-8, 8)),
               (w, y + rng.randint(-8, 8))]
    else:
        x = rng.randint(int(w * 0.1), int(w * 0.9))
        pts = [(x + rng.randint(-8, 8), 0), (x + rng.randint(-8, 8), h // 2),
               (x + rng.randint(-8, 8), h)]
    d.line(pts, fill=shade, width=width)
    layer = layer.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.6, 1.8)))

    dark = Image.new("RGB", (w, h), (0, 0, 0))
    return Image.composite(dark, img, layer.point(lambda v: int(v * 0.35)))


def stain(img: Image.Image, rng: random.Random, cfg: DegradeConfig) -> Image.Image:
    """污渍：几个半透明的浅褐斑点。"""
    n = 0 if rng.random() > cfg.stain_p else rng.randint(1, 4)
    if n == 0:
        return img
    w, h = img.size
    layer = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(layer)
    for _ in range(n):
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        rx, ry = rng.randint(w // 30, w // 9), rng.randint(h // 30, h // 9)
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry],
                  fill=rng.randint(35, 90))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=rng.uniform(2.0, 6.0)))

    color = (rng.randint(150, 200), rng.randint(140, 185), rng.randint(110, 155))
    return Image.composite(Image.new("RGB", (w, h), color), img, layer)


def stamp(img: Image.Image, rng: random.Random, cfg: DegradeConfig) -> Image.Image:
    """红色公章：圆环 + 中心文字。"""
    if rng.random() > cfg.stamp_p:
        return img
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    r = int(min(w, h) * rng.uniform(0.09, 0.15))
    cx = rng.randint(r + 10, max(r + 11, w - r - 10))
    cy = rng.randint(r + 10, max(r + 11, h - r - 10))
    red = (rng.randint(190, 225), rng.randint(20, 60), rng.randint(20, 60),
           rng.randint(110, 175))
    hw = max(2, r // 12)

    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=red, width=hw)
    d.ellipse([cx - int(r * 0.82), cy - int(r * 0.82),
               cx + int(r * 0.82), cy + int(r * 0.82)], outline=red, width=max(1, hw // 2))

    # 中心五角星
    star = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        rad = r * (0.42 if i % 2 == 0 else 0.18)
        star.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    d.polygon(star, fill=red)

    out = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    # 印章略微模糊，模拟印泥扩散
    if rng.random() < 0.6:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.7)))
    return out


def jpeg(img: Image.Image, rng: random.Random, cfg: DegradeConfig) -> Image.Image:
    """JPEG 重压缩：经内存往返一次。"""
    if rng.random() > cfg.jpeg_p:
        return img
    import io
    q = rng.randint(*cfg.jpeg_quality)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ================================================================ 主流程
PIPELINE = [
    rotate, perspective, blur, motion_blur,
    brightness_contrast, add_noise, shadow,
    crease, stain, stamp, jpeg,
]


def degrade(img: Image.Image, rng: random.Random, cfg: DegradeConfig) -> Image.Image:
    for fn in PIPELINE:
        img = fn(img, rng, cfg)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description="合成图退化")
    ap.add_argument("--n", type=int, default=0, help="处理数量（0 = 全部）")
    ap.add_argument("--level", choices=list(LEVELS), default="medium")
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--out-suffix", default="", help="输出文件名后缀，如 _heavy")
    args = ap.parse_args()

    cfg = LEVELS[args.level]
    rng = random.Random(args.seed)
    # 默认以 level 作后缀，避免不同档位互相覆盖
    suffix = args.out_suffix or f"_{args.level}"

    srcs = sorted(SRC_IMG.glob("*.png"))
    if not srcs:
        print(f"[FATAL] {SRC_IMG} 下没有图片，请先运行 src/render.py")
        return 1
    if args.n:
        srcs = srcs[: args.n]

    OUT_IMG.mkdir(parents=True, exist_ok=True)
    records = []

    for p in srcs:
        img = Image.open(p).convert("RGB")
        out = degrade(img, rng, cfg)
        name = f"{p.stem}{suffix}.jpg"
        out.save(OUT_IMG / name, format="JPEG", quality=95)
        records.append({"image": f"images_degraded/{name}", "source": p.name})

    # 把 GT 原样复制过来（退化不改文字内容）
    gt_src = SYN_DIR / "gt.jsonl"
    gt_map = {}
    if gt_src.exists():
        with open(gt_src, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                gt_map[Path(r["image"]).name] = r["gt"]

    out_gt = SYN_DIR / f"gt_degraded{suffix}.jsonl"
    with open(out_gt, "w", encoding="utf-8") as f:
        for r in records:
            src_name = r["source"]
            gt = gt_map.get(src_name)
            if gt is None:
                continue
            f.write(json.dumps({"image": r["image"], "gt": gt}, ensure_ascii=False) + "\n")

    print(f"退化完成：{len(records)} 张（level={args.level}）")
    print(f"  图片目录 : {OUT_IMG}")
    print(f"  GT 文件  : {out_gt}")
    sizes = [Image.open(OUT_IMG / Path(r['image']).name).size for r in records[:5]]
    print(f"  尺寸抽样 : {sizes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
