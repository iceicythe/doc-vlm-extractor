"""合成工程材料清单数据 —— 渲染图片 + 生成 Ground Truth。

设计依据：docs/schema_v1.md

用法：
    venv-gld/Scripts/python.exe src/render.py --n 100
    venv-gld/Scripts/python.exe src/render.py --n 100 --seed 42 --show 3

产出：
    data/synthetic/images/synth_000001.png ...
    data/synthetic/gt.jsonl        每行 {"image": "...", "gt": {...}}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "synthetic"
IMG_DIR = OUT_DIR / "images"
GT_PATH = OUT_DIR / "gt.jsonl"

# ------------------------------------------------------------------ 画布参数
IMG_W, IMG_H = 760, 520
MARGIN_X = 30
FONT_SIZE_TITLE = 26
FONT_SIZE_HEAD = 17
FONT_SIZE_CELL = 16
FONT_SIZE_NOTE = 14

# 列定义：(GT 字段名, 表头显示, 相对宽度权重, 对齐)
COLUMNS = [
    ("序号", "序号", 0.8, "center"),
    ("名称", "材料名称", 3.2, "left"),
    ("规格型号", "规格型号", 2.0, "left"),
    ("单位", "单位", 0.9, "center"),
    ("数量", "数量", 1.2, "right"),
    ("单价", "单价(元)", 1.6, "right"),
    ("金额", "金额(元)", 1.9, "right"),
]

# ------------------------------------------------------------------ 词表
# (材料名, 规格候选, 单位, 单价区间, 数量区间)
MATERIALS = [
    ("螺纹钢 HRB400", ["Φ12", "Φ16", "Φ20", "Φ25"], "吨", (3800, 4500), (5, 60)),
    ("圆钢 HPB300", ["Φ8", "Φ10"], "吨", (3900, 4600), (3, 30)),
    ("混凝土", ["C25", "C30", "C35", "C40"], "m3", (430, 580), (20, 220)),
    ("木模板", ["1830×915×15", "1220×2440×12"], "m2", (42, 68), (80, 900)),
    ("脚手架钢管", ["Φ48×3.0", "Φ48×3.5"], "米", (12, 22), (300, 3000)),
    ("扣件", ["直角", "旋转", "对接"], "个", (3, 7), (500, 5000)),
    ("安全网", ["密目式 1.8×6m"], "m2", (8, 16), (200, 2000)),
    ("水泥", ["P.O 42.5", "P.C 32.5"], "吨", (380, 480), (10, 120)),
    ("中砂", ["中粗"], "m3", (110, 170), (30, 300)),
    ("碎石", ["5-25mm", "10-30mm"], "m3", (95, 150), (30, 300)),
    ("标准砖", ["240×115×53"], "个", (0.4, 0.7), (5000, 80000)),
    ("蒸压加气块", ["600×200×200"], "m3", (240, 330), (20, 200)),
    ("防水卷材", ["SBS 4mm", "APP 3mm"], "m2", (26, 45), (300, 2500)),
    ("挤塑板", ["XPS 50mm", "XPS 30mm"], "m2", (18, 32), (200, 1800)),
    ("电力电缆", ["YJV 4×25", "YJV 4×50"], "米", (78, 145), (100, 1200)),
    ("PPR 给水管", ["DN25", "DN32"], "米", (9, 18), (200, 2000)),
    ("镀锌钢管", ["DN50", "DN80"], "米", (35, 65), (100, 900)),
    ("钢筋套筒", ["Φ16", "Φ20"], "个", (4, 9), (300, 3000)),
    ("土工布", ["200g/m2"], "m2", (4, 9), (500, 4000)),
    ("保温砂浆", ["I 型"], "吨", (620, 850), (5, 60)),
]

# 供应商名 = 地区 + 字号 + 业务后缀
# 组合数 = 20 × 200 × 8 = 32000，避免模型"背词表"
SUPPLIER_REGIONS = [
    "陕西", "西安", "咸阳", "宝鸡", "渭南", "汉中", "安康", "商洛", "延安", "榆林",
    "铜川", "杨凌", "长安", "临潼", "高陵", "鄠邑", "周至", "蓝田", "三原", "兴平",
]
SUPPLIER_SUFFIXES = [
    "建材有限公司", "物资贸易有限公司", "商贸有限公司", "供应链管理有限公司",
    "工程材料有限公司", "贸易有限公司", "建材科技有限公司", "物资有限公司",
]
SUPPLIER_BRANDS = [
    # —— 常见双字字号（混合风格，避免过于规律）
    "宏远", "兴达", "隆昌", "嘉盛", "恒通", "金鼎", "众联", "远洋", "鼎盛", "天成",
    "华建", "中兴", "九州", "广厦", "鲁班", "泰山", "明德", "正大", "利源", "鑫辉",
    "宏图", "腾飞", "永利", "丰泽", "润安", "泰顺", "新宏", "万方", "四海", "环宇",
    "盛世", "博纳", "创辉", "弘毅", "卓越", "昊宇", "宸鑫", "晟泰", "嘉禾", "瑞泽",
    "德坤", "隆瑞", "华信", "诚恒", "久元", "亨通", "聚鑫", "汇通", "金桥", "银海",
    "铜城", "铁马", "钢联", "石基", "木盛", "火旺", "土固", "水润", "东方", "南方",
    "西域", "北辰", "中天", "大地", "长城", "黄河", "长江", "泰岳", "春华", "秋实",
    "明阳", "朝阳", "晨曦", "星辉", "月明", "龙腾", "鹏程", "凤翔", "麒麟", "骏马",
    "蓝天", "白云", "青云", "紫金", "红叶", "绿源", "碧水", "金沙", "银山", "玉龙",
    "锦程", "锦绣", "华美", "华宇", "华丰", "华泰", "华鑫", "华瑞", "华通", "华昌",
    "中泰", "中远", "中鑫", "中盛", "中达", "中联", "中材", "中冶", "泰达", "泰和",
    "泰丰", "泰昌", "泰隆", "泰源", "泰德", "泰华", "鑫源", "鑫达", "鑫盛", "鑫隆",
    "鑫泰", "鑫华", "鑫建", "鑫材", "鑫通", "鑫宇", "金源", "金达", "金盛", "金泰",
    "金华", "金建", "金材", "金宇", "金通", "金昌", "广源", "广达", "广盛", "广泰",
    "广华", "广建", "广材", "广宇", "广通", "广昌", "兴源", "兴隆", "兴盛", "兴泰",
    "兴华", "兴建", "兴材", "兴宇", "兴通", "兴昌", "恒源", "恒达", "恒盛", "恒泰",
    "恒华", "恒建", "恒材", "恒宇", "恒昌", "信源", "信达", "信盛", "信泰", "信华",
    "信建", "信材", "信宇", "信通", "信昌", "润源", "润达", "润盛", "润泰", "润华",
    "润建", "润材", "润宇", "润通", "润昌", "泽源", "泽达", "泽盛", "泽泰", "泽华",
    "泽建", "泽材", "泽宇", "泽通", "泽昌", "安泰", "安顺", "安达", "安盛", "安源",
]

# 项目名 = 地区 + 工程类型，组合数 = 20 × 12 = 240
PROJECT_REGIONS = [
    "曲江", "高新", "经开", "浐灞", "航天基地", "雁塔", "未央", "碑林", "新城", "莲湖",
    "灞桥", "长安", "临潼", "阎良", "鄠邑", "高陵", "周至", "蓝田", "沣东", "沣西",
]
PROJECT_TYPES = [
    "住宅楼主体工程", "市政道路改造工程", "商业综合体项目", "产业园区一期工程",
    "轨道交通站点工程", "医院综合楼工程", "学校教学楼工程", "地下管廊工程",
    "棚户区改造工程", "污水处理厂工程", "体育中心工程", "文化馆建设工程",
]

# 日期显示格式（GT 一律归一化为 YYYY-MM-DD）
DATE_STYLES = [
    lambda d: d.strftime("%Y-%m-%d"),
    lambda d: d.strftime("%Y/%m/%d"),
    lambda d: d.strftime("%Y.%m.%d"),
    lambda d: f"{d.year}年{d.month:02d}月{d.day:02d}日",
    lambda d: f"{d.year}年{d.month}月{d.day}日",
    lambda d: d.strftime("%Y%m%d"),
]


# ================================================================ 数据生成
def gen_header(rng: random.Random) -> tuple[dict, dict]:
    """返回 (GT 表头, 显示用表头)。"""
    project = rng.choice(PROJECT_REGIONS) + rng.choice(PROJECT_TYPES)
    supplier = (rng.choice(SUPPLIER_REGIONS) + rng.choice(SUPPLIER_BRANDS)
                + rng.choice(SUPPLIER_SUFFIXES))
    d = date(2026, 1, 1) + timedelta(days=rng.randint(0, 280))
    doc_no = f"CL-{d.strftime('%Y%m%d')}-{rng.randint(1, 999):03d}"

    gt = {
        "项目名称": project,
        "供应商": supplier,
        "单据编号": doc_no,
        "日期": d.strftime("%Y-%m-%d"),          # GT 统一格式
    }
    shown = dict(gt)
    shown["日期"] = rng.choice(DATE_STYLES)(d)   # 图上随机格式
    return gt, shown


def fmt_money(x: float) -> str:
    return f"{x:.2f}"


def fmt_qty(x: float) -> str:
    return f"{x:g}"


def gen_detail_rows(rng: random.Random, n_rows: int) -> list[dict]:
    picks = rng.sample(MATERIALS, n_rows)
    # 样本级控制：85% 全部等式成立；15% 随机挑一行制造录入误差
    # （行级概率会让多行样本几乎必然出错，统计口径不清）
    err_idx = rng.randrange(n_rows) if rng.random() < 0.15 else -1
    rows = []
    for i, (name, specs, unit, price_rng, qty_rng) in enumerate(picks, start=1):
        spec = rng.choice(specs)
        price = round(rng.uniform(*price_rng), 2)
        # 数量：吨/m3 类带小数，个/米 类多为整数
        if unit in ("吨", "m3"):
            qty = round(rng.uniform(*qty_rng), rng.choice([0, 1, 1, 2]))
        else:
            qty = float(rng.randint(*qty_rng))

        amount = round(qty * price, 2)
        if i - 1 == err_idx:
            amount = round(amount + rng.choice([-1, 1]) * rng.uniform(0.01, 8.0), 2)

        rows.append({
            "序号": str(i),
            "名称": name,
            "规格型号": spec,
            "单位": unit,
            "数量": fmt_qty(qty),
            "单价": fmt_money(price),
            "金额": fmt_money(amount),
        })
    return rows


def gen_record(rng: random.Random) -> tuple[dict, dict]:
    """返回 (GT record, 渲染用 record)。"""
    gt_head, shown_head = gen_header(rng)
    n_rows = rng.randint(1, 8)
    rows = gen_detail_rows(rng, n_rows)

    # 合计 = 明细金额之和（同样保留"照图读"语义）
    total = round(sum(float(r["金额"]) for r in rows), 2)
    if rng.random() < 0.15:
        total = round(total + rng.choice([-1, 1]) * rng.uniform(0.01, 20.0), 2)

    gt = {
        "单据类型": "材料清单",
        "表头": gt_head,
        "明细": rows,
        "合计": {"金额": fmt_money(total)},
    }
    shown = {"表头": shown_head, "明细": rows, "合计": fmt_money(total)}
    return gt, shown


# ================================================================ 渲染
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def load_font(size: int) -> ImageFont.FreeTypeFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
        r"C:\Windows\Fonts\simhei.ttf",    # 黑体
        r"C:\Windows\Fonts\simsun.ttc",    # 宋体
        r"C:\Windows\Fonts\Deng.ttf",      # 等线
    ]
    for p in candidates:
        if Path(p).exists():
            f = ImageFont.truetype(p, size)
            _FONT_CACHE[size] = f
            return f
    raise RuntimeError(
        "未找到中文字体。请确认 C:\\Windows\\Fonts 下存在 msyh.ttc / simhei.ttf / simsun.ttc"
    )


def text_w(draw: ImageDraw.ImageDraw, s: str, font) -> float:
    box = draw.textbbox((0, 0), s, font=font)
    return box[2] - box[0]


def draw_cell(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font,
    align: str = "left",
    pad: int = 6,
) -> None:
    x0, y0, x1, y1 = box
    tw = text_w(draw, text, font)
    th = font.size
    if align == "center":
        tx = x0 + (x1 - x0 - tw) / 2
    elif align == "right":
        tx = x1 - pad - tw
    else:
        tx = x0 + pad
    ty = y0 + (y1 - y0 - th) / 2 - 2
    draw.text((tx, ty), text, fill="black", font=font)


def render(shown: dict, rng: random.Random) -> Image.Image:
    # 高度自适应：标题 + 表头信息 + 表格(表头/明细/合计) + 签章行 + 底部留白
    n_rows = len(shown["明细"])
    row_h = 30
    img_h = (22 + (FONT_SIZE_TITLE + 14) + 2 * (FONT_SIZE_NOTE + 6) + 12
             + (n_rows + 2) * row_h + 16 + FONT_SIZE_NOTE + 30)
    img = Image.new("RGB", (IMG_W, img_h), "white")
    draw = ImageDraw.Draw(img)

    f_title = load_font(FONT_SIZE_TITLE)
    f_head = load_font(FONT_SIZE_HEAD)
    f_cell = load_font(FONT_SIZE_CELL)
    f_note = load_font(FONT_SIZE_NOTE)

    y = 22
    title = "工程材料清单"
    draw.text(((IMG_W - text_w(draw, title, f_title)) / 2, y), title, fill="black", font=f_title)
    y += FONT_SIZE_TITLE + 14

    # ---- 表头信息
    head = shown["表头"]
    draw.text((MARGIN_X, y), f"项目名称：{head['项目名称']}", fill="black", font=f_note)
    draw.text((IMG_W - MARGIN_X - text_w(draw, f"日期：{head['日期']}", f_note), y),
              f"日期：{head['日期']}", fill="black", font=f_note)
    y += FONT_SIZE_NOTE + 6
    draw.text((MARGIN_X, y), f"供应商：{head['供应商']}", fill="black", font=f_note)
    draw.text((IMG_W - MARGIN_X - text_w(draw, f"编号：{head['单据编号']}", f_note), y),
              f"编号：{head['单据编号']}", fill="black", font=f_note)
    y += FONT_SIZE_NOTE + 12

    # ---- 表格
    table_w = IMG_W - 2 * MARGIN_X
    total_w = sum(c[2] for c in COLUMNS)
    col_ws = [int(table_w * c[2] / total_w) for c in COLUMNS]
    col_ws[-1] = table_w - sum(col_ws[:-1])       # 修正舍入误差
    col_x = [MARGIN_X]
    for w in col_ws:
        col_x.append(col_x[-1] + w)

    def draw_row(ry: int, cells: list[str], align_over: list[str], font) -> None:
        draw.rectangle([MARGIN_X, ry, MARGIN_X + table_w, ry + row_h], outline="black")
        for i in range(1, len(col_x) - 1):
            draw.line([col_x[i], ry, col_x[i], ry + row_h], fill="black")
        for i, (txt, al) in enumerate(zip(cells, align_over)):
            draw_cell(draw, (col_x[i], ry, col_x[i + 1], ry + row_h), txt, font, al)

    # 表头行
    draw_row(y, [c[1] for c in COLUMNS], ["center"] * len(COLUMNS), f_head)
    y += row_h

    # 明细行
    aligns = [c[3] for c in COLUMNS]
    for row in shown["明细"]:
        draw_row(y, [row.get(c[0], "") for c in COLUMNS], aligns, f_cell)
        y += row_h

    # 合计行
    total_label = "合计金额(元)"
    draw.rectangle([MARGIN_X, y, MARGIN_X + table_w, y + row_h], outline="black")
    for i in range(1, len(col_x) - 1):
        draw.line([col_x[i], y, col_x[i], y + row_h], fill="black")
    draw_cell(draw, (col_x[0], y, col_x[-2], y + row_h), total_label, f_head, "right")
    draw_cell(draw, (col_x[-2], y, col_x[-1], y + row_h), shown["合计"], f_head, "right")
    y += row_h + 16

    # ---- 签章行
    draw.text((MARGIN_X, y), "制单人：________", fill="black", font=f_note)
    draw.text((IMG_W // 2, y), "审核人：________", fill="black", font=f_note)
    draw.text((IMG_W - MARGIN_X - text_w(draw, "日期：____年__月__日", f_note), y),
              "日期：____年__月__日", fill="black", font=f_note)

    return img


# ================================================================ 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="生成合成工程材料清单")
    ap.add_argument("--n", type=int, default=100, help="生成数量")
    ap.add_argument("--seed", type=int, default=20260915, help="随机种子")
    ap.add_argument("--show", type=int, default=0, help="打印前 N 条 GT 供检查")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for i in range(1, args.n + 1):
        gt, shown = gen_record(rng)
        img = render(shown, rng)
        name = f"synth_{i:06d}.png"
        img.save(IMG_DIR / name)
        records.append({"image": f"images/{name}", "gt": gt})

    with open(GT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 统计
    n_rows = [len(r["gt"]["明细"]) for r in records]
    bad_eq = sum(
        1 for r in records
        if any(abs(float(d["数量"]) * float(d["单价"]) - float(d["金额"])) > 0.01 for d in r["gt"]["明细"])
    )
    print(f"生成完成：{len(records)} 张")
    print(f"  图片目录 : {IMG_DIR}")
    print(f"  GT 文件  : {GT_PATH}")
    print(f"  明细行数 : 最少 {min(n_rows)} / 最多 {max(n_rows)} / 平均 {sum(n_rows)/len(n_rows):.1f}")
    print(f"  等式不成立的样本: {bad_eq} ({bad_eq*100//len(records)}%)")

    for r in records[: args.show]:
        print("\n" + "-" * 60)
        print(r["image"])
        print(json.dumps(r["gt"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
