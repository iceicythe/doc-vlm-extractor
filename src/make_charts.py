"""生成评测对比图表（纯 Python 输出 SVG，无需 matplotlib）。

用法：
    venv-gld/Scripts/python.exe src/make_charts.py

读取 outputs/ 下的评测结果，生成：
    outputs/chart_main.svg      主结果对比（F1）
    outputs/chart_halluc.svg    幻觉率对比
    outputs/chart_summary.svg   汇总看板（含数字表）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

# ---------------------------------------------------------------- 配色
C_BLUE = "#2563eb"
C_GREEN = "#059669"
C_ORANGE = "#ea580c"
C_RED = "#dc2626"
C_GRAY = "#94a3b8"
C_BG = "#f8fafc"
C_TEXT = "#0f172a"
C_SUB = "#64748b"


def load(tag: str) -> dict | None:
    p = OUT / f"eval_{tag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------- 柱状图
def bar_chart(
    title: str,
    subtitle: str,
    bars: list[tuple[str, float, str]],
    y_max: float,
    y_label: str,
    fmt: str = "{:.3f}",
    width: int = 820,
    height: int = 420,
) -> str:
    pad_l, pad_r, pad_t, pad_b = 90, 40, 96, 84
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    n = len(bars)
    slot = plot_w / n
    bar_w = min(88, slot * 0.55)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Segoe UI,Microsoft YaHei,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="{C_BG}"/>',
        f'<text x="{pad_l}" y="40" font-size="20" font-weight="600" fill="{C_TEXT}">{esc(title)}</text>',
        f'<text x="{pad_l}" y="66" font-size="13" fill="{C_SUB}">{esc(subtitle)}</text>',
    ]

    # 网格线 + y 轴刻度
    steps = 5
    for i in range(steps + 1):
        v = y_max * i / steps
        y = pad_t + plot_h - plot_h * i / steps
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 12}" y="{y + 4:.1f}" font-size="11" fill="{C_SUB}" '
            f'text-anchor="end">{v:.2f}</text>'
        )
    parts.append(
        f'<text x="{pad_l - 60}" y="{pad_t + plot_h / 2:.0f}" font-size="12" fill="{C_SUB}" '
        f'text-anchor="middle" transform="rotate(-90 {pad_l - 60} {pad_t + plot_h / 2:.0f})">'
        f'{esc(y_label)}</text>'
    )

    # 柱子
    for i, (label, value, color) in enumerate(bars):
        cx = pad_l + slot * (i + 0.5)
        h = plot_h * min(value, y_max) / y_max
        y = pad_t + plot_h - h
        x = cx - bar_w / 2
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
            f'rx="4" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{y - 8:.1f}" font-size="13" font-weight="600" '
            f'fill="{C_TEXT}" text-anchor="middle">{fmt.format(value)}</text>'
        )
        # 多行标签
        lines = label.split("\n")
        for j, ln in enumerate(lines):
            parts.append(
                f'<text x="{cx:.1f}" y="{pad_t + plot_h + 22 + j * 15:.1f}" font-size="12" '
                f'fill="{C_SUB}" text-anchor="middle">{esc(ln)}</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------- 汇总看板
def summary_board(width: int = 820, height: int = 470) -> str:
    rows = [
        ("合成数据 · clean", "0.994", "—", C_GREEN),
        ("合成数据 · medium", "0.982", "—", C_GREEN),
        ("合成数据 · heavy", "0.961", "—", C_GREEN),
        ("真实收据 · 零样本", "0.319", "16.0%", C_ORANGE),
        ("真实收据 · 微调后", "0.284", "28.0%", C_RED),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Segoe UI,Microsoft YaHei,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="{C_BG}"/>',
        f'<text x="40" y="42" font-size="20" font-weight="600" fill="{C_TEXT}">'
        f'图文多模态文档结构化提取 · 实验结果</text>',
        f'<text x="40" y="66" font-size="13" fill="{C_SUB}">'
        f'Qwen3-VL-2B · QLoRA · 8GB 消费级显卡 · 合成工程单据 + 真实英文收据</text>',
    ]

    # 三档大数字
    cards = [
        ("合成数据 F1", "0.980", C_GREEN, "同分布测试"),
        ("真实数据 F1", "0.319", C_ORANGE, "跨语言跨域"),
        ("微调副作用", "−10.9%", C_RED, "跨域 F1 下降"),
    ]
    cw, gap = 232, 22
    for i, (label, value, color, note) in enumerate(cards):
        x = 40 + i * (cw + gap)
        parts.append(f'<rect x="{x}" y="92" width="{cw}" height="92" rx="8" fill="white"/>')
        parts.append(f'<rect x="{x}" y="92" width="4" height="92" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{x + 20}" y="120" font-size="12" fill="{C_SUB}">{esc(label)}</text>')
        parts.append(
            f'<text x="{x + 20}" y="154" font-size="30" font-weight="700" '
            f'fill="{color}">{esc(value)}</text>'
        )
        parts.append(f'<text x="{x + 20}" y="174" font-size="11" fill="{C_SUB}">{esc(note)}</text>')

    # 明细表
    ty = 218
    parts.append(f'<text x="40" y="{ty}" font-size="14" font-weight="600" fill="{C_TEXT}">分档明细</text>')
    ty += 16
    cols = [(40, "数据来源"), (300, "字段级 F1"), (480, "幻觉率"), (660, "备注")]
    parts.append(f'<rect x="40" y="{ty}" width="{width - 80}" height="30" rx="4" fill="#e2e8f0"/>')
    for x, name in cols:
        parts.append(
            f'<text x="{x + 12}" y="{ty + 20}" font-size="12" font-weight="600" '
            f'fill="{C_TEXT}">{esc(name)}</text>'
        )
    ty += 30

    notes = ["—", "—", "—", "跨语言+跨领域", "微调放大幻觉"]
    for i, ((label, f1, hall, color), note) in enumerate(zip(rows, notes)):
        bg = "white" if i % 2 == 0 else "#f1f5f9"
        parts.append(f'<rect x="40" y="{ty}" width="{width - 80}" height="30" fill="{bg}"/>')
        parts.append(f'<text x="52" y="{ty + 20}" font-size="12" fill="{C_TEXT}">{esc(label)}</text>')
        parts.append(
            f'<text x="312" y="{ty + 20}" font-size="12" font-weight="600" '
            f'fill="{color}">{esc(f1)}</text>'
        )
        parts.append(f'<text x="492" y="{ty + 20}" font-size="12" fill="{C_TEXT}">{esc(hall)}</text>')
        parts.append(f'<text x="672" y="{ty + 20}" font-size="12" fill="{C_SUB}">{esc(note)}</text>')
        ty += 30

    parts.append(
        f'<text x="40" y="{ty + 28}" font-size="11" fill="{C_SUB}">'
        f'注：合成数据 n=40，真实数据 n=100；样本量低于 50 时指标波动大，不可作为结论。</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------- 分组柱状图
def grouped_bar_chart(
    title: str,
    subtitle: str,
    groups: list[str],
    series: list[tuple[str, str, list[float]]],
    y_min: float,
    y_max: float,
    y_label: str,
    note: str = "",
    width: int = 920,
    height: int = 500,
) -> str:
    pad_l, pad_r, pad_t, pad_b = 95, 45, 130, 108
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    n_g = len(groups)
    slot = plot_w / n_g
    inner = slot * 0.74
    bw = inner / len(series)

    def ypos(v: float) -> float:
        v = max(y_min, min(y_max, v))
        return pad_t + plot_h * (y_max - v) / (y_max - y_min)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Segoe UI,Microsoft YaHei,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="{C_BG}"/>',
        f'<text x="40" y="42" font-size="20" font-weight="600" fill="{C_TEXT}">{esc(title)}</text>',
        f'<text x="40" y="68" font-size="13" fill="{C_SUB}">{esc(subtitle)}</text>',
    ]

    # 图例
    lx = 40
    for name, color, _ in series:
        parts.append(f'<rect x="{lx}" y="86" width="12" height="12" rx="2" fill="{color}"/>')
        parts.append(
            f'<text x="{lx + 18}" y="97" font-size="12" fill="{C_SUB}">{esc(name)}</text>'
        )
        lx += 24 + len(name) * 13

    # 网格 + y 轴
    steps = 5
    for i in range(steps + 1):
        v = y_min + (y_max - y_min) * i / steps
        y = ypos(v)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 12}" y="{y + 4:.1f}" font-size="11" fill="{C_SUB}" '
            f'text-anchor="end">{v:.3f}</text>'
        )
    parts.append(
        f'<text x="{pad_l - 62}" y="{pad_t + plot_h / 2:.0f}" font-size="12" fill="{C_SUB}" '
        f'text-anchor="middle" transform="rotate(-90 {pad_l - 62} {pad_t + plot_h / 2:.0f})">'
        f'{esc(y_label)}</text>'
    )

    # 柱组
    for gi, gname in enumerate(groups):
        gx = pad_l + slot * gi
        for si, (_, color, vals) in enumerate(series):
            v = vals[gi]
            x = gx + (slot - inner) / 2 + bw * si
            y = ypos(v)
            h = pad_t + plot_h - y
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw - 3:.1f}" height="{max(h, 1):.1f}" '
                f'rx="3" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{x + (bw - 3) / 2:.1f}" y="{y - 6:.1f}" font-size="9.5" '
                f'fill="{C_TEXT}" text-anchor="middle">{v:.3f}</text>'
            )
        # 组标签（可多行）
        for j, ln in enumerate(gname.split("\n")):
            parts.append(
                f'<text x="{gx + slot / 2:.1f}" y="{pad_t + plot_h + 22 + j * 15:.1f}" '
                f'font-size="12" fill="{C_SUB}" text-anchor="middle">{esc(ln)}</text>'
            )
        # 组分隔虚线
        if gi:
            parts.append(
                f'<line x1="{gx:.1f}" y1="{pad_t}" x2="{gx:.1f}" y2="{pad_t + plot_h}" '
                f'stroke="#e2e8f0" stroke-width="1" stroke-dasharray="3 3"/>'
            )

    if note:
        parts.append(
            f'<text x="40" y="{height - 16}" font-size="11" fill="{C_SUB}">{esc(note)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------- 错误构成图
def err_structure_chart(
    items: list[tuple[str, int, int]],
    width: int = 920,
    height: int = 470,
) -> str:
    """items: [(配置标签, 近似误读数, 显著误读数)]"""
    title = "错误结构：分辨率把「语义误读」清零，数据量不足才会引入语义错误"
    subtitle = "字段级错误构成 · 同一测试子集 n=102（clean/medium/heavy 各 34）· 全量 3387 个待抽字段"
    pad_l, pad_r, pad_t, pad_b = 95, 45, 118, 106
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    y_max = max(a + b for _, a, b in items) * 1.25
    n = len(items)
    slot = plot_w / n
    bar_w = min(74, slot * 0.5)

    def ypos(v: float) -> float:
        return pad_t + plot_h * (1 - v / y_max)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Segoe UI,Microsoft YaHei,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="{C_BG}"/>',
        f'<text x="40" y="42" font-size="19" font-weight="600" fill="{C_TEXT}">{esc(title)}</text>',
        f'<text x="40" y="68" font-size="13" fill="{C_SUB}">{esc(subtitle)}</text>',
    ]
    for i, (name, color) in enumerate((("近似误读（一字之差）", C_BLUE),
                                       ("显著误读（语义混淆）", C_RED))):
        lx = 40 + i * 200
        parts.append(f'<rect x="{lx}" y="86" width="12" height="12" rx="2" fill="{color}"/>')
        parts.append(
            f'<text x="{lx + 18}" y="97" font-size="12" fill="{C_SUB}">{esc(name)}</text>'
        )

    steps = 5
    for i in range(steps + 1):
        v = y_max * i / steps
        y = ypos(v)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 12}" y="{y + 4:.1f}" font-size="11" fill="{C_SUB}" '
            f'text-anchor="end">{v:.0f}</text>'
        )
    parts.append(
        f'<text x="{pad_l - 62}" y="{pad_t + plot_h / 2:.0f}" font-size="12" fill="{C_SUB}" '
        f'text-anchor="middle" transform="rotate(-90 {pad_l - 62} {pad_t + plot_h / 2:.0f})">'
        f'错误字段数</text>'
    )

    for i, (label, approx, severe) in enumerate(items):
        cx = pad_l + slot * (i + 0.5)
        x = cx - bar_w / 2
        y_approx = ypos(approx)
        h_approx = pad_t + plot_h - y_approx
        y_all = ypos(approx + severe)
        h_severe = y_approx - y_all
        parts.append(
            f'<rect x="{x:.1f}" y="{y_approx:.1f}" width="{bar_w:.1f}" '
            f'height="{max(h_approx, 1):.1f}" fill="{C_BLUE}"/>'
        )
        if severe:
            parts.append(
                f'<rect x="{x:.1f}" y="{y_all:.1f}" width="{bar_w:.1f}" '
                f'height="{max(h_severe, 1):.1f}" fill="{C_RED}"/>'
            )
        parts.append(
            f'<text x="{cx:.1f}" y="{y_all - 9:.1f}" font-size="13" font-weight="600" '
            f'fill="{C_TEXT}" text-anchor="middle">{approx + severe}</text>'
        )
        if severe:
            parts.append(
                f'<text x="{cx:.1f}" y="{y_all - 26:.1f}" font-size="10" '
                f'fill="{C_RED}" text-anchor="middle">显著 {severe}</text>'
            )
        for j, ln in enumerate(label.split("\n")):
            parts.append(
                f'<text x="{cx:.1f}" y="{pad_t + plot_h + 22 + j * 15:.1f}" font-size="12" '
                f'fill="{C_SUB}" text-anchor="middle">{esc(ln)}</text>'
            )

    parts.append(
        f'<text x="40" y="{height - 16}" font-size="11" fill="{C_SUB}">'
        f'注：错误字段数由 src/analyze_errors.py 直接统计预测文件得出，可复核。</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------- 消融图入口
def ablation_charts() -> list[str]:
    """读取 outputs/ablation_summary.json + 各 arm 的错误分析，生成消融图。"""
    p = OUT / "ablation_summary.json"
    if not p.exists():
        print("  [跳过] 未找到 ablation_summary.json")
        return []
    summ = json.loads(p.read_text(encoding="utf-8"))

    # tag -> (标签, 配色)
    order = [
        ("v1", "384px\n数据 100%", C_GRAY),
        ("abl_d50", "384px\n数据 50%", C_GRAY),
        ("abl_d25", "384px\n数据 25%", C_GRAY),
        ("abl_r512", "512px\n数据 100%", C_BLUE),
        ("abl_r768", "768px\n数据 100%", C_GREEN),
        ("abl_lang", "384px\n仅语言层", C_ORANGE),
    ]
    groups, series = [], {lv: [] for lv in ("clean", "medium", "heavy")}
    for tag, label, _ in order:
        d = summ.get(tag)
        if not d or d.get("failed"):
            continue
        groups.append(label)
        bl = d.get("by_level", {})
        for lv in series:
            cell = bl.get(lv, 0.0)
            # 兼容两种结构：{"clean": 0.98} 或 {"clean": {"f1": 0.98}}
            series[lv].append(cell.get("f1", 0.0) if isinstance(cell, dict) else float(cell))

    files: list[str] = []
    if groups:
        svg = grouped_bar_chart(
            "消融实验：分辨率是主导因素，数据量要砍到 25% 才显形",
            "Qwen3-VL-2B QLoRA · 同一测试子集 n=102（每档 34）· 固定 seed=3407 · 384px 基线可复现",
            groups,
            [("clean", C_GREEN, series["clean"]),
             ("medium", C_BLUE, series["medium"]),
             ("heavy", C_ORANGE, series["heavy"])],
            0.90, 1.0, "字段级 F1",
            "注：y 轴从 0.90 起，刻意放大差异；柱顶数字为各档 F1。",
        )
        (OUT / "chart_ablation.svg").write_text(svg, encoding="utf-8")
        files.append("chart_ablation.svg")

    # ---- 错误构成（从 preds 现算，不读报告文本）----
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from analyze_errors import analyze  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        analyze = None  # type: ignore[assignment]

    if analyze is not None:
        items: list[tuple[str, int, int]] = []
        # tag -> 预测文件名（基线跑在 sft_v1 上，预测文件叫 eval_abl_base_preds.jsonl）
        preds_name = {"v1": "eval_abl_base_preds.jsonl"}
        for tag, label, _ in order:
            preds = OUT / preds_name.get(tag, f"eval_abl_{tag}_preds.jsonl")
            if not preds.exists():
                continue
            a = analyze(preds)
            ct = a["cat_total"]
            approx = ct.get("数值误读", 0) + ct.get("文本近似误读", 0)
            severe = ct.get("文本显著误读", 0) + ct.get("漏抽", 0) \
                + ct.get("多抽/幻觉", 0) + ct.get("行列错位", 0)
            items.append((label, approx, severe))
        if items:
            (OUT / "chart_ablation_err.svg").write_text(
                err_structure_chart(items), encoding="utf-8")
            files.append("chart_ablation_err.svg")

    return files


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # 主 F1 对比
    main_bars = [
        ("合成\nclean", 0.994, C_GREEN),
        ("合成\nmedium", 0.982, C_GREEN),
        ("合成\nheavy", 0.961, C_GREEN),
        ("真实收据\n零样本", 0.319, C_ORANGE),
        ("真实收据\n微调后", 0.284, C_RED),
    ]
    (OUT / "chart_main.svg").write_text(
        bar_chart(
            "字段级 F1 对比：合成数据 vs 真实数据",
            "合成测试集 n=40 · 真实收据 n=100 · 跨域测试为零样本/微调后在 WildReceipt 上的表现",
            main_bars, 1.0, "字段级 F1",
        ),
        encoding="utf-8",
    )

    # 幻觉率
    hall_bars = [
        ("合成\nclean", 0.000, C_GREEN),
        ("合成\nmedium", 0.000, C_GREEN),
        ("合成\nheavy", 0.000, C_GREEN),
        ("真实收据\n零样本", 0.160, C_ORANGE),
        ("真实收据\n微调后", 0.280, C_RED),
    ]
    (OUT / "chart_halluc.svg").write_text(
        bar_chart(
            "幻觉率对比：微调放大了幻觉倾向",
            "幻觉率 = 输出了 Ground Truth 中不存在字段的样本占比 · 真实收据 n=100",
            hall_bars, 0.5, "幻觉率",
        ),
        encoding="utf-8",
    )

    # 汇总看板
    (OUT / "chart_summary.svg").write_text(summary_board(), encoding="utf-8")

    # 消融实验
    extra = ablation_charts()

    print("图表已生成：")
    for f in ("chart_main.svg", "chart_halluc.svg", "chart_summary.svg", *extra):
        p = OUT / f
        print(f"  {p}  ({p.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
