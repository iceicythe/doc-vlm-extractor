"""Demo —— 一屏看完「零样本 → 知识注入 → SFT」与真值的逐字段差异。

用法（必须在自己终端跑，模型要占 GPU）：
    venv-gld\\Scripts\\python.exe src\\demo.py --check            # 纯 CPU 自检，先跑这个
    venv-gld\\Scripts\\python.exe src\\demo.py --smoke 41         # 真机检查两档 + adapter 切换
    venv-gld\\Scripts\\python.exe src\\demo.py                    # http://127.0.0.1:7860
    venv-gld\\Scripts\\python.exe src\\demo.py --port 7861
    venv-gld\\Scripts\\python.exe src\\demo.py --render-sample 2  # 用已落盘预测渲一张静态预览
    venv-gld\\Scripts\\python.exe src\\demo.py --render-sample 41 --png   # 再截一张 PNG

三个设计取舍（面试会被问到，先写清楚）：
  1. **两个档位共用一个模型**：PeftModel 挂上 SFT LoRA，零样本用 `disable_adapter()`
     临时摘掉适配器。所以 ② 与 ③ 的输入图像、prompt、解码参数完全相同，
     唯一变量就是 adapter —— 这是「③ 的提升归因于微调」的前提。
  2. **不重写打分逻辑**：差异比对复用 `evaluate.flatten/align_rows`，
     知识注入复用 `inject_knowledge.correct_record`。Demo 若自造一套口径，
     屏幕上显示的和报告里的数字就会对不上。
  3. **不做结果预烤**：每次点按钮都是真推理（384px 约十几秒/档）。
     预烤能秒出，但那就不是 demo 了。

自检（`--check`）覆盖 5 件事：
  * `flatten_pairs` 的归一化值与 `evaluate.flatten` 逐路径全等（跨全部历史预测文件）
  * 四种字段状态 + 合计标量 + 解析失败 + HTML 转义 + F1 条 / 图例色点等视觉结构
  * Demo 的注入结果与 `inject_knowledge.correct_record` 逐条一致
  * 全局指标常量与 `outputs/eval_*.json` 对得上（防止报告改了、Demo 没改）
  * Gradio 界面可构建 + 主题可实例化（Gradio 6 把 theme/css 从 Blocks 挪到了
    launch，传错会静默失效；主题名写死也会让界面直接起不来）
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import re
import subprocess
import time
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")   # 本地权重已缓存，禁掉网络探测

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
LEXICON = ROOT / "data" / "lexicon" / "v1.json"

MODEL_NAME = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
SFT_DIR = (OUTPUTS / "sft_layout_v1" / "lora"
           if (OUTPUTS / "sft_layout_v1" / "lora").exists()
           else OUTPUTS / "sft_v1" / "lora")
MAX_NEW_TOKENS = 768
TEST_FILE = PROCESSED / "test_ablation.jsonl"

# 知识注入参数：与 inject_knowledge.py 的默认值保持一致（拐点见 report_inject_sweep.md）
MIN_SIM, MIN_GAP = 0.50, 0.05

# 全局指标（同域 = test_ablation 102 条；跨域 = wildreceipt_test 前 100 条）。
# 数字来源写在第三列，--check 会去核对能核对的那几个。
GLOBAL_TIERS = [
    ("① OCR 规则基线", 0.8245, None, "outputs/ocr-test-v1/summary.json"),
    ("② 2B 零样本", 0.6924, 0.3190, "outputs/evaluation-v2-final/eval_zero_abl.json"),
    ("② + 知识注入", 0.7870, None, "outputs/report_inject_zero_abl.md"),
    ("③ 2B SFT 384px", 0.9785, 0.2842, "outputs/evaluation-v2-final/eval_abl_base.json"),
    ("③ + 版式增强", 0.9794, None, "outputs/scoring-v2.0.0/eval_layout_v1_regression.json"),
    ("③ + 知识注入", 0.9779, None, "outputs/report_inject_abl_base.md"),
    ("④ 8B 零样本", None, None, "未做"),
    ("⑤ API 上界", None, None, "待跑（DeepSeek / 百炼）"),
]

FIELD_ORDER = ["项目名称", "供应商", "单据编号", "日期",
               "序号", "名称", "规格型号", "单位", "数量", "单价", "金额"]

# ------------------------------------------------------------------ 模块加载
# evaluate.py 与 inject_knowledge.py 都按路径加载，保证全项目**只有一份**
# evaluate 模块实例（否则 align_rows 等是不同对象，口径可能悄悄分叉）。
import importlib.util  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


INJ = _load_module("_inj", ROOT / "src" / "inject_knowledge.py")
EV = INJ.load_evaluate()
INJ.EV = EV
INJ.NORM = EV.normalize
import review as REV  # noqa: E402


# ============================================================ 展示用的摊平
def flatten_pairs(gt: dict, pred: dict | None) -> dict:
    """Display raw values, but use the shared scorer for every normalized path."""
    pred_failed = pred is None
    gt, pred = EV._as_dict(gt), EV._as_dict(pred)
    gnorm, pnorm = EV.flatten(gt, pred)

    def raw_paths(obj):
        out = {}
        def visit(value, path):
            out[path] = value
            if isinstance(value, dict):
                for key, child in value.items():
                    key = key.replace("\\", "\\\\").replace(".", "\\.").replace("[", "\\[").replace("]", "\\]")
                    visit(child, f"{path}.{key}" if path else key)
            elif isinstance(value, list):
                for i, child in enumerate(value):
                    visit(child, f"{path}[{i}]")
        visit(obj, "")
        return out

    def entries(obj, norms):
        raw = raw_paths(obj)
        return OrderedDict((key, {"raw": raw.get(key, ""), "norm": value, "shape": False})
                           for key, value in norms.items())

    gout, pout = entries(gt, gnorm), entries(pred, pnorm)
    total = pred.get("合计")
    shape_wrong = total is not None and not isinstance(total, dict)
    if shape_wrong:
        pout["合计.金额"] = {"raw": total, "norm": "", "shape": True}
    rows = pred.get("明细")
    return {"gt": gout, "pred": pout,
            "slot_of_pi": {i: str(i) for i in range(len(rows))} if isinstance(rows, list) else {},
            "shape_wrong": shape_wrong, "parse_failed": pred_failed}


def remap_changed(changed: dict, slot_of_pi: dict) -> dict:
    """把注入记录里的 `明细[pi]` 改写成展示层的 `明细[n]`。

    `inject_knowledge.correct_record` 用的是**预测行下标 pi**，
    而 `evaluate.flatten` 用的是**配对后的槽位 n**（两者在行全部配对时相等，
    有漏配行时就会错位）。不转换的话高亮会标到错误的行上。
    """
    out = {}
    for path, val in changed.items():
        m = re.fullmatch(r"明细\[(\d+)\]\.(.+)", path)
        if m:
            pi, field = int(m.group(1)), m.group(2)
            slot = slot_of_pi.get(pi)
            out[f"明细[{slot}].{field}" if slot is not None else path] = val
        else:
            out[path] = val
    return out


# ================================================================ 知识注入
_TABLES: dict | None = None
_LEX_INFO: dict = {}


def load_tables() -> dict:
    """词表：只取 inject_knowledge 默认纳入的闭集字段。"""
    global _TABLES, _LEX_INFO
    if _TABLES is not None:
        return _TABLES
    lex = json.loads(LEXICON.read_text(encoding="utf-8"))
    want = [f for f in INJ.DEFAULT_FIELDS if f in lex["fields"]]
    _TABLES = {f: lex["fields"][f]["values"] for f in want}
    _LEX_INFO = {"version": lex.get("version"), "n_source": lex.get("n_source"),
                 "n_values": sum(len(v) for v in _TABLES.values()),
                 "fields": want}
    return _TABLES


def inject_pred(raw_text: str, gt: dict) -> dict:
    """对一条模型输出做词表纠正。返回解析后的新对象 / 改动表 / 统计。"""
    obj = EV.parse_json(raw_text or "")
    if obj is None:
        return None, {}, {"fixed": 0, "broke": 0, "changed": 0}
    nobj, inst = INJ.correct_record(obj, gt or {}, load_tables(), MIN_SIM, MIN_GAP, True)
    changed = {i["path"]: (i["before"], i["after"]) for i in inst if i["changed"]}
    stats = {
        "changed": len(changed),
        "fixed": sum(1 for i in inst if i["changed"] and not i["before_ok"] and i["after_ok"]),
        "broke": sum(1 for i in inst if i["changed"] and i["before_ok"] and not i["after_ok"]),
    }
    return nobj, changed, stats


# ================================================================ 视觉层
# 一套 GitHub 浅色系的设计令牌。面板、指标条、图例全部取自这里，
# 避免出现「深色面板 + 浅色外壳」那种两套皮拼接的观感。
PALETTE = {
    "ok": "#1a7f37", "bad": "#cf222e", "miss": "#9a6700",
    "extra": "#8250df", "fix": "#0969da",
    "ink": "#1f2328", "mut": "#656d76", "line": "#d8dee4", "card": "#ffffff",
}

# 状态 → 颜色，供图例与面板共用（图例上的色点和表格里的行必须同源，
# 否则改了颜色忘了改图例，看的人会按错的颜色读）。
STATE_COLOR = [
    ("ok", "正确"), ("bad", "读错"), ("miss", "漏抽"),
    ("extra", "多余"), ("fix", "词表纠正"),
]

SHELL_CSS = """
<style>
.mvw{font-family:ui-monospace,"Cascadia Mono",Consolas,monospace;font-size:12.5px;
     line-height:1.6;background:#fff;color:#1f2328;border:1px solid #d8dee4;
     border-radius:10px;box-shadow:0 1px 2px rgba(27,31,36,.06);overflow:hidden;}
.mvw .ph{display:flex;align-items:center;gap:8px;padding:9px 12px;background:#f6f8fa;
     border-bottom:1px solid #d8dee4;}
.mvw .tier{font-family:"Microsoft YaHei UI",system-ui,sans-serif;font-size:13px;
     font-weight:600;color:#1f2328;}
.mvw .f1{margin-left:auto;font-family:ui-monospace,Consolas,monospace;font-size:12px;
     font-weight:700;padding:1px 9px;border-radius:999px;border:1px solid;}
.mvw .meta{font-family:"Microsoft YaHei UI",system-ui,sans-serif;font-size:11px;
     color:#656d76;padding:7px 12px 0;}
.mvw .body{padding:4px 12px 10px;max-height:540px;overflow:auto;}
.mvw .body.tall{max-height:none;}
.mvw table{width:100%;border-collapse:collapse;}
.mvw td{padding:3px 6px;vertical-align:top;border-bottom:1px solid #eaeef2;}
.mvw td.k{color:#656d76;white-space:nowrap;font-size:11.5px;width:34%;
     font-family:"Microsoft YaHei UI",system-ui,sans-serif;}
.mvw td.v{border-left:3px solid transparent;word-break:break-word;}
/* 字段面板：标签最长 4 个汉字，给个固定窄列即可。
   默认的 34% 在半宽面板上＝约 350px，每个字段后面都是一大片空白。 */
.mvw table.fields td.k{width:96px;padding-right:14px;}
.mvw tr.grp td{background:#f6f8fa;color:#57606a;font-size:10.5px;letter-spacing:.4px;
     padding:5px 6px;border-bottom:1px solid #d8dee4;
     font-family:"Microsoft YaHei UI",system-ui,sans-serif;}
.mvw tr.ok td.v{color:#1a7f37;border-left-color:#1a7f37;}
.mvw tr.bad td.v{color:#cf222e;border-left-color:#cf222e;background:#fff5f5;}
.mvw tr.miss td.v{color:#9a6700;border-left-color:#d4a72c;background:#fffdf2;}
.mvw tr.extra td.v{color:#8250df;border-left-color:#8250df;background:#fbf7ff;}
.mvw tr.shape td.v{color:#9a6700;border-left-color:#d4a72c;background:#fffdf2;}
.mvw tr.fix td.v{color:#0969da;border-left-color:#0969da;background:#f2f8ff;}
.mvw .gtc{color:#1a7f37;font-weight:600;}
.mvw s{color:#8b949e;}
.mvw b{color:#0969da;}
.mvw .arrow{color:#8b949e;padding:0 5px;}
.mvw .none{color:#9a6700;font-style:italic;}
.mvw .tag{font-family:"Microsoft YaHei UI",system-ui,sans-serif;font-size:10px;
     padding:1px 6px;border-radius:999px;background:#eff2f5;color:#57606a;
     margin-left:6px;white-space:nowrap;}
.mvw .tag.fix{background:#ddf4ff;color:#0969da;}
.mvw .tag.warn{background:#fff8c5;color:#7d4e00;}
.mvw .empty{color:#8b949e;font-style:italic;padding:6px 0;}
.mvw .raw{color:#953800;white-space:pre-wrap;word-break:break-all;font-size:11px;
     background:#fff8f2;border:1px solid #ffd8b5;border-radius:6px;padding:8px;margin-top:6px;}
.mvw .note{font-family:"Microsoft YaHei UI",system-ui,sans-serif;font-size:11px;
     color:#656d76;padding:8px 12px;border-top:1px solid #eaeef2;background:#fafbfc;
     line-height:1.7;}
.mvw .note code{background:#eff2f5;padding:1px 5px;border-radius:3px;
     font-family:ui-monospace,Consolas,monospace;}
.mvw .sect{font-family:"Microsoft YaHei UI",system-ui,sans-serif;font-size:12px;
     font-weight:600;color:#1f2328;padding:11px 0 5px;margin-top:4px;
     border-top:1px solid #eaeef2;}
.mvw .sect:first-child{border-top:none;padding-top:2px;}
.sb{font-family:"Microsoft YaHei UI",system-ui,sans-serif;}
/* 条的长度封顶：宽屏下 1fr 会被拉成近千米，比例感反而失真 */
.sbr{display:grid;grid-template-columns:132px minmax(150px,360px) 60px 1fr;gap:10px;
     align-items:center;padding:6px 0;border-bottom:1px solid #f0f2f5;}
.sbr:last-child{border-bottom:none;}
@media (max-width:1280px){
  .sbr{grid-template-columns:108px 130px 54px 1fr;gap:8px;}
  .mvw table.fields td.k{width:76px;padding-right:10px;}
}
.sbn{font-size:12px;color:#1f2328;white-space:nowrap;overflow:hidden;
     text-overflow:ellipsis;}
.sbn i{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;
     vertical-align:1px;}
.sbb{background:#eef1f4;border-radius:999px;height:9px;overflow:hidden;}
.sbb i{display:block;height:100%;border-radius:999px;transition:width .5s ease;}
.sbv{font-family:ui-monospace,Consolas,monospace;font-size:12.5px;font-weight:700;
     text-align:right;}
.sbd{font-size:11px;color:#656d76;white-space:nowrap;overflow:hidden;
     text-overflow:ellipsis;}
.hero{background:linear-gradient(120deg,#0b2540,#123a63 55%,#1a5fa8);color:#eaf2fb;
     border-radius:12px;padding:16px 20px;margin-bottom:12px;
     box-shadow:0 2px 10px rgba(11,37,64,.18);}
.hero .ht{font-family:"Microsoft YaHei UI",system-ui,sans-serif;font-size:19px;
     font-weight:700;letter-spacing:.2px;}
.hero .hs{font-family:"Microsoft YaHei UI",system-ui,sans-serif;font-size:12.5px;
     opacity:.86;margin-top:6px;line-height:1.75;}
.hero .hs code{background:rgba(255,255,255,.16);padding:1px 6px;border-radius:4px;
     font-family:ui-monospace,Consolas,monospace;}
.hero .chips{margin-top:11px;display:flex;flex-wrap:wrap;gap:6px;}
.hero .chip{font-family:"Microsoft YaHei UI",system-ui,sans-serif;font-size:11px;
     padding:2px 10px;border-radius:999px;background:rgba(255,255,255,.13);
     border:1px solid rgba(255,255,255,.22);}
.hero .chip i{display:inline-block;width:7px;height:7px;border-radius:50%;
     margin-right:5px;vertical-align:1px;}
.mv-side{background:#fff;border:1px solid #d8dee4;border-radius:10px;padding:12px;
     box-shadow:0 1px 2px rgba(27,31,36,.06);}
.sidehd{font-family:"Microsoft YaHei UI",system-ui,sans-serif;font-size:13.5px;
     font-weight:600;color:#1f2328;margin:0 0 6px;}
.sidehint{font-family:"Microsoft YaHei UI",system-ui,sans-serif;font-size:11px;
     color:#656d76;line-height:1.7;padding-top:4px;}
</style>
"""


def _f1_color(f1: float) -> str:
    """按 F1 分档取色 —— 让「一眼看出哪档行」不依赖读数字。"""
    if f1 >= 0.95:
        return PALETTE["ok"]
    if f1 >= 0.85:
        return PALETTE["fix"]
    if f1 >= 0.75:
        return PALETTE["miss"]
    return PALETTE["bad"]


def _chips() -> str:
    dots = "".join(
        f'<span class="chip"><i style="background:{PALETTE[st]}"></i>{lab}</span>'
        for st, lab in STATE_COLOR)
    return f'<div class="chips">{dots}</div>'


HEADER_HTML = f"""{SHELL_CSS}
<div class="hero">
  <div class="ht">MiniVLM · 工程材料清单字段抽取</div>
  <div class="hs">同一张单据 · 同一个 2B 模型 · 同一份 prompt —— 唯一变量是 <code>LoRA</code> 适配器
    （零样本靠 <code>disable_adapter()</code> 临时摘掉）。
    下方四个面板与真值逐字段对照，右侧条为 v2 字段级 F1；全局对比表保留历史旧口径，尚未全部重算。</div>
  {_chips()}
</div>
"""


def _esc(v: object) -> str:
    return html.escape(str(v))


def _stray_keys(pred_obj: dict | None) -> list[str]:
    from scoring import extra_fields
    return extra_fields(pred_obj)


def _sort_key(path: str):
    if path.startswith("表头."):
        g, idx, f = 0, 0, path.split(".", 1)[1]
    elif path.startswith("明细[") and "." in path:
        head, f = path.split(".", 1)
        inner = head[len("明细["):-1]
        if inner.startswith("unmatched"):
            g, idx = 2, int(re.sub(r"\D", "", inner) or 0)
        else:
            g, idx = 1, int(re.sub(r"\D", "", inner) or 0)
    else:
        g, idx, f = 3, 0, path.rsplit(".", 1)[-1]
    fi = FIELD_ORDER.index(f) if f in FIELD_ORDER else 99
    return (g, idx, fi)


def _group_title(path: str) -> str:
    if path == "表头" or path.startswith("表头."):
        return "表头"
    if path == "合计" or path.startswith("合计."):
        return "合计"
    if not re.match(r"明细\[\d+\]", path):
        return "单据信息与额外字段"
    inner = path.split(".", 1)[0][len("明细["):-1]
    if inner.startswith("unmatched_g"):
        return "明细 · 只有真值（模型漏整行）"
    if inner.startswith("unmatched_p"):
        return "明细 · 只有预测（模型多整行）"
    return f"明细 第 {int(inner) + 1} 行"


def render_panel(title: str, sub: str, gt: dict, pred_obj: dict | None,
                 raw_text: str = "", changed: dict | None = None,
                 f1: float | None = None) -> str:
    """渲染一个档位面板：与真值逐字段对照，错红 / 漏黄 / 多余紫 / 纠正蓝。

    `f1` 只影响右上角徽章的颜色与数字，不参与任何判定 —— 判定仍走
    `score_pred`，保证屏幕上的红绿与报告里的 F1 同源。
    """
    changed = changed or {}
    flat = flatten_pairs(gt, pred_obj)
    gp, pp = flat["gt"], flat["pred"]

    chip = ""
    if f1 is not None:
        c = _f1_color(f1)
        chip = (f'<span class="f1" style="color:{c};'
                f'background:{c}14;border-color:{c}44">F1 {f1:.3f}</span>')
    head = (f'<div class="ph"><span class="tier">{_esc(title)}</span>{chip}</div>'
            f'<div class="meta">{sub}</div>')

    if flat["parse_failed"]:
        body = ('<div class="empty">JSON 解析失败（模型输出不是合法 JSON，'
                '通常是被 max_new_tokens 截断）。原始输出：</div>'
                f'<div class="raw">{_esc((raw_text or "")[:600])}</div>')
        return f"<div class='mvw'>{head}<div class='body'>{body}</div></div>"

    paths = sorted(set(gp) | set(pp), key=_sort_key)
    rows: list[str] = []
    cur_group = None
    for path in paths:
        gtitle = _group_title(path)
        if gtitle != cur_group:
            rows.append(f'<tr class="grp"><td colspan="2">{_esc(gtitle)}</td></tr>')
            cur_group = gtitle

        label = path.rsplit(".", 1)[-1]
        ge, pe = gp.get(path), pp.get(path)
        st, val = "bad", ""

        if path in changed:
            before, after = changed[path]
            st = "fix"
            val = (f'<s>{_esc(before)}</s><span class="arrow">→</span>'
                   f'<b>{_esc(after)}</b><span class="tag fix">词表纠正</span>')
            if ge and ge["raw"] != after:
                val += (f'<span class="arrow">·</span>真值 '
                        f'<span class="gtc">{_esc(ge["raw"])}</span>')
        elif ge and pe:
            if ge["norm"] == pe["norm"]:
                st, val = "ok", _esc(ge["raw"])
            else:
                st = "bad"
                val = (f'<s>{_esc(pe["raw"])}</s><span class="arrow">→</span>'
                       f'<span class="gtc">{_esc(ge["raw"])}</span>')
        elif ge and not pe:
            st = "miss"
            val = ('<span class="none">未输出</span><span class="arrow">→</span>'
                   f'<span class="gtc">{_esc(ge["raw"])}</span>')
        elif pe and not ge:
            note = ('<span class="tag warn">形状不符：应为 {"金额": …}</span>'
                    if pe.get("shape") else '<span class="tag">多余</span>')
            st, val = ("shape" if pe.get("shape") else "extra"), _esc(pe["raw"]) + note

        # 形状不符的合计：显示了但要明确它在这个口径下不算抽到
        if path == "合计.金额" and flat["shape_wrong"] and path not in changed:
            st = "shape"
            val = (_esc(pp[path]["raw"])
                   + '<span class="tag warn">形状不符：应为 {"金额": …}，严格口径计漏抽</span>')

        rows.append(f'<tr class="{st}"><td class="k">{_esc(label)}</td>'
                    f'<td class="v">{val}</td></tr>')

    if not rows:
        rows.append('<tr><td colspan="2" class="empty">（无可比对字段）</td></tr>')

    foot = ""
    strays = _stray_keys(pred_obj)
    if strays:
        shown = "、".join(strays[:6]) + ("…" if len(strays) > 6 else "")
        foot = ('<div class="note">Schema 之外的键已计入多抽：'
                f'<code>{_esc(shown)}</code> —— 官方口径不计入 F1，故此处也不标色。</div>')

    return (f"<div class='mvw'>{head}"
            f"<div class='body'><table class='fields'>{''.join(rows)}</table></div>"
            f"{foot}</div>")


def _prf(gt: dict, pred_obj: dict | None) -> tuple[float, float, float, dict]:
    r = EV.score_pred(gt, pred_obj, "schema", tolerant=False)
    p, rr, f1 = EV.prf(r["tp"], r["fp"], r["fn"])
    return p, rr, f1, r


def render_metrics(items: list[tuple[str, dict, dict | None, str]]) -> str:
    """本样本四档对比（F1 条）+ 全局指标表。

    items: [(名称, gt, pred_obj, 备注)]

    F1 条、徽章颜色与表格数字来自同一次 `score_pred`，不会出现
    「条是绿的、数字是红的」这种自相矛盾。
    """
    sb: list[str] = []
    base_f1: float | None = None
    for name, gt, obj, note in items:
        if not gt:
            sb.append(f'<div class="sbr"><span class="sbn">{_esc(name)}</span>'
                      f'<span class="sbb"></span><span class="sbv">—</span>'
                      f'<span class="sbd">无真值，无法打分</span></div>')
            continue
        p, r, f1, rr = _prf(gt, obj)
        is_base = base_f1 is None
        if is_base:
            base_f1 = f1
        c = _f1_color(f1)
        if is_base:
            delta = '<span style="color:#656d76">基线</span>'
        else:
            dv = f1 - (base_f1 or 0.0)
            dc = (PALETTE["ok"] if dv > 1e-9
                  else PALETTE["bad"] if dv < -1e-9 else "#656d76")
            delta = f'<span style="color:{dc};font-weight:600">Δ{dv:+.3f}</span>'
        detail = (f'P {p:.2f} · R {r:.2f} · TP{rr["tp"]} FP{rr["fp"]} FN{rr["fn"]}'
                  + (f' · {note}' if note else ''))
        sb.append(
            f'<div class="sbr">'
            f'<span class="sbn"><i style="background:{c}"></i>{_esc(name)}</span>'
            f'<span class="sbb"><i style="width:{max(0.0, min(1.0, f1)) * 100:.1f}%;'
            f'background:{c}"></i></span>'
            f'<span class="sbv" style="color:{c}">{f1:.3f}</span>'
            f'<span class="sbd">{delta} · {_esc(detail)}</span>'
            f'</div>')

    glo = []
    for name, same, cross, src in GLOBAL_TIERS:
        if same is None:
            f_same = f_cross = '<span style="color:#8b949e">—</span>'
            dot = "#c9d1d9"
        else:
            f_same = f'<b style="color:{_f1_color(same)}">{same:.4f}</b>'
            f_cross = ('<span style="color:#8b949e">—</span>' if cross is None
                       else f'{cross:.4f}')
            dot = _f1_color(same)
        glo.append(
            f'<tr><td class="k"><i style="display:inline-block;width:7px;height:7px;'
            f'border-radius:50%;background:{dot};margin-right:6px;vertical-align:1px">'
            f'</i>{_esc(name)}</td><td class="v">{f_same}</td>'
            f'<td class="v">{f_cross}</td><td class="k">{_esc(src)}</td></tr>')

    return (
        f"<div class='mvw'>"
        f'<div class="ph"><span class="tier">结果总览</span>'
        f'<span class="f1" style="color:#656d76;background:#f6f8fa;'
        f'border-color:#d8dee4">严格口径</span></div>'
        f'<div class="body tall">'
        f'<div class="sect">本样本 · 字段级 P / R / F1</div>'
        f'<div class="sb">{"".join(sb)}</div>'
        f'<div class="sect">全局 · 同域 test_ablation 102 条 / '
        f'跨域 WildReceipt 前 100 条</div>'
        f'<table><tr class="grp"><td>档位</td><td>同域 F1</td><td>跨域 F1</td>'
        f'<td>数字来源</td></tr>{"".join(glo)}</table>'
        f'</div>'
        f'<div class="note"><b>② → ③</b> 的差距 = 微调收益；'
        f'<b>② → ②+注入</b> = 词表知识注入收益；'
        f'③ 仅剩 1 处可纠正，说明 SFT 已把词表内化进权重。'
        f'本样本仅供参考，稳定结论看全局表。</div>'
        f"</div>")


# ================================================================ 推理
_MODEL = None
_PROC = None
_PROMPT: str | None = None
_LOAD_ERR: str | None = None


def load_prompt() -> str:
    global _PROMPT
    if _PROMPT is None:
        rec = json.loads(TEST_FILE.open(encoding="utf-8").readline())
        _PROMPT = rec["messages"][0]["content"][-1]["text"]
    return _PROMPT


def load_model(sft_dir: Path = SFT_DIR):
    """常驻一个 4bit 基座 + SFT LoRA。零样本靠 disable_adapter() 切换。"""
    global _MODEL, _PROC, _LOAD_ERR
    if _MODEL is not None or _LOAD_ERR:
        return
    try:
        import torch  # noqa: F401
        from unsloth import FastVisionModel
        from peft import PeftModel

        t0 = time.time()
        print(f"[demo] 加载 {MODEL_NAME} ...", flush=True)
        model, processor = FastVisionModel.from_pretrained(
            MODEL_NAME, load_in_4bit=True, max_seq_length=2048)
        if not sft_dir.exists():
            raise FileNotFoundError(f"找不到 SFT 适配器 {sft_dir}")
        model = PeftModel.from_pretrained(model, str(sft_dir))
        FastVisionModel.for_inference(model)
        _MODEL, _PROC = model, processor
        print(f"[demo] 就绪（{time.time() - t0:.0f}s）。"
              f"零样本走 disable_adapter()。", flush=True)
    except Exception as e:                      # noqa: BLE001
        _LOAD_ERR = f"{type(e).__name__}: {e}"
        print(f"[demo] 模型加载失败：{_LOAD_ERR}", flush=True)


def generate(image_path: str, size: int, use_sft: bool) -> tuple[str, float]:
    """单图单档推理。返回 (原始输出, 秒)。"""
    import torch
    from PIL import Image

    img = Image.open(image_path).convert("RGB").resize((size, size))
    msg = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": load_prompt()},
    ]}]
    text = _PROC.apply_chat_template(msg, add_generation_prompt=True)
    inputs = _PROC(images=[img], text=[text], return_tensors="pt",
                   padding=True).to("cuda")

    t0 = time.time()
    with torch.no_grad():
        if use_sft:
            out = _MODEL.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        else:
            with _MODEL.disable_adapter():
                out = _MODEL.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    dt = time.time() - t0

    n_in = inputs["input_ids"].shape[1]
    raw = _PROC.decode(out[0][n_in:], skip_special_tokens=True)
    return raw, dt


# ================================================================ 数据集
SAMPLES: list[dict] = []


def load_samples() -> list[dict]:
    global SAMPLES
    if SAMPLES:
        return SAMPLES
    if not TEST_FILE.exists():
        return SAMPLES
    for i, line in enumerate(TEST_FILE.open(encoding="utf-8")):
        if not line.strip():
            continue
        r = json.loads(line)
        meta = r.get("meta") or {}
        SAMPLES.append({
            "idx": i,
            "path": r["image"],
            "gt": r["gt"],
            "level": meta.get("level", "?"),
            "stem": meta.get("stem", Path(r["image"]).stem),
        })
    return SAMPLES


def label_of(s: dict) -> str:
    return f"{s['stem']} · {s['level']}"


def find_gt_by_path(image_path: str) -> dict:
    name = Path(image_path).name
    for s in load_samples():
        if Path(s["path"]).name == name:
            return s["gt"]
    return {}


# ================================================================ 主流程
def run(image_path: str | None, size_label: str):
    load_model()
    if _LOAD_ERR:
        err = ("<div class='mvw'><div class='ph'><span class='tier'>模型未就绪</span>"
               "</div><div class='body'>"
               f"<div class='raw'>{_esc(_LOAD_ERR)}</div></div></div>")
        return err, "", "", "", err, "", "", "", {}
    if not image_path:
        msg = ("<div class='mvw'><div class='ph'><span class='tier'>等待输入</span></div>"
               "<div class='body'><div class='empty'>请在左侧上传一张单据，"
               "或从测试集里选一张（带真值）→ 点「开始抽取」。</div></div></div>")
        return msg, "", "", "", msg, "", "", "", {}

    size = int(size_label)
    gt = find_gt_by_path(image_path)
    gt_note = "有真值" if gt else "自由上传，无真值"

    raw_z, t_z = generate(image_path, size, use_sft=False)
    raw_s, t_s = generate(image_path, size, use_sft=True)

    if raw_z.strip() == raw_s.strip():
        print("[demo][警告] 两档输出完全相同 —— adapter 切换可能失效，请检查 "
              "disable_adapter() 是否生效。", flush=True)

    obj_z = EV.parse_json(raw_z)
    obj_s = EV.parse_json(raw_s)
    obj_zi, ch_zi, st_zi = inject_pred(raw_z, gt)
    obj_si, ch_si, st_si = inject_pred(raw_s, gt)

    # 注入记录里的 明细[pi] 要换算成展示层的 明细[n]
    ch_zi = remap_changed(ch_zi, flatten_pairs(gt, obj_z)["slot_of_pi"])
    ch_si = remap_changed(ch_si, flatten_pairs(gt, obj_s)["slot_of_pi"])

    def badge(obj, raw) -> str:
        if gt:
            p, r, f1, _ = _prf(gt, obj)
            return f"P {p:.2f} R {r:.2f} <b>F1 {f1:.2f}</b>"
        return "无真值" if obj is None else "不评分"

    def f1_of(obj) -> float | None:
        return _prf(gt, obj)[2] if gt else None

    p1 = render_panel(
        "② 2B 零样本", f"{t_z:.1f}s · {size}px · {badge(obj_z, raw_z)}",
        gt, obj_z, raw_z, f1=f1_of(obj_z))
    p2 = render_panel(
        "② + 知识注入",
        f"改动 <b>{st_zi['changed']}</b> 处（纠正 {st_zi['fixed']} / "
        f"改坏 {st_zi['broke']}）· 纯 CPU 后处理 · {badge(obj_zi, '')}",
        gt, obj_zi, "", ch_zi, f1=f1_of(obj_zi))
    p3 = render_panel(
        "③ 2B SFT", f"{t_s:.1f}s · {size}px · {badge(obj_s, raw_s)}",
        gt, obj_s, raw_s, f1=f1_of(obj_s))
    p4 = render_panel(
        "③ SFT + 知识注入",
        f"改动 <b>{st_si['changed']}</b> 处 · 词表已被微调内化，SFT 后几乎无可纠正"
        f" · {badge(obj_si, '')}", gt, obj_si, "", ch_si, f1=f1_of(obj_si))
    p5 = render_panel("GT 真值", f"单据真值 · {gt_note}", gt, gt, f1=1.0 if gt else None)

    metrics = render_metrics([
        ("② 2B 零样本", gt, obj_z, f"{t_z:.1f}s"),
        ("② + 知识注入", gt, obj_zi, f"改动 {st_zi['changed']} 处"),
        ("③ 2B SFT", gt, obj_s, f"{t_s:.1f}s"),
        ("③ + 知识注入", gt, obj_si, f"改动 {st_si['changed']} 处"),
    ])
    editable = json.dumps(obj_s, ensure_ascii=False, indent=2) if obj_s is not None else raw_s
    model_hash = hashlib.sha256((SFT_DIR / "adapter_model.safetensors").read_bytes()).hexdigest()[:12]
    review_state = {
        "sample_id": Path(image_path).stem,
        "image_path": str(image_path),
        "raw_output": raw_s,
        "model_version": f"{SFT_DIR.relative_to(ROOT)}@{model_hash}",
        "prompt_sha256": hashlib.sha256(load_prompt().encode("utf-8")).hexdigest(),
    }
    return p1, p2, p3, p4, metrics, p5, editable, REV.validation_html(editable), review_state


def save_review_ui(state, edited_text, error_types, notes):
    import gradio as gr
    try:
        saved = REV.save_review(state or {}, edited_text or "", error_types or [], notes or "")
        choices = REV.list_reviews()
        status = (f"<div style='color:#067647'><b>已保存复核记录</b><br>"
                  f"修改字段 {len(saved['record']['modified_fields'])} 个；记录不会覆盖原始模型输出。"
                  f"</div>")
        return status, [saved["record_path"], saved["csv_path"]], gr.Dropdown(choices=choices, value=saved["record_path"])
    except Exception as exc:  # noqa: BLE001
        return f"<div style='color:#b42318'><b>保存失败：</b>{html.escape(str(exc))}</div>", [], gr.Dropdown(choices=REV.list_reviews())


def refresh_reviews_ui():
    import gradio as gr
    return gr.Dropdown(choices=REV.list_reviews(), value=None)


def load_review_ui(path):
    if not path:
        return "", "", {}, []
    try:
        record = REV.load_review(path)
        text = json.dumps(record["reviewed_document"], ensure_ascii=False, indent=2)
        state = {"sample_id": record.get("sample_id"), "image_path": record.get("image_path"),
                 "raw_output": record.get("original_raw_output"), "model_version": record.get("model_version"),
                 "prompt_sha256": record.get("prompt_sha256")}
        csv_path = str(REV.REVIEW_ROOT / "exports" / (record["review_id"] + ".csv"))
        files = [str(path)] + ([csv_path] if Path(csv_path).exists() else [])
        return text, REV.validation_html(text), state, files
    except Exception as exc:  # noqa: BLE001
        return "", f"<div style='color:#b42318'>载入失败：{html.escape(str(exc))}</div>", {}, []


def build_ui():
    import gradio as gr

    samples = load_samples()
    label2path = {label_of(s): s["path"] for s in samples}

    with gr.Blocks(title="MiniVLM · 工程材料清单抽取") as demo:
        gr.HTML(HEADER_HTML)

        with gr.Row(equal_height=False):
            # ---- 左：输入 + 控制，收成一张卡 ----
            with gr.Column(scale=2, min_width=300, elem_classes="mv-side"):
                gr.HTML('<div class="sidehd">① 选一张单据</div>')
                img_in = gr.Image(type="filepath", label="单据图片", height=320,
                                  sources=["upload", "clipboard"])
                dd = gr.Dropdown(choices=list(label2path),
                                 label="或从测试集选一张（带真值）",
                                 filterable=True, value=None)
                rnd = gr.Button("随机换一张", size="sm")
                size_rd = gr.Radio(choices=["384", "768"], value="384",
                                   label="推理分辨率",
                                   info="384 = 训练一致 ｜ 768 = 消融最优")
                btn = gr.Button("开始抽取", variant="primary", size="lg")
                gr.HTML('<div class="sidehint">模型在启动时已加载（终端打印进度）。'
                        '每档约 10–20s，768px 更慢；知识注入是纯 CPU 后处理，'
                        '瞬时完成。</div>')

            # ---- 右：总览在顶（先看结论），四个面板 2×2，真值单独一行 ----
            with gr.Column(scale=5):
                met = gr.HTML()
                with gr.Row(equal_height=False):
                    p1 = gr.HTML()
                    p2 = gr.HTML()
                with gr.Row(equal_height=False):
                    p3 = gr.HTML()
                    p4 = gr.HTML()
                with gr.Row():
                    p5 = gr.HTML()

        with gr.Accordion("人工复核与导出", open=False):
            gr.Markdown("编辑模型原始提取结果；金额关系只提示，不会自动改值。保存后可重新载入，并导出带审计信息的 JSON 与表格 CSV。")
            review_state = gr.State({})
            editor = gr.Code(label="可编辑 JSON", language="json", lines=20)
            review_status = gr.HTML()
            with gr.Row():
                error_types = gr.CheckboxGroup(choices=REV.ERROR_TYPES, label="错误类型（可多选）")
                notes = gr.Textbox(label="复核备注", lines=3)
            with gr.Row():
                validate_btn = gr.Button("重新校验")
                save_btn = gr.Button("保存复核并导出", variant="primary")
            review_files = gr.File(label="JSON / CSV 导出", file_count="multiple")
            with gr.Row():
                saved_dd = gr.Dropdown(choices=REV.list_reviews(), label="已保存复核记录", filterable=True)
                refresh_btn = gr.Button("刷新记录")
                load_btn = gr.Button("重新载入")

        dd.change(lambda v: label2path.get(v), inputs=dd, outputs=img_in)
        rnd.click(lambda: random.choice(list(label2path)), outputs=dd)
        btn.click(run, inputs=[img_in, size_rd],
                  outputs=[p1, p2, p3, p4, met, p5, editor, review_status, review_state])
        validate_btn.click(REV.validation_html, inputs=editor, outputs=review_status)
        save_btn.click(save_review_ui, inputs=[review_state, editor, error_types, notes],
                       outputs=[review_status, review_files, saved_dd])
        refresh_btn.click(refresh_reviews_ui, outputs=saved_dd)
        load_btn.click(load_review_ui, inputs=saved_dd,
                       outputs=[editor, review_status, review_state, review_files])
    return demo


def make_theme():
    """主题单独抽出，并带兜底 —— 主题名随 Gradio 版本变化，
    写死在 launch 里会让整个界面在版本升级后直接起不来。"""
    import gradio as gr

    try:
        return gr.themes.Soft(
            primary_hue="blue", neutral_hue="slate",
            font=["Microsoft YaHei UI", "system-ui", "sans-serif"],
            font_mono=["Cascadia Mono", "Consolas", "monospace"],
        )
    except Exception:                       # noqa: BLE001
        return gr.themes.Base()


LAUNCH_CSS = """
.gradio-container{max-width:2100px !important;padding:14px 18px 24px !important;}
body,.gradio-container,.main,.app,.gradio-container .prose{background:#f3f5f8 !important;}
footer{display:none !important;}
.gradio-container .prose{color:#1f2328;}
"""

# 强制浅色：卡片是白的，若 Gradio 跟随系统切成深色，外壳和卡片就是两套皮。
# 这里不依赖 Gradio 内部的主题 mode 存储（各版本键名不同），只做一件事：
# 把 <html> 上的 dark 标记去掉，并用 MutationObserver 盯着别被加回来。
LAUNCH_HEAD = """
<style>html,html.dark{color-scheme:light !important;}</style>
<script>
(function () {
  function strip() {
    var el = document.documentElement;
    if (el.classList.contains('dark')) el.classList.remove('dark');
    if (el.getAttribute('data-theme') === 'dark') el.setAttribute('data-theme', 'light');
  }
  strip();
  new MutationObserver(strip).observe(document.documentElement,
      {attributes: true, attributeFilter: ['class', 'data-theme']});
  document.addEventListener('DOMContentLoaded', strip);
})();
</script>
"""


# ================================================================ 自检
def _check() -> int:
    ok = fail = 0

    def ck(name: str, cond: bool, extra: str = ""):
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  [PASS] {name}")
        else:
            fail += 1
            print(f"  [FAIL] {name}  {extra}")

    print("=" * 70)
    print("Demo 自检（纯 CPU，不加载模型）")
    print("=" * 70)

    # ---- 1. flatten_pairs 与 evaluate.flatten 等价 ----
    print("\n[1] 展示层摊平 vs 官方 flatten（全部历史预测文件）")
    files = sorted(OUTPUTS.glob("eval_*_preds.jsonl"))
    ck("找到历史预测文件", bool(files), f"n={len(files)}")
    n_rec = mismatch = 0
    for f in files:
        for line in f.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("_meta"):
                continue
            g = r.get("gt")
            if isinstance(g, str):
                g = json.loads(g)
            obj = EV.parse_json(r.get("pred") or "")
            # 官方口径：严格
            g_off, p_off = EV.flatten(g or {}, obj if obj is not None else {}, False)
            mine = flatten_pairs(g or {}, obj)
            g_mine = {k: v["norm"] for k, v in mine["gt"].items() if v["norm"]}
            p_mine = {k: v["norm"] for k, v in mine["pred"].items() if v["norm"]}
            n_rec += 1
            if g_mine != g_off or p_mine != p_off:
                mismatch += 1
                if mismatch == 1:
                    only_g = set(g_off) ^ set(g_mine)
                    only_p = set(p_off) ^ set(p_mine)
                    print(f"    首个不一致 {f.name} gt差={sorted(only_g)[:3]} "
                          f"pred差={sorted(only_p)[:3]}")
    ck(f"{n_rec} 条预测逐路径全等", mismatch == 0, f"不一致 {mismatch} 条")

    # ---- 2. 渲染不崩 + 命中预期状态 ----
    print("\n[2] 渲染层")
    gt = {"表头": {"项目名称": "沣西市政道路改造工程", "供应商": "汉中盛世建材有限公司",
                   "单据编号": "CL-20260626-029", "日期": "2026-06-26"},
          "明细": [{"序号": "1", "名称": "中砂", "规格型号": "中粗", "单位": "m3",
                    "数量": "200", "单价": "120.27", "金额": "24054.00"}],
          "合计": {"金额": "24054.00"}}
    good = {"表头": dict(gt["表头"]), "明细": [dict(gt["明细"][0])],
            "合计": {"金额": "24054.00"}}
    mixed = {"表头": {"项目名称": "洋市市政道路改造工程",   # 读错
                      "供应商": "汉中盛世建材有限公司",       # 正确
                      "单据编号": "CL-20260626-029", "日期": "2026-06-26"},
             "明细": [{"序号": "1", "名称": "中砂", "规格型号": "中粗", "单位": "m3",
                       "数量": "200", "单价": "120.27", "金额": "24054.00"}],
             "合计": "24054.00"}                               # 形状不符
    h_good = render_panel("t", "s", gt, good)
    ck("全对：无红/黄/紫", 'class="bad"' not in h_good and 'class="miss"' not in h_good
       and 'class="extra"' not in h_good)
    h_mix = render_panel("t", "s", gt, mixed)
    ck("读错→红", 'class="bad"' in h_mix)
    ck("合计标量→形状不符", 'class="shape"' in h_mix and "形状不符" in h_mix)
    h_miss = render_panel("t", "s", gt, {"表头": {}, "明细": [], "合计": {}})
    ck("漏抽→黄", 'class="miss"' in h_miss)

    # 多出目标值与 Schema 外字段均计入 FP。
    gt_nu = {"表头": {}, "明细": [{"序号": "1", "名称": "中砂"}],
             "合计": {"金额": "24054.00"}}
    pr_nu = {"表头": {}, "明细": [{"序号": "1", "名称": "中砂", "单位": "m3"}],
             "合计": {"金额": "24054.00"}}
    ck("多余（GT 空、模型填）→紫", 'class="extra"' in render_panel("t", "s", gt_nu, pr_nu))
    h_stray = render_panel("t", "s", gt, {"表头": {"项目名称": "x"},
                                          "明细": [{"名称": "中砂", "备注": "手写批注"}],
                                          "合计": {"金额": "1"}})
    ck("schema 之外的键被显式提示", "已计入多抽" in h_stray and "备注" in h_stray)

    h_null = render_panel("t", "s", gt, None, "```json\n{truncated")
    ck("解析失败→不抛异常且提示截断", "解析失败" in h_null)
    ck("HTML 转义生效（无裸标签注入）",
       "<script>" not in render_panel("t", "s", gt, {"表头": {"项目名称": "<script>x"}}))

    # 视觉层的结构断言：改样式时别把 F1 条 / Δ / 图例色点改没了。
    # 注意只断言「结构存在」，不断言具体色值 —— 色值会随设计调整。
    h_met = render_metrics([("② t", gt, good, ""),
                            ("③ t", gt, {"表头": {}, "明细": [], "合计": {}}, "改动 1 处")])
    ck("指标条渲染（F1 条 + Δ + 基线）",
       'class="sbb"' in h_met and "Δ" in h_met and "基线" in h_met)
    n_sbr = h_met.count('class="sbr"')
    ck("指标条两行对应两档", n_sbr == 2, f"n={n_sbr}")
    ck("图例色点与状态表同源",
       all(f'background:{PALETTE[st]}' in HEADER_HTML for st, _ in STATE_COLOR))
    ck("面板头部含 F1 徽章", 'class="f1"' in render_panel("t", "s", gt, good, f1=0.9))

    # ---- 3. 注入结果与 inject_knowledge 一致 ----
    print("\n[3] 知识注入复用一致性")
    ck("词表已加载", bool(load_tables()), str(_LEX_INFO))

    # 用零样本档做对照 —— 它是唯一有大量可纠正错误的档位（SFT 只改 1 处）
    src = OUTPUTS / "eval_zero_abl_preds.jsonl"
    if not src.exists():
        src = files[0] if files else None
    if src:
        n_cmp = n_changed = 0
        same_obj = same_ch = True
        for line in src.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("_meta"):
                continue
            g = r.get("gt")
            if isinstance(g, str):
                g = json.loads(g)
            raw = r.get("pred") or ""
            obj = EV.parse_json(raw)
            if obj is None:          # 解析失败：inject_knowledge 原实现也是跳过
                mine, ch, st = inject_pred(raw, g or {})
                if mine is not None or ch:
                    same_obj = False
                n_cmp += 1
                continue
            mine, ch, st = inject_pred(raw, g or {})
            ref, inst = INJ.correct_record(obj, g or {}, load_tables(),
                                           MIN_SIM, MIN_GAP, True)
            ref_ch = {i["path"]: (i["before"], i["after"])
                      for i in inst if i["changed"]}
            n_cmp += 1
            n_changed += st["changed"]
            if json.dumps(mine, sort_keys=True) != json.dumps(ref, sort_keys=True):
                same_obj = False
            if ch != ref_ch:
                same_ch = False
        ck(f"{n_cmp} 条：纠正后的对象逐条一致", same_obj)
        ck(f"{n_cmp} 条：改动集合逐条一致", same_ch)
        ck("零样本档确实有可纠正的改动（非平凡样例）", n_changed > 0,
           f"共 {n_changed} 处")
        print(f"    样例来源 {src.name}：合计 {n_changed} 处改动")

    # ---- 4. 全局指标常量对得上 ----
    print("\n[4] 全局指标常量 vs 落盘结果")
    for name, same, cross, source in GLOBAL_TIERS:
        if not source.endswith(".json"):
            continue
        p = ROOT / source
        if not p.exists():
            ck(f"{name} 数字来源存在", False, source)
            continue
        j = json.loads(p.read_text(encoding="utf-8"))
        if same is not None:
            ck(f"{name} 同域 F1 == {same}", abs(j["f1"] - same) < 5e-5,
               f"落盘 {j['f1']} vs 常量 {same}")

    # ---- 5. Gradio 界面可构建（大版本 API 漂移是这类 Demo 最典型的翻车点）----
    print("\n[5] Gradio 界面")
    try:
        import inspect

        import gradio as gr
        print(f"    gradio {gr.__version__}")
        demo = build_ui()
        cfg = demo.get_config_file()
        ck("Blocks 构建成功", True)
        ck("组件数 > 8", len(cfg.get("components", [])) > 8,
           f"n={len(cfg.get('components', []))}")
        ck("事件链已注册（选图 / 随机 / 抽取）", len(cfg.get("dependencies", [])) >= 3,
           f"n={len(cfg.get('dependencies', []))}")
        lp = inspect.signature(gr.Blocks.launch).parameters
        ck("launch 接受 theme/css", "theme" in lp and "css" in lp)
        ck("launch 接受 head（强制浅色用）", "head" in lp)
        th = make_theme()
        ck("主题可实例化（Soft → Base 兜底）", th is not None, type(th).__name__)
        ck("头部图例色点齐全", 'class="chip"' in HEADER_HTML)
        ck("浅色覆盖脚本会摘掉 dark 标记", "classList.remove('dark')" in LAUNCH_HEAD)
        # Gradio 6 起 theme/css 必须传给 launch：传给 Blocks 会静默失效（不报错、不生效）
        bp = inspect.signature(gr.Blocks.__init__).parameters
        ck("已知：Blocks 不再显式接收 theme/css（故只传 launch）",
           "theme" not in bp and "css" not in bp)
    except ImportError:
        print("    （未安装 gradio，跳过 —— 界面部分无法离线验证）")

    print("\n" + "=" * 70)
    print(f"自检结果：{ok} 通过 / {fail} 失败")
    print("=" * 70)
    return 1 if fail else 0


def _render_sample(idx: int, out_path: Path) -> int:
    """用已落盘的预测渲一张静态预览（不需要 GPU，用于快速核对样式）。

    布局刻意与 Gradio 界面一致（总览在顶、2×2 面板、真值单列），
    这样改样式时能先用它快速迭代，不必反复起服务加载模型。
    """
    z = [json.loads(l) for l in (OUTPUTS / "eval_zero_abl_preds.jsonl")
         .open(encoding="utf-8") if l.strip()]
    s = [json.loads(l) for l in (OUTPUTS / "eval_abl_base_preds.jsonl")
         .open(encoding="utf-8") if l.strip()]
    if idx >= len(z) or idx >= len(s):
        print(f"[FATAL] idx 超范围（0..{min(len(z), len(s)) - 1}）")
        return 1
    rz, rs = z[idx], s[idx]
    gt = rz["gt"] if isinstance(rz["gt"], dict) else json.loads(rz["gt"])
    obj_z, obj_s = EV.parse_json(rz["pred"]), EV.parse_json(rs["pred"])
    obj_zi, ch_zi, st_zi = inject_pred(rz["pred"], gt)
    obj_si, ch_si, st_si = inject_pred(rs["pred"], gt)
    ch_zi = remap_changed(ch_zi, flatten_pairs(gt, obj_z)["slot_of_pi"])
    ch_si = remap_changed(ch_si, flatten_pairs(gt, obj_s)["slot_of_pi"])
    img_name = Path(rz["image"]).name

    def f1_of(o) -> float | None:
        return _prf(gt, o)[2] if gt else None

    panels = [
        render_panel("② 2B 零样本", "历史落盘 · 无 adapter", gt, obj_z, rz["pred"],
                     f1=f1_of(obj_z)),
        render_panel("② + 知识注入",
                     f"改动 {st_zi['changed']} 处（纠正 {st_zi['fixed']} / "
                     f"改坏 {st_zi['broke']}）· 纯 CPU 后处理",
                     gt, obj_zi, "", ch_zi, f1=f1_of(obj_zi)),
        render_panel("③ 2B SFT", "历史落盘 · 384px", gt, obj_s, rs["pred"],
                     f1=f1_of(obj_s)),
        render_panel("③ + 知识注入", f"改动 {st_si['changed']} 处",
                     gt, obj_si, "", ch_si, f1=f1_of(obj_si)),
        render_panel("GT 真值", f"{img_name}", gt, gt, f1=1.0 if gt else None),
    ]
    metrics = render_metrics([
        ("② 2B 零样本", gt, obj_z, ""),
        ("② + 知识注入", gt, obj_zi, f"改动 {st_zi['changed']} 处"),
        ("③ 2B SFT", gt, obj_s, ""),
        ("③ + 知识注入", gt, obj_si, f"改动 {st_si['changed']} 处"),
    ])
    body = ("<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>"
            "<title>MiniVLM Demo 静态预览</title>"
            "<style>body{background:#f3f5f8;margin:0;padding:16px;}"
            ".grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;}"
            ".full{margin-top:12px;}"
            "@media(max-width:1100px){.grid{grid-template-columns:1fr;}}"
            "</style></head><body>"
            + HEADER_HTML +
            f"{metrics}"
            f"<div class='grid full'>{''.join(panels[:4])}</div>"
            f"<div class='full'>{panels[4]}</div>"
            "</body></html>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")
    print(f"静态预览 → {out_path}")
    return 0


BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


def _screenshot(html_path: Path, png_path: Path,
                width: int = 1500, height: int = 2400) -> int:
    """把静态预览截成 PNG（README / 简历可以直接用图，不必让人点开 HTML）。

    用本机已装的 Edge/Chrome 跑无头模式 —— 不引入 playwright 这类重依赖。
    找不到浏览器只提示、不算失败：HTML 本身已经生成好了。
    """
    exe = next((p for p in BROWSERS if Path(p).exists()), None)
    if not exe:
        print("  [跳过] 未找到 Edge/Chrome，只留 HTML（可直接双击打开）")
        return 0
    cmd = [exe, "--headless=new", "--disable-gpu", "--no-sandbox",
           "--hide-scrollbars", f"--window-size={width},{height}",
           f"--screenshot={png_path}", html_path.as_uri()]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=180)
    except subprocess.TimeoutExpired:
        print("  [失败] 截图超时")
        return 1
    if png_path.exists() and png_path.stat().st_size > 0:
        print(f"截图 → {png_path}（{png_path.stat().st_size / 1024:.0f} KB，"
              f"{width}px 宽）")
        return 0
    tail = (r.stderr or r.stdout or "")[-200:]
    print(f"  [失败] 截图未生成：{tail}")
    return 1


def _smoke(idx: int, size: int) -> int:
    """真机检查（需 GPU，约 1 分钟）：同一张图跑两档，验证 adapter 切换生效。

    这是唯一无法在 CPU 上验证的环节，也是 Demo 最坏的失败模式 ——
    `disable_adapter()` 若不生效，界面上四个面板会显示同一份结果而**不报任何错**。
    """
    load_model()
    if _LOAD_ERR:
        print(f"[FATAL] 模型加载失败：{_LOAD_ERR}")
        return 1
    samples = load_samples()
    if not samples:
        print("[FATAL] 没有测试样本")
        return 1
    s = samples[idx % len(samples)]
    gt = s["gt"]
    print(f"\n样本 {label_of(s)}\n  {s['path']}")

    rz, tz = generate(s["path"], size, use_sft=False)
    rs, ts = generate(s["path"], size, use_sft=True)
    print(f"\n--- ② 2B 零样本（{tz:.1f}s）---\n{rz[:400]}")
    print(f"\n--- ③ 2B SFT（{ts:.1f}s）---\n{rs[:400]}")

    for nm, o in (("② 零样本", EV.parse_json(rz)), ("③ SFT", EV.parse_json(rs))):
        p, r, f1, rr = _prf(gt, o)
        print(f"  {nm:<10} F1={f1:.4f}   P/R={p:.3f}/{r:.3f}   "
              f"TP/FP/FN={rr['tp']}/{rr['fp']}/{rr['fn']}")

    if rz.strip() == rs.strip():
        print("\n[adapter 切换] ⚠ 两档输出完全相同 —— disable_adapter() 可能未生效，"
              "请排查后再开界面（换一张更难的图复测：--smoke 2）。")
    else:
        print("\n[adapter 切换] ✓ 两档输出不同，切换生效。")
        print("[性能] 显存与耗时正常即可直接开界面："
              "venv-gld\\Scripts\\python.exe src\\demo.py")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="MiniVLM Demo")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--adapter", default=str(SFT_DIR))
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--check", action="store_true", help="纯 CPU 自检，不加载模型")
    ap.add_argument("--smoke", type=int, default=None, metavar="IDX",
                    help="真机检查：加载模型，第 IDX 条测试样本跑两档并验证 adapter 切换")
    ap.add_argument("--image-size", type=int, default=384, help="--smoke 用的分辨率")
    ap.add_argument("--render-sample", type=int, default=None,
                    help="用已落盘预测渲一张静态 HTML 预览（不加载模型）")
    ap.add_argument("--out", default=None, help="静态预览输出路径")
    ap.add_argument("--png", action="store_true",
                    help="配 --render-sample：再截一张 PNG（README/简历用）")
    ap.add_argument("--png-size", default="1500x2400",
                    help="PNG 视口尺寸，默认 1500x2400")
    args = ap.parse_args()

    if args.check:
        return _check()
    if args.smoke is not None:
        return _smoke(args.smoke, args.image_size)
    if args.render_sample is not None:
        out = Path(args.out) if args.out else OUTPUTS / "demo_preview.html"
        rc = _render_sample(args.render_sample, out)
        if rc == 0 and args.png:
            try:
                w, h = (int(x) for x in str(args.png_size).lower().split("x"))
            except ValueError:
                print("[FATAL] --png-size 需形如 1500x2400")
                return 1
            rc = _screenshot(out, out.with_suffix(".png"), w, h)
        return rc

    load_samples()
    load_model(Path(args.adapter))
    demo = build_ui()
    demo.queue()
    # Gradio 6 把 theme/css 从 Blocks 挪到了 launch —— 传给 Blocks 会静默失效
    demo.launch(server_name=args.host, server_port=args.port, share=args.share,
                inbrowser=True, show_error=True, quiet=False,
                theme=make_theme(), css=LAUNCH_CSS, head=LAUNCH_HEAD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
