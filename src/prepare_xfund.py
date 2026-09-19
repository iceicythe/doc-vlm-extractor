"""解析 XFUND-zh 真实中文表单 —— 转成扁平键值对，供跨域评测使用。

XFUND 原始格式（TSV：文件名 \t JSON）：
    zh_train_0.jpg → {"height":3508,"width":2480,"ocr_info":[
        {"text":"受理时间:","label":"question","bbox":[...],"id":7, "linking":[[7,13]]},
        {"text":"2021年3月","label":"answer",  "bbox":[...],"id":13,"linking":[[7,13]]}
    ]}

关键：`linking` 字段建立 question ↔ answer 的配对关系，据此还原成 {字段: 值}。

用法：
    venv-gld/Scripts/python.exe src/prepare_xfund.py
    venv-gld/Scripts/python.exe src/prepare_xfund.py --show 2

产出：
    data/processed/xfund_val.jsonl    每行 {"image": "...", "gt": {键值对...}, "meta": {...}}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
XFUND = ROOT / "data" / "public" / "xfund"
OUT_DIR = ROOT / "data" / "processed"


def clean(s: str) -> str:
    """去掉键名末尾的冒号与空白。"""
    return re.sub(r"[:：\s]+$", "", str(s).strip())


def parse_file(tsv: Path, img_dir: Path) -> list[dict]:
    """解析一个 TSV 标注文件。"""
    out = []
    if not tsv.exists():
        return out

    with open(tsv, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            fname, payload = parts
            try:
                doc = json.loads(payload)
            except json.JSONDecodeError:
                continue

            img_path = img_dir / fname
            if not img_path.exists():
                continue

            items = {it.get("id"): it for it in doc.get("ocr_info", [])}
            kv: dict[str, str] = {}

            for it in doc.get("ocr_info", []):
                if it.get("label") != "question":
                    continue
                key = clean(it.get("text", ""))
                if not key:
                    continue
                # linking 形如 [[q_id, a_id]]
                for link in it.get("linking") or []:
                    if len(link) != 2:
                        continue
                    a = items.get(link[1])
                    if not a or a.get("label") != "answer":
                        continue
                    val = str(a.get("text", "")).strip()
                    if not val:
                        continue
                    # 同名字段重复出现时加后缀，避免覆盖
                    k = key
                    i = 2
                    while k in kv:
                        k = f"{key}#{i}"
                        i += 1
                    kv[k] = val

            if kv:
                out.append({
                    "image": str(img_path),
                    "gt": kv,
                    "meta": {"source": "xfund", "split": tsv.stem, "n_kv": len(kv)},
                })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=0, help="打印前 N 条")
    args = ap.parse_args()

    splits = [
        ("train", XFUND / "zh_train" / "xfun_normalize_train.json", XFUND / "zh_train" / "image"),
        ("val", XFUND / "zh_val" / "xfun_normalize_val.json", XFUND / "zh_val" / "image"),
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_recs = []

    print("=" * 62)
    print("解析 XFUND-zh")
    print("=" * 62)

    for name, tsv, img_dir in splits:
        recs = parse_file(tsv, img_dir)
        print(f"  {name:<6} 解析出 {len(recs)} 条（原始 {tsv.name}）")
        all_recs.extend(recs)
        if recs:
            out = OUT_DIR / f"xfund_{name}.jsonl"
            with open(out, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"         → {out.name}")

    if not all_recs:
        print("[FATAL] 没有解析出任何样本，检查 data/public/xfund 结构")
        return 1

    # 统计
    n_kv = [r["meta"]["n_kv"] for r in all_recs]
    keys = Counter(k for r in all_recs for k in r["gt"])
    print()
    print(f"  总样本   : {len(all_recs)}")
    print(f"  每图键值对: 最少 {min(n_kv)} / 最多 {max(n_kv)} / 平均 {sum(n_kv)/len(n_kv):.1f}")
    print(f"  不同字段名: {len(keys)} 种")
    print("  高频字段  :")
    for k, c in keys.most_common(12):
        print(f"    {c:>4}×  {k}")

    for r in all_recs[: args.show]:
        print("\n" + "-" * 62)
        print(Path(r["image"]).name)
        print(json.dumps(r["gt"], ensure_ascii=False, indent=2)[:800])
    return 0


if __name__ == "__main__":
    sys.exit(main())
