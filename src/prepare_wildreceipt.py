"""解析 WildReceipt（真实英文收据）—— 转成扁平键值对，用于跨域评测。

WildReceipt 原始格式（TSV：图片路径 \t 文本框列表）：
    image_files/.../xxx.jpeg → [{"label":1,"transcription":"SAFEWELL","points":[[550,190],...]}, ...]

25 类标签（class_list.txt）：
    0 Ignore | 1 Store_name_value | 2 Store_name_key | 3 Store_addr_value | 4 Store_addr_key
    5 Tel_value | 6 Tel_key | 7 Date_value | 8 Date_key | 9 Time_value | 10 Time_key
    11 Prod_item_value | 12 Prod_item_key | 13 Prod_quantity_value | 14 Prod_quantity_key
    15 Prod_price_value | 16 Prod_price_key | 17 Subtotal_value | 18 Subtotal_key
    19 Tax_value | 20 Tax_key | 21 Tips_value | 22 Tips_key | 23 Total_value | 24 Total_key
    25 Others

设计：只提取**单据级单值字段**（store_name / store_addr / tel / date / time /
subtotal / tax / tips / total），不做商品行配对。
理由：① 单值字段评测口径干净 ② 其中 store_name / date / total
与工程单据的「供应商 / 日期 / 合计」语义对齐，可直接做跨域迁移实验。

用法：
    venv-gld/Scripts/python.exe src/prepare_wildreceipt.py
    venv-gld/Scripts/python.exe src/prepare_wildreceipt.py --show 2

产出：
    data/processed/wildreceipt_{train,test}.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WR = ROOT / "data" / "public" / "wildreceipt"
OUT_DIR = ROOT / "data" / "processed"

# label id -> 统一字段名（只取 value 类）
LABEL_TO_FIELD = {
    1: "store_name",
    3: "store_addr",
    5: "tel",
    7: "date",
    9: "time",
    17: "subtotal",
    19: "tax",
    21: "tips",
    23: "total",
}

# 与工程单据语义对齐的共同字段
SHARED_FIELDS = {"store_name": "供应商", "date": "日期", "total": "合计"}


def parse_tsv(tsv: Path) -> list[dict]:
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
            rel_path, payload = parts
            try:
                boxes = json.loads(payload)
            except json.JSONDecodeError:
                continue

            img_path = WR / rel_path
            if not img_path.exists():
                continue

            fields: dict[str, str] = {}
            for b in boxes:
                if not isinstance(b, dict):
                    continue
                field = LABEL_TO_FIELD.get(b.get("label"))
                if not field:
                    continue
                txt = str(b.get("transcription", "")).strip()
                if not txt or txt in fields:      # 去空、去重（同字段多次出现只取首次）
                    continue
                fields[field] = txt

            # 至少要有个 store_name 或 total 才算有效样本
            if not (fields.get("store_name") or fields.get("total")):
                continue

            out.append({
                "image": str(img_path),
                "gt": fields,
                "meta": {"source": "wildreceipt", "n_fields": len(fields)},
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()

    print("=" * 62)
    print("解析 WildReceipt")
    print("=" * 62)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_recs = []

    for split in ("train", "test"):
        tsv = WR / f"wildreceipt_{split}.txt"
        recs = parse_tsv(tsv)
        print(f"  {split:<6} 解析出 {len(recs)} 条（{tsv.name}）")
        all_recs.extend(recs)
        if recs:
            out = OUT_DIR / f"wildreceipt_{split}.jsonl"
            with open(out, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"         → {out.name}")

    if not all_recs:
        print("[FATAL] 未解析出任何样本，检查 data/public/wildreceipt 结构")
        return 1

    n_f = [r["meta"]["n_fields"] for r in all_recs]
    field_cnt = Counter(k for r in all_recs for k in r["gt"])

    print()
    print(f"  总样本      : {len(all_recs)}")
    print(f"  每图字段数  : 最少 {min(n_f)} / 最多 {max(n_f)} / 平均 {sum(n_f)/len(n_f):.1f}")
    print("  各字段覆盖率:")
    for f, c in field_cnt.most_common():
        mark = "  ★ 与工程单据对齐" if f in SHARED_FIELDS else ""
        print(f"    {f:<12} {c:>5} 条 ({c*100//len(all_recs)}%){mark}")

    for r in all_recs[: args.show]:
        print("\n" + "-" * 62)
        print(Path(r["image"]).name)
        print(json.dumps(r["gt"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
