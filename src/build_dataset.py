"""构建训练数据集 —— 合成三档退化数据统一成 Unsloth 对话格式并划分。

关键设计：**按「源样本」而���「图片」划分**
    同底图的 清晰 / medium / heavy 三个版本必须落在同一个 split，
    否则等于数据泄漏（模型在 train 见过该底图，test 再考它是无意义的）。

用法：
    venv-gld/Scripts/python.exe src/build_dataset.py
    venv-gld/Scripts/python.exe src/build_dataset.py --train 0.7 --val 0.15

产出：
    data/processed/{train,val,test}.jsonl
    每行：{"image": "...", "messages": [...], "gt": {...}, "meta": {...}}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYN_DIR = ROOT / "data" / "synthetic"
OUT_DIR = ROOT / "data" / "processed"

GT_FILES = {
    "clean": SYN_DIR / "gt.jsonl",
    "medium": SYN_DIR / "gt_degraded_medium.jsonl",
    "heavy": SYN_DIR / "gt_degraded_heavy.jsonl",
}

DEFAULT_PROMPT = (
    "请把这张工程材料清单提取成 JSON，结构为："
    "单据类型、表头（项目名称/供应商/单据编号/日期）、"
    "明细（数组，每项含 序号/名称/规格型号/单位/数量/单价/金额）、"
    "合计（金额）。所有值用字符串。只输出 JSON，不要添加任何解释或代码块标记。"
)


# ================================================================ 加载
def load_all() -> dict[str, dict[str, dict]]:
    """返回 {源样本名: {level: {"image": 绝对路径, "gt": ...}}}。"""
    buckets: dict[str, dict[str, dict]] = defaultdict(dict)

    for level, gt_file in GT_FILES.items():
        if not gt_file.exists():
            print(f"  [警告] 缺少 {gt_file.name}，跳过 {level}")
            continue
        with open(gt_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                img_rel = rec["image"]                       # images/xxx.png 或 images_degraded/xxx.jpg
                img_path = SYN_DIR / img_rel
                if not img_path.exists():
                    continue
                # 源样本名 = 去掉 level 后缀与扩展名的 stem
                stem = img_path.stem
                for sfx in ("_medium", "_heavy", "_light"):
                    if stem.endswith(sfx):
                        stem = stem[: -len(sfx)]
                        break
                buckets[stem][level] = {"image": str(img_path), "gt": rec["gt"]}

    return buckets


# ================================================================ 组装
def to_sample(stem: str, level: str, item: dict, prompt: str) -> dict:
    """组装成 Unsloth 对话格式。"""
    gt_json = json.dumps(item["gt"], ensure_ascii=False)
    return {
        "image": item["image"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["image"]},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": gt_json}],
            },
        ],
        "gt": item["gt"],
        "meta": {"stem": stem, "level": level},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="构建训练数据集")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = ap.parse_args()

    print("=" * 62)
    print("构建数据集")
    print("=" * 62)

    buckets = load_all()
    if not buckets:
        print("[FATAL] 没有找到任何样本，请先跑 render.py 与 degrade.py")
        return 1

    stems = sorted(buckets)
    rng = random.Random(args.seed)
    rng.shuffle(stems)

    n = len(stems)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    splits = {
        "train": stems[:n_train],
        "val": stems[n_train:n_train + n_val],
        "test": stems[n_train + n_val:],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, int]] = {}

    for split, ss in splits.items():
        rows = []
        counter: dict[str, int] = defaultdict(int)
        for stem in ss:
            for level, item in sorted(buckets[stem].items()):
                rows.append(to_sample(stem, level, item, args.prompt))
                counter[level] += 1
        rng.shuffle(rows)

        out = OUT_DIR / f"{split}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[split] = dict(counter)
        print(f"  {split:<6} 源样本 {len(ss):>3}  →  图片 {len(rows):>4} 条  {dict(counter)}")

    # 泄漏检查
    print()
    leak = set(splits["train"]) & set(splits["test"])
    if leak:
        print(f"  [错误] train/test 存在源样本泄漏：{list(leak)[:5]}")
    else:
        print("  [OK] 无源样本泄漏（同底图的不同退化版本落在同一 split）")

    # 抽样展示
    first = json.loads(open(OUT_DIR / "train.jsonl", encoding="utf-8").readline())
    print()
    print("  样例（train 第一条）：")
    print(f"    图片   : {Path(first['image']).name}")
    print(f"    提问   : {first['messages'][0]['content'][1]['text'][:60]}...")
    print(f"    回答   : {first['messages'][1]['content'][0]['text'][:120]}...")

    print("=" * 62)
    print(f"输出目录：{OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
