"""Build frozen B/C layout augmentation data without using the D holdout."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import render as R
import scoring as S

SEED = 2026092602
NEW_SOURCES = 200
REPLAY_SOURCES = 200
OUT = ROOT / "data/benchmarks/layout_train_v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shown_record(gt: dict, shown_header: dict) -> dict:
    return {"表头": shown_header, "明细": gt["明细"], "合计": gt["合计"]["金额"]}


def draw_table(draw, shown, columns, y, weak=False):
    cell, head = R.load_font(16), R.load_font(17)
    total_width = 700
    weight_sum = sum(c[2] for c in columns)
    widths = [int(total_width * c[2] / weight_sum) for c in columns]
    widths[-1] = total_width - sum(widths[:-1])
    for row_index, row in enumerate([None] + shown["明细"]):
        x = 30
        if row is None:
            draw.rectangle((30, y, 730, y + 31), fill=(239, 242, 246))
        for col, width in zip(columns, widths):
            value = col[1] if row is None else row[col[0]]
            R.draw_cell(draw, (x, y, x + width, y + 31), value,
                        head if row is None else cell,
                        "center" if row is None else col[3])
            if not weak:
                draw.line((x, y, x, y + 31), fill=(110, 110, 110), width=1)
            x += width
        draw.line((30, y + 31, 730, y + 31),
                  fill=(170, 170, 170) if weak else (110, 110, 110), width=1)
        y += 31
    if not weak:
        draw.line((730, y - 31 * (len(shown["明细"]) + 1), 730, y), fill=(110, 110, 110), width=1)
        draw.line((30, y - 31 * (len(shown["明细"]) + 1), 730, y - 31 * (len(shown["明细"]) + 1)), fill=(110, 110, 110), width=1)
    return y


def layout_b(shown: dict) -> Image.Image:
    height = max(520, 238 + 31 * (len(shown["明细"]) + 1))
    image = Image.new("RGB", (760, height), "white")
    draw = ImageDraw.Draw(image)
    title, note = R.load_font(26), R.load_font(14)
    draw.text((30, 20), "材料进场明细表", font=title, fill="black")
    h = shown["表头"]
    draw.text((30, 60), "供应单位：" + h["供应商"], font=note, fill="black")
    draw.text((30, 82), "工程项目：" + h["项目名称"], font=note, fill="black")
    draw.text((470, 60), "日期：" + h["日期"], font=note, fill="black")
    draw.text((470, 82), "单号：" + h["单据编号"], font=note, fill="black")
    columns = [R.COLUMNS[i] for i in (0, 1, 2, 3, 5, 4, 6)]
    y = draw_table(draw, shown, columns, 118, weak=False)
    draw.text((455, y + 14), "合计（元）：" + shown["合计"], font=note, fill="black")
    return image


def layout_c(shown: dict) -> Image.Image:
    height = max(520, 258 + 31 * (len(shown["明细"]) + 1))
    image = Image.new("RGB", (760, height), "white")
    draw = ImageDraw.Draw(image)
    title, note = R.load_font(26), R.load_font(14)
    draw.text((260, 18), "工程物料清单", font=title, fill="black")
    h = shown["表头"]
    # Two-column header cards and weak table rules are deliberately different from D.
    draw.rectangle((30, 58, 730, 112), outline=(170, 170, 170), width=1)
    draw.line((390, 58, 390, 112), fill=(185, 185, 185), width=1)
    draw.text((42, 68), "项目：" + h["项目名称"], font=note, fill="black")
    draw.text((402, 68), "编号：" + h["单据编号"], font=note, fill="black")
    draw.text((42, 90), "供方：" + h["供应商"], font=note, fill="black")
    draw.text((402, 90), "制表日期：" + h["日期"], font=note, fill="black")
    y = draw_table(draw, shown, R.COLUMNS, 132, weak=True)
    draw.line((30, y + 8, 730, y + 8), fill=(200, 200, 200), width=1)
    draw.text((30, y + 18), "制表：________", font=note, fill="black")
    draw.text((485, y + 18), "总金额：" + shown["合计"], font=note, fill="black")
    return image


def make_row(path: Path, gt: dict, prompt: str, source_id: str, template: str) -> dict:
    answer = json.dumps(gt, ensure_ascii=False)
    return {
        "image": str(path),
        "messages": [
            {"role": "user", "content": [{"type": "image", "image": str(path)}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ],
        "gt": gt,
        "meta": {"stem": source_id, "source_id": source_id, "template_id": template,
                 "level": "clean", "split": "train", "degradation": "none"},
    }


def main() -> None:
    if OUT.exists():
        raise SystemExit("layout_train_v1 already exists; refusing to overwrite frozen training data")
    old_hashes = set()
    for split in ("train", "val", "test"):
        for line in (ROOT / f"data/processed/{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                old_hashes.add(S.stable_hash(json.loads(line)["gt"]))
    for line in (ROOT / "data/benchmarks/layout_v1/manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            old_hashes.add(S.stable_hash(json.loads(line)["gt"]))

    base_rows = [json.loads(x) for x in (ROOT / "data/processed/train.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    prompt = base_rows[0]["messages"][0]["content"][-1]["text"]
    clean_by_stem = {r["meta"]["stem"]: r for r in base_rows if r["meta"]["level"] == "clean"}
    rng = random.Random(SEED)
    replay_stems = sorted(rng.sample(sorted(clean_by_stem), REPLAY_SOURCES))
    rows = []
    for stem in replay_stems:
        row = clean_by_stem[stem]
        row["meta"] = {**row["meta"], "source_id": stem, "template_id": "A_replay"}
        rows.append(row)

    (OUT / "images").mkdir(parents=True)
    generated = 0
    while generated < NEW_SOURCES:
        gt, shown = R.gen_record(rng)
        identity = S.stable_hash(gt)
        if identity in old_hashes:
            continue
        old_hashes.add(identity)
        generated += 1
        source_id = f"layout_train_{generated:04d}"
        for template, renderer in (("B", layout_b), ("C", layout_c)):
            path = OUT / "images" / f"{source_id}_{template}.png"
            renderer(shown).save(path)
            rows.append(make_row(path, gt, prompt, source_id, template))

    rng.shuffle(rows)
    train_path = OUT / "train.jsonl"
    train_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    counts = {t: sum(r["meta"]["template_id"] == t for r in rows) for t in ("A_replay", "B", "C")}
    protocol = {
        "seed": SEED,
        "new_source_count": NEW_SOURCES,
        "replay_source_count": REPLAY_SOURCES,
        "row_count": len(rows),
        "template_counts": counts,
        "content_overlap_with_val_test_or_D_holdout": 0,
        "data_sha256": sha(train_path),
        "generator_sha256": sha(Path(__file__)),
        "policy": "B/C plus fixed A replay for training; D holdout images and errors are never read by the renderer or training data builder",
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
