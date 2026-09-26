"""Create a paired A/D layout holdout from new content; never alters existing data."""
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

SEED = 2026092601
SOURCE_COUNT = 12
OUT = ROOT / "data/benchmarks/layout_v1"


def novel_layout(shown, size):
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    title, note, cell, head = R.load_font(26), R.load_font(14), R.load_font(16), R.load_font(17)
    draw.text((30, 22), "工程材料清单", font=title, fill="black")
    h = shown["表头"]
    # Same content/fonts, new header positions and top-right total.
    draw.text((30, 62), "编号：" + h["单据编号"], font=note, fill="black")
    draw.text((450, 62), "日期：" + h["日期"], font=note, fill="black")
    draw.text((30, 82), "供应商：" + h["供应商"], font=note, fill="black")
    draw.text((30, 102), "项目名称：" + h["项目名称"], font=note, fill="black")
    draw.text((475, 102), "合计金额(元)：" + shown["合计"], font=note, fill="black")
    columns = [R.COLUMNS[i] for i in (1, 2, 3, 4, 5, 6, 0)]
    weights = sum(c[2] for c in columns)
    widths = [int(700*c[2]/weights) for c in columns]
    widths[-1] = 700 - sum(widths[:-1])
    y = 136
    draw.rectangle((30,y,730,y+30),fill=(240,240,240))
    for row_index, row in enumerate([None] + shown["明细"]):
        x = 30
        for col, width in zip(columns, widths):
            text = col[1] if row is None else row.get(col[0], "")
            R.draw_cell(draw, (x,y,x+width,y+30), text, head if row is None else cell,
                        "center" if row is None else col[3])
            x += width
        # Weak horizontal rules, no vertical grid.
        draw.line((30,y+30,730,y+30), fill=(175,175,175), width=1)
        y += 30
    draw.text((30,y+14), "备注：请按清单逐项核对。", font=note, fill="black")
    draw.text((430,y+14), "审核人：________", font=note, fill="black")
    return image


def main():
    if OUT.exists():
        raise SystemExit("Holdout directory exists; do not overwrite or regenerate test data")
    old = set()
    for split in ("train", "val", "test"):
        for line in (ROOT / f"data/processed/{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                old.add(S.stable_hash(json.loads(line)["gt"]))
    prompt = json.loads((ROOT / "data/processed/train.jsonl").read_text(encoding="utf-8").splitlines()[0])["messages"][0]["content"][-1]["text"]
    rng = random.Random(SEED)
    generated = []
    while len(generated) < SOURCE_COUNT:
        gt, shown = R.gen_record(rng)
        identity = S.stable_hash(gt)
        if identity not in old:
            old.add(identity)
            generated.append((gt, shown))
    (OUT / "images").mkdir(parents=True)
    records = []
    for i, (gt, shown) in enumerate(generated, 1):
        a = R.render(shown, rng)
        size = (a.width, a.height + 40)
        padded = Image.new("RGB", size, "white"); padded.paste(a, (0,0))
        images = {"A": padded, "D": novel_layout(shown, size)}
        for template, image in images.items():
            path = OUT / "images" / f"layout_{i:03d}_{template}.png"
            image.save(path)
            records.append({"image": str(path), "gt": gt,
                            "meta": {"source_id": f"layout_{i:03d}", "stem": f"layout_{i:03d}",
                                     "template_id": template, "level": "clean", "split": "test",
                                     "degradation": "none"},
                            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "image_size": list(size)})
    (OUT / "manifest.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records),encoding="utf-8")
    (OUT / "prompt.txt").write_text(prompt,encoding="utf-8")
    protocol = {"seed":SEED,"source_count":SOURCE_COUNT,"image_count":len(records),
                "generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "original_renderer_sha256":hashlib.sha256((ROOT/'src/render.py').read_bytes()).hexdigest(),
                "manifest_sha256":hashlib.sha256((OUT/'manifest.jsonl').read_bytes()).hexdigest(),
                "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
                "source_content_overlap_with_existing_splits":0,
                "layout_A":"original renderer, bottom padded 40px",
                "layout_D":"serial column moved to end; header positions changed; total above table; weak horizontal rules without vertical lines; footer remark",
                "controls":"paired identical GT and shown values, font family/sizes, canvas size; no degradation; all inputs resized to 384x384 for VLM",
                "limits":"12 synthetic source documents; bundled layout changes cannot isolate a single layout factor; no real-document claim",
                "policy":"freeze before inference; do not tune prompt/rules/model on this holdout"}
    (OUT / "protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(protocol,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
