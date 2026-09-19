"""探针：dspy.Image 能否透过 DeepSeek 的 OpenAI 兼容端点走通。

这是 DSPy 方案的最大未知数 —— dspy 3.x 的多模态序列化走 litellm，
而 litellm 的 deepseek provider 是否支持 vision 一直是个坑。
成本：1 次调用，约 1 分钱。
"""
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import dspy
from dotenv import load_dotenv
import os

load_dotenv(ROOT / ".env")

key = os.environ.get("DEEPSEEK_API_KEY", "")
print(f"key 长度 = {len(key)}，前缀 = {key[:3] if key else '（空）'}")

lm = dspy.LM(
    "openai/deepseek-flash",
    api_base="https://api.deepseek.com/v1",
    api_key=key,
    max_tokens=4096,
    temperature=0.0,
)
dspy.configure(lm=lm)


class Extract(dspy.Signature):
    """从工程材料清单图片中抽取结构化字段。"""

    image: dspy.Image = dspy.InputField(desc="工程材料清单扫描件")
    json_out: str = dspy.OutputField(
        desc="仅输出 JSON 对象本身，不要解释、不要代码块标记"
    )


# 取第 1 条测试样本
rec = json.loads(
    open(ROOT / "data/processed/test_ablation.jsonl", encoding="utf-8").readline()
)
img_path = ROOT / rec["image"]
blob = img_path.read_bytes()
uri = "data:image/png;base64," + base64.b64encode(blob).decode("ascii")
print(f"图片 = {rec['image']}（{len(blob)//1024} KB）")

img = dspy.Image(uri)
print(f"dspy.Image 构造成功: {type(img).__name__}")

pred = dspy.Predict(Extract)
try:
    out = pred(image=img)
except Exception as e:
    print(f"\n[FAIL] 调用失败：{type(e).__name__}: {e}")
    raise SystemExit(1)

print("\n[OK] 调用成功")
print("原始输出前 300 字符：")
print(repr(out.json_out[:300]))
try:
    parsed = json.loads(out.json_out)
    print(f"\nJSON 可解析，顶层键 = {list(parsed.keys())}")
except Exception as e:
    print(f"\nJSON 不可解析：{e}")
print(f"\n用量: {lm.history[-1].get('usage') if lm.history else '无记录'}")
