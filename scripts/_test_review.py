import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import review as R


def doc(amount="20.00", total="20.00"):
    return {"单据类型": "材料清单", "表头": {"项目名称": "项目", "供应商": "供方", "单据编号": "00123", "日期": "2026-09-26"},
            "明细": [{"序号": "1", "名称": "水泥", "规格型号": "P.O 42.5", "单位": "吨", "数量": "2", "单价": "10.00", "金额": amount}],
            "合计": {"金额": total}}


class ReviewTests(unittest.TestCase):
    def test_consistent(self):
        result = R.validate_document(doc())
        self.assertTrue(result["schema_valid"]); self.assertEqual(result["warnings"], [])

    def test_warning_does_not_change_value(self):
        value = doc("18.00", "18.00")
        before = json.dumps(value, ensure_ascii=False)
        result = R.validate_document(value)
        self.assertEqual(len(result["warnings"]), 1)
        self.assertEqual(before, json.dumps(value, ensure_ascii=False))

    def test_total_warning(self):
        result = R.validate_document(doc("20.00", "19.00"))
        self.assertEqual(result["warnings"][0]["type"], "合计关系异常")

    def test_strict_json(self):
        with self.assertRaises(ValueError): R.parse_document("说明\n" + json.dumps(doc(), ensure_ascii=False))

    def test_save_reload_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = doc("18.00", "18.00"); new = doc()
            state = {"sample_id": "s1", "image_path": "x.png", "raw_output": json.dumps(old, ensure_ascii=False),
                     "model_version": "adapter@sha", "prompt_sha256": "abc"}
            saved = R.save_review(state, json.dumps(new, ensure_ascii=False), ["文字识别错误"], "checked", Path(tmp))
            record = json.loads(Path(saved["record_path"]).read_text(encoding="utf-8"))
            self.assertEqual(record["reviewed_document"], new)
            self.assertTrue(any(x["path"] == "明细[0].金额" for x in record["modified_fields"]))
            self.assertTrue(Path(saved["csv_path"]).exists())
            self.assertEqual(record["original_parsed"], old)
            self.assertEqual(R.load_review(saved["record_path"], Path(tmp))["review_id"], record["review_id"])


if __name__ == "__main__": unittest.main()
