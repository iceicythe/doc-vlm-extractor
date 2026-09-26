"""Independent boundary cases for the versioned scorer. CPU only."""
import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import scoring as S
import evaluate as E
import evaluate_api as API
import check_pref_quality as PREF

GT = {"单据类型": "材料清单", "表头": {"项目名称": "项目 A", "供应商": "供应商",
      "单据编号": "00123", "日期": "2026-09-25"}, "明细": [
      {"序号": "1", "名称": "管材", "规格型号": "DN20", "单位": "米",
       "数量": "2", "单价": "10.00", "金额": "20.00"},
      {"序号": "2", "名称": "螺栓", "规格型号": "M10", "单位": "个",
       "数量": "3", "单价": "2", "金额": "6"}], "合计": {"金额": "26.00"}}


def score(obj):
    return S.score_one(GT, json.dumps(obj, ensure_ascii=False))


class ScoringTests(unittest.TestCase):
    def test_precision_and_identifiers(self):
        self.assertNotEqual(S.normalize("123456.71", "合计.金额"), S.normalize("123456.72", "合计.金额"))
        self.assertNotEqual(S.normalize("00123", "表头.单据编号"), S.normalize("123", "表头.单据编号"))
        self.assertEqual(S.normalize("20.00", "明细[0].金额"), "20")
        self.assertNotEqual(S.normalize("00123", "tel"), S.normalize("123", "tel"))
        self.assertEqual(S.normalize("123456789012345678901234567890.71", "合计.金额"), "123456789012345678901234567890.71")

    def test_text_and_date_policy(self):
        for a, b in [("DN20", "DN200"), ("A B", "AB"), ("A,B", "AB"), ("-20", "20")]:
            self.assertNotEqual(S.normalize(a), S.normalize(b))
        self.assertEqual(S.normalize("2026年9月25日", "表头.日期"), "2026-09-25")
        for value in ["09/10/2026", "2026-02-30", "日期2026-09-25", "2026/09-25"]:
            self.assertTrue(S.normalize(value, "表头.日期").startswith(S.INVALID))

    def test_numeric_grammar(self):
        self.assertEqual(S.normalize("-1,234.500", "合计.金额"), "-1234.5")
        for value in ["1,23", "NaN", "Infinity", "-Infinity", "20元", "￥20", "1e3"]:
            self.assertTrue(S.normalize(value, "合计.金额").startswith(S.INVALID))
        for value in [None, "", " "]:
            self.assertNotEqual(S.normalize(value, "合计.金额"), S.normalize("0", "合计.金额"))

    def test_perfect(self):
        r = score(GT)
        self.assertTrue(r["document_correct"])
        self.assertEqual((r["tp"], r["fp"], r["fn"]), (20, 0, 0))
        self.assertEqual((r["amount_correct"], r["amount_total"]), (3, 3))

    def test_wrong_amount(self):
        obj = copy.deepcopy(GT); obj["合计"]["金额"] = "26.01"
        r = score(obj)
        self.assertEqual((r["tp"], r["fp"], r["fn"]), (19, 1, 1))
        self.assertEqual(r["amount_correct"], 2)
        self.assertFalse(r["document_correct"])

    def test_missing_empty_null_and_zero(self):
        for value in [None, "", "0"]:
            obj = copy.deepcopy(GT); obj["合计"]["金额"] = value
            r = score(obj)
            self.assertEqual(r["fn"], 1)
            self.assertEqual(r["amount_correct"], 2)
            self.assertFalse(r["document_correct"])
        obj = copy.deepcopy(GT); del obj["合计"]["金额"]
        self.assertEqual(score(obj)["fn"], 1)

    def test_strict_and_loose(self):
        good = json.dumps(GT)
        for raw in ["说明：" + good, good + "提取完成", "```json\n" + good + "\n```"]:
            r = S.score_one(GT, raw)
            self.assertFalse(r["strict_json_valid"])
            self.assertTrue(r["loose_parse_success"])
            self.assertTrue(r["schema_valid"])
            self.assertFalse(r["document_correct"])
        r = S.score_one(GT, "[]")
        self.assertTrue(r["strict_json_valid"])
        self.assertFalse(r["loose_parse_success"])
        self.assertFalse(r["schema_valid"])

    def test_invalid_json(self):
        for raw in ["", "garbage", '{"a":NaN}', '{"a":1,"a":2}', '{"a":Infinity}']:
            r = S.score_one(GT, raw)
            self.assertFalse(r["strict_json_valid"])
            self.assertFalse(r["loose_parse_success"])
            self.assertEqual((r["tp"], r["fn"]), (0, 20))

    def test_types_and_extra_fields(self):
        for value in [20, True, ["20"], {"value": "20"}]:
            obj = copy.deepcopy(GT); obj["明细"][0]["金额"] = value
            r = score(obj)
            self.assertTrue(r["strict_json_valid"])
            self.assertFalse(r["schema_valid"])
            self.assertEqual((r["fp"], r["fn"]), (1, 1))
        for location in ["root", "row", "head"]:
            obj = copy.deepcopy(GT)
            target = obj if location == "root" else obj["明细"][0] if location == "row" else obj["表头"]
            target["无关字段"] = None
            r = score(obj)
            self.assertEqual(r["fp"], 1)
            self.assertEqual(len(r["extra_fields"]), 1)
            self.assertFalse(r["document_correct"])

    def test_rows(self):
        obj = copy.deepcopy(GT); obj["明细"].reverse()
        self.assertEqual(score(obj)["fn"], 14)
        obj = copy.deepcopy(GT); obj["明细"].pop()
        self.assertEqual(score(obj)["fn"], 7)
        obj = copy.deepcopy(GT); obj["明细"].append(copy.deepcopy(obj["明细"][0]))
        self.assertEqual(score(obj)["fp"], 7)
        obj = copy.deepcopy(GT); obj["明细"][1]["序号"] = "1"
        self.assertEqual(score(obj)["fn"], 1)
        obj = copy.deepcopy(GT); del obj["明细"][1]["序号"]
        self.assertEqual(score(obj)["fn"], 1)
        obj = copy.deepcopy(GT); obj["明细"].insert(0, "bad row")
        self.assertFalse(score(obj)["schema_valid"])
        self.assertGreater(score(obj)["fn"], 0)

    def test_document_type_is_scored(self):
        obj = copy.deepcopy(GT); obj["单据类型"] = "送货单"
        self.assertEqual(score(obj)["fn"], 1)
        self.assertFalse(score(obj)["document_correct"])

    def test_literal_path_keys_cannot_hide_errors(self):
        obj = copy.deepcopy(GT)
        obj["表头"]["单据编号"] = "wrong"
        obj["表头.单据编号"] = "00123"
        r = score(obj)
        self.assertEqual((r["fp"], r["fn"]), (2, 1))
        self.assertEqual(len(r["extra_fields"]), 1)

    def test_flat(self):
        r = S.score_one({"tel": "00123", "total": "20.00"}, '{"tel":"123","total":"20"}', "flat")
        self.assertEqual((r["tp"], r["fp"], r["fn"]), (1, 1, 1))

    def test_shared_entrypoints_and_denominators(self):
        raw = json.dumps(GT)
        self.assertEqual(E.score_one(GT, raw), API.ev.score_one(GT, raw))
        altered = copy.deepcopy(GT); altered["合计"]["金额"] = "26.01"
        r = score(altered)
        self.assertEqual(PREF.field_f1(altered, GT), (S.prf(r["tp"], r["fp"], r["fn"])[2], r["tp"], r["fp"], r["fn"]))
        report = S.summarize([score(GT), S.score_one(GT, "bad")])
        self.assertEqual(report["n"], 2)
        self.assertEqual(report["schema_valid_rate"], .5)
        self.assertEqual(report["document_correct_rate"], .5)
        self.assertEqual(report["amount_accuracy"], .5)
        self.assertEqual(report["parse_failure_count"], 1)
        self.assertEqual(report["fn"], 20)
        self.assertIsNone(report["inference_failure_count"])


if __name__ == "__main__":
    unittest.main()
