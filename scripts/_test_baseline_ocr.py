"""Rules accept only pixels/OCR. Synthetic box tests do not require OCR inference."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import baseline_ocr as B
import cv2
import numpy as np


def block(text, x, y, width=40):
    return {"text": text, "confidence": .99,
            "box": [[x-width/2,y-8],[x+width/2,y-8],[x+width/2,y+8],[x-width/2,y+8]]}


def fixture(order=None):
    order = order or B.S.ROW_FIELDS
    edges = [20, 70, 220, 340, 400, 470, 560, 680]
    img = np.full((330, 720, 3), 255, np.uint8)
    for edge in edges:
        cv2.line(img, (edge, 105), (edge, 245), (0,0,0), 1)
    texts = [block("工程材料清单",350,20,120),block("项目名称：项目 A",120,48,170),
             block("日期：20260102",585,48,160),block("供应商：供应商 B",140,78,180),
             block("编号：00123",570,78,110)]
    vals = {"序号":"1","名称":"管材","规格型号":"DN20","单位":"米","数量":"2","单价":"10","金额":"18"}
    for i, field in enumerate(order):
        x = (edges[i]+edges[i+1])/2
        texts.extend([block(field, x, 120),block(vals[field], x, 160)])
    texts.extend([block("合计金额(元)",510,230,100),block("18",620,230),
                  block("审核人：张三",300,280,110),block("日期：",550,280)])
    return img, texts


class RulesTests(unittest.TestCase):
    def test_fields_and_no_arithmetic_rewrite(self):
        img, boxes = fixture()
        result, trace = B.extract(img, boxes)
        self.assertEqual(result["表头"]["单据编号"], "00123")
        self.assertEqual(result["表头"]["日期"], "2026-01-02")
        self.assertEqual(result["明细"][0]["金额"], "18")
        self.assertEqual(result["合计"]["金额"], "18")
        self.assertEqual(len(result["明细"]), 1)
        self.assertIn("明细[0].金额", trace["fields"])

    def test_column_reorder(self):
        img, boxes = fixture(["名称","序号","数量","单位","金额","规格型号","单价"])
        result, _ = B.extract(img, boxes)
        self.assertEqual(result["明细"][0], {"序号":"1","名称":"管材","规格型号":"DN20","单位":"米","数量":"2","单价":"10","金额":"18"})

    def test_no_guess_for_missing_values(self):
        img, boxes = fixture()
        boxes = [b for b in boxes if b["text"] != "18"]
        result, _ = B.extract(img, boxes)
        self.assertEqual(result["明细"][0]["金额"], "")
        self.assertEqual(result["合计"]["金额"], "")

    def test_unknown_document_type_and_no_headers(self):
        img, boxes = fixture()
        boxes[0]["text"] = "送货单"
        result, _ = B.extract(img, boxes)
        self.assertEqual(result["单据类型"], "")
        result, trace = B.extract(img, [])
        self.assertEqual(result["明细"], [])
        self.assertIn("fewer_than_four_column_headings", trace["warnings"])

    def test_name_continuation_and_extra_ocr_footer(self):
        img, boxes = fixture()
        boxes.append(block("第二行",120,180))
        result, _ = B.extract(img, boxes)
        self.assertEqual(result["明细"][0]["名称"], "管材 第二行")
        self.assertEqual(len(result["明细"]), 1)

    def test_no_input_mutation(self):
        img, boxes = fixture()
        original = copy.deepcopy(boxes)
        B.extract(img, boxes)
        self.assertEqual(boxes, original)

    def test_missing_heading_does_not_merge_columns(self):
        img, boxes = fixture()
        boxes = [b for b in boxes if b["text"] != "数量"]
        result, trace = B.extract(img, boxes)
        self.assertEqual(result["明细"][0]["数量"], "")
        self.assertEqual(result["明细"][0]["单价"], "10")
        self.assertIn("token_in_unmapped_column", trace["warnings"])


if __name__ == "__main__":
    unittest.main()
