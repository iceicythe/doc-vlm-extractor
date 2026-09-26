# P0 评测修复与历史重算

评分器：`2.0.0`；旧代码提交：`0adab0c6481ad4d5fcb912e9a673f0c8ab4198ad`。
同一固定图片列表和 GT；直接使用历史生成文本，无训练或新推理。

| 方法 | 历史 F1 | 旧规则重算 | 仅修数值精度（诊断） | v2 主 F1 | 金额正确/总数 | 整单正确/总数 |
|---|---:|---:|---:|---:|---|---|
| eval_zero_abl | 0.734877 | 0.734877 | 0.734877 | 0.692387 | 269/513 | 0/102 |
| eval_abl_base | 0.978152 | 0.978152 | 0.977857 | 0.978504 | 491/513 | 58/102 |

## 解释边界

v2 同时改变字段归一化、明细按阅读顺序对齐、单据类型计分、错误结构/额外字段计分。
因此 v2 与旧分数之差不全是金额精度造成，更不能解释为模型提升或退步。
‘仅修数值精度’保留旧类型处理及行匹配，只去掉六位有效数字截断，是定位精度影响的辅助诊断，不是新主指标。
金额准确率只统计明细金额和合计金额；数量、单价计入字段 F1。所有解析失败保留在分母。
Schema 合规基于宽松解析出的对象，整单正确还要求原始输出严格 JSON 合法。

## eval_zero_abl

- 严格 JSON：68/102；宽松解析：98/102；Schema：0/102。
- 额外字段样本：4/102；解析失败：4；推理失败：未知。
- 仅修精度改变了 0 条样本的字段计数。
- [完整汇总](../outputs/evaluation-v2-final/eval_zero_abl.json) · [逐样本结果](../outputs/evaluation-v2-final/eval_zero_abl_preds.jsonl) · [差异明细](../outputs/evaluation-v2-final/eval_zero_abl_diffs.json)

变化最大的样本（字段 F1 差值；可凭 sample_id 在逐样本文件回查 GT 和原始输出）：

- `synth_000877_medium.jpg`：ΔF1=-0.198381；旧 TP/FP/FN={'tp': 7, 'fp': 0, 'fn': 5}；新={'tp': 7, 'fp': 6, 'fn': 6}。
  Schema：表头: required；供应商: extra field
  `供应商`：GT=None，预测='\x00invalid:extra:"鄂州恒华工程材料有限公司"'。
  `单据类型`：GT='材料清单'，预测='工程材料清单'。
- `synth_000506.png`：ΔF1=-0.157191；旧 TP/FP/FN={'tp': 8, 'fp': 3, 'fn': 4}；新={'tp': 7, 'fp': 6, 'fn': 6}。
  Schema：单据类型: expected 材料清单；合计: expected object
  `单据类型`：GT='材料清单'，预测='工程材料清单'。
  `合计`：GT=None，预测='\x00invalid:object'。
- `synth_000506_medium.jpg`：ΔF1=-0.157191；旧 TP/FP/FN={'tp': 8, 'fp': 3, 'fn': 4}；新={'tp': 7, 'fp': 6, 'fn': 6}。
  Schema：单据类型: expected 材料清单；合计: expected object
  `单据类型`：GT='材料清单'，预测='工程材料清单'。
  `合计`：GT=None，预测='\x00invalid:object'。
- `synth_000665.png`：ΔF1=-0.148148；旧 TP/FP/FN={'tp': 8, 'fp': 4, 'fn': 4}；新={'tp': 7, 'fp': 7, 'fn': 6}。
  Schema：单据类型: expected 材料清单；合计: expected object
  `单据类型`：GT='材料清单'，预测='工程材料清单'。
  `合计`：GT=None，预测='\x00invalid:object'。
- `synth_000849.png`：ΔF1=-0.113790；旧 TP/FP/FN={'tp': 25, 'fp': 3, 'fn': 8}；新={'tp': 24, 'fp': 10, 'fn': 10}。
  Schema：表头: required；供应商: extra field
  `供应商`：GT=None，预测='\x00invalid:extra:"鄂州华建科技有限公司"'。
  `单据类型`：GT='材料清单'，预测='工程材料清单'。
## eval_abl_base

- 严格 JSON：102/102；宽松解析：102/102；Schema：102/102。
- 额外字段样本：0/102；解析失败：0；推理失败：未知。
- 仅修精度改变了 1 条样本的字段计数。
- [完整汇总](../outputs/evaluation-v2-final/eval_abl_base.json) · [逐样本结果](../outputs/evaluation-v2-final/eval_abl_base_preds.jsonl) · [差异明细](../outputs/evaluation-v2-final/eval_abl_base_diffs.json)

变化最大的样本（字段 F1 差值；可凭 sample_id 在逐样本文件回查 GT 和原始输出）：

- `synth_000569_heavy.jpg`：ΔF1=+0.021053；旧 TP/FP/FN={'tp': 11, 'fp': 8, 'fn': 8}；新={'tp': 12, 'fp': 8, 'fn': 8}。
  `合计.金额`：GT='192372.01'，预测='182372.01'。
  `明细[0].单价`：GT='522.21'，预测='52221'。
- `synth_000802_heavy.jpg`：ΔF1=-0.020390；旧 TP/FP/FN={'tp': 46, 'fp': 1, 'fn': 1}；新={'tp': 46, 'fp': 2, 'fn': 2}。
  `明细[4].金额`：GT='16545.55'，预测='16545.53'。
  `明细[5].金额`：GT='79732.36'，预测='79722.36'。
- `synth_000668_heavy.jpg`：ΔF1=+0.019231；旧 TP/FP/FN={'tp': 9, 'fp': 3, 'fn': 3}；新={'tp': 10, 'fp': 3, 'fn': 3}。
  `合计.金额`：GT='18661.61'，预测='18861.61'。
  `明细[0].金额`：GT='18661.61'，预测='18861.61'。
- `synth_000665.png`：ΔF1=+0.006410；旧 TP/FP/FN={'tp': 11, 'fp': 1, 'fn': 1}；新={'tp': 12, 'fp': 1, 'fn': 1}。
  `表头.供应商`：GT='西安华信供应链管理有限公司'，预测='西安华昌供应链管理有限公司'。
- `synth_000772_medium.jpg`：ΔF1=+0.005263；旧 TP/FP/FN={'tp': 17, 'fp': 2, 'fn': 2}；新={'tp': 18, 'fp': 2, 'fn': 2}。
  `表头.供应商`：GT='鄠邑泽材建材科技有限公司'，预测='鄠邑泽源建材科技有限公司'。
  `表头.单据编号`：GT='CL-20260220-836'，预测='CL-20260220-036'。

## 复算

```powershell
.\venv-gld\Scripts\python.exe scripts\audit_evaluation.py --out outputs/evaluation-v2-repeat
```

输出目录必须尚不存在；脚本不覆盖旧预测、旧汇总和旧数据。旧规则对照通过 --legacy-commit 固定历史提交，后续提交不会改变对照代码。

后续已完成 OCR＋规则同集对比与合成陌生版式试点，分别见 [OCR 基线报告](ocr-baseline-report.md) 和 [版式增强报告](layout-training-report.md)。真实中文单据仍未提供，本报告不宣称真实单据泛化已验证。
