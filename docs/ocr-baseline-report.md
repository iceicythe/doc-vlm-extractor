# OCR＋规则基线与四方案比较

本轮没有训练。OCR 新运行 102 张图片；其余方案复用历史预测并按同一评分器重算。
评分器 `2.0.0`，样本/GT 哈希 `7351a99945de7526306d4b4d19eb03830ecabe6eb29c16aa1cb9cf74a23baf06`。

## 数据边界

开发集：原 val 的前 8 个源单据组，包含 clean/medium/heavy，共 24 张。
规则在开发集上修订一次：避免标题漏识别时把缺失列并入相邻列，并容许列标题货币后缀括号缺失。
开发集 F1：0.8170 → 0.8436；第二次复用相同 OCR 缓存，只调整规则。
正式测试前冻结源码、评分器、依赖、OCR 模型哈希和配置；测试后未修改规则。
测试集：原 test_ablation 的 102 张，来自 83 个源单据；clean/medium/heavy 各 34 张，并非同一批 34 个源单据各取三档。
开发源单据与测试源单据无重叠；测试中部分源单据重复出现，其退化变体不能当作独立来源。
四方案按顺序图片 ID 和 GT 精确核验一致。历史 VLM 缺输入图片字节哈希，不能补造历史图像指纹。

## 同集结果

| 方案 | 字段 F1 | 金额正确/总数 | Schema 合规/总数 | 整单正确/总数 |
|---|---:|---|---|---|
| OCR＋规则（CPU，本轮新运行） | 0.8245 | 432/513 | 57/102 | 5/102 |
| 原始 VLM（历史预测重算） | 0.6924 | 269/513 | 0/102 | 0/102 |
| 微调 VLM（历史预测重算） | 0.9785 | 491/513 | 102/102 | 58/102 |
| 微调 VLM＋词表（历史后处理重算） | 0.9782 | 491/513 | 102/102 | 58/102 |

## 分退化档位

| 方案 | clean F1 | medium F1 | heavy F1 |
|---|---:|---:|---:|
| OCR＋规则（CPU，本轮新运行） | 0.9256 | 0.8662 | 0.6753 |
| 原始 VLM（历史预测重算） | 0.7443 | 0.7282 | 0.6121 |
| 微调 VLM（历史预测重算） | 0.9889 | 0.9848 | 0.9637 |
| 微调 VLM＋词表（历史后处理重算） | 0.9889 | 0.9848 | 0.9629 |

## 推理配置与时间

OCR：RapidOCR ONNX Runtime 1.4.4；PP-OCRv4 检测/识别、v2 方向分类；CPU 4 个算子内线程、1 个算子间线程。
读原始尺寸图片，OCR 检测保持长宽比并按最短边 736 处理。参数依据 [RapidOCR 官方接口文档](https://rapidai.github.io/RapidOCRDocs/v1.4.4/install_usage/api/RapidOCR/)，实际模型与配置指纹见产物。
OCR 端到端平均 **1.528 秒/张**，中位数 1.529 秒，P95 1.983 秒。
包含读图、解码、OCR、规则提取和 JSON 序列化；排除模型加载（0.221 秒）、一次预热、评分和结果文件写入。
OCR 未使用 GPU。未采集进程峰值内存；本轮未重新测量 VLM 时延或显存，因此不作速度或资源优劣结论。
VLM 历史缓存记录输入 384px，提示词身份等配置缺口沿用 [资产盘点](evaluation-inventory.md)，不把当前默认值冒充历史参数。

## 结构指标和失败分母

OCR 推理/规则异常 0/102；对象解析失败 0/102。
OCR 与词表输出由程序序列化，严格 JSON 合法率不代表模型遵循指令能力；主表因此不拿该指标作四方案能力排名。
空字段、识别错误、结构不合规均保留在分母。金额保持图上 OCR 值，不按数量×单价自动重算。

## 词表收益与误改

按当前评分归一化值，历史微调后处理共有 1 处变化：修复 0，误改 1。
已逐条验证后处理产物的 pred_orig 与原始微调文本一致；后处理原始实验的阈值选择过程仍未完整核验。

- `synth_000286_heavy.jpg` / `表头.项目名称`：'高陵棚户区改造工程' → '高新棚户区改造工程'，GT='高陵棚户区改造工程'。

## 可追溯失败案例

以下按每档最低字段 F1 选取两例，不只展示成功图。错误字段计数是路径事件，可一图多错，不是图片错误率。
OCR 识字错误与行列关联错误可借文字框、字段来源和原图复核；自动统计不把字符不匹配直接等同于视觉原因，也不把多字段直接称为幻觉。

错误路径事件数（按字段汇总）：规格型号 199；名称 145；序号 107；金额 81；单位 70；项目名称 49；数量 44；单价 42；供应商 35；单据编号 16；日期 13；单据类型 1。

- `synth_000158.png`（clean）：Schema 问题 0 项。
  `明细[0].规格型号`：GT='密目式 1.8×6m'，提取='密目式1.8x6m'；分配的 OCR 文本=['密目式1.8x6m']。
  `明细[1].规格型号`：GT='APP 3mm'，提取='APP3mm'；分配的 OCR 文本=['APP3mm']。
- `synth_000506.png`（clean）：Schema 问题 0 项。
  `明细[0].名称`：GT='PPR 给水管'，提取='PPR给水管'；分配的 OCR 文本=['PPR给水管']。
  `表头.项目名称`：GT='沣西产业园区一期工程'，提取='洋西产业园区一期工程'；分配的 OCR 文本=['项目名称：洋西产业园区一期工程']。
- `synth_000374_medium.jpg`（medium）：Schema 问题 3 项。
  `明细[0].单价`：GT='416.28'，提取=None；分配的 OCR 文本=[]。
  `明细[0].单位`：GT='吨'，提取=None；分配的 OCR 文本=[]。
- `synth_000342_medium.jpg`（medium）：Schema 问题 2 项。
  `明细[0].单价`：GT='59.74'，提取=None；分配的 OCR 文本=[]。
  `明细[0].单位`：GT='m2'，提取='个'；分配的 OCR 文本=['个']。
- `synth_000961_heavy.jpg`（heavy）：Schema 问题 6 项。
  `合计.金额`：GT='107619.96'，提取=None；分配的 OCR 文本=[]。
  `明细[0].单价`：GT='0.42'，提取=None；分配的 OCR 文本=[]。
- `synth_000106_heavy.jpg`（heavy）：Schema 问题 6 项。
  `合计.金额`：GT='71208.42'，提取=None；分配的 OCR 文本=[]。
  `明细[0].单价`：GT='21.16'，提取=None；分配的 OCR 文本=[]。

## 产物与复现

- [四方案汇总](../outputs/baseline-comparison-v1/comparison.json)
- [OCR 汇总](../outputs/ocr-test-v1/summary.json) · [逐样本预测](../outputs/ocr-test-v1/predictions.jsonl) · [OCR 文字框和规则中间结果](../outputs/ocr-test-v1/ocr.jsonl)
- [失败案例与字段证据](../outputs/baseline-comparison-v1/failure_cases.json) · [开发集两版结果](../outputs/baseline-comparison-v1/development-summary.json)
- [冻结配置](../data/benchmarks/ocr_v1/frozen_config.json) · [依赖锁定](../requirements-ocr-lock.txt)

```powershell
.\venv-gld\Scripts\python.exe -m venv .venv-ocr
.\.venv-ocr\Scripts\python.exe -m pip install -r requirements-ocr-lock.txt
$env:PYTHONUTF8 = "1"
.\.venv-ocr\Scripts\python.exe scripts\_test_baseline_ocr.py
.\.venv-ocr\Scripts\python.exe src\baseline_ocr.py --data data/processed/test_ablation.jsonl --out outputs/ocr-test-repeat --frozen-config data/benchmarks/ocr_v1/frozen_config.json
```

输出目录必须不存在；原数据清单含本机绝对路径，迁移机器前需制作单独的路径映射清单，不覆盖历史清单。

## 未完成边界

这只是已有合成模板上的比较，不能推出真实中文单据或陌生版式的泛化效果。
四方案准确率比较已完成；同机 VLM 时延/显存重测、真实单据、陌生版式和人工复核流程仍未完成。
测试结果已观察；如后续据此改 OCR 规则，必须将这批数据作为开发参考，并另设最终留出测试集。
