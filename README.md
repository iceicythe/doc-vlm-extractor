# Doc-VLM-Extractor

面向工程材料清单的多模态结构化提取实验：输入单据图片，使用 Qwen3-VL-2B 输出包含表头、明细和合计的 JSON。

项目包含合成数据构建、QLoRA 微调、分辨率与数据规模对比、真实英文收据跨域评测、词表后处理、DPO 探索和 Gradio 演示。主要训练实验使用单张 RTX 4060 Laptop 8GB 显卡。

> **结果状态（2026-09-22）**：本文保留历史实验记录，但评分代码存在已知的数字归一化问题，尚未修复并复算。因此，下列分数是**旧评分口径下的历史结果**，不能作为已校正的准确性结论。本次更新修正文档表述与运行说明，不代表评分实现、历史报告或 Demo 中的数字已同步修复。

## 输入与输出

```text
工程材料清单图片 → Qwen3-VL-2B（可加载 SFT LoRA）→ JSON 解析 → 可选词表纠错 → 字段评测 / 展示
```

目标输出示例（用于说明结构，不是新增实验结果）：

```json
{
  "单据类型": "材料清单",
  "表头": {
    "项目名称": "某住宅楼主体工程",
    "供应商": "某建材有限公司",
    "单据编号": "CL-20260915-001",
    "日期": "2026-09-15"
  },
  "明细": [
    {
      "序号": "1",
      "名称": "螺纹钢 HRB400",
      "规格型号": "Φ12",
      "单位": "吨",
      "数量": "12.5",
      "单价": "4200.00",
      "金额": "52500.00"
    }
  ],
  "合计": {"金额": "52500.00"}
}
```

字段约定见 [schema 文档](docs/schema_v1.md)。当前实现以固定 schema 的工程材料清单为主要任务；真实票据支持程度由跨域实验单独评估，不宣称通用票据或工程图纸识别能力。

![历史 Demo 静态预览](outputs/demo_preview.png)

上图是历史预测的静态展示，图中评分沿用旧口径。无需 GPU 即可查看 [HTML 预览](outputs/demo_preview.html)；重新生成预览需要额外的原图和预测文件，详见复现说明。

## 已实现的工作

| 模块 | 内容 | 主要代码 |
|---|---|---|
| 数据构建 | 材料清单渲染，旋转、模糊、透视、污渍等退化，按源样本划分 | [render.py](src/render.py)、[degrade.py](src/degrade.py)、[build_dataset.py](src/build_dataset.py) |
| SFT | 4bit 基座上的 LoRA 微调，支持分辨率、数据比例及训练层范围配置 | [train_sft.py](src/train_sft.py) |
| 对比实验 | 零样本 / SFT、分辨率、训练数据比例、仅语言侧 LoRA | [run_ablation.py](src/run_ablation.py) |
| 评测 | 本地与 API 推理、JSON 解析、字段配对、增量保存 | [evaluate.py](src/evaluate.py)、[evaluate_api.py](src/evaluate_api.py) |
| 词表后处理 | 从训练集构建候选值集合，按字符串相似度保守纠错 | [build_lexicon.py](src/build_lexicon.py)、[inject_knowledge.py](src/inject_knowledge.py) |
| DPO 探索 | 偏好样本构建、自定义训练循环、参考 adapter 检查 | [build_pref_data.py](src/build_pref_data.py)、[train_dpo.py](src/train_dpo.py) |
| 提示词实验 | 人工完整字段清单与 DSPy/GEPA 优化指令对比 | [optimize_prompt_dspy.py](src/optimize_prompt_dspy.py) |
| 展示 | 零样本、微调及词表处理结果对照 | [demo.py](src/demo.py) |

OCR＋规则基线、8B 零样本对照尚未完成。它们仍是有价值的后续比较，尤其是 OCR 基线可以帮助判断引入 VLM 的收益与成本。

## 数据与任务边界

### 合成工程材料清单

已提交的数据清单包含 1,000 个源样本，每个源样本生成 clean、medium、heavy 三种图像，共 3,000 张。

| 划分 | 源样本数 | 图像数 |
|---|---:|---:|
| 训练 | 700 | 2,100 |
| 验证 | 150 | 450 |
| 测试 | 150 | 450 |

划分单位是源样本：同一底图的不同退化版本放在同一个集合。已提交清单的训练、验证、测试源样本集合互不重叠。

这些是固定渲染流程生成的不同内容样本，**不是 1,000 种独立版式**。样本隔离避免了底图复用，但不等于已经验证未见版式、未见材料词汇或真实拍摄单据上的泛化。

主要消融使用 `test_ablation.jsonl` 的 102 张图片（三档各 34 张），对应 83 个不同源样本。三档分别抽样，并非完全相同的 34 个源样本；按档位比较时应考虑内容组成差异。历史主实验另报告过 n=90 的子集结果，不与 n=102 的结果混用。

### 真实收据跨域测试

使用 WildReceipt 测试清单前 100 条，任务为扁平字段提取。它与训练任务同时存在语言、版式、字段结构及图像来源差异，因此结果反映多种分布变化的综合影响。

完整提示词要求 8 个字段：`store_name`、`store_addr`、`tel`、`date`、`time`、`subtotal`、`tax`、`total`。提示词见 [prompt_manual_8f.txt](outputs/prompt_manual_8f.txt)。不存在的字段应省略。

历史 DSPy 实验将该测试清单第 101–124 条用于优化、第 125–164 条用于验证，与前 100 条评测样本分开。这是项目自定义划分，不应表述为完整遵循官方训练 / 测试协议的 benchmark 成绩。

## 历史实验结果（待评分修复后复算）

以下数值来自已提交的汇总和报告。公开仓库没有完整的逐样本预测与训练日志，当前无法仅凭仓库内容独立复算所有历史指标。

### 同域对比

评测集：合成材料清单消融子集，n=102；沿用历史字段级 micro-F1。

| 方法 | 历史 F1 |
|---|---:|
| Qwen3-VL-2B 零样本 | 0.7349 |
| 零样本＋词表纠错 | 0.7870 |
| Qwen3-VL-2B QLoRA SFT，384px | 0.9782 |
| SFT＋词表纠错 | 0.9779 |
| 商用 API，历史记录标识 `deepseek-flash`，关闭思考 | 0.9681 |
| 商用 API＋词表纠错 | 0.9790 |

旧口径下，SFT 在这一固定任务的合成子集上优于本地零样本。与 API 的差异只适用于记录中的输入、提示词和调用配置，不代表通用模型能力排名。API 精确服务版本、运行日期和完整请求配置仍需随原始记录补齐。

词表处理属于输出端的候选值纠错。SFT 后纠错收益很小，说明它在当前闭集数据上的额外价值有限，不能据此证明词表或提示词已经被模型“内化”。测试值不在词表时，也不存在“绝不改坏”的普遍保证。

### 分辨率与训练配置

数据来源：[ablation_summary.json](outputs/ablation_summary.json)。

| 配置 | clean | medium | heavy | 总体历史 F1 |
|---|---:|---:|---:|---:|
| 384px，100% 训练数据 | 0.9886 | 0.9843 | 0.9635 | 0.9782 |
| 384px，50% 训练数据 | 0.9868 | 0.9784 | 0.9562 | 0.9731 |
| 384px，25% 训练数据 | 0.9850 | 0.9764 | 0.9417 | 0.9666 |
| 512px，100% 训练数据 | 1.0000 | 0.9971 | 0.9789 | 0.9914 |
| 768px，100% 训练数据 | 1.0000 | 1.0000 | 0.9870 | 0.9953 |
| 384px，仅语言侧 LoRA | 0.9894 | 0.9872 | 0.9627 | 0.9790 |

可从历史记录中观察到：

- 提高分辨率后三档均有提升，heavy 档提升最大；不能说收益全部来自 heavy。
- 分辨率实验同时改变训练和推理分辨率，无法单独归因于推理阶段多看到了像素。
- 数据比例实验使用 150 / 300 / 600 步，近似保持训练轮数，同时改变了数据量和更新步数；不能据此得出“数据量不重要”或确定的饱和点。
- 相同训练步数不等于相同计算量。更高分辨率的运行成本需要单独测量。
- 仅语言侧 LoRA 与另一配置的分数接近，但没有多随机种子和置信区间，不能宣称统计等价，也不能推广到所有视觉任务。

历史 SFT 基线记录为 600 步、58.7 分钟，`torch.cuda.max_memory_allocated()` 峰值约 4.03 GiB。该值是 PyTorch 分配显存的统计，不等于整卡总占用；耗时依赖设备、软件环境和温度，不能作为其他机器的速度保证。

### 跨域对比：完整 8 字段提示词

评测集：WildReceipt 前 100 条，使用同一份人工 8 字段提示词。

| 方法 | 历史 F1 | Precision | Recall |
|---|---:|---:|---:|
| 2B 零样本 | 0.4691 | 0.4449 | 0.4961 |
| 2B SFT | 0.3752 | 0.3401 | 0.4184 |
| 商用 API | 0.7085 | 0.6973 | 0.7201 |

该设置下，针对合成材料清单的 SFT 没有改善真实英文收据提取。下降可能涉及领域适配、输出习惯、图像感知及其他因素；现有实验不足以排除能力遗忘，也不能唯一归因为“感知层瓶颈”。

早期内置 flat 提示词只要求 3 个字段，却按更多真值字段计分，得到 0.3190 / 0.2842 / 0.4408。它们保留为任务定义不匹配的诊断记录，不作为完整字段提取的主结果。被提示词点名的真值字段占比是覆盖情况，不是模型召回率的严格数学上限；F1 也不能直接除以召回覆盖率来解释“能力达成率”。

### DPO 与提示词优化

- **DPO**：当前配置在同域的历史 F1 为 0.9782 → 0.9773，早期 3 字段跨域设置为 0.2842 → 0.2863，未观察到明确收益。偏好样本的 chosen 部分含错，且偏好数据与跨域目标不匹配，均值得进一步检查。但 chosen 含错不等于偏好方向错误，仍需比较 chosen 与 rejected；不能据此断言“一半梯度方向是反的”。提高分辨率更有效，也不能证明 DPO 原理上无法改善此类任务。
- **DSPy/GEPA**：API 对照的历史 F1 为三字段提示词 0.4408、人工八字段 0.7085、GEPA 0.6830。补齐字段清单改善了任务覆盖，自动优化指令未超过人工完整基线。验证集与测试集均落后于人工基线，本身不足以证明过拟合；还需分析候选搜索、评价方式与运行波动。

这些负结果用于记录当前实验的边界，不作为对 DPO、自动提示词优化或多模态模型的一般性否定。

## 评分定义与已知问题

当前评分实现在 [evaluate.py](src/evaluate.py)，尚待修复：

1. **数字归一化损失精度。** `normalize()` 对纯数字执行 `float()` 和默认 `g` 格式化。例如 `123456.71` 与 `123456.72` 都变成 `123457`，不同结果可能被判为相同；编号 `00123` 也会与 `123` 合并。应改为按字段类型处理：金额使用精确十进制比较，编号和电话保留字符串意义。影响幅度必须使用原始预测重新计算，当前未知。
2. **`hallucination_rate` 的命名过宽。** 当前主要统计“至少出现一个真值中没有的评分字段路径”的样本比例，本文称其为“额外字段样本率”。已有字段中填入错误值会影响 F1，却不一定增加该指标。schema 模式还会忽略部分未纳入评分的额外键，因此 0% 不代表没有无依据生成。
3. **JSON 可解析不等于 schema 合规。** 解析器容忍代码块和前后文本，并提取 JSON 对象。`json_valid_rate` 更准确地说是“解析器成功取得对象的比例”，不能代替原始输出格式检查或字段类型验证。
4. **字段 F1 不等于整单正确率。** 当前按配对后的字段路径和值计算 TP / FP / FN；schema 模式主要评分表头、明细和合计，并未完整验证所有输出约束。应另报整单全对率、schema 合规率以及关键金额字段的精确匹配结果。

下一次发布校正结果时，需要保存评分版本、输入清单、逐样本原始输出和配置，并同步更新汇总、报告与 Demo。当前不提供“已校正准确率”或“零幻觉”的结论。

## 查看与复现

### 1. 直接查看已发布材料

克隆后可以直接查看代码、JSONL 数据清单、汇总表、Markdown 报告和静态预览。

```powershell
git clone https://github.com/iceicythe/doc-vlm-extractor.git
cd doc-vlm-extractor
```

当前仓库不包含模型权重、LoRA adapter、合成原图、公开数据原始包、完整训练日志和逐样本预测。因而不能在刚克隆后直接运行历史预测复算或静态预览再生成。

### 2. 准备环境

历史运行环境为 Windows + NVIDIA GPU。可以先建立脚本约定的环境目录：

```powershell
python -m venv venv-gld
```

训练涉及 PyTorch、torchvision、transformers、Unsloth、TRL、PEFT、accelerate、bitsandbytes、datasets、Pillow 等；图像退化还依赖 NumPy，网页演示依赖 Gradio，API 与提示词实验另需相应客户端。

**仓库尚未提供经过验证的依赖锁定文件。** 上面的环境创建命令不会安装这些依赖；需要按设备、CUDA 和所选训练栈准备兼容环境，不能保证直接安装各包最新版可以复现。依赖安装完成后运行：

```powershell
.\venv-gld\Scripts\python.exe scripts\check_env.py
```

该脚本用于诊断，不会自动补齐环境。部分调度脚本固定引用 `venv-gld/Scripts/python.exe`，Linux/macOS 使用前需要调整路径。

### 3. 生成合成数据与本机清单

在新克隆中，依次执行渲染、两档退化，再构建划分：

```powershell
.\venv-gld\Scripts\python.exe src\render.py --n 1000 --seed 20260915
.\venv-gld\Scripts\python.exe src\degrade.py --level medium --seed 20260915
.\venv-gld\Scripts\python.exe src\degrade.py --level heavy --seed 20260915
.\venv-gld\Scripts\python.exe src\build_dataset.py --train 0.70 --val 0.15 --seed 20260915
.\venv-gld\Scripts\python.exe src\run_ablation.py --build-subset
```

已提交的 JSONL 含历史机器上的绝对路径。上述步骤会生成本机路径，并重写处理后的清单及消融子集；已有本地实验数据时应先保留自己的清单。字体和图像依赖版本可能影响渲染，未提供图像哈希前，不保证重建图片与历史实验逐字节一致。

### 4. 训练与同域评测

显式指定历史主训练配置，避免落入脚本默认的 60 步 smoke 配置：

```powershell
.\venv-gld\Scripts\python.exe src\train_sft.py --max-steps 600 --tag v1 --image-size 384 --rank 16 --lr 2e-4 --batch 1 --grad-accum 8

.\venv-gld\Scripts\python.exe src\evaluate.py --data data\processed\test_ablation.jsonl --tag zero_abl --image-size 384
.\venv-gld\Scripts\python.exe src\evaluate.py --data data\processed\test_ablation.jsonl --adapter outputs\sft_v1\lora --tag abl_base --image-size 384
```

训练输出为 `outputs/sft_v1/lora`。评测生成的分数仍沿用当前有已知问题的评分器，修复前只能用于调通流程。推理脚本支持续跑；改变配置后应使用新 tag，避免混淆实验产物。

### 5. 真实收据与完整字段提示词

安装下载脚本依赖并确保网络可访问后：

```powershell
.\venv-gld\Scripts\python.exe scripts\download_datasets.py --only wildreceipt
.\venv-gld\Scripts\python.exe src\prepare_wildreceipt.py

.\venv-gld\Scripts\python.exe src\evaluate.py --mode flat --data data\processed\wildreceipt_test.jsonl --limit 100 --prompt-file outputs\prompt_manual_8f.txt --tag zero_wr_8f
.\venv-gld\Scripts\python.exe src\evaluate.py --mode flat --data data\processed\wildreceipt_test.jsonl --limit 100 --prompt-file outputs\prompt_manual_8f.txt --adapter outputs\sft_v1\lora --tag sft_wr_8f
```

API 对照需要额外服务配置和调用额度，参数见 `src/evaluate_api.py --help`。比较时应统一样本、字段清单和评分器，并保存模型标识、图像处理、解码与思考模式等实际配置。

### 6. Demo 与检查

完成环境、数据和 SFT adapter 准备后，可启动交互演示：

```powershell
.\venv-gld\Scripts\python.exe src\demo.py
```

`demo.py --render-sample 41 --png` 还依赖相应原图、`outputs/eval_zero_abl_preds.jsonl` 与 `outputs/eval_abl_base_preds.jsonl` 等预测产物；PNG 导出需要可用的浏览器。这些条件未满足时，请直接查看已提交的静态预览。

已有检查脚本包括：

```powershell
.\venv-gld\Scripts\python.exe scripts\_test_evaluate_scoring.py
.\venv-gld\Scripts\python.exe scripts\_test_evaluate_args.py
.\venv-gld\Scripts\python.exe scripts\_test_dpo_math.py
.\venv-gld\Scripts\python.exe scripts\_test_evaluate_api.py
```

各脚本仍需对应依赖。现有测试覆盖部分评分与训练辅助逻辑，不能证明上述数字归一化漏洞已修复，也不替代 GPU 训练 / 推理验证。

## 后续优先级

- [ ] 修复按字段类型的数字归一化，增加精度与编号回归测试。
- [ ] 明确额外字段、字段值错误、schema 合规和整单全对指标。
- [ ] 发布逐样本预测、实验配置及可复算命令，重新生成历史对比表。
- [ ] 锁定依赖版本，移除数据清单对个人绝对路径的依赖。
- [ ] 补 OCR＋规则基线，并统一完整字段提示词。
- [ ] 增加未见版式、未见词汇和真实单据测试；在验证集上选择后处理参数，再进行独立测试。
- [ ] 对小幅差异增加重复实验或按源样本分组的不确定性估计。

## 代码与历史记录导航

| 路径 | 内容 |
|---|---|
| `src/` | 数据、训练、推理、评测和演示源码 |
| `scripts/` | 环境检查、数据下载及辅助测试 |
| `data/processed/` | 已提交的数据清单，图片需另行准备 |
| `data/lexicon/` | 训练集派生词表 |
| `outputs/ablation_summary.json` | 历史消融汇总 |
| [技术复盘](docs/retrospective.md) | 实现过程与调试记录 |
| [历史 Results](outputs/results_onepager.md) | 旧版结果整理 |
| [DPO 记录](outputs/report_dpo_negative.md) | 当前配置下的负结果分析 |
| [提示词实验记录](outputs/report_dspy_flat.md) | 人工与自动指令优化对照 |
| [报告附录](outputs/reports_appendix.md) | 错误分类及后处理统计 |

历史文档保留实验过程，其中的强因果表述、“幻觉率”和旧评分结论尚未逐份修订。解释项目当前状态时，以本 README 的指标边界和复现限制为准。
