# 模态缺失下的多模态情感分析

本仓库是机器学习课程项目，目标是在 CMU-MOSI 上研究文本（T）、音频（A）、视觉（V）模态缺失时的情感回归。模型输入 MMSA 预提取特征及模态可用性掩码，输出范围为 `[-3, 3]` 的情感分数；统一使用 MAE（越低越好）与 Pearson 相关系数（越高越好）评测。项目不仅要求模型可运行，还要求实验可追溯、比较公平、关键代码可解释，并用实验支持成功或失败的结论。

项目原始要求见 [`pj/requirement.md`](pj/requirement.md)。本文既是协作规范，也是五位组员各自的验收清单。除负责人共同维护的本文件外，每位组员只修改自己任务表中列出的路径，以保证五条开发线可以同时进行。

## 1. 分支与人员分工

五个分支都从同一个初始提交创建。表中的“组员”可替换为真实姓名，但分支名和交付路径不得擅自修改。

| 组员 | 开发分支 | 负责模块 | 主要交付路径 |
|---|---|---|---|
| 组员 1 | `feat/member1-data` | 数据读取、固定划分、归一化、缺失掩码 | `pj/src/data/`、`pj/scripts/prepare_data.py`、`pj/configs/data.yaml`、`pj/tests/test_data.py` |
| 组员 2 | `feat/member2-unimodal` | 文本/音频/视觉三套单模态基线 | `pj/src/models/unimodal.py`、`pj/configs/unimodal/`、`pj/scripts/train_unimodal.py`、`pj/tests/test_unimodal.py` |
| 组员 3 | `feat/member3-concat-trainer` | 特征拼接基线、通用训练循环、检查点 | `pj/src/models/concat.py`、`pj/src/training/`、`pj/configs/concat.yaml`、`pj/scripts/train_concat.py`、`pj/tests/test_training.py` |
| 组员 4 | `feat/member4-missing-fusion` | 自选缺失感知融合模型与消融配置 | `pj/src/models/missing_aware.py`、`pj/configs/fusion/`、`pj/scripts/train_fusion.py`、`pj/tests/test_missing_aware.py` |
| 组员 5 | `feat/member5-evaluation` | 指标、缺失场景、批量评测、汇总绘图 | `pj/src/evaluation/`、`pj/scripts/evaluate.py`、`pj/scripts/run_experiments.py`、`pj/configs/eval.yaml`、`pj/tests/test_evaluation.py` |

### 1.1 每人所需电脑硬件与系统配置

全组统一使用 Python 环境，禁止因为个人机器不同而改变数据格式或接口。推荐公共环境如下：

- 操作系统：Ubuntu 22.04/24.04 LTS（推荐）或 Windows 10/11 64 位；Windows 用户可使用 PowerShell，路径处理必须跨平台；
- Python：`3.10.x`，使用 Conda 或 `venv` 建立独立环境；
- 公共软件：Git 2.40+、PyTorch 2.x、NumPy、SciPy、pandas、PyYAML、scikit-learn、matplotlib、pytest；
- GPU 环境：NVIDIA 驱动、与所装 PyTorch 匹配的 CUDA runtime；不要求单独安装完整 CUDA Toolkit；
- 全组最低资源：4 核 CPU、8 GB 内存、20 GB 可用磁盘、稳定网络；建议 8 核 CPU、16 GB 内存和 SSD；
- 统一依赖最终冻结到 `requirements.txt` 或 `environment.yml`，所有人记录 Python、PyTorch、CUDA、GPU 型号和操作系统版本。

| 组员 | 主要负载 | 最低可用配置 | 推荐配置 | GPU 要求 |
|---|---|---|---|---|
| 组员 1 | 下载、解析、padding、归一化、缺失模拟 | 4 核 CPU、16 GB RAM、20 GB 空间 | 8 核 CPU、32 GB RAM、SSD、30 GB 空间 | 不需要；数据测试必须能在 CPU 完成 |
| 组员 2 | 三个轻量单模态模型的训练与对比 | 4 核 CPU、8 GB RAM；可用 CPU smoke test | 6–8 核 CPU、16 GB RAM、NVIDIA GPU 6 GB+ | 可选但推荐；无 GPU 时减小 batch/epoch 做开发，正式实验借用统一 GPU |
| 组员 3 | 通用训练循环、拼接模型、检查点测试 | 4 核 CPU、16 GB RAM | 8 核 CPU、16 GB RAM、NVIDIA GPU 6 GB+ | 可选；训练器单测必须支持 CPU，正式训练建议 GPU |
| 组员 4 | 缺失感知融合、注意力/门控、消融训练 | 6 核 CPU、16 GB RAM、NVIDIA GPU 6 GB | 8 核 CPU、32 GB RAM、NVIDIA GPU 8–12 GB | 强烈需要，是全组 GPU 优先级最高的任务 |
| 组员 5 | 批量推理、指标聚合、案例分析、绘图 | 4 核 CPU、8 GB RAM、10 GB 空间 | 8 核 CPU、16 GB RAM、SSD | 开发与绘图不需要；全模型批量推理时可使用 GPU 加速 |

显存不足时只能调整 `batch_size`、梯度累积或模型隐藏维度，不能减少固定测试样本或改变评测场景。正式对比应尽量在同一台 GPU、相同软件环境中运行；若使用不同硬件，必须在报告中记录，但 MAE/Pearson 的算法和数据条件保持不变。

### 工作量对齐原则

每位组员均须完成：

1. 一个可独立运行的核心模块（约 2–4 个源文件）；
2. 一个命令行入口和对应 YAML 配置；
3. 不少于 3 个单元测试，覆盖正常输入、模态缺失和非法输入；
4. 一份模块说明文档 `pj/docs/memberN.md`，包含运行命令、实现选择和已知限制；
5. 至少一次小规模 smoke test，并将日志或结果写入约定目录；
6. 一份独立报告章节 `pj/report/sections/0N_*.md`、一份答辩提纲 `pj/report/slides/memberN.md` 和一份 AI 使用记录 `pj/report/contributions/memberN.md`。

不要把数据集、模型权重、缓存或大体积结果提交到 Git。代码评审以接口兼容、测试通过和结果可复现为验收依据，不以代码行数衡量工作量。

## 2. 统一目录结构

```text
pj/
├── configs/                 # YAML 配置；按任务表分区维护
├── data/                    # 本地数据（不提交大文件）
│   ├── raw/
│   └── processed/
├── docs/                    # 每人一份 memberN.md
├── report/
│   ├── sections/            # 五人分别维护的报告章节
│   ├── slides/              # 五人分别维护的答辩提纲
│   └── contributions/       # 成员贡献与 AI 使用记录
├── results/                 # 指标 JSON/CSV 与图片
├── checkpoints/             # 模型权重（不提交）
├── scripts/                 # 训练、评测命令行入口
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── evaluation/          # 注意使用 evaluation，不新增 evalution
├── tests/
└── requirement.md
```

所有脚本必须从仓库根目录通过 `python -m pj.scripts.<脚本名>` 调用。代码不得依赖当前工作目录拼接路径；路径由配置文件或命令行参数传入。

## 3. 跨模块统一接口（冻结）

以下约定是五个分支的合并契约。若确需修改，必须先在主分支更新本文并通知全组，禁止只在个人分支私自变更。

### 3.1 模态与张量约定

- 模态顺序固定为 `text, audio, vision`，缩写顺序固定为 `T, A, V`。
- 所有特征和标签使用 `torch.float32`，掩码使用 `torch.bool`。
- 批大小记作 `B`，序列长度分别为 `Lt/La/Lv`，特征维度分别为 `Dt/Da/Dv`。
- 文本、音频、视觉张量形状依次为 `[B, Lt, Dt]`、`[B, La, Da]`、`[B, Lv, Dv]`。
- 序列 padding 掩码形状依次为 `[B, Lt]`、`[B, La]`、`[B, Lv]`；`True` 表示有效时间步。
- 模态可用性掩码 `modality_mask` 形状为 `[B, 3]`，列顺序为 T/A/V；`True` 表示该模态可用。
- 标签 `label` 和模型预测 `prediction` 的形状统一为 `[B]`，有效范围为 `[-3.0, 3.0]`。
- 缺失模态的特征必须置零，同时以 `modality_mask=False` 标注；模型不得仅通过“是否全零”推断缺失。

### 3.2 Dataset / DataLoader 输出

`Dataset.__getitem__` 返回以下字典；DataLoader 经过 `collate_fn` 后仅在首维增加 batch 维：

```python
sample = {
    "id": str,
    "text": FloatTensor[Lt, Dt],
    "audio": FloatTensor[La, Da],
    "vision": FloatTensor[Lv, Dv],
    "text_padding_mask": BoolTensor[Lt],
    "audio_padding_mask": BoolTensor[La],
    "vision_padding_mask": BoolTensor[Lv],
    "modality_mask": BoolTensor[3],       # [T, A, V]
    "label": FloatTensor[],               # scalar, [-3, 3]
}
```

固定划分文件为 `pj/data/processed/splits.json`，格式如下：

```json
{
  "dataset": "CMU-MOSI",
  "source": "MMSA-pretrained-features",
  "seed": 2026,
  "train": ["segment_id_1"],
  "valid": ["segment_id_2"],
  "test": ["segment_id_3"]
}
```

归一化统计量只能由训练集计算，保存到 `pj/data/processed/normalization.json`。验证集和测试集不得参与统计。

### 3.3 缺失模拟接口

组员 1 实现统一函数，训练与评测都调用它，其他组员不得重复实现：

```python
def apply_missingness(
    batch: dict[str, object],
    *,
    mode: str,                 # complete | drop_text | drop_audio | drop_vision | random
    missing_rate: float,
    generator: torch.Generator,
) -> dict[str, object]:
    """返回新 batch，不原地修改；保证每个样本至少保留一个模态。"""
```

`complete` 的 `missing_rate` 必须为 `0.0`；三种 `drop_*` 模式固定移除对应模态；`random` 按样本、按模态随机移除，并保证至少保留一个模态。相同 `seed + sample id + scenario` 必须得到相同掩码，以便不同模型在完全相同的测试样本和缺失条件下比较。

### 3.4 模型调用与输出

所有模型继承 `torch.nn.Module`，构造参数从 YAML 读取。`forward` 签名完全统一：

```python
def forward(
    self,
    *,
    text: torch.Tensor,
    audio: torch.Tensor,
    vision: torch.Tensor,
    text_padding_mask: torch.Tensor,
    audio_padding_mask: torch.Tensor,
    vision_padding_mask: torch.Tensor,
    modality_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    ...
```

返回字典至少包含：

```python
{
    "prediction": FloatTensor[B],          # clamp/tanh 到 [-3, 3]
    "representation": FloatTensor[B, H],   # 用于分析；H 可由模型决定
}
```

可以增加辅助输出（如 `attention_weights`、`aux_loss`），但训练器只能依赖上述两个必选字段。模型不得接收 `label`，以免训练/推理接口不一致。

统一调用示例：

```python
model_inputs = {
    key: batch[key]
    for key in (
        "text", "audio", "vision",
        "text_padding_mask", "audio_padding_mask", "vision_padding_mask",
        "modality_mask",
    )
}
outputs = model(**model_inputs)
loss = torch.nn.functional.l1_loss(outputs["prediction"], batch["label"])
```

### 3.5 检查点接口

检查点保存到 `pj/checkpoints/<model_name>/<run_id>/best.pt`，内容固定为：

```python
{
    "model_name": str,
    "model_config": dict,
    "model_state_dict": dict,
    "feature_dims": {"text": int, "audio": int, "vision": int},
    "epoch": int,
    "valid_mae": float,
    "seed": int,
}
```

以验证集 MAE 最低的 epoch 作为 `best.pt`。测试集只用于最终评测，不参与早停和模型选择。

### 3.6 配置接口

每个训练脚本接受同样的公共参数：

```powershell
python -m pj.scripts.train_fusion --config pj/configs/fusion/default.yaml --seed 2026 --device auto
```

公共 YAML 字段如下；模块可在自己的命名空间下增加字段，不得改变公共字段含义：

```yaml
seed: 2026
device: auto
data:
  feature_path: pj/data/processed/mosi_features.pkl
  split_path: pj/data/processed/splits.json
  batch_size: 32
  num_workers: 0
missingness:
  train_mode: random
  train_rate: 0.3
training:
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001
  patience: 8
model:
  name: missing_aware
output:
  checkpoint_dir: pj/checkpoints
  result_dir: pj/results
```

### 3.7 评测输出接口

每次“模型 × seed × 缺失场景”生成一个 JSON，路径为：

`pj/results/raw/<model_name>/<run_id>/<scenario>.json`

```json
{
  "schema_version": 1,
  "dataset": "CMU-MOSI",
  "split": "test",
  "model": "missing_aware",
  "run_id": "seed2026",
  "seed": 2026,
  "scenario": "random_0.3",
  "missing_rate": 0.3,
  "n_samples": 686,
  "mae": 0.0,
  "pearson": 0.0,
  "predictions": [
    {"id": "segment_id", "target": 0.0, "prediction": 0.0, "mask": [true, false, true]}
  ]
}
```

场景名固定为：`complete`、`drop_text`、`drop_audio`、`drop_vision`、`random_0.1`、`random_0.3`、`random_0.5`、`random_0.7`。汇总文件写入 `pj/results/summary.csv`，列固定为：

```text
model,run_id,seed,scenario,missing_rate,n_samples,mae,pearson
```

### 3.8 公平评测、实验追溯与 AI 使用

- 数据划分：以 `splits.json` 为唯一依据，一旦生成并由全组确认后冻结；所有模型使用相同 ID；
- 标签预算：只使用固定训练集的 100% 人工标签训练，验证集只用于早停和调参，测试标签只用于最终计算指标；不得引入外部标签、伪标签或人工挑选测试样本；
- 预训练来源：使用 MMSA 发布的 CMU-MOSI 预提取特征，不自行用测试集微调特征提取器；在报告中列出下载链接、特征名称、维度、文件校验值和访问日期；
- 调参纪律：所有模型先在验证集确定超参数并冻结，再统一运行测试矩阵；看过测试结果后不得回头选择模型或超参数；
- 实验清单：每个正式 run 保存完整 YAML、Git commit、随机种子、Python/PyTorch/CUDA 版本、设备型号、checkpoint 路径与开始/结束时间；
- 结果保留：正面和负面结果都保留，不删除不理想 seed；报告中的每个数字必须能追溯到 raw JSON；
- AI 使用：每人在 `pj/report/contributions/memberN.md` 记录 AI 工具、主要使用环节、采纳或修改内容、人工验证方法；关键代码必须能脱离 AI 解释并现场修改；
- 引用规范：数据、MMSA 特征、参考方法和复用代码均记录来源，不把 AI 输出本身当作事实依据。

## 4. 每位组员的详细任务与验收

### 组员 1：数据与缺失模拟

目标：把 MMSA 预提取的 CMU-MOSI 特征转换为统一 batch，并提供全组唯一的可复现缺失模拟实现。

交付内容：

- `pj/src/data/dataset.py`：加载特征、标签与固定划分，实现 `Dataset`；
- `pj/src/data/collate.py`：三种变长序列 padding 与 padding mask；
- `pj/src/data/missingness.py`：实现 3.3 节的 `apply_missingness`；
- `pj/src/data/transforms.py`：仅使用训练集统计量的标准化；
- `pj/scripts/prepare_data.py`：数据完整性检查、生成固定划分和统计量；
- `pj/configs/data.yaml`、`pj/tests/test_data.py`、`pj/docs/member1.md`；
- `pj/report/sections/01_task_data.md`：任务定义、数据来源、预训练特征来源、固定划分、标签预算和数据限制；
- `pj/report/slides/member1.md`：答辩中的任务/数据/输入输出部分；`pj/report/contributions/member1.md`：本人贡献与 AI 使用；
- smoke test 输出 `pj/results/smoke/member1_data.json`，记录样本数、维度、标签范围和各场景实测缺失率。

验收条件：同一 seed 重跑的 mask 完全一致；每个样本至少保留一种模态；三个 split 无 ID 交集；任何统计量均不泄漏测试集；下载来源、访问日期、文件校验值、特征维度和异常样本处理均有记录。答辩时能够解释一条原始样本如何变成模型 batch。

### 组员 2：三种单模态基线

目标：分别实现只使用 T/A/V 的三套可比较基线，作为信息量下限。

交付内容：

- `pj/src/models/unimodal.py`：共享时序池化组件和 `TextOnlyModel`、`AudioOnlyModel`、`VisionOnlyModel`；
- `pj/scripts/train_unimodal.py`：通过 `--modality text|audio|vision` 选择模型；
- `pj/configs/unimodal/text.yaml`、`audio.yaml`、`vision.yaml`；
- `pj/tests/test_unimodal.py`、`pj/docs/member2.md`；
- `pj/report/sections/02_unimodal_baselines.md`：三种基线、参数量、训练设置、对照结果和各模态适用条件；
- `pj/report/slides/member2.md`：答辩中的基线与比较思路；`pj/report/contributions/member2.md`：本人贡献与 AI 使用；
- 三个模型各完成一个小规模训练，结果写入 `pj/results/smoke/member2_{text,audio,vision}.json`。

验收条件：三个模型均严格满足 3.4 节接口；未选择的模态置换为随机值时预测不变；目标模态缺失时仍返回有限值且不报错；参数量与训练时间写入文档；三种模型使用相同训练预算。答辩时能够解释池化、回归头和单模态基线为何是必要对照。

### 组员 3：拼接基线与通用训练器

目标：实现早期特征拼接基线，并提供所有模型共用的训练、验证、早停和检查点逻辑。

交付内容：

- `pj/src/models/concat.py`：分别池化可用模态、按掩码置零后拼接并回归；
- `pj/src/training/trainer.py`：训练/验证循环、梯度裁剪、早停；
- `pj/src/training/checkpoint.py`：严格实现 3.5 节格式；
- `pj/src/training/seed.py`：固定 Python/NumPy/PyTorch 随机性；
- `requirements.txt` 或 `environment.yml`：冻结全组可复现环境；
- `pj/src/training/manifest.py`：保存 Git commit、配置、随机种子、设备和软件版本等 run manifest；
- `pj/scripts/train_concat.py`、`pj/configs/concat.yaml`；
- `pj/tests/test_training.py`、`pj/docs/member3.md`；
- `pj/report/sections/03_training_reproducibility.md`：拼接基线、公平训练预算、早停、调参与复现环境；
- `pj/report/slides/member3.md`：答辩中的训练流程与公平性；`pj/report/contributions/member3.md`：本人贡献与 AI 使用；
- smoke test 输出 `pj/results/smoke/member3_concat.json` 和一个可加载的本地检查点（检查点不提交）。

验收条件：能够用合成数据完成至少 2 个 epoch；保存后加载的预测一致；验证 MAE 改善时更新 best checkpoint；训练器可接收任意满足 3.4 节的模型；每次实验自动落盘完整 manifest。答辩时能够解释测试集为何不能参与早停或调参，并现场修改一个训练超参数。

### 组员 4：缺失感知融合模型

目标：实现一个显式利用 `modality_mask` 的自选融合模型，并通过消融证明掩码与融合机制的作用。

交付内容：

- `pj/src/models/missing_aware.py`：每模态编码器、mask-aware 门控/注意力融合、回归头；
- 缺失模态不得参与 softmax/门控归一化，且全流程避免 NaN；
- `pj/scripts/train_fusion.py`；
- `pj/configs/fusion/default.yaml`、`no_mask.yaml`、`no_gating.yaml`；
- `pj/tests/test_missing_aware.py`、`pj/docs/member4.md`；
- `pj/report/sections/04_missing_aware_method.md`：自选方法、关键公式/结构、消融实验、负面结果和适用条件；
- `pj/report/slides/member4.md`：答辩中的改进方法与消融；`pj/report/contributions/member4.md`：本人贡献与 AI 使用；
- smoke test 输出 `pj/results/smoke/member4_fusion.json`，包含完整输入和至少三种单模态缺失情况。

验收条件：满足 3.4 节接口；更改缺失模态的特征不会改变输出；融合权重在可用模态上归一化、缺失模态权重为 0；完成两项消融的最小可运行验证；无论是否涨分都用实验解释。答辩时能够解释并现场修改 mask-aware 融合的关键代码。

### 组员 5：评测、实验矩阵与可视化

目标：确保所有模型在同一批测试样本、同一组缺失掩码上评测，并自动生成可用于报告的表格和图。

交付内容：

- `pj/src/evaluation/metrics.py`：MAE 和 Pearson（处理零方差边界情况）；
- `pj/src/evaluation/evaluator.py`：加载检查点、推理、按 3.7 节落盘；
- `pj/src/evaluation/scenarios.py`：定义八种固定场景并复用组员 1 的缺失函数；
- `pj/src/evaluation/aggregate.py`：校验 JSON schema、汇总 CSV、均值/标准差；
- `pj/src/evaluation/plot.py`：完整/缺失对比柱状图及随机缺失率曲线；
- `pj/src/evaluation/case_analysis.py`：按误差与场景选择成功/失败案例，不允许手工只挑有利样本；
- `pj/scripts/evaluate.py`、`pj/scripts/run_experiments.py`、`pj/configs/eval.yaml`；
- `pj/tests/test_evaluation.py`、`pj/docs/member5.md`；
- `pj/report/sections/05_results_analysis.md`：核心图表、对照实验、成功/失败案例、结论、局限和适用条件；
- `pj/report/slides/member5.md`：答辩中的结果/案例/结论；`pj/report/contributions/member5.md`：本人贡献与 AI 使用；
- smoke test 输出 `pj/results/smoke/member5_evaluation.json`、`pj/results/summary.csv` 和 `pj/results/figures/smoke.png`。

验收条件：手工样例的 MAE/Pearson 正确；不同模型对同一场景使用完全相同的样本 ID 与 mask；缺文件或 schema 错误时给出明确报错；图表可由原始 JSON 一条命令重新生成；案例选择规则可复现且同时包含成功和失败样本。答辩时能够从图表追溯到逐样本预测并解释 Pearson 与 MAE 的差异。

## 5. Git 协作流程

首次开始工作：

```powershell
git fetch origin
git switch feat/memberN-模块名
git pull --ff-only origin feat/memberN-模块名
```

日常提交：

```powershell
git status
git add <仅添加本人负责的路径>
git commit -m "feat(module): 简述改动"
git push origin HEAD
```

提交信息使用 `feat`、`fix`、`test`、`docs`、`refactor` 前缀。每个人在自己的分支完成后向 `main` 提交 Pull Request，并在 PR 描述中附上：

- 交付文件清单；
- 复现命令；
- 测试结果；
- smoke test 结果路径；
- 报告章节、答辩提纲、本人贡献及 AI 使用记录；
- 尚未解决的问题。

禁止直接 push 到 `main`。不要在个人分支大范围格式化或移动他人目录。若必须使用尚未合并的其他模块，先依据本文接口用最小 mock 开发，等对应 PR 合并后再做联调。

## 6. 推荐合并与联调顺序

五人可同时开发；最终建议按 `组员 1 → 组员 3 → 组员 2 → 组员 4 → 组员 5` 的顺序合并。该顺序只用于降低联调成本，不表示开发存在串行依赖。

合并后由负责人执行统一实验矩阵：

- 模型：Text-only、Audio-only、Vision-only、Concat、Missing-aware；
- 随机种子：`2026, 2027, 2028`；
- 测试场景：3.7 节规定的八种场景；
- 主结果：每组设置报告 MAE/Pearson 的三次均值与标准差；
- 公平性：固定测试 ID、固定缺失 mask、相同数据归一化与评测代码。

## 7. 全项目完成标准

- 数据划分固定、无泄漏，README 中的命令可从干净环境运行；
- 五类模型均使用同一输入/输出接口并可被统一评测器加载；
- 完整输入、分别缺少 T/A/V、四档随机缺失均有结果；
- 原始逐样本预测、汇总 CSV 和图表之间可以相互追溯；
- MAE 与 Pearson 均按三次独立种子报告均值和标准差；
- 每个模型都报告公平对照；自选方法至少有 `no_mask`、`no_gating` 两项消融，负面结果不得隐去；
- 报告包含任务、数据、方法、结果、成功/失败案例、结论、局限、成员贡献、AI 使用和引用来源；
- 五位成员都能解释本人关键代码和实验选择，并准备至少一个可能的答辩问题及答案；
- 所有测试通过，仓库中不包含数据集、缓存或模型权重等大文件。

## 8. 决定项目质量的关键任务

按风险和对最终成绩的影响排序：

| 优先级 | 关键任务 | 主要负责人 | 为什么关键 | 失败后果 |
|---|---|---|---|---|
| P0 | 固定且无泄漏的数据管线 | 组员 1 | 所有结果的地基；划分、归一化或 mask 出错会污染全部模型 | 指标失真，整套实验需要重跑 |
| P0 | 公平、确定性的缺失场景 | 组员 1 + 组员 5 | 不同模型必须面对完全相同的样本与 mask 才能比较 | “改进”可能只是抽到了更容易的缺失样本 |
| P0 | 统一训练/记录/测试隔离 | 组员 3 | 防止测试集调参，并保证每个数字可由配置和 checkpoint 复现 | 结果无法自证，答辩容易被质疑 |
| P1 | 扎实的单模态与拼接基线 | 组员 2 + 组员 3 | 没有强且公平的基线，就无法说明自选融合是否有价值 | 即使新模型分数高，也缺乏可信对照 |
| P1 | 真正使用可用性掩码的融合与消融 | 组员 4 | 这是项目主题和主要技术亮点，必须证明增益来自哪里 | 项目退化为普通特征拼接，主题不突出 |
| P1 | 系统评测与成功/失败案例分析 | 组员 5 | 新要求明确看重证据、适用条件、负面结果和局限 | 只有数字没有结论，报告和答辩说服力弱 |

最值得投入时间的不是继续堆复杂模型，而是先守住三个底线：数据无泄漏、比较条件完全一致、每张表都能追溯。底线满足后，再优化组员 4 的融合方法。自选方法即使没有超过拼接基线，只要消融、失败案例和适用条件分析可信，仍然符合项目要求；相反，单次高分但划分不公平或无法复现，会严重降低项目质量。
