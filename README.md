# 模态缺失下的多模态情感分析

本仓库是机器学习课程项目，目标是在 CMU-MOSI 上研究文本（T）、音频（A）、视觉（V）模态缺失时的情感回归。模型输入 MMSA 预提取特征及模态可用性掩码，输出范围为 `[-3, 3]` 的情感分数；统一使用 MAE（越低越好）与 Pearson 相关系数（越高越好）评测。

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

### 工作量对齐原则

每位组员均须完成：

1. 一个可独立运行的核心模块（约 2–4 个源文件）；
2. 一个命令行入口和对应 YAML 配置；
3. 不少于 3 个单元测试，覆盖正常输入、模态缺失和非法输入；
4. 一份模块说明文档 `pj/docs/memberN.md`，包含运行命令、实现选择和已知限制；
5. 至少一次小规模 smoke test，并将日志或结果写入约定目录。

不要把数据集、模型权重、缓存或大体积结果提交到 Git。代码评审以接口兼容、测试通过和结果可复现为验收依据，不以代码行数衡量工作量。

## 2. 统一目录结构

```text
pj/
├── configs/                 # YAML 配置；按任务表分区维护
├── data/                    # 本地数据（不提交大文件）
│   ├── raw/
│   └── processed/
├── docs/                    # 每人一份 memberN.md
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
- smoke test 输出 `pj/results/smoke/member1_data.json`，记录样本数、维度、标签范围和各场景实测缺失率。

验收条件：同一 seed 重跑的 mask 完全一致；每个样本至少保留一种模态；三个 split 无 ID 交集；任何统计量均不泄漏测试集。

### 组员 2：三种单模态基线

目标：分别实现只使用 T/A/V 的三套可比较基线，作为信息量下限。

交付内容：

- `pj/src/models/unimodal.py`：共享时序池化组件和 `TextOnlyModel`、`AudioOnlyModel`、`VisionOnlyModel`；
- `pj/scripts/train_unimodal.py`：通过 `--modality text|audio|vision` 选择模型；
- `pj/configs/unimodal/text.yaml`、`audio.yaml`、`vision.yaml`；
- `pj/tests/test_unimodal.py`、`pj/docs/member2.md`；
- 三个模型各完成一个小规模训练，结果写入 `pj/results/smoke/member2_{text,audio,vision}.json`。

验收条件：三个模型均严格满足 3.4 节接口；未选择的模态置换为随机值时预测不变；目标模态缺失时仍返回有限值且不报错；参数量与训练时间写入文档。

### 组员 3：拼接基线与通用训练器

目标：实现早期特征拼接基线，并提供所有模型共用的训练、验证、早停和检查点逻辑。

交付内容：

- `pj/src/models/concat.py`：分别池化可用模态、按掩码置零后拼接并回归；
- `pj/src/training/trainer.py`：训练/验证循环、梯度裁剪、早停；
- `pj/src/training/checkpoint.py`：严格实现 3.5 节格式；
- `pj/src/training/seed.py`：固定 Python/NumPy/PyTorch 随机性；
- `pj/scripts/train_concat.py`、`pj/configs/concat.yaml`；
- `pj/tests/test_training.py`、`pj/docs/member3.md`；
- smoke test 输出 `pj/results/smoke/member3_concat.json` 和一个可加载的本地检查点（检查点不提交）。

验收条件：能够用合成数据完成至少 2 个 epoch；保存后加载的预测一致；验证 MAE 改善时更新 best checkpoint；训练器可接收任意满足 3.4 节的模型。

### 组员 4：缺失感知融合模型

目标：实现一个显式利用 `modality_mask` 的自选融合模型，并通过消融证明掩码与融合机制的作用。

交付内容：

- `pj/src/models/missing_aware.py`：每模态编码器、mask-aware 门控/注意力融合、回归头；
- 缺失模态不得参与 softmax/门控归一化，且全流程避免 NaN；
- `pj/scripts/train_fusion.py`；
- `pj/configs/fusion/default.yaml`、`no_mask.yaml`、`no_gating.yaml`；
- `pj/tests/test_missing_aware.py`、`pj/docs/member4.md`；
- smoke test 输出 `pj/results/smoke/member4_fusion.json`，包含完整输入和至少三种单模态缺失情况。

验收条件：满足 3.4 节接口；更改缺失模态的特征不会改变输出；融合权重在可用模态上归一化、缺失模态权重为 0；完成两项消融的最小可运行验证。

### 组员 5：评测、实验矩阵与可视化

目标：确保所有模型在同一批测试样本、同一组缺失掩码上评测，并自动生成可用于报告的表格和图。

交付内容：

- `pj/src/evaluation/metrics.py`：MAE 和 Pearson（处理零方差边界情况）；
- `pj/src/evaluation/evaluator.py`：加载检查点、推理、按 3.7 节落盘；
- `pj/src/evaluation/scenarios.py`：定义八种固定场景并复用组员 1 的缺失函数；
- `pj/src/evaluation/aggregate.py`：校验 JSON schema、汇总 CSV、均值/标准差；
- `pj/src/evaluation/plot.py`：完整/缺失对比柱状图及随机缺失率曲线；
- `pj/scripts/evaluate.py`、`pj/scripts/run_experiments.py`、`pj/configs/eval.yaml`；
- `pj/tests/test_evaluation.py`、`pj/docs/member5.md`；
- smoke test 输出 `pj/results/smoke/member5_evaluation.json`、`pj/results/summary.csv` 和 `pj/results/figures/smoke.png`。

验收条件：手工样例的 MAE/Pearson 正确；不同模型对同一场景使用完全相同的样本 ID 与 mask；缺文件或 schema 错误时给出明确报错；图表可由原始 JSON 一条命令重新生成。

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
- 所有测试通过，仓库中不包含数据集、缓存或模型权重等大文件。

