# 中文新闻情感分类与评估

**Chinese Sentiment Classification & Evaluation**

基于 **jieba + FastText + CNN** 的中文新闻标题三分类课程项目，涵盖数据预处理、重复实验、多指标评估与 PyQt 桌面展示。本仓库为团队课程项目的匿名化作品集版本。

`TensorFlow / Keras` · `scikit-learn` · `PyQt5` · `NLP` · `Model Evaluation`

![Historical experiment comparison](assets/experiment-comparison.svg)

## 项目概览

| 内容 | 说明 |
|---|---|
| 任务 | 判断新闻标题为积极、消极或中性 |
| 历史数据 | 原始 7,200 条；按标题去重后 7,067 条 |
| 表示 | jieba 分词，300 维 FastText 词向量，可训练 Embedding |
| 模型 | 单层一维卷积、全局最大池化、全连接三分类 |
| 实验 | A/B/C 三组配置，Notebook 设置每组重复 5 次 |
| 评估 | Accuracy、Macro Precision / Recall / F1、OVR ROC-AUC |
| 界面 | 数据及词向量选择、参数配置、训练日志、最终结果展示 |

## 从数据到评价

```mermaid
flowchart LR
    A[带标签新闻标题] --> B[去重与文本清洗]
    B --> C[分层划分 80/10/10]
    C --> D[仅训练集进行增强]
    D --> E[jieba 分词与词表]
    E --> F[300 维 Embedding]
    F --> G[Conv1D 与最大池化]
    G --> H[三类 Softmax]
    H --> I[分类指标与 ROC]
```

模型层次：`Embedding → SpatialDropout1D → Conv1D → BatchNorm → ReLU → Dropout → GlobalMaxPooling1D → Dense → Dropout → Softmax`。

## 历史实验结果

下表由保留的历史演示运行画面转录，**不是本次发布时重新训练的结果**。原始完整数据、模型权重和逐样本预测未随仓库发布，因此此处不是可直接复现的 benchmark。

| 配置 | Accuracy | Macro-F1 | Macro-AUC |
|---|---:|---:|---:|
| A | 73.24% ± 0.80 pp | 66.51% ± 1.14 pp | 0.8521 ± 0.0087 |
| B | **74.20% ± 0.53 pp** | 65.28% ± 1.81 pp | 0.8510 ± 0.0066 |
| C | 72.31% ± 1.50 pp | **66.54% ± 1.57 pp** | **0.8575 ± 0.0076** |

`±` 为重复运行的标准差；pp 为百分点。粗体仅表示表内均值最高，不代表统计显著。

- B 的总体准确率最高，但 Macro-F1 未领先。
- A/C 的 Macro-F1 非常接近，不能据此宣称 C 显著更优。
- 历史 C 第 5 次运行的分类型 AUC：negative 0.91、neutral 0.79、positive 0.89；中性类别仍需重点分析。

完整参数、结果来源和限制见 [实验说明](docs/experiments.md)；精确转录数值见 [CSV](results/historical_metrics.csv)。

## 本项目展示的能力

- **数据处理**：分层划分、训练集增强、训练词表构建、类别加权。
- **模型评价**：结合整体表现、类别均衡表现与概率排序能力解读实验。
- **实验组织**：多组配置与重复运行，汇总均值及标准差。
- **流程展示**：通过桌面界面连接数据选择、训练与最终评价。
- **复盘与改进**：识别标签冲突、评测可追溯性和脚本/界面配置一致性问题。

## 目录

```text
src/          PyQt 桌面原型与 UI 文件
notebooks/    历史训练、评估 Notebook（已清除输出和本机路径）
data/         数据格式说明与人工编写的格式示例
results/      匿名历史指标转录
assets/       由指标重新绘制的比较图
docs/         实验设计、限制与后续改进
scripts/      示例数据转换、指标绘图与静态检查
```

## 运行方式

这是一套使用 Keras 2 接口的历史代码。建议从 **Python 3.10 + TensorFlow 2.15 / Keras 2.15 的 CPU 环境**尝试运行。`requirements.txt` 是整理后的兼容性目标，尚未在干净环境完成全部安装与端到端验证。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**准备数据：** 将含 `text`、`label` 两列的数据放到 `data/sentiment.xlsx`。标签必须为 `negative`、`neutral`、`positive`。原始数据及大型词向量不包含在仓库中，见 [数据说明](data/README.md)。

**打开桌面原型：**

```bash
python src/main.py
```

选择 Excel 数据集、可选 FastText `.vec` 文件和 A/B/C 配置，然后开始训练。需要图形桌面环境。原型已知限制见下文，不承诺所有按钮/实时曲线均可用。

**运行 Notebook：**

```bash
# 在仓库根目录执行
export SENTIMENT_DATASET="$PWD/data/sentiment.xlsx"
export FASTTEXT_VECTORS="$PWD/data/cc.zh.300.vec"
jupyter notebook notebooks/assess.ipynb
```

Notebook 使用预训练词向量，必须先准备相应文件。`train.ipynb` 汇总准确率，`assess.ipynb` 汇总完整指标并保存 ROC。输出目录为 `outputs/`。不要用示例数据推导历史模型表现。

原生 Windows GPU 的旧环境与上述 CPU 建议环境不同。CUDA/cuDNN 的选择请参照 [TensorFlow 官方安装说明](https://www.tensorflow.org/install/pip)，不要直接混用最新依赖。

## 已知限制与后续方向

1. **数据质量**：复核原始数据发现 16 组同标题标签冲突；原流程直接保留首条，未做冲突裁决。清洗也可能损失数字、英文等信息。
2. **实验归因**：三组同时修改多个超参数，属于组合比较，不是单因素消融。
3. **可复现性**：原始数据、完整预测与 checkpoint 未发布；原代码未统一固定所有随机源。
4. **实现差异**：Notebook 使用 class_weight 并监控 val_accuracy；GUI 未传入 class_weight，监控 val_loss。不能假定两条路径结果相同。
5. **界面原型**：停止标记未接入训练中止；实时指标回调的数据传递存在缺口；ROC 计算与 UI 显示没有完整连接。

可继续推进：统一训练/评估模块，保存模型与 tokenizer，建立数据版本和逐样本错误分析，完善线程通信及取消训练，增加独立测试与基线模型对照。上述为改进方向，非已实现功能。

## 发布整理范围

保留课程实现与实验参数；清除 Notebook 输出/非必要元数据，将硬编码路径改为环境变量；补充文档、格式示例和指标图。没有重新训练模型，也没有将尚未实现的工程改进包装为历史成果。

本仓库不包含人员名单、个人分工、聊天、简历或未经处理的演示文件。第三方数据和模型资源遵循各自的来源与使用条件。
