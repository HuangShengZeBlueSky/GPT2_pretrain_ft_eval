代码文件太大啦，放在压缩包里，包含所有代码，数据，可以跟着教程/问ai来运行：
https://cloud.tsinghua.edu.cn/d/685ff55a55da462fac97/

## 技术报告（仓库功能、原理、输入输出、运行示例）

### 1. 功能概览

本仓库实现了一个简化版 GPT-2 训练体系，并基于同一套模型与嵌入进行多项下游任务实验：

- 迷你 GPT 训练（Shakespeare 语料）
- 释义识别（Quora Paraphrase）
- 情感分类（SST / CFIMDB）
- 诗歌（sonnet）生成

适用场景：课程作业/教学型 Transformer 训练、文本生成与下游分类任务。

---

### 2. 核心原理与结构

#### 2.1 GPT-2 简化模型结构

- 词嵌入 + 位置嵌入 → 多层 GPT2Layer → LayerNorm → 输出隐藏状态。
- GPT2Layer 使用 Pre-LN 结构：先 LayerNorm，再注意力/前馈网络，每层均有残差与 dropout。
- 注意力为因果自注意力（下三角 mask），并叠加 padding mask。
- 词嵌入权重与输出层权重共享（weight tying）。

#### 2.2 训练与下游任务原理

- 语言模型训练：自回归预测下一个 token，损失为交叉熵（CE）。
- 释义识别（Quora）：取句对表示 → 分类器判断是否语义等价。
- 情感分类（SST / CFIMDB）：取序列末端或池化表示 → 二分类/多分类头。
- Sonnet 生成：用自回归解码采样生成文本。

---

### 3. 输入与输出

#### 3.1 输入数据位置

- 语言模型训练：`downstream-tasks/data/tinyshakespeare.txt`
- 释义识别：`downstream-tasks/data/quora-*.csv`
- 情感分类：`downstream-tasks/data/ids-sst-*.csv`、`downstream-tasks/data/ids-cfimdb-*.csv`
- Sonnet：`downstream-tasks/data/sonnets*.txt`

#### 3.2 输出产物

- 训练模型权重：`best_model_*.pt`、`downstream-tasks/*.pt`
- 预测结果：`downstream-tasks/predictions/*.csv`
- 生成文本：`downstream-tasks/predictions/generated_sonnets.txt`

---

### 4. 代码结构说明

- `models/`：GPT-2 模型定义（base_gpt、gpt2）
- `modules/`：注意力、GPT2Layer 等核心模块
- `train_mini_gpt.py`：自回归语言模型训练入口
- `downstream-tasks/`：释义识别、情感分类、sonnet 生成等任务脚本
- `optimizer.py`：自定义优化器
- `config.py`：统一超参配置入口
- `utils.py`：通用工具与辅助函数

---

### 5. 环境与依赖

建议使用 conda（项目内提供 `env.yml` 与 `setup.sh`）。

```bash
# 创建环境
conda env create -f env.yml
conda activate <env_name>

# 或使用 setup.sh
bash setup.sh
```

---

### 6. 运行示例

#### 6.1 迷你 GPT 训练（Shakespeare）

```bash
python train_mini_gpt.py --use_gpu
```

#### 6.2 释义识别（Quora）

```bash
python downstream-tasks/paraphrase_detection.py --use_gpu
```

#### 6.3 情感分类（SST + CFIMDB）

```bash
python downstream-tasks/sentiment_classifier.py --use_gpu
```

#### 6.4 Sonnet 训练与生成

```bash
python downstream-tasks/sonnet_generation.py --use_gpu
```

---

### 7. 数据格式说明

#### 7.1 Quora Paraphrase（CSV）

- `quora-train.csv`、`quora-dev.csv`、`quora-test-student.csv`
- 常见字段：`sentence1`, `sentence2`, `label`（脚本内解析）

#### 7.2 SST / CFIMDB

- `ids-sst-*.csv`、`ids-cfimdb-*.csv`
- 通常包含 `id` 与 `label`，文本由 ID 映射到原始数据（脚本内处理）

#### 7.3 Sonnet

- 文本文件，一行一个诗句或一首诗段落

---

### 8. 配置与调参

- 统一参数建议从 `config.py` 修改。
- 常见项：`batch_size`、`lr`、`n_layer`、`n_head`、`n_embd`、`max_seq_len`。
- 显存不足：减小 `batch_size` 或 `max_seq_len`。

---

### 9. 评估指标

- 语言模型：Perplexity / CE loss（脚本输出）
- 释义识别 / 情感分类：Accuracy / F1（依赖脚本实现）
- 生成任务：人工阅读 + 保存文本（`generated_sonnets.txt`）

---

### 10. 常见问题

- **权重加载失败**：确认路径与模型结构一致。
- **GPU 显存不足**：降低 batch_size，或使用 CPU 运行。
- **输出为空**：检查数据路径、文件名与脚本参数。

---

## Acknowledgement

This project is adapted from the Stanford CS 224N final project: https://github.com/cfifty/public_cs224n_gpt .
