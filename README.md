# IMDB 电影评论情感分析：TF-IDF Baseline 与 BERT 微调

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-blue)]([https://www.kaggle.com/code/jiatyan/bert-bag-of-words-meets-bags-of-popcorn])

本仓库对应一个完整的 Kaggle Notebook，用于参加经典 NLP 竞赛 [Bag of Words Meets Bags of Popcorn](https://www.kaggle.com/c/word2vec-nlp-tutorial)。Notebook 中包含了从数据探索、简单基线模型到 BERT 微调的完整流程，并针对长文本处理、多 GPU 训练、混合精度加速等实际问题给出了解决方案。**Best Score0.97666**

## 📋 项目简介

- **任务**：对 IMDB 电影评论进行情感二分类（1 = 正面，0 = 负面）
- **评价指标**：AUC（Area Under ROC Curve）
- **数据**：25,000 条训练评论，25,000 条测试评论，评论文本可能包含 HTML 标签且长度较长
- **方案**：
  1. 基于 TF-IDF + 逻辑回归的快速基线
  2. 使用 HuggingFace Transformers 微调 BERT（`bert-base-uncased`）
  3. 多 GPU 训练（DataParallel）与混合精度加速
  4. 不依赖阈值调优（AUC 评估使用概率）

## 🔗 Notebook 地址

所有代码和详细说明均在一个 Kaggle Notebook 中：  
[https://www.kaggle.com/code/jiatyan/bert-bag-of-words-meets-bags-of-popcorn]

## 📁 Notebook 结构

Notebook 按顺序包含以下主要部分：

1. **数据探索分析（EDA）**
   - 情感标签分布（完美平衡）
   - 缺失值检查（无缺失）
   - 文本长度分布（字符数、单词数）
   - HTML 标签统计与清理
   - 正负面评论词云对比

2. **Baseline：TF-IDF + Logistic Regression**
   - 文本清洗（去除 HTML、标点，小写化）
   - TF-IDF 向量化（n-gram 1-2，最大特征 50000）
   - 逻辑回归（saga 求解器）与 5 折交叉验证
   - 生成二值预测提交文件

3. **BERT 微调**
   - 最小化文本清洗（仅去除 HTML 标签，保留标点和大写）
   - 自定义 `ReviewDataset` 与 DataLoader（包含训练、验证、测试集）
   - 加载 `bert-base-uncased` 并添加二分类头
   - 多 GPU 训练（`DataParallel`）与损失向量处理
   - 学习率调度（warmup + 线性衰减）
   - 混合精度训练（AMP）加速
   - 早停与最佳模型保存
   - 测试集预测概率并生成提交文件

4. **常见错误与解决方案**  
   Notebook 中注释详细，且针对以下问题给出了处理：
   - `AdamW` 导入错误
   - 多 GPU 下损失向量问题
   - 模型加载时键名不匹配
   - tqdm 进度条不更新

## 🚀 如何使用

### 在 Kaggle 上运行

1. 打开 Notebook 链接，点击 **Copy & Edit** 创建自己的副本。
2. 确保 Notebook 的 Accelerator 设置为 **GPU**（推荐 T4 x2）。
3. 依次运行所有单元格即可。最终会生成 `submission.csv` 文件，可直接提交。

### 在本地运行（需自行调整路径）

若想在本地环境运行，需要安装依赖：

```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch transformers tqdm
```

并将数据文件放在指定路径（或修改代码中的路径）。

## 🧠 主要技术细节

### 1. 文本预处理
- **Baseline**：去除 HTML 标签、标点，只保留字母和数字，小写化。
- **BERT**：仅去除 HTML 标签，保留其他所有字符（包括标点、大小写），以充分利用 BERT 的 tokenizer。

### 2. 长文本处理
- 统计显示评论平均约 231 词，最大可达 2470 词，但 BERT 最大输入长度为 512。
- 采用截断策略：设置 `MAX_LEN = 256`（或 512），超出部分直接丢弃，保留前部信息。  
  对于本任务，截断前 256 token 已能取得不错效果（验证 AUC > 0.97）。

### 3. 多 GPU 训练
- 使用 `nn.DataParallel` 包装模型，自动将数据拆分到两块 T4 GPU 上并行计算。
- **损失向量处理**：DataParallel 会返回每个 GPU 的标量损失组成向量，需调用 `.mean()` 转为标量再反向传播。
- **模型保存**：保存时使用 `model.module.state_dict()` 去除 `module.` 前缀，便于后续单卡加载。

### 4. 混合精度训练（AMP）
- 利用 T4 的 Tensor Cores，加速训练并减少显存占用。
- 使用 `torch.cuda.amp.autocast` 和 `GradScaler`，代码改动小，通常能提速 30%~50%。

### 5. 优化器与学习率调度
- 优化器：PyTorch 自带的 `AdamW`（注意不是从 transformers 导入）
- 学习率：2e-5，配合线性预热（前 10% 步数）和线性衰减
- 批量大小：16（可调大至 32 或 64，但需注意显存）

### 6. 早停与模型选择
- 监控验证集 AUC，若连续 1 个 epoch 无提升则提前停止。
- 保存验证 AUC 最高的模型用于最终预测。

## 📊 结果

- **TF-IDF + Logistic Regression**：交叉验证 AUC 约 0.88~0.90
- **BERT 微调**：验证 AUC 可达 0.97 以上（公开测试集 AUC 通常也在 0.95+）

## ❗ 常见问题

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| 训练速度慢 | 批量太小、未使用混合精度、num_workers 不足 | 增大 batch size（注意同时调整学习率）、启用 AMP、设置 `num_workers=2~4` |
| `ImportError: cannot import name 'AdamW' from 'transformers'` | 新版 transformers 移除了 AdamW | 改用 `from torch.optim import AdamW` |
| `RuntimeError: grad can be implicitly created only for scalar outputs` | DataParallel 返回向量损失 | 获取 loss 后添加 `loss = loss.mean()` |

## 🙏 致谢

- Kaggle 提供竞赛平台与数据
- HuggingFace 提供 Transformers 库
- 数据集来源于 IMDB，由竞赛组织者整理发布

---

欢迎 Star 和 Fork，如有问题请提 Issue。
