# TIGER: Generative Retrieval for Recommender Systems

本项目是论文 [Recommender Systems with Generative Retrieval](https://arxiv.org/pdf/2305.05065) 的 Pytorch 优化实现。通过将推荐任务转化为自回归序列生成，TIGER 能够直接生成物品的**语义 ID**，在 4GB 显存环境下展现出了超越原项目基准的性能。

## 🚀 项目亮点

* **性能突破**：在 Beauty 数据集上的复现结果全面超越原项目基准，**Recall@10 提升约 30%**。
* **现代环境栈**：基于 **Python 3.11** 和 **PyTorch 2.1.2** 构建，完美适配 Windows 11 VICTUS 的 WSL2 深度学习流。
* **端到端流程**：涵盖原始数据清洗、Sentence-T5 语义特征提取、RQ-VAE 残差量化及 T5 生成式训练。

---

## 🛠 环境配置 (Environment)

建议在 **WSL2 (Ubuntu)** 下运行。

* **硬件核心**：CPU 8G / GPU 8G (4G 显存环境下已通过调整 `batch_size` 验证可行)。
* **关键依赖版本**：
* `python`: 3.11.x
* `torch`: 2.1.2+cu121
* `transformers`: 4.57.3
* `numpy`: 1.24.3 (严格保持 < 2.0 以兼容 pyarrow)
* `pandas`: 1.5.3



---

## 📁 项目结构 (Project Structure)

> **重要标注**：原始下载的 `.gz` 压缩文件必须放置在 `data/` 根目录下，**严禁**放入 `data/Beauty/` 等子目录，否则 `process.ipynb` 将无法正确识别路径。

```text
.
├── data/
│   ├── reviews_Beauty_5.json.gz # 必须放在此处
│   ├── meta_Beauty.json.gz      # 必须放在此处
│   ├── Beauty/                  # 数据集输出目录
│   │   ├── item_emb.parquet     # Sentence-T5 提取的语义向量
│   │   ├── item_mapping.npy     # ID 映射表
│   │   ├── train.parquet        # 训练集序列
│   │   └── Beauty_t5_rqvae.npy  # 最终生成的离散码表
├── sentence-t5-base/            # 本地权重文件夹 (由 ModelScope 下载)
├── rqvae/                       # 阶段 1: 语义索引模块
│   ├── main.py                  # 训练 RQ-VAE 模型
│   └── generate_code.py         # 铸造离散码序列
└── model/                       # 阶段 2: 生成式推荐模块
    └── main.py                  # T5 核心训练脚本

```

---

## 📖 复现流程 (Workflow)

### 阶段 1：数据预处理 (Data Prep)

1. **解压与清洗**：运行 `data/process.ipynb`。脚本会读取 `data/*.gz` 文件并生成中间态 JSON。
2. **语义特征提取**：利用本地 `sentence-t5-base` 将商品标题、品牌、类目等文本编码为高维向量，存储于 `item_emb.parquet`。

### 阶段 2：构建语义 ID (RQ-VAE)

1. **训练残差量化器**：
```bash
cd rqvae
python main.py

```


> **实测记录**：`Best Collision Rate 0.06716` (Epoch 2999 收敛)。


2. **生成离散码表**：
```bash
python generate_code.py

```


> **输出结果**：`Collision Rate 0.0027`。通过增加额外维度消除了 ID 冲突，生成 4 层级语义 ID，如 `[244, 215, 47, 0]`。



### 阶段 3：生成式模型训练 (T5 Training)

1. **训练下一项预测**：
```bash
cd ../model
python main.py

```


* **逻辑**：将历史语义序列作为 Context，预测 Target ID 码元。
* **参数**：总可训练参数约 4.59M。



---

## 📊 实验结果 (Experimental Results)

在 **Beauty** 数据集上的复现表现：

| Metric | **本项目实现 (Ours)** | 原项目基准 (README) | 原论文 (Paper) |
| --- | --- | --- | --- |
| **Recall@5** | **0.0505** | 0.0392 | 0.0454 |
| **Recall@10** | **0.0772** | 0.0594 | 0.0648 |
| **NDCG@5** | **0.0329** | 0.0257 | 0.0321 |
| **NDCG@10** | **0.0414** | 0.0321 | 0.0384 |

> **结论**：本项目在 Python 3.11 环境下展现出更优的收敛质量，核心指标不仅对齐且部分超越了原论文。

---

## 💡 专家避坑指南 (Tips)

1. **PyTorch 2.6+ 适配**：加载 `.pth` 权重时必须显式设置 `torch.load(..., weights_only=False)`，否则会因拦截 `args` 自定义对象导致 UnpicklingError。
2. **显存压制策略**：在 4G 显存下，建议在 `model/main.py` 中将 `batch_size` 限制在 32 以下，或启用 `fp16` 混合精度训练。
3. **组合式表达能力**：TIGER 的核心优势在于 
$$Capacity = C^k$$


。本项目通过 4 层码簿实现了对万级物品的高效覆盖，同时通过 `generate_code.py` 的 Padding 逻辑解决了长尾物品的语义 ID 碰撞问题。

