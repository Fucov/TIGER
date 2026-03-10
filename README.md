# TIGER: Generative Retrieval for Recommender Systems

本项目是论文 [Recommender Systems with Generative Retrieval (NeurIPS 2023)](https://arxiv.org/pdf/2305.05065) 的 PyTorch 优化复现，并在原始架构基础上引入了 **InfoNCE 对比学习** 等改进。通过将推荐任务转化为自回归序列生成，TIGER 能够直接生成物品的**语义 ID**，在 4GB 显存环境下展现出了超越原项目基准的性能。

## 🚀 项目亮点

* **性能突破**：Beauty 数据集 Recall@10 达 **0.0772**，超越原项目基准约 **30%**。
* **类目感知对比学习 SCL（新增）**：在 RQ-VAE 阶段引入 Category-aware Supervised Contrastive Learning，同类目物品不参与负样本排斥，保护语义 ID 的层次前缀共享结构。
* **完整文件日志**：三个训练阶段均写入带时间戳的 `.log` 文件，方便实验追踪。
* **冷启动接口（新增）**：新物品无需重训 RQ-VAE，通过 `project_new_item()` 直接推理分配语义 ID。
* **现代环境栈**：基于 Python 3.11 + PyTorch 2.1.2，适配 WSL2 深度学习流。

---

## 🛠 环境配置

建议在 **WSL2 (Ubuntu)** 下运行。

* **硬件**：CPU 8G / GPU ≥ 4GB 显存（已验证，通过调整 `batch_size` 适配）
* **关键依赖**：

| 包 | 版本 |
|---|---|
| `python` | 3.11.x |
| `torch` | 2.1.2+cu121 |
| `transformers` | 4.57.3 |
| `numpy` | 1.24.3（严格 < 2.0 兼容 pyarrow）|
| `pandas` | 1.5.3 |

```bash
pip install -r requirements.txt
```

---

## 📁 项目结构

> **重要**：原始 `.gz` 压缩文件必须放在 `data/` 根目录，**不能放入子目录**，否则 `process.ipynb` 无法识别。

```text
.
├── data/
│   ├── reviews_Beauty_5.json.gz    # 必须放在此处
│   ├── meta_Beauty.json.gz         # 必须放在此处
│   ├── add_category.py             # 类目提取脚本（追加 category_id 到 parquet）
│   └── Beauty/
│       ├── item_emb.parquet        # Sentence-T5 语义向量 + category_id
│       ├── category_mapping.npy    # 类目名称 → 整数 ID 映射
│       ├── item_mapping.npy        # ID 映射表
│       ├── train/valid/test.parquet
│       └── Beauty_t5_rqvae.npy    # 最终语义 ID 码表（4 层级）
├── sentence-t5-base/               # 本地权重（ModelScope 下载）
├── rqvae/                          # 阶段一：语义 ID 构建
│   ├── models/
│   │   ├── rqvae.py               # RQ-VAE + Category-aware SCL 对比学习
│   │   ├── rq.py                  # 残差向量量化器
│   │   ├── vq.py                  # 单层向量量化器
│   │   └── layers.py              # MLP、K-Means、Sinkhorn
│   ├── main.py                    # 训练 RQ-VAE
│   ├── generate_code.py           # 铸造并去冲突化语义 ID
│   ├── trainer.py                 # 训练控制器
│   ├── datasets.py                # 数据加载
│   ├── utils.py                   # 工具函数
│   └── logs/                      # 自动生成的训练日志
└── model/                         # 阶段二：生成式推荐
    ├── main.py                    # TIGER（T5）训练与评估
    ├── dataset.py                 # 物品→码字映射、序列构造
    ├── dataloader.py              # Collate 展平为 token 流
    └── logs/                      # 自动生成的训练日志
```

---

## 📖 复现流程

### 阶段 0：数据预处理

```bash
jupyter notebook data/process.ipynb
```

脚本读取 `data/*.gz`，完成数据清洗、Sentence-T5 特征提取，输出 `item_emb.parquet`。

**追加类目信息**（仅首次需要）：

```bash
cd data
python add_category.py
```

脚本自动从 `meta_Beauty.json.gz` 提取二级类目，给 `item_emb.parquet` 追加 `category_id` 列。

---

### 阶段 1：训练 RQ-VAE（构建语义 ID）

```bash
cd rqvae
python main.py
```

**关键可调参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--epochs` | 3000 | 训练轮数 |
| `--batch_size` | 1024 | 批大小 |
| `--num_emb_list` | 256 256 256 | 各层码本大小 |
| `--cl_weight` | **0.1** | Category-aware SCL 权重（0 = 禁用）|
| `--temperature` | **0.07** | SCL 温度系数 |
| `--log_dir` | ./logs | 日志输出目录 |

日志自动写入 `rqvae/logs/rqvae_<timestamp>.log`。

> **实测记录**：`Best Collision Rate 0.06716`（Epoch 2999 收敛）。

---

### 阶段 2：铸造语义 ID 码表

修改 `generate_code.py` 中 `ckpt_path` 指向最新检查点，然后：

```bash
python generate_code.py
```

日志自动写入 `rqvae/logs/generate_<timestamp>.log`，记录碰撞率和输出路径。

> **实测结果**：`Collision Rate 0.0027`。通过 Sinkhorn 去冲突 + Padding 第 4 维，生成 4 层级唯一语义 ID，例：`[244, 215, 47, 0]`。

---

### 阶段 3：训练生成式推荐模型（T5）

```bash
cd ../model
python main.py
```

**关键可调参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--batch_size` | 256 | 训练批大小 |
| `--num_epochs` | 200 | 最大训练轮数 |
| `--beam_size` | 30 | Beam Search 候选数 |
| `--early_stop` | 10 | NDCG@20 不提升时的早停轮数 |
| `--log_path` | ./logs/tiger.log | 日志路径（目录自动创建）|

日志同时输出到文件和控制台。

---

## 📊 实验结果

在 **Beauty** 数据集上：

| Metric | **本项目（Ours）** | 原项目基准 | 原论文 |
|---|---|---|---|
| **Recall@5** | **0.0505** | 0.0392 | 0.0454 |
| **Recall@10** | **0.0772** | 0.0594 | 0.0648 |
| **NDCG@5** | **0.0329** | 0.0257 | 0.0321 |
| **NDCG@10** | **0.0414** | 0.0321 | 0.0384 |

> 结论：本项目在 Python 3.11 环境下展现出更优的收敛质量，核心指标全面超越原论文。

---

## 🔬 改进设计

### Category-aware Supervised Contrastive Learning (SCL)

标准 InfoNCE 将 batch 内所有其他物品视为负样本，会破坏同类目物品共享语义 ID 前缀的层次结构。

**本项目改进**：引入类目 Mask 矩阵，同类目物品不参与负样本排斥：

$$\mathcal{L}_{SCL} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\text{sim}(z_i, z_{q,i})/\tau)}{\exp(\text{sim}(z_i, z_{q,i})/\tau) + \sum_{j \neq i} M_{ij} \cdot \exp(\text{sim}(z_i, z_{q,j})/\tau)}$$

其中 $M_{ij} = 1$ 当 $\text{Category}(i) \neq \text{Category}(j)$，否则为 $0$。

**实现特点**：
- 纯矩阵运算（`torch.eq` + `logical_not`），无 for 循环
- 自动退化：无 `category_id` 列时等价于标准 InfoNCE
- 边界保护：全同类目 batch 时分母 `logsumexp` 自然退化为正样本项，loss = 0
- 通过 `--cl_weight` 控制叠加权重（设为 0 可完全禁用）

### 冷启动接口

新物品有 Sentence-T5 向量时，无需重训 RQ-VAE：

```python
ckpt = torch.load("rqvae/ckpt/.../best_collision_model.pth", weights_only=False)
model = RQVAE(**ckpt['args'].__dict__).eval()
model.load_state_dict(ckpt['state_dict'])

new_emb = torch.tensor(...)               # (N, 768) Sentence-T5 向量
codes   = model.project_new_item(new_emb) # (N, num_quantizers)
```

---

## 💡 避坑指南

1. **PyTorch 2.6+ 适配**：加载 `.pth` 权重时须显式设置 `torch.load(..., weights_only=False)`。
2. **显存压制**：4G 显存下 `--batch_size` 建议 ≤ 32，或启用 `fp16` 混合精度。
3. **语义 ID 容量**：码本容量 $= C^k = 256^3 = 16.7M$，通过 4 层级实现万级物品高效覆盖。
