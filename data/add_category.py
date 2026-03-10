"""从 meta_Beauty.json.gz 提取二级类目，追加 category_id 列到 item_emb.parquet。

用法:
    cd data
    python add_category.py

依赖:
    - data/meta_Beauty.json.gz（原始元数据）
    - data/Beauty/item_mapping.npy（ASIN → 整数 ItemID 映射）
    - data/Beauty/item_emb.parquet（现有嵌入文件）

输出:
    - data/Beauty/item_emb.parquet（追加 category_id 列，原地覆写）
    - data/Beauty/category_mapping.npy（类目名称 → 整数 ID 映射）
"""

import gzip
import numpy as np
import pandas as pd

# ============================================================
# 配置
# ============================================================
DATASET = "Beauty"
META_GZ = f"meta_{DATASET}.json.gz"
ITEM_MAPPING_PATH = f"./{DATASET}/item_mapping.npy"
EMB_PARQUET_PATH = f"./{DATASET}/item_emb.parquet"
CATEGORY_MAPPING_PATH = f"./{DATASET}/category_mapping.npy"

# 类目层级：取 categories[0] 的第几级作为 category_id（0-indexed）
# level=1 → 二级类目（如 Makeup, Fragrance, Hair Care）
CATEGORY_LEVEL = 1

# ============================================================
# 1. 加载 item_mapping（ASIN → 整数 ItemID）
# ============================================================
item_mapping = np.load(ITEM_MAPPING_PATH, allow_pickle=True).item()
# item_mapping: {'7806397051': 1, '9759091062': 2, ...}

# ============================================================
# 2. 读取 meta 文件，提取每个 item 的二级类目
# ============================================================
item_category_raw = {}  # {ItemID(int): category_name(str)}

with gzip.open(META_GZ, "rt", encoding="utf-8") as f:
    for line in f:
        meta = eval(line.strip())  # 原始 Amazon meta 格式
        asin = meta.get("asin", "")
        if asin not in item_mapping:
            continue

        item_id = item_mapping[asin]

        # 提取类目：categories 格式为 [['Beauty', 'Makeup', 'Face', ...]]
        cats = meta.get("categories", [[]])
        if cats and len(cats[0]) > CATEGORY_LEVEL:
            cat_name = cats[0][CATEGORY_LEVEL]
        else:
            cat_name = "Unknown"

        item_category_raw[item_id] = cat_name

# ============================================================
# 3. 构建 category_name → category_id 映射
# ============================================================
unique_cats = sorted(set(item_category_raw.values()))
cat_to_id = {name: idx for idx, name in enumerate(unique_cats)}
print(f"发现 {len(cat_to_id)} 个二级类目: {unique_cats}")

# 保存映射表
np.save(CATEGORY_MAPPING_PATH, cat_to_id)
print(f"类目映射已保存: {CATEGORY_MAPPING_PATH}")

# ============================================================
# 4. 追加 category_id 列到 item_emb.parquet
# ============================================================
df = pd.read_parquet(EMB_PARQUET_PATH)
print(f"原始 parquet 列: {df.columns.tolist()}, shape: {df.shape}")

# 为每个 ItemID 分配 category_id（缺失类目标记为 -1）
df["category_id"] = df["ItemID"].map(
    lambda x: cat_to_id.get(item_category_raw.get(x, "Unknown"), -1)
)

# 统计
print(f"category_id 分布:\n{df['category_id'].value_counts().head(20)}")
print(f"缺失类目数量: {(df['category_id'] == -1).sum()}")

# 原地覆写
df.to_parquet(EMB_PARQUET_PATH, index=False)
print(f"已追加 category_id 列并保存: {EMB_PARQUET_PATH}")
print(f"最终列: {df.columns.tolist()}, shape: {df.shape}")
