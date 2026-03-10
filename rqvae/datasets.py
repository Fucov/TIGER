import pandas as pd
import torch
import torch.utils.data as data
import numpy as np


class EmbDataset(data.Dataset):
    """物品嵌入数据集。

    支持两种模式（自动检测）：
    - parquet 含 category_id 列 → __getitem__ 返回 (embedding, category_id)
    - parquet 无 category_id 列 → __getitem__ 返回 (embedding, -1)

    Trainer 通过 category_id == -1 判断是否退化为标准 InfoNCE。
    """

    def __init__(self, data_path):
        self.data_path = data_path
        df = pd.read_parquet(data_path)

        self.embeddings = np.stack(df["embedding"].values, axis=0)
        self.dim = self.embeddings.shape[-1]

        # 类目 ID（若 parquet 含该列则加载，否则全部填 -1）
        if "category_id" in df.columns:
            self.category_ids = df["category_id"].values.astype(np.int64)
        else:
            self.category_ids = np.full(len(self.embeddings), -1, dtype=np.int64)

    def __getitem__(self, index):
        emb = torch.FloatTensor(self.embeddings[index])
        cat_id = torch.LongTensor([self.category_ids[index]]).squeeze(0)
        return emb, cat_id

    def __len__(self):
        return len(self.embeddings)
