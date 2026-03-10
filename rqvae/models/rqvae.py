import torch
import torch.nn.functional as F
from torch import nn

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer


class RQVAE(nn.Module):
    def __init__(
        self,
        in_dim=768,
        # num_emb_list=[256,256,256,256],
        num_emb_list=None,
        e_dim=64,
        # layers=[512,256,128],
        layers=None,
        dropout_prob=0.0,
        bn=False,
        loss_type="mse",
        quant_loss_weight=1.0,
        beta=0.25,
        kmeans_init=False,
        kmeans_iters=100,
        # sk_epsilons=[0,0,0.003,0.01]],
        sk_epsilons=None,
        sk_iters=100,
    ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(
            layers=self.encode_layer_dims, dropout=self.dropout_prob, bn=self.bn
        )

        self.rq = ResidualVectorQuantizer(
            num_emb_list,
            e_dim,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
        )

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(
            layers=self.decode_layer_dims, dropout=self.dropout_prob, bn=self.bn
        )

    def forward(self, x, use_sk=True):
        z = self.encoder(x)  # 连续编码向量（用于对比学习）
        x_q, rq_loss, indices = self.rq(z, use_sk=use_sk)
        out = self.decoder(x_q)

        return out, rq_loss, indices, z, x_q  # 额外返回 z 和 x_q 用于 InfoNCE

    @staticmethod
    def cl_loss(
        z: torch.Tensor,
        z_q: torch.Tensor,
        labels: torch.Tensor = None,
        temperature: float = 0.07,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """类目感知监督对比学习损失 (Category-aware SCL)。

        当提供 labels（类目 ID）时，同类目物品不参与负样本排斥，
        保护 TIGER 语义 ID 的层次前缀共享结构。
        当 labels=None 或全为 -1 时，退化为标准 InfoNCE。

        公式：
        L = -1/N Σᵢ log[ exp(sim(zᵢ,z_qᵢ)/τ) /
                         ( exp(sim(zᵢ,z_qᵢ)/τ) + Σⱼ≠ᵢ Mᵢⱼ · exp(sim(zᵢ,z_qⱼ)/τ) ) ]

        其中 Mᵢⱼ = 1 当 Category(i) ≠ Category(j)，否则为 0。

        Args:
            z:    (N, D) encoder 连续输出
            z_q:  (N, D) 量化后的离散表示
            labels: (N,) 类目 ID 整数张量（-1 表示无类目信息）
            temperature: 温度系数
            epsilon: 分母保护值，防止全同类目时 log(0)

        Returns:
            scalar loss
        """
        N = z.size(0)

        # ---- Step 1: L2 归一化 ----
        z = F.normalize(z.float(), dim=-1)  # (N, D)
        z_q = F.normalize(z_q.float(), dim=-1)  # (N, D)

        # ---- Step 2: 相似度矩阵 ----
        # sim_matrix[i, j] = cos(zᵢ, z_qⱼ) / τ
        sim_matrix = torch.matmul(z, z_q.T) / temperature  # (N, N)

        # ---- Step 3: 构建类目 Mask 矩阵 ----
        # neg_mask[i, j] = True  →  j 是 i 的有效负样本（不同类目）
        # neg_mask[i, j] = False →  j 与 i 同类目，不参与排斥
        if labels is not None and not (labels < 0).all():
            # same_cat[i, j] = True 当 labels[i] == labels[j]
            same_cat = torch.eq(
                labels.unsqueeze(1),  # (N, 1)
                labels.unsqueeze(0),  # (1, N)
            )  # (N, N) bool

            # 负样本 mask = 不同类目 AND 不是自己
            eye_mask = torch.eye(N, dtype=torch.bool, device=z.device)
            neg_mask = torch.logical_not(same_cat) & torch.logical_not(eye_mask)
            # neg_mask: (N, N) True = 有效负样本
        else:
            # 无类目信息 → 退化为标准 InfoNCE：除自身外全是负样本
            neg_mask = torch.logical_not(
                torch.eye(N, dtype=torch.bool, device=z.device)
            )

        # ---- Step 4: 计算 SCL Loss ----
        # 正样本 logit = 对角线
        pos_logits = torch.diag(sim_matrix)  # (N,)

        # 负样本 logit：被 mask 掉的位置填 -inf（不参与 softmax 分母）
        neg_logits = sim_matrix.clone()
        neg_logits[~neg_mask] = float("-inf")  # 同类目 + 自身位置 → -inf

        # 分母 = exp(正样本) + Σ_有效负样本 exp(负样本)
        # 为了数值稳定，先减去 max
        all_logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)  # (N, 1+N)
        log_denom = torch.logsumexp(all_logits, dim=1)  # (N,)

        # ---- Step 5: 边界保护 ----
        # 如果某行全是 -inf（整个 batch 同类目），logsumexp 退化为 pos_logits 本身
        # 此时 loss_i = pos - pos = 0，是合理的（无需排斥任何人）
        # 但为安全起见加 epsilon 防止极端数值
        loss = -(pos_logits - log_denom)  # (N,)
        loss = loss.mean()

        return loss

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, use_sk=use_sk)
        return indices

    @torch.no_grad()
    def project_new_item(self, emb: torch.Tensor, use_sk: bool = False) -> torch.Tensor:
        """为新物品（冷启动）直接推理语义 ID，无需重训。

        Args:
            emb: (N, in_dim) Sentence-T5 语义向量
            use_sk: 是否启用 Sinkhorn 均匀分配（默认关闭以保持一致性）

        Returns:
            (N, num_quantizers) 整数码字张量
        """
        return self.get_indices(emb, use_sk=use_sk)

    def compute_loss(self, out, quant_loss, xs=None):

        if self.loss_type == "mse":
            loss_recon = F.mse_loss(out, xs, reduction="mean")
        elif self.loss_type == "l1":
            loss_recon = F.l1_loss(out, xs, reduction="mean")
        else:
            raise ValueError("incompatible loss type")

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon
