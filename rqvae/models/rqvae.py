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
        z: torch.Tensor, z_q: torch.Tensor, temperature: float = 0.07
    ) -> torch.Tensor:
        """InfoNCE Contrastive Loss.

        正样本对：同一物品的 z（encoder 输出）与 z_q（量化输出）。
        负样本对：batch 内其他物品的 z_q。

        Args:
            z:   (N, D) encoder 连续输出，L2 归一化前
            z_q: (N, D) 量化后的离散表示，L2 归一化前
            temperature: InfoNCE 温度系数

        Returns:
            scalar loss
        """
        z = F.normalize(z.float(), dim=-1)  # (N, D)
        z_q = F.normalize(z_q.float(), dim=-1)  # (N, D)

        # 相似度矩阵 (N, N)  —— 对角线为正样本对
        logits = torch.matmul(z, z_q.T) / temperature  # (N, N)
        labels = torch.arange(z.size(0), device=z.device)

        # 双向对称 InfoNCE
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
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
