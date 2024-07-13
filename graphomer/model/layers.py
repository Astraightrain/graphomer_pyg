import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from einops import rearrange, einsum
from typing import Optional
from torch import Tensor


class GraphomerBlock(nn.Module):
    def __init__(self, d_model: int = 768, nheads: int = 8, d_ff: int = 768, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.attn = MHA(d_model, num_heads, attn_dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.post_norm1 = RMSNorm(d_model)
        self.post_norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        x = x + self.dropout(self.attn(x, x, x, mask))
        x = self.post_norm1(x)
        x = x + self.dropout(self.ffn(x))
        x = self.post_norm2(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, d_model: int = 768, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        x = x / (torch.sqrt(torch.mean(x**2, -1, keepdim=True) + self.eps))
        x = self.alpha * x + self.gamma
        return x


class MHA(nn.Module):
    """
    Multi Head Attention module
    """

    def __init__(self, d_model: int = 768, nheads: int = 8, bias=True, dropout=0.0):
        super().__init__()
        assert d_model % nheads == 0, "d_model must be divisible by nheads"

        self.d_model = d_model
        self.nheads = nheads
        self.d_h = d_model // nheads
        self.scaling = self.d_h**-0.5

        self.wq = nn.Linear(d_model, d_model, bias)
        self.wk = nn.Linear(d_model, d_model, bias)
        self.wv = nn.Linear(d_model, d_model, bias)
        self.wo = nn.Linear(d_model, d_model, bias)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        nn.init.xavier_uniform_(self.wo.weight)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_bias: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        need_weights=False,
    ):

        q = rearrange(self.wq(q), "b n (h d) -> b h n d", h=self.nheads)
        k = rearrange(self.wk(k), "b n (h d) -> b h n d", h=self.nheads)
        v = rearrange(self.wv(v), "b n (h d) -> b h n d", h=self.nheads)
        # Scaled Dot Product Attention
        attn_weights = einsum(q, k, "b h q d, b h k d -> b h q k") * self.scaling
        if attn_bias is not None:
            attn_weights += rearrange(attn_bias, "b i j -> b () i j")
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = einsum(attn_weights, v, "b h i j, b h j d -> b h i d")
        attn_output = rearrange(attn_output, "b h n d -> b n (h d)")
        attn_output = self.wo(attn_output)
        if need_weights:
            return attn_output, attn_weights
        else:
            return attn_output


class GQA(nn.Module):
    """
    Grouped Query Attention module
    """

    def __init__(self, d_model, qheads, kvheads, bias=True, dropout=0.0):
        super().__init__()
        assert d_model % qheads == 0, "d_model must be divisible by nheads"
        assert qheads % kvheads == 0, "qheads must be divisible by kvheads"

        self.d_model = d_model
        self.qheads = qheads
        self.kvheads = kvheads

        d_q = d_model // qheads
        d_kv = d_model // qheads * kvheads
        self.scaling = d_q**-0.5
        self.num_head_groups = qheads // kvheads

        self.wq = nn.Linear(d_model, d_model, bias)
        self.wk = nn.Linear(d_model, d_kv, bias)
        self.wv = nn.Linear(d_model, d_kv, bias)
        self.wo = nn.Linear(d_model, d_model, bias)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        nn.init.xavier_uniform_(self.wo.weight)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_bias: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        need_weights=False,
    ):

        q = rearrange(self.wq(x), "b n (h d) -> b h n d", h=self.qheads)
        k = rearrange(self.wk(x), "b s (h d) -> b h s d", h=self.kvheads)
        v = rearrange(self.wv(x), "b s (h d) -> b h s d", h=self.kvheads)
        # Grouped Query Attention
        q = rearrange(q, "b (h g) n d -> b g h n d", g=self.num_head_groups)
        attn_weights = einsum(q, k, "b g h n d, b h s d -> b g h n s") * self.scaling
        if attn_bias is not None:
            attn_weights += rearrange(attn_bias, "b i j -> b () i j")
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = einsum(attn_weights, v, "b g h n s, b h s d -> b g h n d")
        attn_output = rearrange(attn_output, "b g h n d -> b n (h g d)")
        attn_output = self.wo(attn_output)
        if need_weights:
            attn_weights = rearrange(attn_weights, "b g h n s -> b n s (h g)")
            return attn_output, attn_weights
        else:
            return attn_output


class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    from torch_geometric.data import Data
    from einops import repeat
    from einops.layers.torch import Rearrange

    d_model = 128
    num_heads = 8
    d_ff = 512
    num_atoms = 10
    num_bonds = 20

    x = torch.randn(num_atoms, d_model)
    edge_index = torch.randint(0, num_atoms, (2, num_bonds))
    edge_attr = torch.randn(num_bonds, d_model)
    mask = torch.ones(num_atoms, num_atoms)
    mask = mask.masked_fill(torch.eye(num_atoms).bool(), 0)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, mask=mask)

    block = GraphomerBlock(d_model, num_heads, d_ff)
    block(data)
