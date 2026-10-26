import os
import torch
import torch.nn as nn
from torch.nn import Linear, MultiheadAttention

class CrossAttention(nn.Module):
    def __init__(self, input_size, num_heads):
        super(CrossAttention, self).__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=input_size,
            num_heads=num_heads,
            batch_first=False
        )
        self.last_attn_weights_per_head = None
        self.last_attn_mask = None

    def forward(self, input1, input2, idr_matrix=None, lambda_weight=None):
        """Attend from input1 (Lq, E) to input2 (Sk, E).

        The optional additive bias has shape (Lq, Sk) or (num_heads, Lq, Sk).
        Return updated query features and attention weights averaged over heads.
        """
        q = input1.unsqueeze(0).transpose(0, 1)  # (Lq, 1, E)
        k = input2.unsqueeze(0).transpose(0, 1)  # (Sk, 1, E)
        v = k

        # Scale the additive attention bias, optionally per head.
        attn_mask = None
        if idr_matrix is not None:
            this_attn_mask = idr_matrix.to(dtype=q.dtype, device=q.device)

            if isinstance(lambda_weight, (list, tuple)):
                attn_mask = torch.stack([
                    w * this_attn_mask for w in lambda_weight
                ], dim=0)   # (num_heads, Lq, Sk)
            else:
                attn_mask = (lambda_weight * this_attn_mask)

        Lq, Nq, _ = q.shape
        Sk, Nk, _ = k.shape
        assert Nq == Nk, f"q/k batch 维不一致: {Nq} vs {Nk}"
        # Accept either orientation of the query/key axes.
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                if attn_mask.shape == (Lq, Sk):
                    pass
                elif attn_mask.shape == (Sk, Lq):
                    attn_mask = attn_mask.transpose(0, 1).contiguous()
                else:
                    raise ValueError(
                        f"2D attn_mask 形状不匹配: got {tuple(attn_mask.shape)}, "
                        f"expected {(Lq, Sk)} 或 {(Sk, Lq)}"
                    )


            elif attn_mask.dim() == 3:
                num_heads = getattr(self.mha, "num_heads", None)
                if num_heads is None:
                    raise RuntimeError("无法读取 MultiheadAttention.num_heads")
                expected_b = Nq * num_heads

                if attn_mask.shape == (expected_b, Lq, Sk):
                    pass
                elif attn_mask.shape == (expected_b, Sk, Lq):
                    attn_mask = attn_mask.transpose(1, 2).contiguous()
                else:
                    raise ValueError(
                        f"3D attn_mask 形状不匹配: got {tuple(attn_mask.shape)}, "
                        f"expected {(expected_b, Lq, Sk)} 或 {(expected_b, Sk, Lq)}"
                    )
            else:
                raise ValueError(f"attn_mask 维度必须为 2 或 3，got {attn_mask.dim()}")


        capture_per_head = os.environ.get("FLEXMEC_CAPTURE_HEAD_EXPORT", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
            "y",
        }
        if capture_per_head:
            out, attn_weights_per_head = self.mha(
                q,
                k,
                v,
                attn_mask=attn_mask,
                average_attn_weights=False,
            )
            if attn_weights_per_head.dim() == 4 and attn_weights_per_head.shape[0] == 1:
                attn_weights_per_head = attn_weights_per_head.squeeze(0)
            self.last_attn_weights_per_head = attn_weights_per_head.detach()
            self.last_attn_mask = attn_mask.detach() if attn_mask is not None else None
            attn_weights = attn_weights_per_head.mean(dim=0)
        else:
            out, attn_weights = self.mha(
                q, k, v,
                attn_mask=attn_mask,
            )
            self.last_attn_weights_per_head = None
            self.last_attn_mask = None
            attn_weights = attn_weights.squeeze(0)
        output_1 = out.transpose(0, 1).squeeze(0)
        return output_1, attn_weights
