from ._backend import *
from .settings import _env_flag

class FlexPairAttnBias(torch.nn.Module):
    """Compute per-head flexibility bias for antigen queries and antibody keys.

    bias[h, i, j] = lambda_pair[h] * alpha_ab[j] *
                    tanh(cosine(U_h ag_h1[i], V_h ab_h1[j]))

    Zero-initialized lambda_pair gives zero bias at initialization.
    """
    def __init__(self, embed_dim, num_heads, pair_dim=8, max_bias=1.0, bound_type="hard"):
        super().__init__()
        self.num_heads = num_heads
        self.pair_dim = pair_dim
        self.max_bias = float(max_bias)
        self.bound_type = str(bound_type).strip().lower()
        if self.bound_type in {"off", "unbound", "unbounded"}:
            self.bound_type = "none"
        if self.bound_type not in {"hard", "soft", "none"}:
            raise ValueError(f"Unsupported FLEX_PAIR_BIAS_BOUND_TYPE={bound_type!r}")
        if self.max_bias <= 0 and self.bound_type != "none":
            raise ValueError(f"FLEX_PAIR_BIAS_MAX must be > 0 for {self.bound_type} bound, got {self.max_bias}")
        self.lambda_pair = torch.nn.Parameter(torch.zeros(num_heads))
        self.ag_proj = Linear(embed_dim, num_heads * pair_dim, bias=False)
        self.ab_proj = Linear(embed_dim, num_heads * pair_dim, bias=False)

    def forward(self, alpha_token_ab, ag_h1, ab_h1):
        L_ag = ag_h1.shape[0]
        L_ab = ab_h1.shape[0]
        if alpha_token_ab.numel() != L_ab:
            raise ValueError(
                f"alpha_token_ab length {alpha_token_ab.numel()} != AB length {L_ab}"
            )

        q = self.ag_proj(ag_h1).view(L_ag, self.num_heads, self.pair_dim)
        k = self.ab_proj(ab_h1).view(L_ab, self.num_heads, self.pair_dim)
        q = q.permute(1, 0, 2).contiguous()  # (num_heads, L_ag, pair_dim)
        k = k.permute(1, 0, 2).contiguous()  # (num_heads, L_ab, pair_dim)
        q = torch.nn.functional.normalize(q, dim=-1, eps=1e-6)
        k = torch.nn.functional.normalize(k, dim=-1, eps=1e-6)

        pair_score = torch.einsum("hid,hjd->hij", q, k)
        pair_score = torch.tanh(pair_score)
        alpha = alpha_token_ab.to(device=ag_h1.device, dtype=ag_h1.dtype).view(1, 1, L_ab)
        lambdas = self.lambda_pair.to(dtype=ag_h1.dtype).view(-1, 1, 1)
        raw_bias_ag = lambdas * alpha * pair_score
        if self.bound_type == "hard":
            bias_ag = torch.clamp(raw_bias_ag, min=-self.max_bias, max=self.max_bias)
        elif self.bound_type == "soft":
            bias_ag = self.max_bias * torch.tanh(raw_bias_ag / self.max_bias)
        else:
            bias_ag = raw_bias_ag
        if _env_flag("FLEXMEC_CAPTURE_HEAD_EXPORT", False):
            self.last_pair_score = pair_score.detach()
            self.last_raw_bias_ag = raw_bias_ag.detach()
            self.last_bias_ag = bias_ag.detach()
            self.last_alpha_token_ab = alpha_token_ab.detach()
        return bias_ag
