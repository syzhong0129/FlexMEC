from ._backend import *
from .settings import _env_flag

class EarlyCDRFeatureGate(torch.nn.Module):
    """Pool six CDRs, infer flexibility, and broadcast six modality weights.

    Output order: alpha_cdr, alpha_sample, alpha_token, w_cdr, w_token, cdr_feat.
    Framework weights stay one; framework alpha uses a detached CDR mean.
    """
    def __init__(
        self,
        node_dim,
        seq_dim,
        hidden_dim,
        num_cdr=6,
        num_feature_groups=6,
        group_dim=16,
        num_heads=4,
    ):
        super().__init__()
        self.num_cdr = num_cdr
        self.num_feature_groups = num_feature_groups
        self.group_dim = group_dim
        self.patch_dim = max(node_dim - 62, 0)
        if self.patch_dim <= 0:
            raise ValueError(f"Expected node_dim > 62 to include patch features, got {node_dim}")
        self.gate_input_dim = num_feature_groups * group_dim
        if self.gate_input_dim % num_heads != 0:
            raise ValueError(
                f"FLEXMEC_CDR_GATE_HEADS={num_heads} must divide gate_input_dim={self.gate_input_dim}"
            )

        self.patch_proj = Linear(self.patch_dim, group_dim)
        self.local_proj = Linear(20, group_dim)
        self.asa_proj = Linear(2, group_dim)
        self.pssm_proj = Linear(20, group_dim)
        self.aa_proj = Linear(20, group_dim)
        self.seq_proj = Linear(seq_dim, group_dim)

        self.patch_norm = LayerNorm(group_dim, elementwise_affine=False)
        self.local_norm = LayerNorm(group_dim, elementwise_affine=False)
        self.asa_norm = LayerNorm(group_dim, elementwise_affine=False)
        self.pssm_norm = LayerNorm(group_dim, elementwise_affine=False)
        self.aa_norm = LayerNorm(group_dim, elementwise_affine=False)
        self.seq_norm = LayerNorm(group_dim, elementwise_affine=False)
        self.cdr_selfattention = torch.nn.MultiheadAttention(
            embed_dim=self.gate_input_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.cdr_gate_mlp = Sequential(
            Linear(self.gate_input_dim, hidden_dim),
            GELU(),
            Linear(hidden_dim, 1),
        )

        # Feature order: patch, local profile, ASA/RSA, PSSM, amino acid, sequence.
        # Initialize rigid and flexible profiles uniformly.
        rigid_base = torch.ones(num_feature_groups, dtype=torch.float32)
        flex_base = torch.ones(num_feature_groups, dtype=torch.float32)
        rigid_prior = rigid_base / rigid_base.sum()
        flex_prior = flex_base / flex_base.sum()
        self.theta_rigid = torch.nn.Parameter(torch.log(rigid_prior).repeat(num_cdr, 1))
        self.theta_flex = torch.nn.Parameter(torch.log(flex_prior).repeat(num_cdr, 1))

        # Calibration knots: raw score 0.06 maps to 0.50; 0.12 maps to 0.80.
        self.register_buffer(
            "_calib_x",
            torch.tensor([0.0, 0.02, 0.03, 0.06, 0.12, 1.00], dtype=torch.float32),
        )
        self.register_buffer(
            "_calib_y",
            torch.tensor([0.0, 0.10, 0.30, 0.50, 0.80, 0.95], dtype=torch.float32),
        )

    def calibrate_alpha(self, alpha):
        """Calibrate flexibility scores by monotonic piecewise-linear interpolation."""
        x = self._calib_x.to(device=alpha.device, dtype=alpha.dtype)
        y = self._calib_y.to(device=alpha.device, dtype=alpha.dtype)
        a = torch.clamp(alpha, min=float(x[0].item()), max=float(x[-1].item()))
        idx = torch.clamp(
            torch.searchsorted(x, a.detach()) - 1,
            min=0,
            max=x.numel() - 2,
        )
        x_lo = x[idx]
        x_hi = x[idx + 1]
        y_lo = y[idx]
        y_hi = y[idx + 1]
        t = (a - x_lo) / (x_hi - x_lo + 1e-12)
        return y_lo + t * (y_hi - y_lo)

    def _span_mean(self, x, start, end):
        return x[start:end + 1].mean(dim=0)

    def _pool_by_cdr(self, x, spans):
        pooled = []
        for idx in range(self.num_cdr):
            start = int(spans[idx, 0].item())
            end = int(spans[idx, 1].item())
            pooled.append(self._span_mean(x, start, end))
        return torch.stack(pooled, dim=0)

    def _build_drive_outputs(self, alpha_cdr, spans, seq_len, device, dtype):
        if seq_len <= 0:
            alpha_sample = torch.tensor(0.5, dtype=dtype, device=device)
            alpha_token = torch.zeros((0,), dtype=dtype, device=device)
            w_cdr = torch.ones((self.num_cdr, self.num_feature_groups), dtype=dtype, device=device)
            w_token = torch.ones((0, self.num_feature_groups), dtype=dtype, device=device)
            return alpha_sample, alpha_token, w_cdr, w_token

        alpha_cdr = alpha_cdr.to(device=device, dtype=dtype).reshape(-1)
        if alpha_cdr.numel() != self.num_cdr:
            raise ValueError(
                f"Expected {self.num_cdr} CDR alpha values, got {alpha_cdr.numel()}"
            )
        alpha_cdr = torch.clamp(alpha_cdr, min=1e-4, max=1 - 1e-4)

        profile_rigid = torch.softmax(self.theta_rigid, dim=-1).to(device=device, dtype=dtype)
        profile_flex = torch.softmax(self.theta_flex, dim=-1).to(device=device, dtype=dtype)
        interpolated_profile = (1 - alpha_cdr[:, None]) * profile_rigid + alpha_cdr[:, None] * profile_flex
        w_cdr = self.num_feature_groups * interpolated_profile

        alpha_token = torch.full(
            (seq_len,),
            float(alpha_cdr.mean().item()),
            dtype=dtype,
            device=device,
        )
        w_token = torch.ones(
            (seq_len, self.num_feature_groups),
            dtype=dtype,
            device=device,
        )
        for idx in range(self.num_cdr):
            start = int(spans[idx, 0].item())
            end = int(spans[idx, 1].item())
            alpha_token[start:end + 1] = alpha_cdr[idx]
            w_token[start:end + 1] = w_cdr[idx]

        alpha_sample = torch.clamp(alpha_cdr.mean(), min=1e-4, max=1 - 1e-4)
        alpha_token = torch.clamp(alpha_token, min=1e-4, max=1 - 1e-4)
        return alpha_sample, alpha_token, w_cdr, w_token

    def drive_from_alpha(self, alpha_cdr, spans, seq_len, device, dtype):
        return self._build_drive_outputs(alpha_cdr, spans, seq_len, device=device, dtype=dtype)

    def forward(self, node_tokens, seq_tokens, spans):
        """Predict CDR flexibility and modality weights from antibody features.

        Inputs: node_tokens (L_ab, 122), seq_tokens (L_ab, seq_dim), and
        inclusive CDR spans (6, 2) in H1, H2, H3, L1, L2, L3 order.
        """
        seq_len = node_tokens.shape[0]
        device = node_tokens.device
        dtype = node_tokens.dtype


        cdr_node = self._pool_by_cdr(node_tokens, spans)
        cdr_seq = self._pool_by_cdr(seq_tokens, spans)

        aa_feat = self.aa_norm(self.aa_proj(cdr_node[:, 0:20]))
        pssm_feat = self.pssm_norm(self.pssm_proj(cdr_node[:, 20:40]))
        asa_feat = self.asa_norm(self.asa_proj(cdr_node[:, 40:42]))
        local_feat = self.local_norm(self.local_proj(cdr_node[:, 42:62]))
        patch_feat = self.patch_norm(self.patch_proj(cdr_node[:, 62:62 + self.patch_dim]))
        seq_feat = self.seq_norm(self.seq_proj(cdr_seq))
        cdr_group_feat = torch.stack(
            [patch_feat, local_feat, asa_feat, pssm_feat, aa_feat, seq_feat],
            dim=1,
        )
        cdr_feat = cdr_group_feat.flatten(1)
        z, _ = self.cdr_selfattention(cdr_feat.unsqueeze(0), cdr_feat.unsqueeze(0), cdr_feat.unsqueeze(0))
        z = z.squeeze(0)

        gate_in = z
        alpha_cdr = torch.sigmoid(self.cdr_gate_mlp(gate_in)).squeeze(-1)
        alpha_sample, alpha_token, w_cdr, w_token = self._build_drive_outputs(
            alpha_cdr,
            spans,
            seq_len,
            device=device,
            dtype=dtype,
        )
        return alpha_cdr, alpha_sample, alpha_token, w_cdr, w_token, cdr_feat
