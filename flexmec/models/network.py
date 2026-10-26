from ._backend import *
from .encoders import encode_ag_sequence, encode_ab_sequence
from .settings import _get_fail_log_path, _env_flag, _env_float, _env_int
from .flexmetric import EarlyCDRFeatureGate
from .flexenergy import FlexPairAttnBias
from .normalization import ChannelLayerNorm
from .attention import CrossAttention

class FlexMEC(torch.nn.Module):
    """Predict antigen and antibody residue scores for one complex.

    Node features: (L_ag, 122) and (L_ab, 122).
    Graph inputs: [Graph, None, valid_mask].
    PLM inputs normalize to (1, 1280, L_ag) and (1, 1024, L_ab).
    CDR spans follow H1, H2, H3, L1, L2, L3 order; numeric ends are inclusive.
    Returns 11 tensors: AG/AB probabilities, both attention maps, the
    predicted CDR alpha values and group weights, and AG/AB logits.
    """
    def __init__(self, share_weight=False, dropout=0.3, heads=4):
        super(FlexMEC, self).__init__()
        self.node_attr_dim = 122
        self.hidden_dim = 64
        self.esm_dim = 1280
        self.ablang_dim = 1024
        self.hidden_dim_cnn = 64
        self.hidden_dim_cnn2 = 64
        self.hidden_dim_cnn3 = 128
        self.concat_cnn = 256
        self.h1_dim = 64
        self.h2_dim = 64
        self.share_weight = share_weight
        self.dropout = dropout
        self.heads = 1
        self.multiheads = _env_int("FLEXMEC_MULTIHEADS", 16, min_value=1)
        if self.h1_dim % self.multiheads != 0:
            raise ValueError(
                f"FLEXMEC_MULTIHEADS={self.multiheads} must divide h1_dim={self.h1_dim}"
            )
        # Flexibility prediction, CDR feature gating and pairwise attention bias.
        self.enable_ab_cdr_gate = _env_flag("ENABLE_AB_CDR_GATE", True)
        self.enable_flex_attn_bias = _env_flag("ENABLE_FLEX_ATTN_BIAS", True)
        self.ab_cdr_gate_beta = float(os.environ.get("AB_CDR_GATE_BETA", "1.0"))
        self.flex_drive_neutral_ratio = 0.95

        cdr_gate_group_dim = _env_int("FLEXMEC_CDR_GATE_GROUP_DIM", 16, min_value=4)
        cdr_gate_heads = _env_int("FLEXMEC_CDR_GATE_HEADS", 4, min_value=1)
        self.cdr_key_order = ("h1", "h2", "h3", "l1", "l2", "l3")
        self.early_cdr_feature_gate = EarlyCDRFeatureGate(
            node_dim=self.node_attr_dim,
            seq_dim=self.ablang_dim,
            hidden_dim=self.h1_dim,
            num_cdr=6,
            num_feature_groups=6,
            group_dim=cdr_gate_group_dim,
            num_heads=cdr_gate_heads,
        )
        # Normalize antigen features without a learned feature gate.
        self.ag_feature_gate = torch.nn.Identity()
        self.ag_node_gate_norm = LayerNorm(self.node_attr_dim)
        self.ag_seq_gate_norm = LayerNorm(self.esm_dim)

        self.ab_seq_gate_norm = LayerNorm(self.ablang_dim)
        self.ab_pssm_norm = LayerNorm(20, elementwise_affine=False)
        self.ab_local_norm = LayerNorm(20, elementwise_affine=False)
        self.ab_patch_norm = LayerNorm(60, elementwise_affine=False)
        self.register_buffer(
            "ab_exposure_mean",
            torch.tensor([2.8900, 0.2669], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "ab_exposure_std",
            torch.tensor([1.6560, 0.2582], dtype=torch.float32),
            persistent=False,
        )


        ag_node_dropout_p = _env_float("AG_NODE_INPUT_DROPOUT", 0.15, 0.0, 1.0)
        ag_seq_dropout_p = _env_float("AG_SEQ_INPUT_DROPOUT", 0.10, 0.0, 1.0)
        self.ag_node_input_dropout = Dropout(ag_node_dropout_p)
        self.ag_seq_input_dropout = Dropout(ag_seq_dropout_p)

        # Additional dropout applies only to the antigen branch.
        ag_struct_dp = _env_float("AG_STRUCT_DROPOUT", 0.3, 0.0, 1.0)
        ag_seq_out_dp = _env_float("AG_SEQ_OUT_DROPOUT", 0.3, 0.0, 1.0)
        ag_pred_dp = _env_float("AG_PRED_DROPOUT", 0.2, 0.0, 1.0)
        self.ag_struct_dropout = Dropout(ag_struct_dp)
        self.ag_seq_out_dropout = Dropout(ag_seq_out_dp)
        self.ag_pred_dropout = Dropout(ag_pred_dp)

        # GearNet
        self.gearnet = models.GearNet(input_dim=self.node_attr_dim, hidden_dims=[64, 64, 64], num_relation=3)

        # Learned structural tokens replace features at missing coordinates.
        self._gearnet_out_dim = 64
        self.ag_occluded_token = torch.nn.Parameter(torch.empty(self._gearnet_out_dim))
        self.ab_occluded_token = torch.nn.Parameter(torch.empty(self._gearnet_out_dim))
        torch.nn.init.normal_(self.ag_occluded_token, std=0.02)
        torch.nn.init.normal_(self.ab_occluded_token, std=0.02)

        # CNN
        self.ag_cnn1 = Conv1d(in_channels=self.esm_dim, out_channels=self.hidden_dim_cnn, kernel_size=5, padding="same")
        if self.share_weight:
            self.ab_cnn1 = self.ag_cnn1
        else:
            self.ab_cnn1 = Conv1d(in_channels=self.ablang_dim, out_channels=self.hidden_dim_cnn, kernel_size=5, padding="same")

        # Dilated CNN
        self.ag_cnn2 = Conv1d(self.hidden_dim_cnn, self.hidden_dim_cnn2, 3, dilation=2, padding="same")
        self.ag_cnn3 = Conv1d(self.hidden_dim_cnn2, self.hidden_dim_cnn3, 3, dilation=3, padding="same")
        if self.share_weight:
            self.ab_cnn2 = self.ag_cnn2
            self.ab_cnn3 = self.ag_cnn3
        else:
            self.ab_cnn2 = Conv1d(self.hidden_dim_cnn, self.hidden_dim_cnn2, 3, dilation=2, padding="same")
            self.ab_cnn3 = Conv1d(self.hidden_dim_cnn2, self.hidden_dim_cnn3, 3, dilation=3, padding="same")

        # BiLSTM
        self.ag_lstm = LSTM(input_size=self.concat_cnn, hidden_size=self.h1_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        if self.share_weight:
            self.ab_lstm = self.ag_lstm
        else:
            self.ab_lstm = LSTM(input_size=self.concat_cnn, hidden_size=self.h1_dim // 2, num_layers=1, batch_first=True, bidirectional=True)

        self.ab_seq_from_node = Sequential(
            Linear(self.node_attr_dim, self.ablang_dim),
            GELU(),
            Dropout(_env_float("AB_SEQ_FROM_NODE_DROPOUT", 0.1, 0.0, 1.0)),
        )
        self.linear_ag = Linear(self.h1_dim + self.h1_dim, self.h1_dim)
        self.linear_ab = Linear(self.h1_dim + self.h1_dim, self.h1_dim)

        use_layer_norm_cnn = _env_flag("USE_LAYER_NORM_CNN", False)

        if use_layer_norm_cnn:
            self.ag_bnorm1 = ChannelLayerNorm(self.hidden_dim_cnn)
            self.ag_bnorm2 = ChannelLayerNorm(self.hidden_dim_cnn2)
            self.ag_bnorm3 = ChannelLayerNorm(self.hidden_dim_cnn3)
            self.ag_bnorm4 = LayerNorm(self.hidden_dim)
            self.ab_bnorm1 = ChannelLayerNorm(self.hidden_dim_cnn)
            self.ab_bnorm2 = ChannelLayerNorm(self.hidden_dim_cnn2)
            self.ab_bnorm3 = ChannelLayerNorm(self.hidden_dim_cnn3)
            self.ab_bnorm4 = LayerNorm(self.hidden_dim)
        else:
            self.ag_bnorm1 = BatchNorm1d(self.hidden_dim_cnn, track_running_stats=False)
            self.ag_bnorm2 = BatchNorm1d(self.hidden_dim_cnn2, track_running_stats=False)
            self.ag_bnorm3 = BatchNorm1d(self.hidden_dim_cnn3, track_running_stats=False)
            self.ag_bnorm4 = BatchNorm1d(self.hidden_dim, track_running_stats=False)
            self.ab_bnorm1 = BatchNorm1d(self.hidden_dim_cnn, track_running_stats=False)
            self.ab_bnorm2 = BatchNorm1d(self.hidden_dim_cnn2, track_running_stats=False)
            self.ab_bnorm3 = BatchNorm1d(self.hidden_dim_cnn3, track_running_stats=False)
            self.ab_bnorm4 = BatchNorm1d(self.hidden_dim, track_running_stats=False)

        # CrossAttention
        self.ag_crossattention = CrossAttention(self.h1_dim, self.multiheads)
        self.ab_crossattention = CrossAttention(self.h1_dim, self.multiheads)
        self.flex_attn_bias = FlexPairAttnBias(
            embed_dim=self.h1_dim,
            num_heads=self.multiheads,
            pair_dim=_env_int("FLEX_PAIR_BIAS_PAIR_DIM", 8, min_value=1),
            max_bias=_env_float("FLEX_PAIR_BIAS_MAX", 1.0, 0.0),
            bound_type=os.environ.get("FLEX_PAIR_BIAS_BOUND_TYPE", "hard"),
        )
        self.ag_feed_forward = Sequential(
            Linear(self.h1_dim, 4 * self.h1_dim),
            ReLU(),
            Linear(4 * self.h1_dim, self.h1_dim),
        )
        self.ab_feed_forward = Sequential(
            Linear(self.h1_dim, 4 * self.h1_dim),
            ReLU(),
            Linear(4 * self.h1_dim, self.h1_dim),
        )

        self.ag_fuse = Sequential(
            Linear(2 * self.h1_dim, self.h1_dim),
            GELU(),
        )
        self.ab_fuse = Sequential(
            Linear(2 * self.h1_dim, self.h1_dim),
            GELU(),
        )

        self.ag_norm1 = LayerNorm(self.h1_dim)
        self.ag_norm2 = LayerNorm(self.h1_dim)
        self.ab_norm1 = LayerNorm(self.h1_dim)
        self.ab_norm2 = LayerNorm(self.h1_dim)

        self.ag_linearsigmoid_linear = Linear(self.h1_dim, 1)
        self.ag_linearsigmoid_sigmoid = Sigmoid()
        self.ab_linearsigmoid_linear = Linear(self.h1_dim, 1)
        self.ab_linearsigmoid_sigmoid = Sigmoid()
        self.ag_linearsigmoid = Sequential(Linear(self.h1_dim, 1), Sigmoid())
        self.ab_linearsigmoid = Sequential(Linear(self.h1_dim, 1), Sigmoid())

        self.ag_cross_drop_path_enabled = _env_flag("AG_CROSS_DROP_PATH_ENABLED", True)
        self.ag_cross_drop_path = _env_float("AG_CROSS_DROP_PATH", 0.1, 0.0, 1.0)

    def _default_cdr_spans(self, seq_len, device):
        if seq_len <= 0:
            return torch.zeros((6, 2), dtype=torch.long, device=device)

        # Chothia base CDR lengths: H1/H2/H3/L1/L2/L3 = 11/9/12/15/11/13
        base_lens = torch.tensor([11, 9, 12, 15, 11, 13], dtype=torch.float32, device=device)
        raw = base_lens / base_lens.sum() * float(seq_len)
        lens = torch.floor(raw).to(torch.long)
        lens = torch.clamp(lens, min=1)

        diff = int(seq_len - int(lens.sum().item()))
        if diff > 0:
            frac = raw - torch.floor(raw)
            add_idx = torch.argsort(frac, descending=True)
            for k in range(diff):
                lens[add_idx[k % 6]] += 1
        elif diff < 0:
            removable = lens - 1
            rem_idx = torch.argsort(removable, descending=True)
            for k in range(-diff):
                idx = rem_idx[k % 6]
                if lens[idx] > 1:
                    lens[idx] -= 1

        starts = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            torch.cumsum(lens, dim=0)[:-1],
        ], dim=0)
        ends = torch.cumsum(lens, dim=0) - 1

        starts = torch.clamp(starts, min=0, max=max(seq_len - 1, 0))
        ends = torch.clamp(ends, min=0, max=max(seq_len - 1, 0))
        ends = torch.maximum(ends, starts)
        return torch.stack([starts, ends], dim=1)

    def _sanitize_cdr_spans(self, cdr_spans, seq_len, device):
        assert cdr_spans is not None, "Missing cdr_spans/cdr_ranges; refusing to use default CDR spans"
        assert seq_len > 0, f"Invalid antibody sequence length for CDR spans: {seq_len}"

        spans = cdr_spans
        if isinstance(spans, dict):
            parsed = []
            for key in self.cdr_key_order:
                span = spans.get(key)
                if span is None:
                    span = spans.get(key.upper())
                assert span is not None, f"Missing CDR span for key {key!r}"
                if isinstance(span, str):
                    parts = span.split(":")
                    assert len(parts) == 2, f"Invalid CDR range string for {key!r}: {span!r}"
                    try:
                        start, stop = int(parts[0]), int(parts[1])
                    except ValueError as exc:
                        raise AssertionError(f"Invalid CDR range string for {key!r}: {span!r}") from exc
                    parsed.append([start, stop - 1])
                    continue
                if torch.is_tensor(span):
                    span = span.detach().cpu().tolist()
                assert isinstance(span, (list, tuple)) and len(span) >= 2, (
                    f"Invalid CDR span for key {key!r}: {span!r}"
                )
                parsed.append([int(span[0]), int(span[1])])
            spans = torch.tensor(parsed, dtype=torch.long, device=device)
        elif isinstance(spans, (list, tuple)) and all(isinstance(span, str) for span in spans):
            assert len(spans) == 6, f"Expected 6 CDR ranges, got {len(spans)}: {spans!r}"
            parsed = []
            for idx, span in enumerate(spans):
                parts = span.split(":")
                assert len(parts) == 2, f"Invalid CDR range string at index {idx}: {span!r}"
                try:
                    start, stop = int(parts[0]), int(parts[1])
                except ValueError as exc:
                    raise AssertionError(f"Invalid CDR range string at index {idx}: {span!r}") from exc
                parsed.append([start, stop - 1])
            spans = torch.tensor(parsed, dtype=torch.long, device=device)
        elif not torch.is_tensor(spans):
            try:
                spans = torch.tensor(spans, dtype=torch.long, device=device)
            except Exception as exc:
                raise AssertionError(f"Invalid CDR spans format: {spans!r}") from exc
        else:
            spans = spans.to(device=device, dtype=torch.long)

        assert spans.dim() == 2 and spans.shape[0] == 6 and spans.shape[1] == 2, (
            f"Expected CDR spans shape (6, 2), got {tuple(spans.shape)}"
        )

        spans = spans.clone()
        spans[:, 0] = torch.clamp(spans[:, 0], min=0, max=seq_len - 1)
        spans[:, 1] = torch.clamp(spans[:, 1], min=0, max=seq_len - 1)
        spans[:, 1] = torch.maximum(spans[:, 1], spans[:, 0])
        return spans

    def _to_token_features(self, x):
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            if x.shape[0] == 1:
                return x.squeeze(0)
            return x[0]
        raise ValueError(f"Unsupported node feature shape: {tuple(x.shape)}")


    def _prepare_ab_esm(self, ab_esm, ab_x):
        # Output shape must be (1, C_ab, L_ab) for Conv1d.
        if ab_esm is None:
            if ab_x.dim() == 3:
                seq_len = ab_x.shape[1]
            elif ab_x.dim() == 2:
                seq_len = ab_x.shape[0]
            else:
                raise ValueError(f"Unsupported ab_x shape for ab_esm=None: {tuple(ab_x.shape)}")
            return torch.zeros(
                (1, self.ablang_dim, seq_len),
                dtype=ab_x.dtype,
                device=ab_x.device,
            )

        if not torch.is_tensor(ab_esm):
            ab_esm = torch.tensor(ab_esm, dtype=ab_x.dtype, device=ab_x.device)
        else:
            ab_esm = ab_esm.to(device=ab_x.device, dtype=ab_x.dtype)

        if ab_esm.dim() == 3:
            if ab_esm.shape[0] != 1:
                ab_esm = ab_esm[:1]
            if ab_esm.shape[1] == self.ablang_dim:
                return ab_esm
            if ab_esm.shape[2] == self.ablang_dim:
                ab_tokens = ab_esm.squeeze(0)
            else:
                ab_tokens = ab_esm.squeeze(0).transpose(0, 1)
        elif ab_esm.dim() == 2:
            if ab_esm.shape[1] == self.ablang_dim:
                ab_tokens = ab_esm
            elif ab_esm.shape[0] == self.ablang_dim:
                return ab_esm.unsqueeze(0)
            else:
                ab_tokens = ab_esm if ab_esm.shape[0] >= ab_esm.shape[1] else ab_esm.transpose(0, 1)
        else:
            raise ValueError(f"Unsupported ab_esm shape: {tuple(ab_esm.shape)}")

        return ab_tokens.transpose(0, 1).unsqueeze(0)

    def _prepare_ag_esm(self, ag_esm, ag_x):
        # Output shape must be (1, C_ag, L_ag) for Conv1d.
        if ag_esm is None:
            if ag_x.dim() == 3:
                seq_len = ag_x.shape[1]
            elif ag_x.dim() == 2:
                seq_len = ag_x.shape[0]
            else:
                raise ValueError(f"Unsupported ag_x shape for ag_esm=None: {tuple(ag_x.shape)}")
            return torch.zeros(
                (1, self.esm_dim, seq_len),
                dtype=ag_x.dtype,
                device=ag_x.device,
            )

        if not torch.is_tensor(ag_esm):
            ag_esm = torch.tensor(ag_esm, dtype=ag_x.dtype, device=ag_x.device)
        else:
            ag_esm = ag_esm.to(device=ag_x.device, dtype=ag_x.dtype)

        if ag_esm.dim() == 3:
            if ag_esm.shape[0] != 1:
                ag_esm = ag_esm[:1]
            if ag_esm.shape[1] == self.esm_dim:
                return ag_esm
            if ag_esm.shape[2] == self.esm_dim:
                ag_tokens = ag_esm.squeeze(0)
            else:
                ag_tokens = ag_esm.squeeze(0).transpose(0, 1)
        elif ag_esm.dim() == 2:
            if ag_esm.shape[1] == self.esm_dim:
                ag_tokens = ag_esm
            elif ag_esm.shape[0] == self.esm_dim:
                return ag_esm.unsqueeze(0)
            else:
                ag_tokens = ag_esm if ag_esm.shape[0] >= ag_esm.shape[1] else ag_esm.transpose(0, 1)
        else:
            raise ValueError(f"Unsupported ag_esm shape: {tuple(ag_esm.shape)}")


        return ag_tokens.transpose(0, 1).unsqueeze(0)

    def _apply_ag_feature_gate(self, ag_tokens, ag_seq_tokens, w_token=None):
        ag_node_gated = self.ag_node_gate_norm(ag_tokens)

        if self.training:
            ag_node_gated = self.ag_node_input_dropout(ag_node_gated)

        ag_esm_tokens_gated = self.ag_seq_gate_norm(ag_seq_tokens)
        if self.training:
            ag_esm_tokens_gated = self.ag_seq_input_dropout(ag_esm_tokens_gated)

        return ag_node_gated, ag_esm_tokens_gated

    def _normalize_ab_feature_groups(self, ab_tokens):
        # aa_onehot: [0:20], pssm: [20:40], exposure: [40:42], local_profile: [42:62], patch_matrix_AB: [62:]
        if ab_tokens.shape[1] < 62:
            pad = torch.zeros(
                (ab_tokens.shape[0], 62 - ab_tokens.shape[1]),
                dtype=ab_tokens.dtype,
                device=ab_tokens.device,
            )
            ab_tokens = torch.cat([ab_tokens, pad], dim=1)

        aa_onehot = ab_tokens[:, 0:20]
        pssm = self.ab_pssm_norm(ab_tokens[:, 20:40]) / (20 ** 0.5)

        asa_raw = ab_tokens[:, 40:41].clamp_min(0.0)
        rsa = ab_tokens[:, 41:42].clamp(0.0, 1.0)
        exposure = torch.cat([torch.log1p(asa_raw), rsa], dim=-1)
        exposure_mean = self.ab_exposure_mean.to(device=exposure.device, dtype=exposure.dtype)
        exposure_std = self.ab_exposure_std.to(device=exposure.device, dtype=exposure.dtype).clamp_min(1e-6)
        exposure = (exposure - exposure_mean) / exposure_std
        exposure = exposure / (2 ** 0.5)

        local_profile = self.ab_local_norm(ab_tokens[:, 42:62]) / (20 ** 0.5)

        patch_matrix = ab_tokens[:, 62:]
        if patch_matrix.shape[1] > 0:
            if patch_matrix.shape[1] == 60:
                patch_matrix = self.ab_patch_norm(patch_matrix)
            else:
                patch_matrix = torch.nn.functional.layer_norm(
                    patch_matrix,
                    (patch_matrix.shape[1],),
                )
            patch_matrix = patch_matrix / (patch_matrix.shape[1] ** 0.5)

        vertex = torch.cat([aa_onehot, pssm, exposure, local_profile], dim=-1)
        return torch.cat([vertex, patch_matrix], dim=-1) if patch_matrix.shape[1] > 0 else vertex

    def _normalize_ab_seq_tokens(self, ab_seq_tokens):
        ab_seq_tokens = self.ab_seq_gate_norm(ab_seq_tokens)
        return ab_seq_tokens / (self.ablang_dim ** 0.5)

    def _apply_ab_feature_gate(self, ab_tokens, ab_seq_tokens, w_token):
        # Apply modality weights in the order defined by EarlyCDRFeatureGate.
        gate_beta = min(max(float(getattr(self, "ab_cdr_gate_beta", 1.0)), 0.0), 1.0)
        beta_tensor = w_token.new_tensor(gate_beta)
        w_token = 1.0 + beta_tensor * (w_token - 1.0)

        aa_onehot = ab_tokens[:, 0:20]
        pssm = ab_tokens[:, 20:40]
        asa_features = ab_tokens[:, 40:42]
        local_profile = ab_tokens[:, 42:62]
        patch_matrix = ab_tokens[:, 62:]

        patch_matrix_feat = w_token[:, 0:1] * patch_matrix if patch_matrix.shape[1] > 0 else patch_matrix
        local_profile_feat = w_token[:, 1:2] * local_profile
        asa_feat = w_token[:, 2:3] * asa_features
        pssm_feat = w_token[:, 3:4] * pssm
        aa_onehot_feat = w_token[:, 4:5] * aa_onehot

        vertex_gated = torch.cat([aa_onehot_feat, pssm_feat, asa_feat, local_profile_feat], dim=-1)
        ab_node_gated = torch.cat([vertex_gated, patch_matrix_feat], dim=-1) if patch_matrix.shape[1] > 0 else vertex_gated

        if ab_node_gated.shape[1] < self.node_attr_dim:
            pad = torch.zeros(
                (ab_node_gated.shape[0], self.node_attr_dim - ab_node_gated.shape[1]),
                dtype=ab_node_gated.dtype,
                device=ab_node_gated.device,
            )
            ab_node_gated = torch.cat([ab_node_gated, pad], dim=1)
        elif ab_node_gated.shape[1] > self.node_attr_dim:
            ab_node_gated = ab_node_gated[:, :self.node_attr_dim]

        antiberta2_gated = w_token[:, 5:6] * ab_seq_tokens
        return ab_node_gated, antiberta2_gated

    def _resolve_ab_flex_drive(self, alpha_cdr_pred):
        alpha_cdr_pred = torch.clamp(alpha_cdr_pred, min=1e-4, max=1 - 1e-4)
        if not self.training:
            return alpha_cdr_pred

        neutral_ratio = float(getattr(self, "flex_drive_neutral_ratio", 0.0))
        neutral_ratio = min(max(neutral_ratio, 0.0), 1.0)
        if neutral_ratio <= 0.0:
            return alpha_cdr_pred

        neutral_alpha = torch.full_like(alpha_cdr_pred, 0.5)
        return neutral_ratio * neutral_alpha + (1.0 - neutral_ratio) * alpha_cdr_pred

    def forward(
        self,
        ag_x,
        ag_edge_index,
        ab_x,
        ab_edge_index,
        ag_esm,
        ab_esm,
        sample_id,
        cdr_spans=None,
    ):
        # Antigen structure and sequence encoders.
        ag_token_feat = self._to_token_features(ag_x)
        ag_esm = self._prepare_ag_esm(ag_esm, ag_x)
        ag_esm_tokens = ag_esm.squeeze(0).transpose(0, 1)


        ag_node_gated, ag_esm_tokens_gated = self._apply_ag_feature_gate(ag_token_feat, ag_esm_tokens)
        ag_node_for_gearnet = torch.nn.functional.normalize(ag_node_gated, dim=1)
        try:
            ag_h1_struct = self.gearnet(ag_edge_index[0], ag_node_for_gearnet)["node_feature"]
        except Exception as e:
            ag_edge_0_shape = ag_edge_index[0].shape if hasattr(ag_edge_index[0], "shape") else "无shape属性"
            ag_edge_1_shape = ag_node_for_gearnet.shape if hasattr(ag_node_for_gearnet, "shape") else "无shape属性"
            print(f"样本 {sample_id} 在 gearnet(ag) 处理时出错: {str(e)}，跳过该样本")
            print(f"ag_edge_index[0] 形状: {ag_edge_0_shape}")
            print(f"ag_node_for_gearnet 形状: {ag_edge_1_shape}")

            fail_file = _get_fail_log_path()
            os.makedirs(os.path.dirname(fail_file), exist_ok=True)
            with open(fail_file, "a") as f:
                f.write(f"{sample_id}\tAG_edge_index[0]:{ag_edge_0_shape}\tAG_node:{ag_edge_1_shape}\tError:{str(e)}\n")

            return (None,) * 11

        # Replace missing structural features with a learned token.
        # Keep dimension checks outside the handler for graph failures.
        if len(ag_edge_index) >= 3 and ag_edge_index[2] is not None:
            ag_valid = ag_edge_index[2]
            assert ag_h1_struct.shape[1] == self.ag_occluded_token.shape[0], (
                f"gearnet output dim {ag_h1_struct.shape[1]} != occluded_token dim "
                f"{self.ag_occluded_token.shape[0]}; 请修改 model.py 里 "
                f"self._gearnet_out_dim (常见值 64 或 192) 并重建 ag/ab_occluded_token"
            )
            if (~ag_valid).any():
                ag_h1_struct = ag_h1_struct.clone()
                ag_h1_struct[~ag_valid] = self.ag_occluded_token.to(ag_h1_struct.dtype)

        ag_h1_struct = self.ag_struct_dropout(ag_h1_struct)
        ag_h2 = encode_ag_sequence(self, ag_esm_tokens_gated)

        ag_h1 = self.linear_ag(torch.cat((ag_h1_struct, ag_h2), dim=1))

        # Antibody feature gating and encoders.
        ab_token_feat = self._normalize_ab_feature_groups(self._to_token_features(ab_x))
        ab_esm = self._prepare_ab_esm(ab_esm, ab_x)
        ab_esm_tokens = self._normalize_ab_seq_tokens(ab_esm.squeeze(0).transpose(0, 1))


        spans = self._sanitize_cdr_spans(cdr_spans, ab_token_feat.shape[0], ab_token_feat.device)

        alpha_cdr_pred, alpha_sample_pred, alpha_token_pred, w_cdr_pred, w_token_pred, cdr_ctx_pred = self.early_cdr_feature_gate(
            ab_token_feat,
            ab_esm_tokens,
            spans,
        )
        alpha_cdr_drive = self._resolve_ab_flex_drive(alpha_cdr_pred)
        _alpha_sample_drive, alpha_token_drive, _w_cdr_drive, w_token_drive = self.early_cdr_feature_gate.drive_from_alpha(
            alpha_cdr_drive,
            spans,
            ab_token_feat.shape[0],
            device=ab_token_feat.device,
            dtype=ab_token_feat.dtype,
        )
        if self.enable_ab_cdr_gate:
            ab_node_gated, ab_esm_tokens_gated = self._apply_ab_feature_gate(
                ab_token_feat,
                ab_esm_tokens,
                w_token_drive,
            )
        else:
            ab_node_gated, ab_esm_tokens_gated = ab_token_feat, ab_esm_tokens

        ab_node_for_gearnet = torch.nn.functional.normalize(ab_node_gated, dim=1)
        try:
            ab_h1_struct = self.gearnet(ab_edge_index[0], ab_node_for_gearnet)["node_feature"]
        except Exception as e:
            ab_edge_0_shape = ab_edge_index[0].shape if hasattr(ab_edge_index[0], "shape") else "无shape属性"
            ab_edge_1_shape = ab_node_for_gearnet.shape if hasattr(ab_node_for_gearnet, "shape") else "无shape属性"
            print(f"样本 {sample_id} 在 gearnet(ab) 处理时出错: {str(e)}，跳过该样本")
            print(f"ab_edge_index[0] 形状: {ab_edge_0_shape}")
            print(f"ab_node_for_gearnet 形状: {ab_edge_1_shape}")

            fail_file = _get_fail_log_path()
            os.makedirs(os.path.dirname(fail_file), exist_ok=True)
            with open(fail_file, "a") as f:
                f.write(f"{sample_id}\tAB_edge_index[0]:{ab_edge_0_shape}\tAB_node:{ab_edge_1_shape}\tError:{str(e)}\n")

            return (None,) * 11

        # Replace missing structural features with a learned token.
        # Keep dimension checks outside the handler for graph failures.
        if len(ab_edge_index) >= 3 and ab_edge_index[2] is not None:
            ab_valid = ab_edge_index[2]
            assert ab_h1_struct.shape[1] == self.ab_occluded_token.shape[0], (
                f"gearnet output dim {ab_h1_struct.shape[1]} != occluded_token dim "
                f"{self.ab_occluded_token.shape[0]}; 请修改 model.py 里 "
                f"self._gearnet_out_dim (常见值 64 或 192) 并重建 ag/ab_occluded_token"
            )
            if (~ab_valid).any():
                ab_h1_struct = ab_h1_struct.clone()
                ab_h1_struct[~ab_valid] = self.ab_occluded_token.to(ab_h1_struct.dtype)

        ab_h2 = encode_ab_sequence(self, ab_esm_tokens_gated)

        ab_h1 = self.linear_ab(torch.cat((ab_h1_struct, ab_h2), dim=1))
        ab_h1_kv = ab_h1

        ag_attn_bias = None
        if self.enable_flex_attn_bias and alpha_token_drive.numel() != 0:
            flex_bias_ag = self.flex_attn_bias(alpha_token_drive, ag_h1, ab_h1_kv)
            ag_attn_bias = flex_bias_ag if ag_attn_bias is None else ag_attn_bias + flex_bias_ag

        if ag_attn_bias is None:
            ag_attention, ag_attention_weights = self.ag_crossattention(ag_h1, ab_h1_kv)
        else:
            ag_attention, ag_attention_weights = self.ag_crossattention(
                ag_h1, ab_h1_kv, idr_matrix=ag_attn_bias, lambda_weight=1.0
            )
        # AB <- AG branch does not use flex bias.
        # A query-wise constant additive term is cancelled by row-wise softmax.
        ab_attention, ab_attention_weights = self.ab_crossattention(ab_h1, ag_h1)

        if (
            self.training
            and self.ag_cross_drop_path_enabled
            and self.ag_cross_drop_path > 0.0
            and torch.rand(1, device=ag_h1.device).item() < self.ag_cross_drop_path
        ):
            ag_res1 = ag_h1
        else:
            ag_res1 = self.ag_norm1(ag_h1 + ag_attention)
        ab_res1 = self.ab_norm1(ab_h1 + ab_attention)

        ag_res2 = self.ag_norm2(ag_res1 + self.ag_feed_forward(ag_res1))
        ab_res2 = self.ab_norm2(ab_res1 + self.ab_feed_forward(ab_res1))

        ag_out_0 = self.ag_linearsigmoid_linear(self.ag_pred_dropout(ag_res2))
        ag_out = self.ag_linearsigmoid_sigmoid(ag_out_0)

        ab_out_0 = self.ab_linearsigmoid_linear(ab_res2)
        ab_out = self.ab_linearsigmoid_sigmoid(ab_out_0)

        return (
            ag_out,
            ab_out,
            ag_attention_weights,
            ab_attention_weights,
            alpha_sample_pred, # Sample-level flexibility; retained in the output interface.
            alpha_token_pred,
            alpha_cdr_pred,
            w_cdr_pred,
            w_token_pred,
            ag_out_0,
            ab_out_0,
        )
