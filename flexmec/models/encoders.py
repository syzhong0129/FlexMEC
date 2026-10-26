"""Sequence encoders using parameters owned by FlexMEC.

Separate modules would change checkpoint keys. Antigen and antibody branches
use different operations.
"""
import torch

def encode_ag_sequence(self: torch.nn.Module, ag_esm_tokens_gated: torch.Tensor) -> torch.Tensor:
    """Encode antigen PLM tokens as residue features with 64 channels."""
    ag_esm = ag_esm_tokens_gated.transpose(0, 1).unsqueeze(0)
    ag_esm = torch.nn.functional.normalize(ag_esm, dim=1)
    ag_esm_out1 = self.ag_bnorm1(self.ag_cnn1(ag_esm))
    ag_esm_out2 = self.ag_bnorm2(self.ag_cnn2(ag_esm_out1))
    ag_esm_out3 = self.ag_bnorm3(self.ag_cnn3(ag_esm_out2))

    ag_esm_features = torch.cat([ag_esm_out1, ag_esm_out2, ag_esm_out3], dim=1)
    ag_esm_features = ag_esm_features.transpose(1, 2)
    ag_lstm_out, _ = self.ag_lstm(ag_esm_features)

    ag_h2 = torch.cat((ag_lstm_out[:, :, :64 // 2], ag_lstm_out[:, :, 64 // 2:]), dim=2)
    ag_h2 = torch.squeeze(ag_h2, dim=0)
    ag_h2 = self.ag_bnorm4(ag_h2)
    ag_h2 = self.ag_seq_out_dropout(ag_h2)
    return ag_h2

def encode_ab_sequence(self: torch.nn.Module, ab_esm_tokens_gated: torch.Tensor) -> torch.Tensor:
    """Encode antibody PLM tokens as residue features with 64 channels."""
    ab_esm = ab_esm_tokens_gated.transpose(0, 1).unsqueeze(0)
    ab_esm_out1 = self.ab_bnorm1(self.ab_cnn1(ab_esm))
    ab_esm_out2 = self.ab_bnorm2(self.ab_cnn2(ab_esm_out1))
    ab_esm_out3 = self.ab_bnorm3(self.ab_cnn3(ab_esm_out2))

    ab_esm_features = torch.cat([ab_esm_out1, ab_esm_out2, ab_esm_out3], dim=1)
    ab_esm_features = ab_esm_features.transpose(1, 2)
    ab_lstm_out, _ = self.ab_lstm(ab_esm_features)

    ab_h2 = torch.cat((ab_lstm_out[:, :, :64 // 2], ab_lstm_out[:, :, 64 // 2:]), dim=2)
    ab_h2 = torch.squeeze(ab_h2, dim=0)
    ab_h2 = self.ab_bnorm4(ab_h2)
    return ab_h2
