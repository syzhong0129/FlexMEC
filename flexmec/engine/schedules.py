from ..configs.runtime import *

def gate_pref_weight_for_epoch(epoch_index):
    if GATE_PREF_WEIGHT <= 0.0:
        return 0.0
    if GATE_PREF_WARMUP_EPOCHS <= 0:
        return GATE_PREF_WEIGHT
    progress = min(max((epoch_index + 1) / GATE_PREF_WARMUP_EPOCHS, 0.0), 1.0)
    return GATE_PREF_WEIGHT * progress


def alpha_attr_weight_for_epoch(epoch_index):
    if ALPHA_ATTR_WEIGHT <= 0.0:
        return 0.0
    if ALPHA_ATTR_WARMUP_EPOCHS <= 0:
        return ALPHA_ATTR_WEIGHT
    progress = min(max((epoch_index + 1) / ALPHA_ATTR_WARMUP_EPOCHS, 0.0), 1.0)
    return ALPHA_ATTR_WEIGHT * progress


def flex_drive_neutral_ratio_for_epoch(epoch_index):
    if not FLEX_DRIVE_NEUTRAL_BLEND_ENABLED:
        return 0.0
    epoch_number = epoch_index + 1
    decay_epochs = max(1, int(FLEX_DRIVE_NEUTRAL_DECAY_EPOCHS))
    if epoch_number > decay_epochs:
        return 0.0
    if decay_epochs == 1:
        return FLEX_DRIVE_NEUTRAL_RATIO_FINAL
    progress = min(max((epoch_number - 1) / (decay_epochs - 1), 0.0), 1.0)
    return (
        FLEX_DRIVE_NEUTRAL_RATIO_START
        + progress * (FLEX_DRIVE_NEUTRAL_RATIO_FINAL - FLEX_DRIVE_NEUTRAL_RATIO_START)
    )


def get_gate_endpoint_gaps(model):
    gate = model.early_cdr_feature_gate
    device = gate.theta_rigid.device
    struct_idx = torch.tensor([0, 1, 2], device=device)
    seq_idx = torch.tensor([3, 4, 5], device=device)
    with torch.no_grad():
        w_rigid = gate.num_feature_groups * torch.softmax(gate.theta_rigid, dim=-1)
        w_flex = gate.num_feature_groups * torch.softmax(gate.theta_flex, dim=-1)
        rigid_gap_by_cdr = w_rigid[:, struct_idx].mean(dim=-1) - w_rigid[:, seq_idx].mean(dim=-1)
        flex_gap_by_cdr = w_flex[:, seq_idx].mean(dim=-1) - w_flex[:, struct_idx].mean(dim=-1)
    return {
        "rigid_gap": float(rigid_gap_by_cdr.mean().item()),
        "flex_gap": float(flex_gap_by_cdr.mean().item()),
    }


def get_flex_pair_bias_stats(model):
    stats = {
        "flex_pair_bias_bound_type": "nan",
        "flex_pair_bias_max": float("nan"),
        "flex_pair_lambda_mean": float("nan"),
        "flex_pair_lambda_max_abs": float("nan"),
        "flex_pair_lambda_mean_abs": float("nan"),
    }
    with torch.no_grad():
        flex_bias = getattr(model, "flex_attn_bias", None)
        if flex_bias is not None and hasattr(flex_bias, "lambda_pair"):
            lambdas = flex_bias.lambda_pair.detach().float().cpu()
            stats["flex_pair_bias_bound_type"] = getattr(flex_bias, "bound_type", "nan")
            stats["flex_pair_bias_max"] = float(getattr(flex_bias, "max_bias", float("nan")))
            stats["flex_pair_lambda_mean"] = float(lambdas.mean().item())
            stats["flex_pair_lambda_max_abs"] = float(lambdas.abs().max().item())
            stats["flex_pair_lambda_mean_abs"] = float(lambdas.abs().mean().item())
    return stats


def parse_run_folds():
    return [int(item) for item in RUN_FOLDS.split(",")]
