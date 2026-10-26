from ..configs.runtime import *

def edge_bce_with_neg_sampling(pred, label, neg_ratio):
    if neg_ratio <= 0:
        return F.binary_cross_entropy(pred, label)

    pos_mask = label > 0.5
    neg_mask = ~pos_mask
    pos_count = int(pos_mask.sum().item())
    neg_count = int(neg_mask.sum().item())
    max_neg = int(pos_count * neg_ratio)

    if pos_count <= 0 or neg_count <= 0 or max_neg <= 0 or max_neg >= neg_count:
        return F.binary_cross_entropy(pred, label)

    neg_index = neg_mask.nonzero(as_tuple=False)
    sampled = neg_index[torch.randperm(neg_index.shape[0], device=label.device)[:max_neg]]
    sample_mask = pos_mask.clone()
    sample_mask[sampled[:, 0], sampled[:, 1]] = True
    return F.binary_cross_entropy(pred[sample_mask], label[sample_mask])


def binary_focal_loss_from_probs(pred_probs, targets, alpha, gamma):
    probs = torch.clamp(pred_probs, min=1e-6, max=1.0 - 1e-6)
    targets = targets.float()
    pt = torch.where(targets > 0.5, probs, 1.0 - probs)
    alpha_t = torch.where(targets > 0.5, torch.full_like(probs, alpha), torch.full_like(probs, 1.0 - alpha))
    focal = -alpha_t * torch.pow(1.0 - pt, gamma) * torch.log(pt)
    return focal.mean()


def _mask_ambiguous_labels(pred_probs, targets, pred_logits=None):
    """Mask negative labels in probabilities, targets and optional logits.

    Return the filtered tensors and a flag indicating whether any labels remain.
    """
    valid = targets >= 0
    if not valid.any():
        return pred_probs, targets, pred_logits, False
    pp = pred_probs[valid]
    tt = targets[valid]
    pl = pred_logits[valid] if pred_logits is not None else None
    return pp, tt, pl, True


def smooth_labels(targets, eps):
    """Map binary targets to eps and 1 - eps; leave them unchanged when eps <= 0."""
    if eps <= 0.0:
        return targets
    return targets * (1.0 - 2.0 * eps) + eps


def compute_ag_node_loss(pred_probs, targets, bce_loss, pred_logits=None, bce_with_logits_loss=None):
    pred_probs, targets, pred_logits, any_valid = _mask_ambiguous_labels(
        pred_probs, targets, pred_logits
    )
    if not any_valid:
        return torch.zeros((), dtype=pred_probs.dtype, device=pred_probs.device, requires_grad=True)
    if LABEL_SMOOTH_NODE:
        targets = smooth_labels(targets, LABEL_SMOOTH_EPS)
        if pred_logits is not None:
            pred_logits = pred_logits
    if AG_POS_WEIGHT_ENABLED and pred_logits is not None and bce_with_logits_loss is not None:
        bce_val = bce_with_logits_loss(pred_logits, targets)
    else:
        bce_val = bce_loss(pred_probs, targets)
    if not AG_FOCAL_ENABLED:
        return bce_val
    focal_val = binary_focal_loss_from_probs(
        pred_probs,
        targets,
        alpha=AG_FOCAL_ALPHA,
        gamma=AG_FOCAL_GAMMA,
    )
    return (1.0 - AG_FOCAL_WEIGHT) * bce_val + AG_FOCAL_WEIGHT * focal_val


def compute_ab_node_loss(pred_probs, targets, bce_loss, pred_logits=None, bce_with_logits_loss=None):
    """Compute antibody node loss with label masking and optional smoothing."""
    pred_probs, targets, pred_logits, any_valid = _mask_ambiguous_labels(
        pred_probs, targets, pred_logits
    )
    if not any_valid:
        return torch.zeros((), dtype=pred_probs.dtype, device=pred_probs.device, requires_grad=True)
    if LABEL_SMOOTH_NODE:
        targets = smooth_labels(targets, LABEL_SMOOTH_EPS)
    if AB_POS_WEIGHT_ENABLED and pred_logits is not None and bce_with_logits_loss is not None:
        bce_val = bce_with_logits_loss(pred_logits, targets)
    else:
        bce_val = bce_loss(pred_probs, targets)
    if not AB_FOCAL_ENABLED:
        return bce_val
    focal_val = binary_focal_loss_from_probs(
        pred_probs,
        targets,
        alpha=AB_FOCAL_ALPHA,
        gamma=AB_FOCAL_GAMMA,
    )
    return (1.0 - AB_FOCAL_WEIGHT) * bce_val + AB_FOCAL_WEIGHT * focal_val


def compute_flex_loss(pred_cdr, target_cdr, loss_flex):
    pred = pred_cdr.reshape(-1)
    target = target_cdr.reshape(-1)
    if pred.numel() != 6 or target.numel() != 6:
        return torch.tensor(0.0, dtype=pred.dtype, device=pred.device)

    target = torch.clamp(target, min=0.0, max=1.0)
    return loss_flex(pred, target).mean()


def compute_gate_preference_loss(model, alpha_cdr_pred, margin):
    """Regularize CDR gate endpoints using predicted flexibility ranks.

    Confident rigid and flexible groups are selected from model predictions.
    """
    gate = model.early_cdr_feature_gate
    dtype = gate.theta_rigid.dtype
    device = gate.theta_rigid.device
    if alpha_cdr_pred is None:
        return torch.zeros((), dtype=dtype, device=device)

    y = torch.clamp(alpha_cdr_pred.to(device=device, dtype=dtype), min=0.0, max=1.0).detach().reshape(-1)
    if y.numel() != gate.num_cdr:
        return torch.zeros((), dtype=dtype, device=device)

    struct_idx = torch.tensor([0, 1, 2], device=device)
    seq_idx = torch.tensor([3, 4, 5], device=device)
    w_rigid = gate.num_feature_groups * torch.softmax(gate.theta_rigid, dim=-1)
    w_flex = gate.num_feature_groups * torch.softmax(gate.theta_flex, dim=-1)

    losses = []
    rigid_mask = y <= GATE_PREF_RIGID_ALPHA_MAX
    if torch.any(rigid_mask):
        rigid_struct = w_rigid[:, struct_idx].mean(dim=-1)
        rigid_seq = w_rigid[:, seq_idx].mean(dim=-1)
        losses.append(torch.relu(margin - (rigid_struct - rigid_seq))[rigid_mask].pow(2).mean())

    flex_mask = y >= GATE_PREF_FLEX_ALPHA_MIN
    if torch.any(flex_mask):
        flex_struct = w_flex[:, struct_idx].mean(dim=-1)
        flex_seq = w_flex[:, seq_idx].mean(dim=-1)
        losses.append(torch.relu(margin - (flex_seq - flex_struct))[flex_mask].pow(2).mean())

    if not losses:
        return torch.zeros((), dtype=dtype, device=device)
    return sum(losses)


def compute_alpha_feature_attribution_loss(
    model,
    ab_node_attr,
    ab_esm,
    cdr_spans,
    flex_target,
    margin,
):
    """Contrast structural and sequence evidence for the flexibility predictor.

    Rigid CDRs favor structural groups; flexible CDRs favor sequence groups.
    """
    gate = model.early_cdr_feature_gate
    dtype = gate.theta_rigid.dtype
    device = gate.theta_rigid.device
    if flex_target is None:
        return torch.zeros((), dtype=dtype, device=device)

    y = gate.calibrate_alpha(flex_target.to(device=device, dtype=dtype)).detach().reshape(-1)
    if y.numel() != gate.num_cdr:
        return torch.zeros((), dtype=dtype, device=device)

    ab_token_feat = model._normalize_ab_feature_groups(model._to_token_features(ab_node_attr))
    ab_esm_prepared = model._prepare_ab_esm(ab_esm, ab_node_attr)
    ab_seq_tokens = model._normalize_ab_seq_tokens(
        ab_esm_prepared.squeeze(0).transpose(0, 1)
    )
    spans = model._sanitize_cdr_spans(cdr_spans, ab_token_feat.shape[0], ab_token_feat.device)

    alpha_base = gate(ab_token_feat, ab_seq_tokens, spans)[0]

    no_struct_tokens = ab_token_feat.clone()
    no_struct_tokens[:, 40:42] = 0.0
    no_struct_tokens[:, 42:62] = 0.0
    if no_struct_tokens.shape[1] > 62:
        no_struct_tokens[:, 62:] = 0.0
    alpha_no_struct = gate(no_struct_tokens, ab_seq_tokens, spans)[0]

    no_seq_tokens = ab_token_feat.clone()
    no_seq_tokens[:, 0:20] = 0.0
    no_seq_tokens[:, 20:40] = 0.0
    no_seq_embeddings = torch.zeros_like(ab_seq_tokens)
    alpha_no_seq = gate(no_seq_tokens, no_seq_embeddings, spans)[0]

    losses = []
    rigid_mask = y <= 0.4
    if torch.any(rigid_mask):
        struct_rigid_evidence = alpha_no_struct - alpha_base
        seq_rigid_evidence = alpha_no_seq - alpha_base
        diff = struct_rigid_evidence - seq_rigid_evidence
        losses.append(torch.relu(margin - diff)[rigid_mask].pow(2).mean())

    flex_mask = y >= 0.6
    if torch.any(flex_mask):
        seq_flex_evidence = alpha_base - alpha_no_seq
        struct_flex_evidence = alpha_base - alpha_no_struct
        diff = seq_flex_evidence - struct_flex_evidence
        losses.append(torch.relu(margin - diff)[flex_mask].pow(2).mean())

    if not losses:
        return torch.zeros((), dtype=dtype, device=device)
    return sum(losses)
