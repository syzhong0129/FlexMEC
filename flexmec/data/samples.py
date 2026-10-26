from ..configs.runtime import *
from .graph import CreateGearnetGraph

def to_conv1d_plm_tensor(raw_value, device):
    tensor = torch.tensor(raw_value, dtype=torch.float).to(device)
    if tensor.dim() == 2:
        # Convert residue features from (L, C) to Conv1d layout (1, C, L).
        if tensor.shape[1] > tensor.shape[0]:
            tensor = tensor.t()
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 3:
        if tensor.shape[0] != 1:
            tensor = tensor[:1]
    else:
        raise ValueError(f"Unexpected embedding shape: {tuple(tensor.shape)}")
    return tensor


def get_ag_esm_tensor(sample, device):
    for key in AG_EMBED_KEYS:
        if key in sample:
            return to_conv1d_plm_tensor(sample[key], device)
    raise KeyError(f"Missing AG embedding keys: {AG_EMBED_KEYS}")


def get_ab_esm_tensor(sample, device):
    for key in AB_EMBED_KEYS:
        if key in sample:
            return to_conv1d_plm_tensor(sample[key], device)
    return None


def get_cdr_spans(sample):
    if "cdr_ranges" in sample:
        return sample["cdr_ranges"]
    for key in ("ab_cdr_spans", "cdr_spans", "AB_CDR_SPANS"):
        if key in sample:
            return sample[key]
    return None


def get_flex_array(sample):
    if "flex_rigid" not in sample:
        return None
    try:
        target = np.asarray(sample["flex_rigid"], dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if target.size != 6:
        return None
    return np.clip(target, 0.0, 1.0)


def get_flex_target(sample, device):
    raw_target = get_flex_array(sample)
    if raw_target is None:
        return None
    return torch.tensor(raw_target, dtype=torch.float, device=device)


def rank_normalize_tensor_01(values):
    x = values.reshape(-1)
    n = int(x.numel())
    if n <= 1:
        return torch.zeros_like(values)

    unique_vals, inverse, counts = torch.unique(x, sorted=True, return_inverse=True, return_counts=True)
    if unique_vals.numel() == 1:
        return torch.full_like(values, 0.5)

    counts = counts.to(device=x.device, dtype=torch.float32)
    cumulative = torch.cumsum(counts, dim=0)
    rank_start = cumulative - counts
    avg_rank = 0.5 * (rank_start + cumulative - 1.0)
    rank_values = avg_rank.to(dtype=x.dtype, device=x.device) / float(n - 1)
    return rank_values[inverse].reshape_as(values)


def maybe_zscore_patch_matrix(patch_tensor):
    if not PATCH_MATRIX_ZSCORE or patch_tensor.numel() == 0:
        return patch_tensor
    mean = patch_tensor.mean(dim=0, keepdim=True)
    std = patch_tensor.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (patch_tensor - mean) / std


def get_sample_id(sample, default_value):
    return sample.get("pdb_id", sample.get("complex_code", default_value))


def ensure_ab_node_dim(ab_node_attr, device):
    if ab_node_attr.shape[1] == AB_NODE_DIM:
        return ab_node_attr
    if ab_node_attr.shape[1] < AB_NODE_DIM:
        pad = torch.zeros((ab_node_attr.shape[0], AB_NODE_DIM - ab_node_attr.shape[1]), dtype=ab_node_attr.dtype).to(device)
        return torch.cat([ab_node_attr, pad], dim=1)
    return ab_node_attr[:, :AB_NODE_DIM]


def _as_float_tensor(value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _as_long_tensor(value, device):
    return torch.as_tensor(value, dtype=torch.long, device=device)


def build_edge_label_tensor(sample, ag_len, ab_len, device):
    edge_label = torch.zeros((ag_len, ab_len), dtype=torch.float32, device=device)
    inter_edge_index = sample["edge_AGAB"]
    if len(inter_edge_index[0]) > 0:
        edge_label[
            _as_long_tensor(inter_edge_index[0], device),
            _as_long_tensor(inter_edge_index[1], device),
        ] = 1.0
    return edge_label


def prepare_sample_cache(sample, sample_idx, device):
    ag_node_attr = _as_float_tensor(sample["vertex_AG"], device)
    ag_targets = _as_float_tensor(sample["label_AG"], device)
    ab_node_attr = _as_float_tensor(sample["vertex_AB"], device)
    ab_targets = _as_float_tensor(sample["label_AB"], device)

    if "patch_matrix_AG" in sample:
        ag_patch_node = _as_float_tensor(sample["patch_matrix_AG"], device)
    else:
        ag_patch_node = torch.zeros((ag_node_attr.shape[0], 60), dtype=torch.float32, device=device)

    if "patch_matrix_AB" in sample:
        ab_patch_node = _as_float_tensor(sample["patch_matrix_AB"], device)
    else:
        ab_patch_node = torch.zeros((ab_node_attr.shape[0], 60), dtype=torch.float32, device=device)

    ag_patch_node = maybe_zscore_patch_matrix(ag_patch_node)
    ab_patch_node = maybe_zscore_patch_matrix(ab_patch_node)

    ag_node_attr = torch.cat([ag_node_attr, ag_patch_node], dim=1)
    ab_node_attr = torch.cat([ab_node_attr, ab_patch_node], dim=1)
    ab_node_attr = ensure_ab_node_dim(ab_node_attr, device)

    ag_esm = get_ag_esm_tensor(sample, device)
    ab_esm = get_ab_esm_tensor(sample, device)

    sample_id = get_sample_id(sample, sample_idx)

    ag_edge_ind, ab_edge_ind = CreateGearnetGraph(sample, ag_node_attr, ab_node_attr)
    edge_label = build_edge_label_tensor(sample, ag_targets.shape[0], ab_targets.shape[0], device)
    flex_target = get_flex_target(sample, device)
    assert flex_target is not None, f"Missing valid flex_rigid target for sample {sample_id}"

    return {
        "sample_id": sample_id,
        "ag_node_attr": ag_node_attr,
        "ab_node_attr": ab_node_attr,
        "ag_targets": ag_targets,
        "ab_targets": ab_targets,
        "ag_edge_ind": ag_edge_ind,
        "ab_edge_ind": ab_edge_ind,
        "ag_esm": ag_esm,
        "ab_esm": ab_esm,
        "cdr_spans": get_cdr_spans(sample),
        "flex_target": flex_target,
        "edge_label": edge_label,
    }


def resolve_cached_sample(cache_list, raw_samples, idx, device, *, split_name):
    sample = cache_list[idx]
    if sample is None:
        sample = prepare_sample_cache(
            raw_samples[idx],
            idx,
            device,
        )
        cache_list[idx] = sample
        cached_count = sum(item is not None for item in cache_list)
        if cached_count == 1 or cached_count % 100 == 0 or cached_count == len(cache_list):
            print(f"[CacheLazy] split={split_name}, cached={cached_count}/{len(cache_list)}")
    return sample
