from ..configs.runtime import *
from ..data.samples import *

def safe_evalution_prot(preds, targets):
    try:
        return evalution_prot(preds, targets)
    except Exception as e:
        print(f"[warning] metric eval failed: {e}")
        return float("nan"), float("nan"), float("nan")


def compute_composite_score(metrics, prefix="test"):
    metric_keys = (
        f"{prefix}_auprc_ag",
        f"{prefix}_auroc_ag",
        f"{prefix}_mcc_ag",
        f"{prefix}_auprc_ab",
        f"{prefix}_auroc_ab",
        f"{prefix}_mcc_ab",
    )
    values = [float(metrics[k]) for k in metric_keys if np.isfinite(float(metrics.get(k, float("nan"))))]
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def compute_auc_mean_score(metrics, prefix="val"):
    metric_keys = (f"{prefix}_auroc_ag", f"{prefix}_auroc_ab")
    values = [float(metrics[k]) for k in metric_keys if np.isfinite(float(metrics.get(k, float("nan"))))]
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def compute_mcc_mean_score(metrics, prefix="val"):
    metric_keys = (f"{prefix}_mcc_ag", f"{prefix}_mcc_ab")
    values = [float(metrics[k]) for k in metric_keys if np.isfinite(float(metrics.get(k, float("nan"))))]
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def evaluate_on_test_data(
    model,
    test_data,
    device,
    split_name="test",
    prediction_path=None,
):
    output_ag_parts, target_ag_parts = [], []
    output_ab_parts, target_ab_parts = [], []
    prediction_records = []
    valid_samples = 0
    skipped_samples = 0

    model.eval()
    with torch.no_grad():
        for idx in range(len(test_data)):
            sample = prepare_sample_cache(
                test_data[idx],
                idx,
                device,
            )

            ag_node_attr = sample["ag_node_attr"]
            ag_edge_ind = sample["ag_edge_ind"]
            ag_targets = sample["ag_targets"]
            ab_node_attr = sample["ab_node_attr"]
            ab_edge_ind = sample["ab_edge_ind"]
            ab_targets = sample["ab_targets"]
            ag_esm = sample["ag_esm"]
            ab_esm = sample["ab_esm"]
            sample_id = sample["sample_id"]
            flex_target = sample["flex_target"]

            outputs = model(
                ag_node_attr,
                ag_edge_ind,
                ab_node_attr,
                ab_edge_ind,
                ag_esm,
                ab_esm,
                sample_id,
                sample["cdr_spans"],
            )
            if outputs[0] is None:
                skipped_samples += 1
                print(f"跳过测试样本 {sample_id} 的后续处理")
                continue

            valid_samples += 1
            ag_pred_cpu = torch.flatten(outputs[0]).detach().cpu()
            ag_target_cpu = ag_targets.long().detach().cpu()
            ab_pred_cpu = torch.flatten(outputs[1]).detach().cpu()
            ab_target_cpu = ab_targets.long().detach().cpu()
            output_ag_parts.append(ag_pred_cpu)
            target_ag_parts.append(ag_target_cpu)
            output_ab_parts.append(ab_pred_cpu)
            target_ab_parts.append(ab_target_cpu)
            prediction_records.append(
                {
                    "sample_id": sample_id,
                    "ag_pred": ag_pred_cpu,
                    "ag_target": ag_target_cpu,
                    "ab_pred": ab_pred_cpu,
                    "ab_target": ab_target_cpu,
                    "alpha_cdr_pred": outputs[6].detach().cpu(),
                    "w_cdr_pred": outputs[7].detach().cpu(),
                    "cdr_spans": sample["cdr_spans"],
                    "flex_target": (
                        flex_target.detach().cpu()
                        if flex_target is not None
                        else None
                    ),
                }
            )

    if not output_ag_parts or not output_ab_parts:
        metrics = {
            "test_auprc_ag": float("nan"),
            "test_auroc_ag": float("nan"),
            "test_mcc_ag": float("nan"),
            "test_auprc_ab": float("nan"),
            "test_auroc_ab": float("nan"),
            "test_mcc_ab": float("nan"),
        }
    else:
        output_ag_all = torch.cat(output_ag_parts, dim=0)
        target_ag_all = torch.cat(target_ag_parts, dim=0)
        output_ab_all = torch.cat(output_ab_parts, dim=0)
        target_ab_all = torch.cat(target_ab_parts, dim=0)
        test_auprc_ag, test_auroc_ag, test_mcc_ag = safe_evalution_prot(output_ag_all, target_ag_all)
        test_auprc_ab, test_auroc_ab, test_mcc_ab = safe_evalution_prot(output_ab_all, target_ab_all)
        metrics = {
            "test_auprc_ag": float(test_auprc_ag),
            "test_auroc_ag": float(test_auroc_ag),
            "test_mcc_ag": float(test_mcc_ag),
            "test_auprc_ab": float(test_auprc_ab),
            "test_auroc_ab": float(test_auroc_ab),
            "test_mcc_ab": float(test_mcc_ab),
        }

    metrics["test_composite"] = compute_composite_score(metrics)
    metrics["test_auc_mean"] = compute_auc_mean_score(metrics, prefix="test")
    metrics["valid_samples"] = int(valid_samples)
    metrics["skipped_samples"] = int(skipped_samples)
    metrics["split_name"] = split_name
    if prediction_path is not None:
        torch.save(prediction_records, prediction_path)
        metrics["prediction_path"] = prediction_path
    return metrics
