from .runtime import *

def evalution_prot(preds, targets, threshold=0.5):
    """Return AUPRC, AUROC and MCC for residue predictions.

    Negative labels are ignored; MCC uses the given probability threshold.
    """

    if hasattr(preds, 'detach'):  
        preds = preds.detach().cpu().numpy()
    else:
        preds = np.asarray(preds)
    preds = preds.reshape(-1)

    if hasattr(targets, 'detach'):  
        targets = targets.detach().cpu().numpy()
    else:
        targets = np.asarray(targets)
    targets = targets.reshape(-1)

    # Exclude ambiguous labels (-1) from all metrics.
    valid = targets >= 0
    preds = preds[valid]
    targets = targets[valid]

    preds_bin = (preds >= threshold).astype(int)

    auroc = roc_auc_score(targets, preds)

    auprc = average_precision_score(targets, preds)

    precision = precision_score(
        targets, preds_bin, zero_division=0
    )
    recall = recall_score(
        targets, preds_bin, zero_division=0
    )

    mcc = matthews_corrcoef(targets, preds_bin)

    return auprc, auroc, mcc


def consine_inter(A, B):
    dot_product = torch.sum(A * B, dim=1)
    norm_A = torch.norm(A, dim=1)
    norm_B = torch.norm(B, dim=1)
    cosine_similarity = dot_product / ((norm_A * norm_B) + 1e-8)
    return (cosine_similarity)
