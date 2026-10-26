"""Loss and metric plots for training runs."""
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_loss_history(train_losses, metrics_dir, val_losses, loss_term_order, train_loss_terms, val_loss_terms, loss_term_titles, gate_gap_history) -> None:
    """Save total losses, weighted loss terms and gate endpoint gaps."""
    plt.figure(figsize=(8, 6))
    epochs_axis = range(1, len(train_losses) + 1)
    plt.plot(epochs_axis, train_losses, marker='o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(metrics_dir, "train_loss.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    epochs_axis_val = range(1, len(val_losses) + 1)
    plt.plot(epochs_axis_val, val_losses, marker='o', label='Val Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(metrics_dir, "val_loss.png"))
    plt.close()

    # Compare weighted training and validation loss terms.
    n_loss_terms = len(loss_term_order)
    n_cols = 3
    n_rows = int(np.ceil(n_loss_terms / n_cols))
    plt.figure(figsize=(14, 4 * n_rows))
    epochs_loss_terms = range(1, len(train_loss_terms[loss_term_order[0]]) + 1)
    for idx, key in enumerate(loss_term_order, start=1):
        plt.subplot(n_rows, n_cols, idx)
        plt.plot(epochs_loss_terms, train_loss_terms[key], marker='o', label='Train')
        plt.plot(epochs_loss_terms, val_loss_terms[key], marker='o', label='Val')
        plt.title(loss_term_titles[key])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, "loss_terms.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    gap_epochs = [item["epoch"] for item in gate_gap_history]
    plt.plot(gap_epochs, [item["rigid_gap"] for item in gate_gap_history], marker='o', label='rigid_gap')
    plt.plot(gap_epochs, [item["flex_gap"] for item in gate_gap_history], marker='o', label='flex_gap')
    plt.axhline(0.0, color='black', linewidth=1, linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Endpoint gap")
    plt.title("Gate Endpoint Gaps")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, "gate_endpoint_gaps.png"))
    plt.close()


def plot_metric_history(val_auprc_ag_list, val_auroc_ag_list, val_mcc_ag_list, val_auprc_ab_list, val_auroc_ab_list, val_mcc_ab_list, metrics_dir) -> None:
    """Save antigen and antibody validation metric curves."""
    epochs = range(1, len(val_auprc_ag_list) + 1)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.plot(epochs, val_auprc_ag_list, marker='o')
    plt.title("Validation AUPRC (AG)")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(epochs, val_auroc_ag_list, marker='o')
    plt.title("Validation AUROC (AG)")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(epochs, val_mcc_ag_list, marker='o')
    plt.title("Validation MCC (AG)")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(epochs, val_auprc_ab_list, marker='o')
    plt.title("Validation AUPRC (AB)")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(epochs, val_auroc_ab_list, marker='o')
    plt.title("Validation AUROC (AB)")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.plot(epochs, val_mcc_ab_list, marker='o')
    plt.title("Validation MCC (AB)")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, "all_metrics.png"))
    plt.close()
