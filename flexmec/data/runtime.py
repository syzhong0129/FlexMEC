import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import csv
import torch
import numpy as np


from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)


if os.environ.get("FLEXMEC_LEGACY_PATH_SUFFIX"):
    os.environ["PATH"] += os.environ["FLEXMEC_LEGACY_PATH_SUFFIX"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
