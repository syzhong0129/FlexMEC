import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("TORCHDRUG_DISABLE_OPENMP", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("KMP_AFFINITY", "disabled")
    os.environ.setdefault("KMP_USE_SHM", "0")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import torch
from torch.nn import Sequential, BatchNorm1d, Dropout, Sigmoid, Conv1d, LSTM, LayerNorm, ReLU, GELU

if sys.platform == "darwin":
    from torch.nn import Linear
else:
    try:
        from torch_geometric.nn import Linear
    except Exception:
        from torch.nn import Linear

from torchdrug import models
