"""Environment configuration and runtime defaults for FlexMEC."""
import os
import random

RUN_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_RUN_DIR = os.path.join(RUN_CODE_DIR, "result_noLeak", "v2_noleak_perf")


def _set_env_default(name, value):
    if not os.environ.get(name):
        os.environ[name] = str(value)


def _apply_builtin_run_config():
    # Set defaults before importing model and data modules.
    _set_env_default("CUDA_VISIBLE_DEVICES", "7")
    _set_env_default("RUN_DIR", DEFAULT_RUN_DIR)
    _set_env_default("OUTPUT_BASE_DIR", os.environ["RUN_DIR"])
    defaults = {
        "RUN_FOLDS": "0,1,2,3,4",
        "EPOCHS": "25",
        "PLOT_EVERY_EPOCHS": "5",
        "USE_SAMPLE_CACHE": "1",
        "SAMPLE_CACHE_ON_DEVICE": "1",
        "PREPARE_ALL_SAMPLES_EAGERLY": "0",
    }
    for name, value in defaults.items():
        _set_env_default(name, value)
    os.makedirs(os.path.join(os.environ["RUN_DIR"], "logs"), exist_ok=True)


_apply_builtin_run_config()

from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np
import pickle

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from ..data.io import try_read_pkl
from ..models.network import FlexMEC
from ..data.runtime import *
from ..data.splits import (
    get_split_group_id, get_split_group_set, assert_group_disjoint,
    split_train_test_data, get_val_fixed_data, get_k_fold_data_from_pdb_id_csv,
    get_k_fold_data,
)
from ..data.graph import dis_pairs, dis_pairs_safe, index_mink, CreateGearnetGraph, CreateKnearestEdge
from ..data.metrics import evalution_prot, consine_inter

# Antibody node features before GearNet.
# vertex_AB: aa_onehot(0:20) + pssm(20:40) + asa(40:42) + local_profile(42:62)
# patch_matrix_AB: 60 dims
AB_VERTEX_DIM = 62
AB_PATCH_DIM = 60
AB_NODE_DIM = AB_VERTEX_DIM + AB_PATCH_DIM
AG_EMBED_KEYS = ("ESM2_AG", "ESM1b_AG")
AB_EMBED_KEYS = ("Antiberta2_AB", "AbLang_AB")


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Input and output paths.
code_dir = RUN_CODE_DIR
ORIGINAL_DATA_FILENAME = "unbound_1099_with_flex_rigid_PSSM_Patch_20260424.pkl"
DEFAULT_EXTERNAL_TEST_DATA_FILENAME = (
    "MIPE_64test_260612_unbound_processed_data_with_flex_rigid_addPSSM_addMaSIFPatch.pkl"
)
project_root_candidates = [
    os.environ.get("PROJECT_ROOT_DIR"),
    os.path.abspath(os.path.join(code_dir, "..")),
    os.path.abspath(os.path.join(code_dir, "..", "..")),
    os.path.abspath(os.path.join(code_dir, "..", "..", "..")),
]
project_root_dir = next(
    (
        path
        for path in project_root_candidates
        if path
        and os.path.isfile(
            os.path.join(
                path,
                "data",
                ORIGINAL_DATA_FILENAME,
            )
        )
    ),
    os.path.abspath(os.path.join(code_dir, "..")),
)
default_data_file_path = os.path.join(
    project_root_dir,
    "data",
    ORIGINAL_DATA_FILENAME,
)
default_external_test_data_file_path = os.path.join(
    project_root_dir,
    "data",
    DEFAULT_EXTERNAL_TEST_DATA_FILENAME,
)
data_file_path = os.environ.get(
    "DATA_FILE_PATH",
    default_data_file_path,
)


def _resolve_test_data_file_path(default_path):
    raw_value = os.environ.get("TEST_DATA_FILE_PATH")
    if raw_value is None:
        return default_path

    value = raw_value.strip()
    if value.lower() in {"", "0", "false", "no", "none", "original", "old", "split", "csv"}:
        return None
    return value


test_data_file_path = _resolve_test_data_file_path(default_external_test_data_file_path)
default_split_csv_dir = os.path.join(DEFAULT_RUN_DIR, "data")
split_csv_dir = os.environ.get("SPLIT_CSV_DIR", default_split_csv_dir)

default_output_base_dir = os.environ.get("RUN_DIR", DEFAULT_RUN_DIR)
output_base_dir = os.environ.get("OUTPUT_BASE_DIR", default_output_base_dir)


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}

# Cross-validation and caching.
TEST_RATIO = 0.15
SEED = 42
TEST_FOLD_INDEX = 0
GROUP_SPLIT_BY_PDB = _env_flag("GROUP_SPLIT_BY_PDB", True)
USE_SAMPLE_CACHE = _env_flag("USE_SAMPLE_CACHE", True)
SAMPLE_CACHE_ON_DEVICE = _env_flag("SAMPLE_CACHE_ON_DEVICE", True)
PREPARE_ALL_SAMPLES_EAGERLY = _env_flag("PREPARE_ALL_SAMPLES_EAGERLY", False)
PLOT_EVERY_EPOCHS = int(os.environ.get("PLOT_EVERY_EPOCHS", "5"))
EPOCHS = int(os.environ.get("EPOCHS", "25"))
ACCUM_STEPS = max(1, int(os.environ.get("ACCUM_STEPS", "8")))
RUN_SEED = os.environ.get("RUN_SEED")
if RUN_SEED is not None:
    run_seed = int(RUN_SEED)
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

# Label smoothing.
LABEL_SMOOTH_EPS = float(os.environ.get("LABEL_SMOOTH_EPS", "0.05"))
LABEL_SMOOTH_NODE = _env_flag("LABEL_SMOOTH_NODE", True)
LABEL_SMOOTH_EDGE = _env_flag("LABEL_SMOOTH_EDGE", True)

# Flexibility modules and regularization.
ENABLE_FLEX_AUX_LOSS = _env_flag("ENABLE_FLEX_AUX_LOSS", True)
ENABLE_AB_CDR_GATE = _env_flag("ENABLE_AB_CDR_GATE", True)
ENABLE_FLEX_ATTN_BIAS = _env_flag("ENABLE_FLEX_ATTN_BIAS", True)
AB_CDR_GATE_BETA = float(os.environ.get("AB_CDR_GATE_BETA", "1.0"))
if AB_CDR_GATE_BETA < 0.0:
    raise ValueError(f"AB_CDR_GATE_BETA must be non-negative, got {AB_CDR_GATE_BETA}")
FLEX_AUX_WEIGHT = float(os.environ.get("FLEX_AUX_WEIGHT", "3.0"))
if not ENABLE_FLEX_AUX_LOSS:
    FLEX_AUX_WEIGHT = 0.0
GATE_PREF_WEIGHT = float(os.environ.get("GATE_PREF_WEIGHT", "2.0"))
GATE_PREF_MARGIN = float(os.environ.get("GATE_PREF_MARGIN", "0.05"))
GATE_PREF_WARMUP_EPOCHS = int(os.environ.get("GATE_PREF_WARMUP_EPOCHS", "5"))
E06_ENABLED_MODULES = tuple(
    name
    for enabled, name in (
        (ENABLE_FLEX_AUX_LOSS and FLEX_AUX_WEIGHT > 0.0, "flex_aux_loss"),
        (ENABLE_AB_CDR_GATE, "ab_cdr_gate"),
        (ENABLE_FLEX_ATTN_BIAS, "flex_pair_attention_bias"),
        (GATE_PREF_WEIGHT > 0.0, "gate_preference"),
    )
    if enabled
)
GATE_PREF_RIGID_ALPHA_MAX = float(os.environ.get("GATE_PREF_RIGID_ALPHA_MAX", "0.3"))
GATE_PREF_FLEX_ALPHA_MIN = float(os.environ.get("GATE_PREF_FLEX_ALPHA_MIN", "0.5"))
if GATE_PREF_RIGID_ALPHA_MAX >= GATE_PREF_FLEX_ALPHA_MIN:
    raise ValueError(
        "GATE_PREF_RIGID_ALPHA_MAX must be smaller than GATE_PREF_FLEX_ALPHA_MIN "
        f"(got {GATE_PREF_RIGID_ALPHA_MAX} >= {GATE_PREF_FLEX_ALPHA_MIN})"
    )
FLEX_DRIVE_NEUTRAL_BLEND_ENABLED = _env_flag("FLEX_DRIVE_NEUTRAL_BLEND_ENABLED", True)
FLEX_DRIVE_NEUTRAL_DECAY_EPOCHS = int(os.environ.get("FLEX_DRIVE_NEUTRAL_DECAY_EPOCHS", "4"))
FLEX_DRIVE_NEUTRAL_RATIO_START = float(os.environ.get("FLEX_DRIVE_NEUTRAL_RATIO_START", "0.95"))
FLEX_DRIVE_NEUTRAL_RATIO_FINAL = float(os.environ.get("FLEX_DRIVE_NEUTRAL_RATIO_FINAL", "0.05"))
ALPHA_ATTR_WEIGHT = float(os.environ.get("ALPHA_ATTR_WEIGHT", "0.0"))
ALPHA_ATTR_MARGIN = float(os.environ.get("ALPHA_ATTR_MARGIN", "0.02"))
ALPHA_ATTR_WARMUP_EPOCHS = int(os.environ.get("ALPHA_ATTR_WARMUP_EPOCHS", "5"))
RUN_FOLDS = os.environ.get("RUN_FOLDS", "0,1,2,3,4")

# Loss weights and optimization.
AG_LOSS_WEIGHT = float(os.environ.get("AG_LOSS_WEIGHT", "6.0"))
AB_LOSS_WEIGHT = float(os.environ.get("AB_LOSS_WEIGHT", "5.0"))
EDGE_AGAB_LOSS_WEIGHT = float(os.environ.get("EDGE_AGAB_LOSS_WEIGHT", "4.0"))
EDGE_ABAG_LOSS_WEIGHT = float(os.environ.get("EDGE_ABAG_LOSS_WEIGHT", "4.0"))
EDGE_NEG_RATIO = float(os.environ.get("EDGE_NEG_RATIO", "3.0"))
PATCH_MATRIX_ZSCORE = _env_flag("PATCH_MATRIX_ZSCORE", True)
AG_FOCAL_ENABLED = _env_flag("AG_FOCAL_ENABLED", False)
AG_FOCAL_ALPHA = float(os.environ.get("AG_FOCAL_ALPHA", "0.25"))
AG_FOCAL_GAMMA = float(os.environ.get("AG_FOCAL_GAMMA", "2.0"))
AG_FOCAL_WEIGHT = float(os.environ.get("AG_FOCAL_WEIGHT", "0.5"))
AG_POS_WEIGHT_ENABLED = _env_flag("AG_POS_WEIGHT_ENABLED", False)
AG_POS_RATE = float(os.environ.get("AG_POS_RATE", "0.067"))
AG_POS_WEIGHT_MAX = float(os.environ.get("AG_POS_WEIGHT_MAX", "15.0"))
AB_FOCAL_ENABLED = _env_flag("AB_FOCAL_ENABLED", False)
AB_FOCAL_ALPHA = float(os.environ.get("AB_FOCAL_ALPHA", "0.35"))
AB_FOCAL_GAMMA = float(os.environ.get("AB_FOCAL_GAMMA", "2.0"))
AB_FOCAL_WEIGHT = float(os.environ.get("AB_FOCAL_WEIGHT", "0.3"))
AB_POS_WEIGHT_ENABLED = _env_flag("AB_POS_WEIGHT_ENABLED", False)
AB_POS_RATE = float(os.environ.get("AB_POS_RATE", "0.15"))
AB_POS_WEIGHT_MAX = float(os.environ.get("AB_POS_WEIGHT_MAX", "5.0"))
LEARN_RATE = float(os.environ.get("LEARN_RATE", "5e-5"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
SCHEDULER_ETA_MIN = float(os.environ.get("SCHEDULER_ETA_MIN", "1e-6"))
# Select by validation mean MCC; environment overrides do not apply.
CHECKPOINT_METRIC = "val_mcc_mean"
