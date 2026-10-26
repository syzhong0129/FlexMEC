# FlexMEC

FlexMEC predicts antigen epitopes and antibody paratopes from precomputed residue features and graphs. FlexMetric uses predicted CDR flexibility to modulate antibody feature groups; FlexEnergy uses it to bias antigen-side cross-attention.

## Repository structure

```text
flexmec/
  models/     # FlexMEC network, FlexMetric, FlexEnergy and supporting layers
  data/       # Sample loading, graph construction and dataset splits
  engine/     # Training, losses, evaluation and reporting
  configs/    # Runtime configuration
configs/      # Example environment settings
scripts/      # Training and evaluation entry points; source export utility
tests/        # API, checkpoint-policy and refactor verification checks
```

The model entry point is `flexmec.models.FlexMEC`; its implementation is in [network.py](flexmec/models/network.py).

## Installation

Reference environment: Python 3.10, PyTorch 1.13.0 with CUDA 11.7, TorchDrug 0.2.1 and PyG 2.6.1. TorchDrug requires a C++ compiler, Ninja and a writable extension cache.

```bash
git clone https://github.com/syzhong0129/FlexMEC.git
cd FlexMEC
conda create -n flexmec python=3.10 pip
conda activate flexmec
pip install 'torch==1.13.0+cu117' --extra-index-url https://download.pytorch.org/whl/cu117
pip install 'torch-scatter==2.1.1+pt113cu117' 'torch-cluster==1.6.1+pt113cu117' \
  -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install -r requirements.txt
pip install --no-deps --no-build-isolation -e .
```

## Input data

Training and evaluation use a pickle containing a list of sample dictionaries. Required fields are:

- `pdb_id`: sample identifier used in split CSV files.
- `vertex_AG`, `vertex_AB`: residue features; `coord_AG`, `coord_AB`: coordinates.
- `edge_AG`, `edge_AB`: within-partner edges; `edge_AGAB`: antigen–antibody contact labels.
- `label_AG`, `label_AB`: residue labels; `flex_rigid`: six CDR flexibility targets.
- `ESM2_AG` or `ESM1b_AG`: antigen PLM embeddings.
- `cdr_ranges`: six numeric `[start, end]` pairs in H1, H2, H3, L1, L2, L3 order, using zero-based indices with inclusive ends.

`patch_matrix_AG`, `patch_matrix_AB` and antibody PLM embeddings (`Antiberta2_AB` or `AbLang_AB`) are optional in the loader and use zero-filled fallbacks when absent. See [samples.py](flexmec/data/samples.py) for loading and conversion, and [network.py](flexmec/models/network.py) for model input shapes.

Each selected fold requires `foldN_train.csv`, `foldN_val.csv` and the shared `test.csv`, each with a `pdb_id` column. Within a fold, the three splits must be disjoint and together cover every sample in the input pickle, without duplicate IDs.

## Training

The commands below require prepared data and split files. Set environment variables before starting Python; training options are defined in [configs/e06.env.example](configs/e06.env.example).

```bash
source configs/e06.env.example
export CUDA_VISIBLE_DEVICES=0
export DATA_FILE_PATH=/path/to/dataset.pkl
export SPLIT_CSV_DIR=/path/to/splits
export TEST_DATA_FILE_PATH=original
export RUN_DIR="$PWD/runs/run01"
export OUTPUT_BASE_DIR="$RUN_DIR"
export RUN_SEED=42
python scripts/train.py
```

`TEST_DATA_FILE_PATH=original` uses the samples selected by `test.csv` from `DATA_FILE_PATH`. Set it to a separate pickle path to evaluate an external test set instead; the split CSV files are still required. The example runs folds 0–4 for 25 epochs each; set `RUN_FOLDS=0` to run only fold 0.

Each fold retains the checkpoint with the highest mean antigen/antibody validation MCC (threshold 0.5), then reloads it for test evaluation. Outputs are saved under `$RUN_DIR/perf/`:

- `best_models/repeat1_foldN_best_mcc.pt`: selected checkpoint.
- `repeat1_foldN/test_metrics.txt` and `test_predictions.pt`: test metrics and predictions.
- `best_models/all_repeat_fold_best_summary.tsv`: fold summary.

## Evaluation

Use the same model configuration as training and a prepared, labeled test pickle with the fields above. This entry point evaluates labeled data; it is not an interface for raw sequences or structures.

```bash
source configs/e06.env.example
export CUDA_VISIBLE_DEVICES=0
python scripts/evaluate.py \
  --checkpoint /path/to/repeat1_fold0_best_mcc.pt \
  --data /path/to/test.pkl \
  --output /path/to/evaluation01
```

Evaluation exports `metrics.json` (including antigen/antibody AUPRC, AUROC and MCC) and `test_predictions.pt` (residue predictions, labels and CDR outputs).
