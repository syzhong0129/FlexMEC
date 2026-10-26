"""Test MCC selection edge cases with real synthetic-data training.

Control reported metrics while keeping training and checkpoint evaluation real.
"""
import argparse
import json
import math
import os
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    from capture_behavior import synthetic_samples
    samples = synthetic_samples()
    dataset = output / "samples.pkl"
    dataset.write_bytes(pickle.dumps(samples))
    splits = output / "splits"
    splits.mkdir()
    for filename, ids in [("fold0_train.csv", [0, 1, 2]), ("fold0_val.csv", [3]), ("test.csv", [4])]:
        (splits / filename).write_text("pdb_id\n" + "".join(f"synthetic{i}\n" for i in ids))
    os.environ.update(RUN_DIR=str(output), OUTPUT_BASE_DIR=str(output), DATA_FILE_PATH=str(dataset),
                      TEST_DATA_FILE_PATH="original", SPLIT_CSV_DIR=str(splits), RUN_FOLDS="0",
                      RUN_SEED="712", ACCUM_STEPS="2", CHECKPOINT_METRIC="val_loss")
    sys.path.insert(0, str(ROOT))
    import torch
    from flexmec.engine import runner
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert runner.CHECKPOINT_METRIC == "val_mcc_mean", "Legacy override changed selection"
    actual_metrics = runner.safe_evalution_prot
    actual_evaluate = runner.evaluate_on_test_data
    cases = [
        ("mean_tie_invalid", [(0.25, 0.5), (0.875, -0.25), (0.125, 0.625), (float("nan"), 0.99)], 1),
        ("negative", [(-0.5, -0.25), (-0.25, -0.25)], 2),
        ("all_invalid", [(float("nan"), 0.99)], None),
    ]
    results = {}
    for name, scores, expected_epoch in cases:
        dest = output / name
        dest.mkdir()
        runner.output_base_dir = str(dest)
        runner.EPOCHS = len(scores)
        calls = []
        tests = []

        def controlled_metrics(preds, targets):
            result = actual_metrics(preds, targets)
            index = len(calls)
            calls.append(index)
            phase = index % 4
            if phase >= 2:
                return result[0], result[1], scores[index // 4][phase - 2]
            return result

        def evaluate(*a, **kw):
            tests.append(kw["split_name"])
            return actual_evaluate(*a, **kw)

        runner.safe_evalution_prot = controlled_metrics
        runner.evaluate_on_test_data = evaluate
        try:
            runner.run()
        except RuntimeError as exc:
            assert expected_epoch is None and "no finite antigen/antibody validation MCC pair" in str(exc), str(exc)
        else:
            assert expected_epoch is not None, "Invalid MCC pair was accepted"
        assert len(calls) == 4 * len(scores)
        checkpoints = list((dest / "perf/best_models").glob("*.pt"))
        if expected_epoch is None:
            assert not checkpoints and not tests
        else:
            assert [p.name for p in checkpoints] == ["repeat1_fold0_best_mcc.pt"]
            checkpoint = torch.load(checkpoints[0], map_location="cpu")
            assert checkpoint["epoch"] == expected_epoch
            assert checkpoint["best_metric_type"] == "val_mcc_mean"
            assert math.isclose(checkpoint["best_metric"], sum(scores[expected_epoch - 1]) / 2)
            assert tests == ["test_best_mcc_repeat1_fold0"]
        results[name] = "PASS"
    (output / "report.json").write_text(json.dumps(results, indent=2) + "\n")
    print("PASS:", results, "and legacy CHECKPOINT_METRIC override ignored")


if __name__ == "__main__":
    main()
