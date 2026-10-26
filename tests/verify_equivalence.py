"""Isolated, exact differential verification against a SHA256-pinned original."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def first_difference(left, right, path="root"):
    if type(left) is not type(right):
        return path + ": type"
    if isinstance(left, dict):
        if list(left) != list(right):
            return path + ": keys/order"
        for key in left:
            found = first_difference(left[key], right[key], path + "." + str(key))
            if found:
                return found
    elif isinstance(left, list):
        if len(left) != len(right):
            return path + ": length"
        for i, (a, b) in enumerate(zip(left, right)):
            found = first_difference(a, b, f"{path}[{i}]")
            if found:
                return found
    elif isinstance(left, float) and math.isnan(left) and math.isnan(right):
        return None
    elif left != right:
        return path + f": {left!r} != {right!r}"
    return None


def checkpoint_policy_comparison(traces, directories):
    """Compare unchanged computation and the original MCC branch independently."""
    training, evaluations, checkpoints = [], [], []
    for trace in traces:
        stop = next(i for i, e in enumerate(trace) if e["node"] == "evaluation.start")
        training.append([e for e in trace[:stop] if not e["node"].startswith("save:")])
        checkpoints.append([e for e in trace[:stop] if e["node"].endswith("_best_mcc.pt")])
        start = next(i for i, e in enumerate(trace) if e["node"] == "evaluation.start"
                     and e["value"] == "test_best_mcc_repeat1_fold0")
        end = next(i for i in range(start + 1, len(trace))
                   if trace[i]["node"] == "function:evaluate_on_test_data:return")
        # Only the prediction artifact's name moves to the primary output name.
        segment = json.loads(json.dumps(trace[start:end + 1]).replace("test_predictions_best_mcc.pt", "test_predictions.pt"))
        evaluations.append(segment)
    for label, values in [("training", training), ("MCC checkpoint saves", checkpoints), ("MCC evaluation", evaluations)]:
        assert values[0] and values[1], f"Missing {label}"
        diff = first_difference(*values)
        assert diff is None, f"{label}: {diff}"
    original, public = [d / "run" / "perf" for d in directories]
    files = list((public / "best_models").glob("*.pt"))
    assert [p.name for p in files] == ["repeat1_fold0_best_mcc.pt"], files
    old_metrics = dict(line.split("=", 1) for line in (original / "repeat1_fold0/test_metrics_best_mcc.txt").read_text().splitlines())
    new_metrics = dict(line.split("=", 1) for line in (public / "repeat1_fold0/test_metrics.txt").read_text().splitlines())
    for key in ["best_epoch", "checkpoint_selection", "best_metric_type", "best_metric",
                "test_auprc_ag", "test_auroc_ag", "test_mcc_ag", "test_auprc_ab", "test_auroc_ab", "test_mcc_ab",
                "valid_samples", "skipped_samples"]:
        assert old_metrics[key] == new_metrics[key], key
    assert new_metrics["best_metric_type"] == "val_mcc_mean"
    assert new_metrics["checkpoint_selection"] == "highest_val_mcc_mean"
    assert new_metrics["skipped_samples"] == "0" and int(new_metrics["valid_samples"]) > 0
    assert not list((public / "repeat1_fold0").glob("test_*_best_*.txt"))
    summary = dict(line.split("=", 1) for line in (public / "best_models/repeat1_fold0_best_summary.txt").read_text().splitlines())
    with (public / "best_models/all_repeat_fold_best_summary.tsv").open() as handle:
        table = list(csv.DictReader(handle, delimiter="\t"))
    assert len(table) == 1
    for values in [summary, table[0]]:
        assert values["best_checkpoint"].endswith("_best_mcc.pt")
        assert values["best_epoch"] == new_metrics["best_epoch"]
        assert values["best_metric_type"] == "val_mcc_mean"
        assert values["checkpoint_selection"] == "highest_val_mcc_mean"
        assert values["test_mcc_ag"] == new_metrics["test_mcc_ag"]
        assert values["test_mcc_ab"] == new_metrics["test_mcc_ab"]
        assert not any(k.startswith(("best_auc_", "best_ab_mcc_", "best_ab_auc_mcc_")) for k in values)
    for name in ["gate_endpoint_gaps.csv", "metrics.txt"]:
        aa, bb = [(p / "repeat1_fold0" / name).read_text() for p in [original, public]]
        if name == "metrics.txt":
            aa = re.sub(r"checkpoint_selection=[^\s]+", "checkpoint_selection=<POLICY>", aa)
            bb = re.sub(r"checkpoint_selection=[^\s]+", "checkpoint_selection=<POLICY>", bb)
        assert aa == bb, name
    for path in (public / "repeat1_fold0").glob("*.png"):
        assert path.read_bytes() == (original / "repeat1_fold0" / path.name).read_bytes(), path.name
    return sum(len(v[0]) for v in [training, evaluations, checkpoints])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / ".verification" / "comparison")
    parser.add_argument("--scenarios", nargs="+", default=["probe", "train", "alternate"])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    manifest = json.loads((ROOT / "tests/reference_sha256.json").read_text())
    for name, expected in manifest.items():
        actual = hashlib.sha256((baseline / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"Reference changed: {name}")
    env = dict(os.environ)
    name_map = json.loads((ROOT / "tests/reference_names.json").read_text())
    # Do not accidentally inherit experiment overrides from the calling shell.
    for file in baseline.glob("*.py"):
        for key in re.findall(r'(?:_env_\w+|_set_env_default|os.environ.get)\([\s]*[\"\']([A-Z][A-Z0-9_]+)', file.read_text()):
            env.pop(key, None)
    for old, new in name_map["environment_names"].items():
        env.pop(old, None)
        env.pop(new, None)
    env.update(PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    report = {"reference_sha256": manifest, "reference_name_map": name_map,
              "criterion": "exact tensor bytes; zero tolerance", "scenarios": {}}
    for scenario in args.scenarios:
        traces = []
        directories = []
        for label, source in [("original", baseline), ("refactored", ROOT)]:
            dest = args.output.resolve() / scenario / label
            if dest.exists():
                raise SystemExit(f"Use a new output directory; refusing to mix runs: {dest}")
            dest.mkdir(parents=True)
            command = [sys.executable, str(ROOT / "tests/capture_behavior.py"), "--source", str(source),
                       "--output", str(dest), "--scenario", scenario, "--device", args.device,
                       "--implementation", "original" if label == "original" else "package"]
            if label == "refactored" and scenario == "probe":
                command += ["--checkpoint", str(directories[0] / "initial_state.pt")]
            with (dest / "process.log").open("w") as log:
                subprocess.run(command, env=env, cwd=dest, stdout=log, stderr=subprocess.STDOUT, check=True)
            traces.append(json.loads((dest / "trace.json").read_text()))
            directories.append(dest)
        compared_events = len(traces[0])
        if scenario != "probe":
            compared_events = checkpoint_policy_comparison(traces, directories)
        diff = first_difference(*traces) if scenario == "probe" else None
        if diff:
            # Each trace is ordered; the first mismatching event identifies the
            # earliest observed divergence. --full capture can retain values.
            raise AssertionError(f"{scenario}: {diff}")
        counts = {name: sum(event["node"] == name for event in traces[0]) for name in ["optimizer.state", "scheduler.state"]}
        if scenario != "probe" and counts["optimizer.state"] < 2:
            raise AssertionError("No meaningful training updates; skipped samples cannot pass")
        original_files = {str(f.relative_to(directories[0] / "run")) for f in (directories[0] / "run").rglob("*") if f.is_file()}
        new_files = {str(f.relative_to(directories[1] / "run")) for f in (directories[1] / "run").rglob("*") if f.is_file()}
        if scenario == "probe":
            assert original_files == new_files, "Output file list differs"
        for name in original_files if scenario == "probe" else []:
            a, b = [d / "run" / name for d in directories]
            if a.suffix in {".txt", ".tsv", ".csv"}:
                texts = [f.read_text().replace(str(d), "<OUTPUT>").replace(str(s), "<SOURCE>") for f, d, s in zip([a, b], directories, [baseline, ROOT])]
                assert texts[0] == texts[1], f"Text artifact differs: {name}"
            elif a.suffix == ".png":
                assert a.read_bytes() == b.read_bytes(), f"Plot artifact differs: {name}"
            # .pt contents are compared at torch.save; ZIP bytes include serializer details.
        report["scenarios"][scenario] = {"status": "PASS", "compared_events": compared_events, **counts,
                                          "checkpoint_policy": "public MCC-only; independent original MCC branch",
                                          "public_output_files": len(new_files), "max_abs_error": 0.0, "max_relative_error": 0.0}
        print(scenario, report["scenarios"][scenario], flush=True)
    report["environment"] = json.loads((directories[0] / "environment.json").read_text())
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
