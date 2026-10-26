"""Compare fresh original evaluation with the public CLI using one old checkpoint."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-child", action="store_true")
    args = parser.parse_args()
    if args.original_child:
        name_map = json.loads((Path(__file__).parent / "reference_names.json").read_text())
        for old, new in name_map["environment_names"].items():
            if new in os.environ:
                os.environ[old] = os.environ.pop(new)
        sys.path.insert(0, str(args.baseline.resolve()))
        import main as engine
        import torch
        model = engine.MIPE().to("cpu")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        metrics = engine.evaluate_on_test_data(model, engine.try_read_pkl(args.data), torch.device("cpu"),
                                                prediction_path=str(args.output / "test_predictions.pt"))
        (args.output / "metrics.json").write_text(json.dumps(metrics))
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="-1",
               RUN_DIR=str(output / "run"), OUTPUT_BASE_DIR=str(output / "run"))
    # Model flags remain caller-supplied and are identical for the two processes.
    for name in ["original", "cli"]:
        dest = output / name
        dest.mkdir()
        if name == "original":
            command = [sys.executable, str(Path(__file__).resolve()), "--original-child", "--baseline", str(args.baseline.resolve())]
        else:
            command = [sys.executable, str(ROOT / "scripts/evaluate.py")]
        command += ["--checkpoint", str(args.checkpoint.resolve()), "--data", str(args.data.resolve()), "--output", str(dest)]
        with (dest / "process.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    import torch
    def compare(a, b):
        assert type(a) is type(b)
        if torch.is_tensor(a):
            assert a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
        elif isinstance(a, dict):
            assert list(a) == list(b)
            for key in a:
                if key != "prediction_path":
                    compare(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                compare(x, y)
        elif not (isinstance(a, float) and math.isnan(a) and math.isnan(b)):
            assert a == b
    compare(*[torch.load(output / label / "test_predictions.pt") for label in ["original", "cli"]])
    compare(*[json.loads((output / label / "metrics.json").read_text()) for label in ["original", "cli"]])
    print("PASS: fresh original evaluator and public CLI predictions/metrics exactly match")


if __name__ == "__main__":
    main()
