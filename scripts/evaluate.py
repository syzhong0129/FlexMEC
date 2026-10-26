"""Evaluate a state_dict checkpoint and export labeled-data predictions."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Import after parsing so --help works without loading scientific dependencies.
    from flexmec.engine.runner import FlexMEC, try_read_pkl, evaluate_on_test_data, torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FlexMEC().to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_on_test_data(model, try_read_pkl(args.data), device,
                                    prediction_path=str(args.output / "test_predictions.pt"))
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
