"""Verify package imports and serialization without any root-level adapters."""
import argparse
import io
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT, help="Checkout or extracted wheel to test")
    parser.add_argument("--original-state", type=Path, help="An optional original state_dict")
    args = parser.parse_args()
    source = args.source.resolve()
    for filename in ["main.py", "model.py", "utils.py", "CrossAttention.py", "flexmec/_compat.py", "flexmec/data/legacy.py"]:
        assert not (source / filename).exists(), filename
    sys.path.insert(0, str(source))
    import torch
    import flexmec
    assert source in Path(flexmec.__file__).resolve().parents
    from flexmec.models import FlexMEC
    from flexmec import models as public_models
    from flexmec.engine.runner import run
    from flexmec.data.samples import prepare_sample_cache
    assert callable(run) and callable(prepare_sample_cache)
    assert FlexMEC.__module__ == "flexmec.models.network"
    assert FlexMEC.__name__ == "FlexMEC"
    assert public_models.__all__ == ["FlexMEC"]
    model = FlexMEC()
    for module in model.modules():
        assert module.__class__.__module__ not in {"model", "CrossAttention"}
    if args.original_state:
        state = torch.load(args.original_state, map_location="cpu")
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=True)
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, map_location="cpu")
    assert isinstance(restored, FlexMEC)
    assert list(model.state_dict()) == list(restored.state_dict())
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key]), key
    assert not ({"main", "model", "utils", "CrossAttention"} & set(sys.modules))
    overrides = {"FLEXMEC_MULTIHEADS": "8", "FLEXMEC_CDR_GATE_GROUP_DIM": "8", "FLEXMEC_CDR_GATE_HEADS": "2"}
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        configured = FlexMEC()
        assert configured.multiheads == 8
        assert configured.early_cdr_feature_gate.group_dim == 8
        assert configured.early_cdr_feature_gate.cdr_selfattention.num_heads == 2
        os.environ["FLEXMEC_MULTIHEADS"] = "3"
        try:
            FlexMEC()
        except ValueError as exc:
            assert "FLEXMEC_MULTIHEADS=3" in str(exc)
        else:
            raise AssertionError("Invalid head count was accepted")
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print("PASS: package-only imports, natural class paths, new-object serialization and strict weights")
    print("PASS: FlexMEC public class and renamed configuration overrides/validation")


if __name__ == "__main__":
    main()
