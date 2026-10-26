"""Trace baseline and public entrypoints with synthetic inputs.

Record model, optimizer, scheduler and save calls using tensor byte hashes.
Use --full to retain tensor values for debugging.
"""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import runpy
import sys

import numpy as np
import torch


def synthetic_samples():
    """Non-biological random fixtures; no sequences, structures or private IDs."""
    rng = np.random.RandomState(813)
    samples = []
    for index, (na, nb) in enumerate([(13, 12), (17, 18), (11, 15), (14, 12), (16, 18)]):
        sample = {"pdb_id": f"synthetic{index}", "flex_rigid": np.array([0, .02, .03, .06, .12, 1], np.float32),
                  "cdr_ranges": [[int(x[0]), int(x[-1])] for x in np.array_split(np.arange(nb), 6)],
                  "edge_AGAB": [[0, 2], [1, 3]]}
        for side, length, channels in [("AG", na, 1280), ("AB", nb, 1024)]:
            sample[f"vertex_{side}"] = rng.normal(size=(length, 62)).astype(np.float32)
            sample[f"label_{side}"] = np.array([-1, 1] + [i % 2 for i in range(length-2)], np.float32)
            sample[f"coord_{side}"] = rng.normal(size=(length, 3)).astype(np.float32)
            sample[f"edge_{side}"] = np.array([[0, 1], [1, 2]], np.int64)
            if index != 1:
                sample[f"patch_matrix_{side}"] = rng.normal(size=(length, 60)).astype(np.float32)
            if index == 2:
                sample[f"coord_{side}"][-1] = np.nan
            if side == "AG" or index != 1:
                key = "ESM2_AG" if side == "AG" else "Antiberta2_AB"
                sample[key] = rng.normal(size=(length, channels)).astype(np.float32)
        samples.append(sample)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--implementation", choices=["original", "package"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=["probe", "train", "alternate"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="Strictly load an original state_dict in the probe")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    implementation = args.implementation or ("package" if (source / "flexmec").is_dir() else "original")
    names = json.loads((Path(__file__).parent / "reference_names.json").read_text())
    env_names = names["environment_names"]
    fail_log_key = "MIPE_FAIL_LOG" if implementation == "original" else env_names["MIPE_FAIL_LOG"]
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / "run"
    split_dir = output / "splits"
    split_dir.mkdir(exist_ok=True)
    samples = synthetic_samples()
    dataset = output / "synthetic.pkl"
    with dataset.open("wb") as handle:
        pickle.dump(samples, handle)
    for filename, ids in [("fold0_train.csv", [0, 1, 2]), ("fold0_val.csv", [3]), ("test.csv", [4])]:
        (split_dir / filename).write_text("pdb_id\n" + "".join(f"synthetic{i}\n" for i in ids))
    os.environ.update(RUN_DIR=str(run_dir), OUTPUT_BASE_DIR=str(run_dir), DATA_FILE_PATH=str(dataset),
                      TEST_DATA_FILE_PATH="original", SPLIT_CSV_DIR=str(split_dir), RUN_FOLDS="0",
                      RUN_SEED="712", EPOCHS="5", ACCUM_STEPS="2", PLOT_EVERY_EPOCHS="5")
    os.environ[fail_log_key] = str(output / "failures.txt")
    if args.scenario == "alternate":
        os.environ.update(USE_LAYER_NORM_CNN="1", AG_FOCAL_ENABLED="1", AB_FOCAL_ENABLED="1",
                          AG_POS_WEIGHT_ENABLED="1", AB_POS_WEIGHT_ENABLED="1", ALPHA_ATTR_WEIGHT="1",
                          ENABLE_AB_CDR_GATE="0", ENABLE_FLEX_ATTN_BIAS="0", ENABLE_FLEX_AUX_LOSS="0",
                          FLEX_DRIVE_NEUTRAL_BLEND_ENABLED="0", EPOCHS="2", MIPE_CAPTURE_HEAD_EXPORT="1")
    # Translate only configuration names for the new implementation; keep the
    # original source and its expected environment unchanged in its own process.
    if implementation == "package":
        for old, new in env_names.items():
            if old in os.environ:
                os.environ[new] = os.environ.pop(old)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    sys.path.insert(0, str(source))
    if implementation == "original":
        import model as model_module
    else:
        from flexmec.models import network as model_module
    model_class = getattr(model_module, "MIPE" if implementation == "original" else "FlexMEC")
    events = []
    actual_save = torch.save

    def snapshot(value):
        if torch.is_tensor(value):
            cpu = value.detach().cpu().contiguous()
            data = cpu.numpy().tobytes()
            result = {"tensor": True, "shape": list(value.shape), "dtype": str(value.dtype),
                      "device": str(value.device), "stride": list(value.stride()),
                      "requires_grad": value.requires_grad, "sha256": hashlib.sha256(data).hexdigest()}
            if args.full:
                result["values"] = cpu.clone()
            return result
        if isinstance(value, np.ndarray):
            return {"array": True, "shape": list(value.shape), "dtype": str(value.dtype),
                    "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
        if isinstance(value, dict):
            return {str(k): snapshot(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return {"container": type(value).__name__, "items": [snapshot(v) for v in value]}
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, (torch.device, torch.dtype)):
            return str(value)
        if isinstance(value, str):
            return value.replace(str(output), "<OUTPUT>").replace(str(source), "<SOURCE>")
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if hasattr(value, "edge_list"):
            return {"graph": True, "edges": snapshot(value.edge_list), "num_node": snapshot(value.num_node),
                    "num_relation": snapshot(value.num_relation), "edge_weight": snapshot(value.edge_weight)}
        raise TypeError(f"Unrecorded type: {type(value)}")

    def record(name, value):
        events.append({"node": name, "value": snapshot(value)})

    def rng_state():
        return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

    def aliases(model):
        groups = {}
        def visit(module, prefix=""):
            for name, parameter in module._parameters.items():
                if parameter is not None:
                    groups.setdefault(id(parameter), []).append(prefix + name)
            for name, child in module._modules.items():
                if child is not None:
                    visit(child, prefix + name + ".")
        visit(model)
        return list(groups.values())

    original_init = model_class.__init__

    @functools.wraps(original_init)
    def traced_init(self, *a, **kw):
        original_init(self, *a, **kw)
        record("model.initial_state", self.state_dict())
        record("model.buffers", dict(self.named_buffers()))
        record("model.parameter_aliases", aliases(self))
        record("model.init_rng", rng_state())
        for name, module in self.named_modules():
            module.register_forward_pre_hook(lambda m, a, name=name: record(f"module:{name}:input", a))
            module.register_forward_hook(lambda m, a, out, name=name: record(f"module:{name}:output", out))
        for name, parameter in self.named_parameters():
            parameter.register_hook(lambda grad, name=name: record(f"gradient:{name}", grad))

    model_class.__init__ = traced_init
    for cls in (model_class, model_module.EarlyCDRFeatureGate):
        for name, method in list(cls.__dict__.items()):
            if callable(method) and name not in {"__init__", "forward"} and not name.startswith("__"):
                def wrap(method, name, cls):
                    @functools.wraps(method)
                    def traced(self, *a, **kw):
                        class_name = names["class_names"].get(cls.__name__, cls.__name__)
                        record(f"method:{class_name}.{name}:input", (a, kw))
                        result = method(self, *a, **kw)
                        record(f"method:{class_name}.{name}:output", result)
                        record(f"method:{class_name}.{name}:input_after", (a, kw))
                        return result
                    return traced
                setattr(cls, name, wrap(method, name, cls))

    original_step = torch.optim.AdamW.step

    @functools.wraps(original_step)
    def traced_step(self, *a, **kw):
        record("optimizer.gradients", [[p.grad for p in g["params"]] for g in self.param_groups])
        result = original_step(self, *a, **kw)
        record("optimizer.parameters", [g["params"] for g in self.param_groups])
        record("optimizer.state", self.state_dict())
        return result

    torch.optim.AdamW.step = traced_step
    original_schedule = torch.optim.lr_scheduler.CosineAnnealingLR.step

    @functools.wraps(original_schedule)
    def traced_schedule(self, *a, **kw):
        result = original_schedule(self, *a, **kw)
        record("scheduler.state", self.state_dict())
        return result

    torch.optim.lr_scheduler.CosineAnnealingLR.step = traced_schedule

    def traced_save(value, path, *a, **kw):
        record("save:" + str(Path(path).relative_to(output)), value)
        return actual_save(value, path, *a, **kw)

    torch.save = traced_save
    watched_returns = {
        "prepare_sample_cache", "CreateGearnetGraph", "build_edge_label_tensor",
        "compute_ag_node_loss", "compute_ab_node_loss", "edge_bce_with_neg_sampling",
        "compute_flex_loss", "compute_gate_preference_loss", "compute_alpha_feature_attribution_loss",
        "gate_pref_weight_for_epoch", "alpha_attr_weight_for_epoch", "flex_drive_neutral_ratio_for_epoch",
        "safe_evalution_prot", "evaluate_on_test_data",
    }

    def profile(frame, event, result):
        if event == "call" and frame.f_code.co_name == "evaluate_on_test_data" and str(source) in frame.f_code.co_filename:
            record("evaluation.start", frame.f_locals.get("split_name", "test"))
        if event == "return" and frame.f_code.co_name in watched_returns:
            if str(source) in frame.f_code.co_filename:
                record("function:" + frame.f_code.co_name + ":return", result)

    sys.setprofile(profile)
    if args.scenario in {"train", "alternate"}:
        # Actual entrypoint, including splitting, cache, loss accumulation, plots,
        # checkpoint selection, strict reload, and checkpoint evaluation.
        entrypoint = source / ("main.py" if implementation == "original" else "scripts/train.py")
        previous_argv = sys.argv
        try:
            sys.argv = [str(entrypoint)]
            runpy.run_path(str(entrypoint), run_name="__main__")
        finally:
            sys.argv = previous_argv
    else:
        if implementation == "original":
            import main as engine
        else:
            from flexmec.engine import runner as engine
        random.seed(712)
        np.random.seed(712)
        torch.manual_seed(712)
        model = model_class().to(device)
        actual_save(model.state_dict(), output / "initial_state.pt")
        state = torch.load(args.checkpoint, map_location=device) if args.checkpoint else model.state_dict()
        record("strict_load", tuple(model.load_state_dict(state, strict=True)))
        record("default_run_layout", (engine.RUN_CODE_DIR, engine.DEFAULT_RUN_DIR, engine.default_split_csv_dir))
        fail_log = os.environ.pop(fail_log_key)
        record("default_failure_log", model_module._get_fail_log_path())
        os.environ[fail_log_key] = fail_log
        for index, sample in enumerate(samples):
            prepared = engine.prepare_sample_cache(sample, index, device)
            record("prepared_sample", prepared)
            for training in [False, True]:
                model.train(training)
                full_capture = args.full
                args.full = False
                before = snapshot(prepared)
                args.full = full_capture
                args_model = [prepared[k] for k in ["ag_node_attr", "ag_edge_ind", "ab_node_attr", "ab_edge_ind", "ag_esm", "ab_esm", "sample_id", "cdr_spans"]]
                result = model(*args_model)
                assert result[0] is not None, "GearNet failed: a skipped sample is not a valid forward test"
                record("outputs", result)
                args.full = False
                modified = before != snapshot(prepared)
                args.full = full_capture
                record("input_modified", modified)
                record("model.mode_buffers", (training, dict(model.named_buffers())))
        for bound in ["hard", "soft", "none"]:
            model.flex_attn_bias.bound_type = bound
            with torch.no_grad():
                model.flex_attn_bias.lambda_pair.fill_(4.0)
            record("nonzero_pair_bias:" + bound, model.flex_attn_bias(torch.rand(9, device=device), torch.randn(7, 64, device=device), torch.randn(9, 64, device=device)))
        for mask in [None, torch.randn(7, 9, device=device), torch.randn(9, 7, device=device), torch.randn(16, 9, 7, device=device)]:
            record("cross_attention_masks", model.ag_crossattention(torch.randn(7, 64, device=device), torch.randn(9, 64, device=device), mask, 1.0))
        for spans in [None, ["0:2", "2:4", "4:6", "6:8", "8:10", "10:12"], {k: [-1, 99] for k in model.cdr_key_order}]:
            try:
                record("sanitize_spans", model._sanitize_cdr_spans(spans, 12, device))
            except (AssertionError, ValueError) as exc:
                record("sanitize_spans_error", (type(exc).__name__, str(exc)))
        x = torch.randn(2, 12, 122, device=device)
        record("batch_two_truncation", model._to_token_features(x))
        for length in [1, 7, 12]:
            record("default_span_legacy_helper", model._default_cdr_spans(length, device))
        for side, channels in [("ag", 1280), ("ab", 1024)]:
            method = getattr(model, "_prepare_" + side + "_esm")
            for shape in [(12, channels), (channels, 12), (1, 12, channels), (2, channels, 12)]:
                record("plm_layout:" + side, method(torch.randn(*shape, device=device), torch.zeros(12, 122, device=device)))
        for length in [1, 3]:
            sample = {f"coord_{side}": np.full((length, 3), np.nan) for side in ["AG", "AB"]}
            sample.update({f"edge_{side}": np.empty((2, 0), dtype=int) for side in ["AG", "AB"]})
            record("all_missing_graph", engine.CreateGearnetGraph(sample, None, None))
        for targets in [torch.full((4,), -1., device=device), torch.tensor([-1, 0, 1, 1.], device=device)]:
            probs = torch.tensor([.2, .4, .7, .9], device=device, requires_grad=True)
            record("node_loss", engine.compute_ag_node_loss(probs, targets, torch.nn.BCELoss()))
        for label in [torch.zeros(4, 5, device=device), torch.ones(4, 5, device=device)]:
            record("edge_loss_boundary", engine.edge_bce_with_neg_sampling(torch.full_like(label, .4), label, 3.))
        record("rank_ties", engine.rank_normalize_tensor_01(torch.tensor([1., 1., 3., 2.], device=device)))
        record("epoch_schedules", [(engine.gate_pref_weight_for_epoch(i), engine.alpha_attr_weight_for_epoch(i), engine.flex_drive_neutral_ratio_for_epoch(i)) for i in range(7)])
        record("evaluation", engine.evaluate_on_test_data(model, [samples[-1]], device))
        sample = engine.prepare_sample_cache(samples[0], 0, device)
        for broken_side in ["ag", "ab"]:
            values = [sample[k] for k in ["ag_node_attr", "ag_edge_ind", "ab_node_attr", "ab_edge_ind", "ag_esm", "ab_esm", "sample_id", "cdr_spans"]]
            values[1 if broken_side == "ag" else 3] = [None, None, None]
            failed = model(*values)
            assert failed == (None,) * 11
            record("failed_graph:" + broken_side, failed)
        record("failed_graph_log", (output / "failures.txt").read_text())
        shared = model_class(share_weight=True)
        record("shared_parameter_aliases", aliases(shared))
    sys.setprofile(None)
    record("final_rng", rng_state())
    if args.scenario in {"train", "alternate"}:
        assert sum(e["node"] == "optimizer.state" for e in events) >= 2, "No meaningful optimization updates"
    import importlib.metadata
    versions = {name: importlib.metadata.version(name) for name in ["torch", "torchdrug", "torch-geometric", "torch-scatter", "numpy", "scipy", "scikit-learn", "matplotlib", "pandas", "tqdm"]}
    versions["python"] = sys.version.split()[0]
    (output / "environment.json").write_text(json.dumps(versions, indent=2))
    if args.full:
        actual_save(events, output / "trace.pt")
    else:
        (output / "trace.json").write_text(json.dumps(events, ensure_ascii=False, allow_nan=True))
    print(f"Recorded {len(events)} events: {args.scenario}")


if __name__ == "__main__":
    main()
