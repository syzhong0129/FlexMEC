"""Ordered cross-validation lifecycle; environment configuration is read on import."""
from ..configs.runtime import *
from ..configs.runtime import _env_flag, _set_env_default, _apply_builtin_run_config, _resolve_test_data_file_path
from ..data.samples import *
from ..data.samples import _as_float_tensor, _as_long_tensor
from .losses import *
from .losses import _mask_ambiguous_labels
from .schedules import *
from .evaluation import *

from .reporting import plot_loss_history, plot_metric_history


def run() -> None:
    """Train each fold, select its best validation mean-MCC model and test it."""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_PECAN = try_read_pkl(data_file_path)
    external_test_data_PECAN = None
    if test_data_file_path is not None:
        if not os.path.isfile(test_data_file_path):
            raise FileNotFoundError(f"External test data PKL not found: {test_data_file_path}")
        external_test_data_PECAN = try_read_pkl(test_data_file_path)
        print(
            "[Config] "
            f"external_test_data_file_path={test_data_file_path}, "
            f"external_test_samples={len(external_test_data_PECAN)}"
        )
    else:
        print("[Config] TEST_DATA_FILE_PATH requests original CSV test split")

    K = 5

    fold_best_records = []
    split_group_by = "pdb_base" if GROUP_SPLIT_BY_PDB else None
    fold_indices = parse_run_folds()
    print(f"[Config] run_folds={fold_indices}")

    for fold_run_idx in fold_indices:
        repeat_id = 0
        kfold = fold_run_idx
        split_seed = 42

        train_data_PECAN, val_data_PECAN, test_data_PECAN = get_k_fold_data_from_pdb_id_csv(
            data_PECAN,
            kfold,
            split_dir=split_csv_dir,
        )
        test_data_source = "csv_split"
        effective_test_data_file_path = data_file_path
        if external_test_data_PECAN is not None:
            original_test_samples = len(test_data_PECAN)
            test_data_PECAN = external_test_data_PECAN
            test_data_source = "external_pkl"
            effective_test_data_file_path = test_data_file_path
            train_groups_now = get_split_group_set(train_data_PECAN)
            val_groups_now = get_split_group_set(val_data_PECAN)
            test_groups_now = get_split_group_set(test_data_PECAN)
            train_overlap = train_groups_now.intersection(test_groups_now)
            val_overlap = val_groups_now.intersection(test_groups_now)
            if train_overlap or val_overlap:
                print(
                    "[warning] external test groups overlap fixed CSV split: "
                    f"train_overlap={sorted(train_overlap)[:10]}, "
                    f"val_overlap={sorted(val_overlap)[:10]}"
                )
            print(
                "[TestData] "
                f"fold={kfold}, source={test_data_source}, "
                f"external_samples={len(test_data_PECAN)}, "
                f"original_csv_test_samples={original_test_samples}, "
                f"path={effective_test_data_file_path}"
            )


        run_tag = f"repeat{repeat_id + 1}_fold{kfold}"
        metrics_dir = os.path.join(output_base_dir, "perf", run_tag)
        best_dir = os.path.join(output_base_dir, "perf", "best_models")
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(best_dir, exist_ok=True)

        train_losses = []
        val_losses = []
        val_task_losses = []

        model = FlexMEC().to(device)
        model.enable_ab_cdr_gate = ENABLE_AB_CDR_GATE
        model.enable_flex_attn_bias = ENABLE_FLEX_ATTN_BIAS
        model.ab_cdr_gate_beta = AB_CDR_GATE_BETA
        print(
            "[Config] "
            f"data_file_path={data_file_path}, "
            f"test_data_source={test_data_source}, "
            f"test_data_file_path={effective_test_data_file_path}, "
            f"output_base_dir={output_base_dir}, "
            f"enabled_modules={','.join(E06_ENABLED_MODULES) if E06_ENABLED_MODULES else 'none'}, "
            f"enable_ab_cdr_gate={ENABLE_AB_CDR_GATE}, "
            f"ab_cdr_gate_beta={AB_CDR_GATE_BETA:g}, "
            f"enable_flex_attn_bias={ENABLE_FLEX_ATTN_BIAS}, "
            f"flex_aux_weight={FLEX_AUX_WEIGHT:g}, "
            f"gate_pref_weight={GATE_PREF_WEIGHT:g}, "
            f"ag_loss_weight={AG_LOSS_WEIGHT:g}, "
            f"ab_loss_weight={AB_LOSS_WEIGHT:g}, "
            f"edge_agab_loss_weight={EDGE_AGAB_LOSS_WEIGHT:g}, "
            f"edge_abag_loss_weight={EDGE_ABAG_LOSS_WEIGHT:g}, "
            f"edge_neg_ratio={EDGE_NEG_RATIO:g}, "
            f"ag_pos_weight_enabled={AG_POS_WEIGHT_ENABLED}, "
            f"ag_focal_enabled={AG_FOCAL_ENABLED}, "
            f"ab_pos_weight_enabled={AB_POS_WEIGHT_ENABLED}, "
            f"ab_focal_enabled={AB_FOCAL_ENABLED}, "
            f"learn_rate={LEARN_RATE:g}, "
            f"weight_decay={WEIGHT_DECAY:g}, "
            f"scheduler_eta_min={SCHEDULER_ETA_MIN:g}"
        )
        loss_BCE = torch.nn.BCELoss()
        loss_BCE_ag_logits = None
        loss_BCE_ab_logits = None
        if AG_POS_WEIGHT_ENABLED:
            ag_pos_weight = torch.tensor([(1.0 - AG_POS_RATE) / AG_POS_RATE], dtype=torch.float, device=device)
            ag_pos_weight = torch.clamp(ag_pos_weight, max=AG_POS_WEIGHT_MAX)
            loss_BCE_ag_logits = torch.nn.BCEWithLogitsLoss(pos_weight=ag_pos_weight)
            print(f"[AG pos_weight] enabled=True, pos_weight={float(ag_pos_weight.item()):.4f}")
        if AB_POS_WEIGHT_ENABLED:
            ab_pos_weight = torch.tensor([(1.0 - AB_POS_RATE) / AB_POS_RATE], dtype=torch.float, device=device)
            ab_pos_weight = torch.clamp(ab_pos_weight, max=AB_POS_WEIGHT_MAX)
            loss_BCE_ab_logits = torch.nn.BCEWithLogitsLoss(pos_weight=ab_pos_weight)
            print(f"[AB pos_weight] enabled=True, pos_weight={float(ab_pos_weight.item()):.4f}")
        loss_flex = torch.nn.SmoothL1Loss(reduction='none')
        flex_aux_weight = FLEX_AUX_WEIGHT
        ag_loss_weight = AG_LOSS_WEIGHT
        learnrate = LEARN_RATE
        optim = torch.optim.AdamW(model.parameters(), learnrate, weight_decay=WEIGHT_DECAY)
        epoch = EPOCHS
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epoch, eta_min=SCHEDULER_ETA_MIN)

        loss_term_order = [
            "ag_node_bce_x6",
            "ab_node_bce_x5",
            "edge_agab_bce_x8",
            "edge_abag_bce_x8",
            "flex_aux",
            "gate_pref",
            "alpha_attr",
        ]
        loss_term_titles = {
            "ag_node_bce_x6": f"{ag_loss_weight:g} * BCE(outputs[0], AG)",
            "ab_node_bce_x5": f"{AB_LOSS_WEIGHT:g} * BCE(outputs[1], AB)",
            "edge_agab_bce_x8": f"{EDGE_AGAB_LOSS_WEIGHT:g} * BCE(outputs[2], edge)",
            "edge_abag_bce_x8": f"{EDGE_ABAG_LOSS_WEIGHT:g} * BCE(outputs[3].T, edge)",
            "flex_aux": f"{flex_aux_weight:g} * flex_loss",
            "gate_pref": f"{GATE_PREF_WEIGHT:g} * gate_pref_loss",
            "alpha_attr": f"{ALPHA_ATTR_WEIGHT:g} * alpha_attr_loss",
        }
        train_loss_terms = {key: [] for key in loss_term_order}
        val_loss_terms = {key: [] for key in loss_term_order}
        gate_gap_history = []

        cache_device = device
        train_sample_cache = [None] * len(train_data_PECAN)
        val_sample_cache = [None] * len(val_data_PECAN)


        # Select exactly one checkpoint per fold by validation mean MCC.
        best_checkpoint_specs = {
            "mcc": {
                "path": os.path.join(best_dir, f"{run_tag}_best_mcc.pt"),
                "score": -float("inf"),
                "epoch": -1,
                "metric_type": CHECKPOINT_METRIC,
                "checkpoint_selection": "highest_val_mcc_mean",
                "metrics": None,
            },
        }
        best_checkpoint_path = best_checkpoint_specs["mcc"]["path"]
        if os.path.isfile(best_checkpoint_path):
            os.remove(best_checkpoint_path)

        val_auprc_ag_list, val_auroc_ag_list, val_mcc_ag_list = [], [], []
        val_auprc_ab_list, val_auroc_ab_list, val_mcc_ab_list = [], [], []

        torch.cuda.empty_cache()
        for e in tqdm(range(epoch)):
            gate_pref_weight_eff = gate_pref_weight_for_epoch(e)
            alpha_attr_weight_eff = alpha_attr_weight_for_epoch(e)
            flex_drive_neutral_ratio_eff = flex_drive_neutral_ratio_for_epoch(e)
            model.flex_drive_neutral_ratio = flex_drive_neutral_ratio_eff
            result_loss_train_all = 0.0
            train_auprc = 0.0
            train_auroc = 0.0
            train_precision = 0.0
            train_recall = 0.0
            train_mcc = 0.0
            ab_train_auprc = 0.0
            ab_train_auroc = 0.0
            ab_train_precision = 0.0
            ab_train_recall = 0.0
            ab_train_mcc = 0.0
            train_term_sums = {key: 0.0 for key in loss_term_order}
            train_valid_samples = 0
            output_ag_parts, target_ag_parts = [], []
            output_ab_parts, target_ab_parts = [], []

            model.train()
            optim.zero_grad(set_to_none=True)
            for i in range(len(train_data_PECAN)):
                sample = resolve_cached_sample(
                    train_sample_cache,
                    train_data_PECAN,
                    i,
                    device,
                    split_name=f"train_repeat{repeat_id + 1}_fold{kfold}",
                )

                ag_node_attr = sample["ag_node_attr"]
                ag_edge_ind = sample["ag_edge_ind"]
                ag_targets = sample["ag_targets"]
                ab_node_attr = sample["ab_node_attr"]
                ab_edge_ind = sample["ab_edge_ind"]
                ab_targets = sample["ab_targets"]
                ag_esm = sample["ag_esm"]
                ab_esm = sample["ab_esm"]
                sample_id = sample["sample_id"]
                flex_target = sample["flex_target"]
                outputs = model(
                    ag_node_attr,
                    ag_edge_ind,
                    ab_node_attr,
                    ab_edge_ind,
                    ag_esm,
                    ab_esm,
                    sample_id,
                    sample["cdr_spans"],
                )

                # Skip samples whose graph encoding failed.
                if outputs[0] is None:
                    print(f"跳过样本 {sample_id} 的后续处理")
                    continue

                edge_label = sample["edge_label"]
                ag_pred = (outputs[0]).squeeze(dim=1)
                ag_logits = None
                if len(outputs) > 9 and outputs[9] is not None:
                    ag_logits = outputs[9].squeeze(dim=1)
                ab_logits = None
                if len(outputs) > 10 and outputs[10] is not None:
                    ab_logits = outputs[10].squeeze(dim=1)
                zero_loss = torch.zeros((), dtype=ag_pred.dtype, device=ag_pred.device)
                if flex_aux_weight > 0.0:
                    flex_target_calibrated = model.early_cdr_feature_gate.calibrate_alpha(flex_target)
                    flex_loss = compute_flex_loss(
                        outputs[6],
                        flex_target_calibrated,
                        loss_flex,
                    )
                    loss_flex_term = flex_aux_weight * flex_loss
                else:
                    flex_loss = zero_loss
                    loss_flex_term = zero_loss
                loss_ag_raw = compute_ag_node_loss(
                    ag_pred,
                    ag_targets,
                    loss_BCE,
                    pred_logits=ag_logits,
                    bce_with_logits_loss=loss_BCE_ag_logits,
                )
                loss_ag_term = ag_loss_weight * loss_ag_raw
                loss_ab_term = AB_LOSS_WEIGHT * compute_ab_node_loss(
                    (outputs[1]).squeeze(dim=1),
                    ab_targets,
                    loss_BCE,
                    pred_logits=ab_logits,
                    bce_with_logits_loss=loss_BCE_ab_logits,
                )
                edge_agab_raw = edge_bce_with_neg_sampling(outputs[2], edge_label, EDGE_NEG_RATIO)
                edge_abag_raw = edge_bce_with_neg_sampling(outputs[3].t(), edge_label, EDGE_NEG_RATIO)
                loss_edge_agab_term = EDGE_AGAB_LOSS_WEIGHT * edge_agab_raw
                loss_edge_abag_term = EDGE_ABAG_LOSS_WEIGHT * edge_abag_raw
                gate_pref_loss = (
                    compute_gate_preference_loss(model, outputs[6], GATE_PREF_MARGIN)
                    if gate_pref_weight_eff > 0.0
                    else zero_loss
                )
                loss_gate_pref_term = gate_pref_weight_eff * gate_pref_loss
                alpha_attr_loss = (
                    compute_alpha_feature_attribution_loss(
                        model,
                        ab_node_attr,
                        ab_esm,
                        sample["cdr_spans"],
                        flex_target,
                        ALPHA_ATTR_MARGIN,
                    )
                    if alpha_attr_weight_eff > 0.0
                    else zero_loss
                )
                loss_alpha_attr_term = alpha_attr_weight_eff * alpha_attr_loss
                result_loss = (
                    loss_ag_term
                    + loss_ab_term
                    + loss_edge_agab_term
                    + loss_edge_abag_term
                    + loss_flex_term
                    + loss_gate_pref_term
                    + loss_alpha_attr_term
                )
                (result_loss / ACCUM_STEPS).backward()
                if ((i + 1) % ACCUM_STEPS == 0 or i == len(train_data_PECAN) - 1):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                    optim.zero_grad(set_to_none=True)
                result_loss_train_all += float(result_loss.item())
                train_valid_samples += 1
                train_term_sums["ag_node_bce_x6"] += float(loss_ag_term.item())
                train_term_sums["ab_node_bce_x5"] += float(loss_ab_term.item())
                train_term_sums["edge_agab_bce_x8"] += float(loss_edge_agab_term.item())
                train_term_sums["edge_abag_bce_x8"] += float(loss_edge_abag_term.item())
                train_term_sums["flex_aux"] += float(loss_flex_term.item())
                train_term_sums["gate_pref"] += float(loss_gate_pref_term.item())
                train_term_sums["alpha_attr"] += float(loss_alpha_attr_term.item())


                output_ag_parts.append(torch.flatten(outputs[0]).detach())
                target_ag_parts.append(ag_targets.long().detach())
                output_ab_parts.append(torch.flatten(outputs[1]).detach())
                target_ab_parts.append(ab_targets.long().detach())

            output_ag = torch.cat(output_ag_parts, dim=0) if output_ag_parts else None
            target_ag = torch.cat(target_ag_parts, dim=0) if target_ag_parts else None
            output_ab = torch.cat(output_ab_parts, dim=0) if output_ab_parts else None
            target_ab = torch.cat(target_ab_parts, dim=0) if target_ab_parts else None


            train_denominator = max(train_valid_samples, 1)
            train_losses.append(result_loss_train_all / train_denominator)
            for key in loss_term_order:
                train_loss_terms[key].append(train_term_sums[key] / train_denominator)

            model.eval()
            val_total_loss = 0
            val_total_loss_all = 0
            val_total_task_loss_all = 0
            right_number = 0

            val_auprc = 0.0
            val_auroc = 0.0
            val_precision = 0.0
            val_recall = 0.0
            val_mcc = 0.0
            ab_val_auprc = 0.0
            ab_val_auroc = 0.0
            ab_val_precision = 0.0
            ab_val_recall = 0.0
            ab_val_mcc = 0.0
            val_term_sums = {key: 0.0 for key in loss_term_order}
            val_valid_samples = 0
            output_ag_val_parts, target_ag_val_parts = [], []
            output_ab_val_parts, target_ab_val_parts = [], []

            with torch.no_grad():
                for j in range(len(val_data_PECAN)):
                    sample = resolve_cached_sample(
                        val_sample_cache,
                        val_data_PECAN,
                        j,
                        device,
                        split_name=f"val_repeat{repeat_id + 1}_fold{kfold}",
                    )

                    ag_node_attr = sample["ag_node_attr"]
                    ag_edge_ind = sample["ag_edge_ind"]
                    ag_targets = sample["ag_targets"]
                    ab_node_attr = sample["ab_node_attr"]
                    ab_edge_ind = sample["ab_edge_ind"]
                    ab_targets = sample["ab_targets"]
                    ag_esm = sample["ag_esm"]
                    ab_esm = sample["ab_esm"]
                    val_sample_id = sample["sample_id"]
                    val_flex_target = sample["flex_target"]

                    outputs = model(
                        ag_node_attr,
                        ag_edge_ind,
                        ab_node_attr,
                        ab_edge_ind,
                        ag_esm,
                        ab_esm,
                        val_sample_id,
                        sample["cdr_spans"],
                    )

                    # Skip samples whose graph encoding failed.
                    if outputs[0] is None:
                        print(f"跳过验证样本索引 {j} 的后续处理")
                        continue

                    edge_label = sample["edge_label"]
                    val_ag_pred = (outputs[0]).squeeze(dim=1)
                    val_ag_logits = None
                    if len(outputs) > 9 and outputs[9] is not None:
                        val_ag_logits = outputs[9].squeeze(dim=1)
                    val_ab_logits = None
                    if len(outputs) > 10 and outputs[10] is not None:
                        val_ab_logits = outputs[10].squeeze(dim=1)
                    val_zero_loss = torch.zeros((), dtype=val_ag_pred.dtype, device=val_ag_pred.device)
                    if flex_aux_weight > 0.0:
                        val_flex_target_calibrated = model.early_cdr_feature_gate.calibrate_alpha(val_flex_target)
                        val_flex_loss = compute_flex_loss(
                            outputs[6],
                            val_flex_target_calibrated,
                            loss_flex,
                        )
                        val_loss_flex_term = flex_aux_weight * val_flex_loss
                    else:
                        val_flex_loss = val_zero_loss
                        val_loss_flex_term = val_zero_loss
                    val_loss_ag_raw = compute_ag_node_loss(
                        val_ag_pred,
                        ag_targets,
                        loss_BCE,
                        pred_logits=val_ag_logits,
                        bce_with_logits_loss=loss_BCE_ag_logits,
                    )
                    val_loss_ag_term = ag_loss_weight * val_loss_ag_raw
                    val_loss_ab_term = AB_LOSS_WEIGHT * compute_ab_node_loss(
                        (outputs[1]).squeeze(dim=1),
                        ab_targets,
                        loss_BCE,
                        pred_logits=val_ab_logits,
                        bce_with_logits_loss=loss_BCE_ab_logits,
                    )
                    val_edge_agab_raw = edge_bce_with_neg_sampling(outputs[2], edge_label, EDGE_NEG_RATIO)
                    val_edge_abag_raw = edge_bce_with_neg_sampling(outputs[3].t(), edge_label, EDGE_NEG_RATIO)
                    val_loss_edge_agab_term = EDGE_AGAB_LOSS_WEIGHT * val_edge_agab_raw
                    val_loss_edge_abag_term = EDGE_ABAG_LOSS_WEIGHT * val_edge_abag_raw
                    val_gate_pref_loss = (
                        compute_gate_preference_loss(model, outputs[6], GATE_PREF_MARGIN)
                        if gate_pref_weight_eff > 0.0
                        else val_zero_loss
                    )
                    val_loss_gate_pref_term = gate_pref_weight_eff * val_gate_pref_loss
                    val_alpha_attr_loss = (
                        compute_alpha_feature_attribution_loss(
                            model,
                            ab_node_attr,
                            ab_esm,
                            sample["cdr_spans"],
                            val_flex_target,
                            ALPHA_ATTR_MARGIN,
                        )
                        if alpha_attr_weight_eff > 0.0
                        else val_zero_loss
                    )
                    val_loss_alpha_attr_term = alpha_attr_weight_eff * val_alpha_attr_loss
                    val_task_loss = (
                        val_loss_ag_term
                        + val_loss_ab_term
                        + val_loss_edge_agab_term
                        + val_loss_edge_abag_term
                    )
                    val_result_loss = (
                        val_task_loss
                        + val_loss_flex_term
                        + val_loss_gate_pref_term
                        + val_loss_alpha_attr_term
                    )
                    val_result_loss_j = float(val_result_loss.item())
                    val_total_loss_all = val_total_loss_all + val_result_loss_j
                    val_total_task_loss_all += float(val_task_loss.item())
                    val_valid_samples += 1
                    val_term_sums["ag_node_bce_x6"] += float(val_loss_ag_term.item())
                    val_term_sums["ab_node_bce_x5"] += float(val_loss_ab_term.item())
                    val_term_sums["edge_agab_bce_x8"] += float(val_loss_edge_agab_term.item())
                    val_term_sums["edge_abag_bce_x8"] += float(val_loss_edge_abag_term.item())
                    val_term_sums["flex_aux"] += float(val_loss_flex_term.item())
                    val_term_sums["gate_pref"] += float(val_loss_gate_pref_term.item())
                    val_term_sums["alpha_attr"] += float(val_loss_alpha_attr_term.item())

                    output_ag_val_parts.append(torch.flatten(outputs[0]).detach())
                    target_ag_val_parts.append(ag_targets.long().detach())
                    output_ab_val_parts.append(torch.flatten(outputs[1]).detach())
                    target_ab_val_parts.append(ab_targets.long().detach())

                output_ag_val = torch.cat(output_ag_val_parts, dim=0) if output_ag_val_parts else None
                target_ag_val = torch.cat(target_ag_val_parts, dim=0) if target_ag_val_parts else None
                output_ab_val = torch.cat(output_ab_val_parts, dim=0) if output_ab_val_parts else None
                target_ab_val = torch.cat(target_ab_val_parts, dim=0) if target_ab_val_parts else None


                val_denominator = max(val_valid_samples, 1)
                val_losses.append(val_total_loss_all / val_denominator)
                val_task_losses.append(val_total_task_loss_all / val_denominator)
                for key in loss_term_order:
                    val_loss_terms[key].append(val_term_sums[key] / val_denominator)
                gate_gaps = get_gate_endpoint_gaps(model)
                flex_pair_stats = get_flex_pair_bias_stats(model)
                gate_gap_history.append(
                    {
                        "epoch": e + 1,
                        "gate_pref_weight_eff": gate_pref_weight_eff,
                        "alpha_attr_weight_eff": alpha_attr_weight_eff,
                        **gate_gaps,
                        **flex_pair_stats,
                    }
                )
                gap_path = os.path.join(metrics_dir, "gate_endpoint_gaps.csv")
                with open(gap_path, "w", encoding="utf-8") as fgap:
                    fgap.write(
                        "epoch,gate_pref_weight_eff,"
                        "alpha_attr_weight_eff,rigid_gap,flex_gap,"
                        "flex_pair_bias_bound_type,flex_pair_bias_max,"
                        "flex_pair_lambda_mean,flex_pair_lambda_max_abs,"
                        "flex_pair_lambda_mean_abs\n"
                    )
                    for item in gate_gap_history:
                        fgap.write(
                            f"{item['epoch']},{item['gate_pref_weight_eff']},"
                            f"{item['alpha_attr_weight_eff']},"
                            f"{item['rigid_gap']},{item['flex_gap']},"
                            f"{item['flex_pair_bias_bound_type']},"
                            f"{item['flex_pair_bias_max']},"
                            f"{item['flex_pair_lambda_mean']},"
                            f"{item['flex_pair_lambda_max_abs']},"
                            f"{item['flex_pair_lambda_mean_abs']}\n"
                        )

                scheduler.step()

                if True:
                    should_plot = (
                        (e + 1) % PLOT_EVERY_EPOCHS == 0
                        or e + 1 == epoch
                    )
                    if should_plot:
                        plot_loss_history(train_losses, metrics_dir, val_losses, loss_term_order, train_loss_terms, val_loss_terms, loss_term_titles, gate_gap_history)

                    if (
                        output_ag is None
                        or target_ag is None
                        or output_ab is None
                        or target_ab is None
                        or output_ag_val is None
                        or target_ag_val is None
                        or output_ab_val is None
                        or target_ab_val is None
                    ):
                        print(f"[warning] fold={kfold}, epoch={e+1}: no valid samples for metrics, set to NaN")
                        auprc_ag = auroc_ag = mcc_ag = float("nan")
                        auprc_ab = auroc_ab = mcc_ab = float("nan")
                        val_auprc_ag = val_auroc_ag = val_mcc_ag = float("nan")
                        val_auprc_ab = val_auroc_ab = val_mcc_ab = float("nan")
                    else:
                        auprc_ag, auroc_ag, mcc_ag = safe_evalution_prot(output_ag, target_ag)
                        auprc_ab, auroc_ab, mcc_ab = safe_evalution_prot(output_ab, target_ab)
                        val_auprc_ag, val_auroc_ag, val_mcc_ag = safe_evalution_prot(output_ag_val, target_ag_val)
                        val_auprc_ab, val_auroc_ab, val_mcc_ab = safe_evalution_prot(output_ab_val, target_ab_val)

                    val_auprc_ag_list.append(val_auprc_ag)
                    val_auroc_ag_list.append(val_auroc_ag)
                    val_mcc_ag_list.append(val_mcc_ag)
                    val_auprc_ab_list.append(val_auprc_ab)
                    val_auroc_ab_list.append(val_auroc_ab)
                    val_mcc_ab_list.append(val_mcc_ab)

                    if should_plot:
                        plot_metric_history(val_auprc_ag_list, val_auroc_ag_list, val_mcc_ag_list, val_auprc_ab_list, val_auroc_ab_list, val_mcc_ab_list, metrics_dir)

                    _metrics_now = {
                        "val_auprc_ag": float(val_auprc_ag),
                        "val_auroc_ag": float(val_auroc_ag),
                        "val_mcc_ag": float(val_mcc_ag),
                        "val_auprc_ab": float(val_auprc_ab),
                        "val_auroc_ab": float(val_auroc_ab),
                        "val_mcc_ab": float(val_mcc_ab),
                    }
                    _metrics_now["val_composite"] = compute_composite_score(_metrics_now, prefix="val")
                    _metrics_now["val_auc_mean"] = compute_auc_mean_score(_metrics_now, prefix="val")
                    _metrics_now["val_mcc_mean"] = compute_mcc_mean_score(_metrics_now, prefix="val")
                    if not all(np.isfinite(_metrics_now[key]) for key in ("val_mcc_ag", "val_mcc_ab")):
                        _metrics_now["val_mcc_mean"] = float("nan")
                    val_ab_auc_mcc_values = [
                        float(_metrics_now.get("val_auroc_ab", float("nan"))),
                        float(_metrics_now.get("val_mcc_ab", float("nan"))),
                    ]
                    val_ab_auc_mcc_values = [
                        item for item in val_ab_auc_mcc_values if np.isfinite(item)
                    ]
                    _metrics_now["val_ab_auc_mcc"] = (
                        float(np.mean(val_ab_auc_mcc_values))
                        if val_ab_auc_mcc_values
                        else float("nan")
                    )
                    current_val_loss = float(val_losses[-1]) if val_losses else float("nan")
                    current_val_task_loss = float(val_task_losses[-1]) if val_task_losses else float("nan")
                    _metrics_now["val_task_loss"] = current_val_task_loss
                    current_best_metric = _metrics_now["val_mcc_mean"]
                    checkpoint_selection = "highest_val_mcc_mean"

                    metrics_line = [
                        f"repeat={repeat_id + 1}",
                        f"fold={kfold}",
                        f"epoch={e + 1}",
                        f"val_loss={current_val_loss:.6g}",
                        f"val_task_loss={current_val_task_loss:.6g}",
                        f"val_auprc_ag={_metrics_now['val_auprc_ag']:.3f}",
                        f"val_auroc_ag={_metrics_now['val_auroc_ag']:.3f}",
                        f"val_mcc_ag={_metrics_now['val_mcc_ag']:.3f}",
                        f"val_auprc_ab={_metrics_now['val_auprc_ab']:.3f}",
                        f"val_auroc_ab={_metrics_now['val_auroc_ab']:.3f}",
                        f"val_mcc_ab={_metrics_now['val_mcc_ab']:.3f}",
                        f"val_auc_mean={_metrics_now['val_auc_mean']:.3f}",
                        f"val_mcc_mean={_metrics_now['val_mcc_mean']:.3f}",
                        f"val_ab_auc_mcc={_metrics_now['val_ab_auc_mcc']:.3f}",
                        f"val_composite={_metrics_now['val_composite']:.3f}",
                        f"gate_pref_weight_eff={gate_pref_weight_eff:.6g}",
                        f"ab_cdr_gate_beta={AB_CDR_GATE_BETA:.6g}",
                        f"gate_pref_rigid_alpha_max={GATE_PREF_RIGID_ALPHA_MAX:.6g}",
                        f"gate_pref_flex_alpha_min={GATE_PREF_FLEX_ALPHA_MIN:.6g}",
                        f"alpha_attr_weight_eff={alpha_attr_weight_eff:.6g}",
                        f"flex_drive_neutral_ratio_eff={flex_drive_neutral_ratio_eff:.6g}",
                        f"rigid_gap={gate_gaps['rigid_gap']:.6g}",
                        f"flex_gap={gate_gaps['flex_gap']:.6g}",
                        f"flex_pair_bias_bound_type={flex_pair_stats['flex_pair_bias_bound_type']}",
                        f"flex_pair_bias_max={flex_pair_stats['flex_pair_bias_max']:.6g}",
                        f"flex_pair_lambda_mean={flex_pair_stats['flex_pair_lambda_mean']:.6g}",
                        f"flex_pair_lambda_max_abs={flex_pair_stats['flex_pair_lambda_max_abs']:.6g}",
                        f"flex_pair_lambda_mean_abs={flex_pair_stats['flex_pair_lambda_mean_abs']:.6g}",
                        f"checkpoint_selection={checkpoint_selection}",
                    ]
                    with open(os.path.join(metrics_dir, "metrics.txt"), "a", encoding="utf-8") as ftxt:
                        ftxt.write("\t".join(metrics_line) + "\n")

                    # Save only the highest validation mean-MCC checkpoint.
                    current_val_composite = float(_metrics_now["val_composite"])
                    current_val_mcc_mean = float(_metrics_now["val_mcc_mean"])
                    current_val_auc_mean = float(_metrics_now["val_auc_mean"])
                    current_val_ab_auc_mcc = float(_metrics_now["val_ab_auc_mcc"])
                    current_gap_score = float(gate_gaps["rigid_gap"] + gate_gaps["flex_gap"])
                    current_checkpoint = {
                        "epoch": e + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                        "best_metric_type": CHECKPOINT_METRIC,
                        "best_metric": current_best_metric,
                        "checkpoint_selection": checkpoint_selection,
                        "val_loss": current_val_loss,
                        "val_task_loss": current_val_task_loss,
                        "val_auc_mean": current_val_auc_mean,
                        "val_mcc_mean": current_val_mcc_mean,
                        "val_composite": current_val_composite,
                        "val_metrics": dict(_metrics_now),
                        "gate_pref_weight_eff": gate_pref_weight_eff,
                        "ab_cdr_gate_beta": AB_CDR_GATE_BETA,
                        "gate_pref_rigid_alpha_max": GATE_PREF_RIGID_ALPHA_MAX,
                        "gate_pref_flex_alpha_min": GATE_PREF_FLEX_ALPHA_MIN,
                        "alpha_attr_weight_eff": alpha_attr_weight_eff,
                        "flex_drive_neutral_ratio_eff": flex_drive_neutral_ratio_eff,
                        "alpha_attr_margin": ALPHA_ATTR_MARGIN,
                        "rigid_gap": gate_gaps["rigid_gap"],
                        "flex_gap": gate_gaps["flex_gap"],
                        "gate_gap_score": current_gap_score,
                        **flex_pair_stats,
                        "use_matrix": "none",
                        "kfold": kfold,
                        "repeat_id": repeat_id,
                        "split_seed": split_seed,
                        "test_ratio": TEST_RATIO,
                        "test_split_seed": SEED,
                        "test_fold_index": TEST_FOLD_INDEX,
                        "group_split_by_pdb": GROUP_SPLIT_BY_PDB,
                        "train_val_data_file_path": data_file_path,
                        "test_data_source": test_data_source,
                        "test_data_file_path": effective_test_data_file_path,
                        "train_samples": len(train_data_PECAN),
                        "val_samples": len(val_data_PECAN),
                        "test_samples": len(test_data_PECAN),
                    }
                    checkpoint_candidates = {"mcc": current_val_mcc_mean}
                    for checkpoint_name, checkpoint_spec in best_checkpoint_specs.items():
                        candidate_score = float(checkpoint_candidates[checkpoint_name])
                        if not np.isfinite(candidate_score):
                            continue
                        is_better = candidate_score > checkpoint_spec["score"]
                        if not is_better:
                            continue

                        checkpoint_spec["score"] = candidate_score
                        checkpoint_spec["epoch"] = e + 1
                        metric_type = checkpoint_spec["metric_type"]
                        checkpoint_selection_now = checkpoint_spec["checkpoint_selection"]
                        best_metrics_now = {
                            "val_loss": current_val_loss,
                            "val_task_loss": current_val_task_loss,
                            "val_auprc_ag": _metrics_now["val_auprc_ag"],
                            "val_auroc_ag": _metrics_now["val_auroc_ag"],
                            "val_mcc_ag": _metrics_now["val_mcc_ag"],
                            "val_auprc_ab": _metrics_now["val_auprc_ab"],
                            "val_auroc_ab": _metrics_now["val_auroc_ab"],
                            "val_mcc_ab": _metrics_now["val_mcc_ab"],
                            "val_auc_mean": current_val_auc_mean,
                            "val_mcc_mean": current_val_mcc_mean,
                            "val_ab_auc_mcc": current_val_ab_auc_mcc,
                            "val_composite": current_val_composite,
                            "best_metric_type": metric_type,
                            "best_metric": candidate_score,
                            "checkpoint_selection": checkpoint_selection_now,
                            "ab_cdr_gate_beta": AB_CDR_GATE_BETA,
                            "gate_pref_rigid_alpha_max": GATE_PREF_RIGID_ALPHA_MAX,
                            "gate_pref_flex_alpha_min": GATE_PREF_FLEX_ALPHA_MIN,
                            **flex_pair_stats,
                        }
                        checkpoint_spec["metrics"] = best_metrics_now
                        best_checkpoint = dict(current_checkpoint)
                        best_checkpoint.update(
                            {
                                "best_metric_type": metric_type,
                                "best_metric": candidate_score,
                                "checkpoint_selection": checkpoint_selection_now,
                                "best_val_loss": current_val_loss,
                                "best_val_task_loss": current_val_task_loss,
                                "best_val_auc_mean": current_val_auc_mean,
                                "best_val_mcc_mean": current_val_mcc_mean,
                                "best_val_ab_auc_mcc": current_val_ab_auc_mcc,
                                "best_val_composite": current_val_composite,
                                "best_val_metrics": best_metrics_now,
                            }
                        )
                        torch.save(best_checkpoint, checkpoint_spec["path"])
                        print(
                            f"[Checkpoint:{checkpoint_name}] save {checkpoint_spec['path']} "
                            f"(repeat={repeat_id + 1}, fold={kfold}, epoch={e + 1}, "
                            f"selection={checkpoint_selection_now}, "
                            f"best_metric={candidate_score:.6f}, "
                            f"val_loss={current_val_loss:.6f}, "
                            f"val_mcc_mean={current_val_mcc_mean:.4f}, "
                            f"val_auc_mean={current_val_auc_mean:.4f}, "
                            f"val_composite={current_val_composite:.4f})"
                        )


        mcc_checkpoint_spec = best_checkpoint_specs["mcc"]
        if mcc_checkpoint_spec["metrics"] is None:
            raise RuntimeError(
                f"fold={kfold}: no finite antigen/antibody validation MCC pair; "
                "cannot select best_mcc.pt"
            )
        best_epoch = mcc_checkpoint_spec["epoch"]
        best_checkpoint_path = mcc_checkpoint_spec["path"]
        best_val_metrics = mcc_checkpoint_spec["metrics"]

        def _empty_test_metrics():
            return {
                "test_auprc_ag": float("nan"),
                "test_auroc_ag": float("nan"),
                "test_mcc_ag": float("nan"),
                "test_auprc_ab": float("nan"),
                "test_auroc_ab": float("nan"),
                "test_mcc_ab": float("nan"),
                "test_auc_mean": float("nan"),
                "test_composite": float("nan"),
                "valid_samples": 0,
                "skipped_samples": 0,
            }

        best_test_metrics_by_checkpoint = {
            checkpoint_name: _empty_test_metrics()
            for checkpoint_name in best_checkpoint_specs
        }
        for checkpoint_name, checkpoint_spec in best_checkpoint_specs.items():
            checkpoint_path = checkpoint_spec["path"]
            checkpoint_metrics = checkpoint_spec["metrics"] or {}
            if os.path.isfile(checkpoint_path):
                best_checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
                model.load_state_dict(best_checkpoint_data["model_state_dict"], strict=True)
                test_prediction_path = os.path.join(metrics_dir, "test_predictions.pt")
                test_metrics_path = os.path.join(metrics_dir, "test_metrics.txt")
                checkpoint_test_metrics = evaluate_on_test_data(
                    model,
                    test_data_PECAN,
                    device,
                    split_name=f"test_best_{checkpoint_name}_repeat{repeat_id + 1}_fold{kfold}",
                    prediction_path=test_prediction_path,
                )
                best_test_metrics_by_checkpoint[checkpoint_name] = checkpoint_test_metrics
                with open(test_metrics_path, "w", encoding="utf-8") as ftest:
                    ftest.write(f"checkpoint_name={checkpoint_name}\n")
                    ftest.write(f"best_checkpoint={checkpoint_path}\n")
                    ftest.write(f"best_epoch={checkpoint_spec['epoch']}\n")
                    ftest.write(f"train_val_data_file_path={data_file_path}\n")
                    ftest.write(f"test_data_source={test_data_source}\n")
                    ftest.write(f"test_data_file_path={effective_test_data_file_path}\n")
                    ftest.write(f"checkpoint_selection={checkpoint_metrics.get('checkpoint_selection', checkpoint_spec['checkpoint_selection'])}\n")
                    ftest.write(f"best_metric_type={checkpoint_metrics.get('best_metric_type', checkpoint_spec['metric_type'])}\n")
                    ftest.write(f"best_metric={checkpoint_metrics.get('best_metric', checkpoint_spec['score'])}\n")
                    ftest.write(f"ab_cdr_gate_beta={AB_CDR_GATE_BETA}\n")
                    ftest.write(f"gate_pref_rigid_alpha_max={GATE_PREF_RIGID_ALPHA_MAX}\n")
                    ftest.write(f"gate_pref_flex_alpha_min={GATE_PREF_FLEX_ALPHA_MIN}\n")
                    ftest.write(f"best_val_loss={checkpoint_metrics.get('val_loss', float('nan'))}\n")
                    ftest.write(f"best_val_mcc_mean={checkpoint_metrics.get('val_mcc_mean', float('nan'))}\n")
                    ftest.write(f"best_val_auc_mean={checkpoint_metrics.get('val_auc_mean', float('nan'))}\n")
                    ftest.write(f"best_val_mcc_ab={checkpoint_metrics.get('val_mcc_ab', float('nan'))}\n")
                    ftest.write(f"best_val_ab_auc_mcc={checkpoint_metrics.get('val_ab_auc_mcc', float('nan'))}\n")
                    ftest.write(f"prediction_path={checkpoint_test_metrics.get('prediction_path', '')}\n")
                    for key, value in get_flex_pair_bias_stats(model).items():
                        ftest.write(f"{key}={value}\n")
                    for key in (
                        "test_auprc_ag",
                        "test_auroc_ag",
                        "test_mcc_ag",
                        "test_auprc_ab",
                        "test_auroc_ab",
                        "test_mcc_ab",
                        "test_auc_mean",
                        "test_composite",
                        "valid_samples",
                        "skipped_samples",
                    ):
                        ftest.write(f"{key}={checkpoint_test_metrics.get(key, float('nan'))}\n")
                print(
                    f"[Test:{checkpoint_name}] repeat={repeat_id + 1}, fold={kfold}, "
                    f"best_epoch={checkpoint_spec['epoch']}, "
                    f"test_auc_mean={checkpoint_test_metrics['test_auc_mean']:.4f}, "
                    f"test_composite={checkpoint_test_metrics['test_composite']:.4f}"
                )
            else:
                raise FileNotFoundError(f"Selected MCC checkpoint not found: {checkpoint_path}")
        best_test_metrics = best_test_metrics_by_checkpoint["mcc"]

        current_flex_pair_stats = get_flex_pair_bias_stats(model)
        best_flex_pair_stats = {
            key: best_val_metrics.get(key, value)
            for key, value in current_flex_pair_stats.items()
        }
        for key, value in best_flex_pair_stats.items():
            best_val_metrics.setdefault(key, value)

        fold_summary_path = os.path.join(best_dir, f"{run_tag}_best_summary.txt")
        train_group_count = len(get_split_group_set(train_data_PECAN))
        val_group_count = len(get_split_group_set(val_data_PECAN))
        test_group_count = len(get_split_group_set(test_data_PECAN))
        with open(fold_summary_path, "w", encoding="utf-8") as fsum:
            fsum.write(f"repeat={repeat_id + 1}\n")
            fsum.write(f"fold={kfold}\n")
            fsum.write(f"split_seed={split_seed}\n")
            fsum.write(f"test_ratio={TEST_RATIO}\n")
            fsum.write(f"test_split_seed={SEED}\n")
            fsum.write(f"test_fold_index={TEST_FOLD_INDEX}\n")
            fsum.write(f"group_split_by_pdb={GROUP_SPLIT_BY_PDB}\n")
            fsum.write(f"train_val_data_file_path={data_file_path}\n")
            fsum.write(f"test_data_source={test_data_source}\n")
            fsum.write(f"test_data_file_path={effective_test_data_file_path}\n")
            fsum.write(f"train_samples={len(train_data_PECAN)}\n")
            fsum.write(f"val_samples={len(val_data_PECAN)}\n")
            fsum.write(f"test_samples={len(test_data_PECAN)}\n")
            fsum.write(f"train_groups={train_group_count}\n")
            fsum.write(f"val_groups={val_group_count}\n")
            fsum.write(f"test_groups={test_group_count}\n")
            fsum.write(f"best_epoch={best_epoch}\n")
            fsum.write(f"best_checkpoint={best_checkpoint_path}\n")
            fsum.write(f"checkpoint_selection={best_val_metrics.get('checkpoint_selection', 'highest_val_mcc_mean')}\n")
            fsum.write(f"best_metric_type={best_val_metrics.get('best_metric_type', CHECKPOINT_METRIC)}\n")
            fsum.write(f"best_metric={best_val_metrics.get('best_metric', float('nan'))}\n")
            fsum.write(f"ab_cdr_gate_beta={AB_CDR_GATE_BETA}\n")
            fsum.write(f"gate_pref_rigid_alpha_max={GATE_PREF_RIGID_ALPHA_MAX}\n")
            fsum.write(f"gate_pref_flex_alpha_min={GATE_PREF_FLEX_ALPHA_MIN}\n")
            fsum.write(f"best_val_loss={best_val_metrics['val_loss']}\n")
            fsum.write(f"best_val_auc_mean={best_val_metrics['val_auc_mean']}\n")
            fsum.write(f"best_val_mcc_mean={best_val_metrics['val_mcc_mean']}\n")
            fsum.write(f"best_val_ab_auc_mcc={best_val_metrics['val_ab_auc_mcc']}\n")
            fsum.write(f"best_val_composite={best_val_metrics['val_composite']}\n")
            fsum.write(f"val_auprc_ag={best_val_metrics['val_auprc_ag']}\n")
            fsum.write(f"val_auroc_ag={best_val_metrics['val_auroc_ag']}\n")
            fsum.write(f"val_mcc_ag={best_val_metrics['val_mcc_ag']}\n")
            fsum.write(f"val_auprc_ab={best_val_metrics['val_auprc_ab']}\n")
            fsum.write(f"val_auroc_ab={best_val_metrics['val_auroc_ab']}\n")
            fsum.write(f"val_mcc_ab={best_val_metrics['val_mcc_ab']}\n")
            for key, value in best_flex_pair_stats.items():
                fsum.write(f"{key}={value}\n")
            for checkpoint_name, checkpoint_spec in best_checkpoint_specs.items():
                checkpoint_metrics = checkpoint_spec["metrics"] or {}
                fsum.write(f"best_{checkpoint_name}_checkpoint={checkpoint_spec['path']}\n")
                fsum.write(f"best_{checkpoint_name}_epoch={checkpoint_spec['epoch']}\n")
                fsum.write(f"best_{checkpoint_name}_metric_type={checkpoint_metrics.get('best_metric_type', checkpoint_spec['metric_type'])}\n")
                fsum.write(f"best_{checkpoint_name}_metric={checkpoint_metrics.get('best_metric', float('nan'))}\n")
                fsum.write(f"best_{checkpoint_name}_test_auc_mean={best_test_metrics_by_checkpoint[checkpoint_name].get('test_auc_mean', float('nan'))}\n")
                fsum.write(f"best_{checkpoint_name}_test_composite={best_test_metrics_by_checkpoint[checkpoint_name].get('test_composite', float('nan'))}\n")
            fsum.write(f"test_auprc_ag={best_test_metrics['test_auprc_ag']}\n")
            fsum.write(f"test_auroc_ag={best_test_metrics['test_auroc_ag']}\n")
            fsum.write(f"test_mcc_ag={best_test_metrics['test_mcc_ag']}\n")
            fsum.write(f"test_auprc_ab={best_test_metrics['test_auprc_ab']}\n")
            fsum.write(f"test_auroc_ab={best_test_metrics['test_auroc_ab']}\n")
            fsum.write(f"test_mcc_ab={best_test_metrics['test_mcc_ab']}\n")
            fsum.write(f"test_auc_mean={best_test_metrics['test_auc_mean']}\n")
            fsum.write(f"test_composite={best_test_metrics['test_composite']}\n")
            fsum.write(f"test_valid_samples={best_test_metrics['valid_samples']}\n")
            fsum.write(f"test_skipped_samples={best_test_metrics['skipped_samples']}\n")
            fsum.write(f"test_prediction_path={best_test_metrics.get('prediction_path', '')}\n")

        fold_best_records.append(
            {
                "repeat": repeat_id + 1,
                "fold": kfold,
                "split_seed": split_seed,
                "test_ratio": TEST_RATIO,
                "test_split_seed": SEED,
                "test_fold_index": TEST_FOLD_INDEX,
                "train_val_data_file_path": data_file_path,
                "test_data_source": test_data_source,
                "test_data_file_path": effective_test_data_file_path,
                "train_samples": len(train_data_PECAN),
                "val_samples": len(val_data_PECAN),
                "test_samples": len(test_data_PECAN),
                "train_groups": train_group_count,
                "val_groups": val_group_count,
                "test_groups": test_group_count,
                "best_epoch": best_epoch,
                "best_checkpoint": best_checkpoint_path,
                "checkpoint_selection": best_val_metrics.get("checkpoint_selection", "highest_val_mcc_mean"),
                "best_metric_type": best_val_metrics.get("best_metric_type", CHECKPOINT_METRIC),
                "best_metric": best_val_metrics.get("best_metric", float("nan")),
                "best_mcc_epoch": best_checkpoint_specs["mcc"]["epoch"],
                "best_mcc_checkpoint": best_checkpoint_specs["mcc"]["path"],
                "best_mcc_metric": (
                    best_checkpoint_specs["mcc"]["metrics"] or {}
                ).get("best_metric", float("nan")),
                "ab_cdr_gate_beta": AB_CDR_GATE_BETA,
                "gate_pref_rigid_alpha_max": GATE_PREF_RIGID_ALPHA_MAX,
                "gate_pref_flex_alpha_min": GATE_PREF_FLEX_ALPHA_MIN,
                "best_val_loss": best_val_metrics["val_loss"],
                "val_auc_mean": best_val_metrics["val_auc_mean"],
                "val_mcc_mean": best_val_metrics["val_mcc_mean"],
                "val_ab_auc_mcc": best_val_metrics["val_ab_auc_mcc"],
                "val_composite": best_val_metrics["val_composite"],
                "val_auprc_ag": best_val_metrics["val_auprc_ag"],
                "val_auroc_ag": best_val_metrics["val_auroc_ag"],
                "val_mcc_ag": best_val_metrics["val_mcc_ag"],
                "val_auprc_ab": best_val_metrics["val_auprc_ab"],
                "val_auroc_ab": best_val_metrics["val_auroc_ab"],
                "val_mcc_ab": best_val_metrics["val_mcc_ab"],
                **best_flex_pair_stats,
                "test_auprc_ag": best_test_metrics["test_auprc_ag"],
                "test_auroc_ag": best_test_metrics["test_auroc_ag"],
                "test_mcc_ag": best_test_metrics["test_mcc_ag"],
                "test_auprc_ab": best_test_metrics["test_auprc_ab"],
                "test_auroc_ab": best_test_metrics["test_auroc_ab"],
                "test_mcc_ab": best_test_metrics["test_mcc_ab"],
                "test_auc_mean": best_test_metrics["test_auc_mean"],
                "test_composite": best_test_metrics["test_composite"],
                "test_valid_samples": best_test_metrics["valid_samples"],
                "test_skipped_samples": best_test_metrics["skipped_samples"],
                "test_prediction_path": best_test_metrics.get("prediction_path", ""),
            }
        )
        print(
            f"[FoldDone] repeat={repeat_id + 1}, fold={kfold}, "
            f"best_epoch={best_epoch}, "
            f"best_val_loss={float(best_val_metrics.get('val_loss', float('nan'))):.6f}, "
            f"best_mcc_epoch={best_checkpoint_specs['mcc']['epoch']}, "
            f"test_auc_mean={best_test_metrics['test_auc_mean']:.4f}"
        )

    overall_summary_path = os.path.join(output_base_dir, "perf", "best_models", "all_repeat_fold_best_summary.tsv")
    os.makedirs(os.path.dirname(overall_summary_path), exist_ok=True)
    with open(overall_summary_path, "w", encoding="utf-8") as fout:
        fout.write(
            "repeat\tfold\tsplit_seed\ttest_ratio\ttest_split_seed\ttest_fold_index\t"
            "train_val_data_file_path\ttest_data_source\ttest_data_file_path\t"
            "train_samples\tval_samples\ttest_samples\ttrain_groups\tval_groups\ttest_groups\t"
            "best_epoch\tbest_checkpoint\tcheckpoint_selection\tbest_metric_type\tbest_metric\t"
            "best_mcc_epoch\tbest_mcc_checkpoint\tbest_mcc_metric\t"
            "ab_cdr_gate_beta\tgate_pref_rigid_alpha_max\tgate_pref_flex_alpha_min\t"
            "best_val_loss\tval_auc_mean\tval_mcc_mean\tval_ab_auc_mcc\tval_composite\t"
            "val_auprc_ag\tval_auroc_ag\tval_mcc_ag\tval_auprc_ab\tval_auroc_ab\tval_mcc_ab\t"
            "flex_pair_bias_bound_type\tflex_pair_bias_max\t"
            "flex_pair_lambda_mean\tflex_pair_lambda_max_abs\tflex_pair_lambda_mean_abs\t"
            "test_auc_mean\ttest_composite\ttest_auprc_ag\ttest_auroc_ag\ttest_mcc_ag\t"
            "test_auprc_ab\ttest_auroc_ab\ttest_mcc_ab\ttest_valid_samples\ttest_skipped_samples\t"
            "test_prediction_path\n"
        )
        for rec in sorted(fold_best_records, key=lambda x: (x["repeat"], x["fold"])):
            fout.write(
                f"{rec['repeat']}\t{rec['fold']}\t{rec['split_seed']}\t"
                f"{rec['test_ratio']}\t{rec['test_split_seed']}\t{rec['test_fold_index']}\t"
                f"{rec['train_val_data_file_path']}\t{rec['test_data_source']}\t"
                f"{rec['test_data_file_path']}\t"
                f"{rec['train_samples']}\t{rec['val_samples']}\t{rec['test_samples']}\t"
                f"{rec['train_groups']}\t{rec['val_groups']}\t{rec['test_groups']}\t"
                f"{rec['best_epoch']}\t"
                f"{rec['best_checkpoint']}\t{rec['checkpoint_selection']}\t"
                f"{rec['best_metric_type']}\t{rec['best_metric']}\t"
                f"{rec['best_mcc_epoch']}\t{rec['best_mcc_checkpoint']}\t{rec['best_mcc_metric']}\t"
                f"{rec['ab_cdr_gate_beta']}\t"
                f"{rec['gate_pref_rigid_alpha_max']}\t{rec['gate_pref_flex_alpha_min']}\t"
                f"{rec['best_val_loss']}\t"
                f"{rec['val_auc_mean']}\t{rec['val_mcc_mean']}\t{rec['val_ab_auc_mcc']}\t{rec['val_composite']}\t"
                f"{rec['val_auprc_ag']}\t{rec['val_auroc_ag']}\t{rec['val_mcc_ag']}\t"
                f"{rec['val_auprc_ab']}\t{rec['val_auroc_ab']}\t{rec['val_mcc_ab']}\t"
                f"{rec['flex_pair_bias_bound_type']}\t{rec['flex_pair_bias_max']}\t"
                f"{rec['flex_pair_lambda_mean']}\t{rec['flex_pair_lambda_max_abs']}\t"
                f"{rec['flex_pair_lambda_mean_abs']}\t"
                f"{rec['test_auc_mean']}\t{rec['test_composite']}\t"
                f"{rec['test_auprc_ag']}\t{rec['test_auroc_ag']}\t{rec['test_mcc_ag']}\t"
                f"{rec['test_auprc_ab']}\t{rec['test_auroc_ab']}\t{rec['test_mcc_ab']}\t"
                f"{rec['test_valid_samples']}\t{rec['test_skipped_samples']}\t"
                f"{rec['test_prediction_path']}\n"
            )
    print(f"[Done] all fold summaries -> {overall_summary_path}")
