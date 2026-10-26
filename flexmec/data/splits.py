from .runtime import *

def get_split_group_id(item, default_value=None):
    """Return the leakage-control group id used by train/val/test splits."""
    raw_value = None
    if isinstance(item, dict):
        raw_value = (
            item.get("pdb_id")
            or item.get("complex_code")
            or item.get("sample_id")
        )

    if raw_value is None:
        return f"__object_{id(item)}"

    group_id = str(raw_value).strip().lower()
    if not group_id:
        return f"__object_{id(item)}"

    # Examples: 6okp_3PH and 6OKP should stay in the same split.
    return group_id.split("_")[0]


def get_split_group_set(samples):
    return {
        get_split_group_id(item, idx)
        for idx, item in enumerate(samples)
    }


def assert_group_disjoint(split_a, split_b, name_a="split_a", name_b="split_b"):
    groups_a = get_split_group_set(split_a)
    groups_b = get_split_group_set(split_b)
    overlap = groups_a.intersection(groups_b)
    assert not overlap, (
        f"{name_a} and {name_b} overlap by split group: "
        f"{sorted(overlap)[:10]}"
    )


def _build_group_folds(K, X, *, shuffle=False, seed=42):
    assert K > 1, "K must be > 1"

    group_to_indices = {}
    for idx, item in enumerate(X):
        group_id = get_split_group_id(item, idx)
        group_to_indices.setdefault(group_id, []).append(idx)

    if len(group_to_indices) < K:
        raise ValueError(
            f"Need at least K={K} split groups, got {len(group_to_indices)}."
        )

    groups = list(group_to_indices.items())
    if shuffle:
        rng = np.random.RandomState(seed)
        tie_order = rng.permutation(len(groups))
    else:
        rng = None
        tie_order = np.arange(len(groups))

    groups_with_order = [
        (group_id, indices, int(order))
        for order, (group_id, indices) in zip(tie_order, groups)
    ]
    groups_with_order.sort(key=lambda item: (-len(item[1]), item[2]))

    folds = [[] for _ in range(K)]
    fold_sizes = [0 for _ in range(K)]
    for _, indices, _ in groups_with_order:
        fold_id = min(range(K), key=lambda j: (fold_sizes[j], j))
        folds[fold_id].extend(indices)
        fold_sizes[fold_id] += len(indices)

    if shuffle:
        for fold in folds:
            rng.shuffle(fold)

    return [np.asarray(fold, dtype=int) for fold in folds]


def split_train_test_data(
    X,
    *,
    test_ratio=0.2,
    seed=42,
    test_fold=0,
    group_by=None,
):
    """Split an independent holdout test set before cross-validation."""
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}.")

    if group_by:
        holdout_folds = max(2, int(round(1.0 / test_ratio)))
        test_fold = int(test_fold) % holdout_folds
        folds = _build_group_folds(
            holdout_folds,
            X,
            shuffle=True,
            seed=seed,
        )
        test_idx = folds[test_fold]
        train_idx = np.concatenate(
            [folds[j] for j in range(holdout_folds) if j != test_fold],
            axis=0,
        )
    else:
        N = len(X)
        idx = np.arange(N)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(idx)
        test_size = max(1, int(round(N * test_ratio)))
        test_idx = idx[:test_size]
        train_idx = idx[test_size:]

    test_idx_set = set(map(int, test_idx))
    assert test_idx_set.isdisjoint(set(map(int, train_idx)))

    train_data = [X[int(i)] for i in train_idx]
    test_data = [X[int(i)] for i in test_idx]

    if group_by:
        assert_group_disjoint(train_data, test_data, "TrainVal", "Test")

    return train_data, test_data


def get_val_fixed_data(data_PECAN):
    valData_complex_code = ['3o6l', '1a3r', '3jau', '3h0t', '1dee', '7rah', '4n8c', '6mtt', '6osh', '6k68', '2h32', '2hfg', '7rt9', '5u3l', '6ogx', '2j4w', '7rxl', '5hi3', '7ox3', '6wdt', '7y1b', '4zfo', '6z7y', '7ur6', '5u3m', '6wio', '2jel', '3j1s', '6blh', '6cyf', '6wex', '7w71', '6r0x', '5epm', '5w3e', '4f37', '7bsd', '6k7o', '4lu5', '6t3j', '7lk4', '7usl', '7vux', '3ffd', '5wk3', '7nx3', '6z7w', '8db4', '2bdn', '6ktr']

    val_data = [item for item in data_PECAN if item['complex_code'] in valData_complex_code]
    train_data = [i for i in data_PECAN if i['complex_code'] not in valData_complex_code]


    def _idset(a):
        return {id(x) for x in np.asarray(a, dtype=object).ravel()}

    assert _idset(train_data).isdisjoint(_idset(val_data)), "Train and Val overlap (by identity)"


    return train_data, val_data


def _normalize_pdb_id(value):
    if value is None:
        return None
    pdb_id = str(value).strip().lower()
    return pdb_id or None


def _get_sample_pdb_id(item):
    if not isinstance(item, dict):
        return None
    for key in ("pdb_id", "complex_code", "sample_id"):
        pdb_id = _normalize_pdb_id(item.get(key))
        if pdb_id is not None:
            return pdb_id
    return None


def _read_pdb_id_split_csv(csv_path, pdb_id_column="pdb_id"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Split CSV has no header: {csv_path}")

        column = pdb_id_column
        if column not in reader.fieldnames:
            column = reader.fieldnames[0]

        pdb_ids = []
        for row_idx, row in enumerate(reader, start=2):
            pdb_id = _normalize_pdb_id(row.get(column))
            if pdb_id is None:
                raise ValueError(f"Empty pdb_id in {csv_path} line {row_idx}")
            pdb_ids.append(pdb_id)

    seen = set()
    duplicated = set()
    for pdb_id in pdb_ids:
        if pdb_id in seen:
            duplicated.add(pdb_id)
        seen.add(pdb_id)
    if duplicated:
        raise ValueError(
            f"Duplicate pdb_id values in {csv_path}: {sorted(duplicated)[:10]}"
        )
    return pdb_ids


def get_k_fold_data_from_pdb_id_csv(
    data_PECAN,
    fold,
    *,
    split_dir,
    pdb_id_column="pdb_id",
    strict=True,
):
    """Load fixed train/val/test splits from pdb_id CSV files."""
    fold = int(fold)
    train_ids = _read_pdb_id_split_csv(
        os.path.join(split_dir, f"fold{fold}_train.csv"),
        pdb_id_column=pdb_id_column,
    )
    val_ids = _read_pdb_id_split_csv(
        os.path.join(split_dir, f"fold{fold}_val.csv"),
        pdb_id_column=pdb_id_column,
    )
    test_ids = _read_pdb_id_split_csv(
        os.path.join(split_dir, "test.csv"),
        pdb_id_column=pdb_id_column,
    )

    split_pairs = [
        ("Train", set(train_ids), "Val", set(val_ids)),
        ("Train", set(train_ids), "Test", set(test_ids)),
        ("Val", set(val_ids), "Test", set(test_ids)),
    ]
    for name_a, ids_a, name_b, ids_b in split_pairs:
        overlap = ids_a.intersection(ids_b)
        if overlap:
            raise ValueError(
                f"{name_a} and {name_b} CSV splits overlap by pdb_id: "
                f"{sorted(overlap)[:10]}"
            )

    pdb_id_to_sample = {}
    missing_sample_ids = 0
    duplicate_sample_ids = []
    for item in data_PECAN:
        pdb_id = _get_sample_pdb_id(item)
        if pdb_id is None:
            missing_sample_ids += 1
            continue
        if pdb_id in pdb_id_to_sample:
            duplicate_sample_ids.append(pdb_id)
        pdb_id_to_sample[pdb_id] = item

    if missing_sample_ids:
        raise ValueError(
            f"{missing_sample_ids} samples do not contain pdb_id/complex_code/sample_id."
        )
    if duplicate_sample_ids:
        raise ValueError(
            f"Duplicate sample pdb_id values in data_PECAN: "
            f"{sorted(set(duplicate_sample_ids))[:10]}"
        )

    def _select(split_name, pdb_ids):
        missing = [pdb_id for pdb_id in pdb_ids if pdb_id not in pdb_id_to_sample]
        if missing:
            raise ValueError(
                f"{split_name} CSV contains pdb_id not found in data_PECAN: "
                f"{missing[:10]}"
            )
        return [pdb_id_to_sample[pdb_id] for pdb_id in pdb_ids]

    train_data = _select("Train", train_ids)
    val_data = _select("Val", val_ids)
    test_data = _select("Test", test_ids)

    assert set(map(id, train_data)).isdisjoint(set(map(id, val_data)))
    assert set(map(id, train_data)).isdisjoint(set(map(id, test_data)))
    assert set(map(id, val_data)).isdisjoint(set(map(id, test_data)))

    if strict:
        assigned_ids = set(train_ids).union(val_ids).union(test_ids)
        unassigned = sorted(set(pdb_id_to_sample).difference(assigned_ids))
        if unassigned:
            raise ValueError(
                f"CSV splits do not cover {len(unassigned)} data_PECAN samples; "
                f"first missing pdb_id values: {unassigned[:10]}"
            )

    print(
        f"[CSV split] fold={fold}, train={len(train_data)}, "
        f"val={len(val_data)}, test={len(test_data)}, split_dir={split_dir}"
    )
    return train_data, val_data, test_data


def get_k_fold_data(
    K,
    val_fold,
    X,
    *,
    shuffle=False,
    seed=42,
    group_by=None,
    test_ratio=0.2,
    test_seed=None,
    test_fold=0,
):
    assert K > 1
    if not (0 <= val_fold < K):
        raise ValueError(f"val_fold must be in [0, {K}), got {val_fold}.")

    # Hold out the test set before cross-validation.
    test_seed = seed if test_seed is None else test_seed
    train_val_data, X_test = split_train_test_data(
        X,
        test_ratio=test_ratio,
        seed=test_seed,
        test_fold=test_fold,
        group_by=group_by,
    )

    # Split the remaining data into training and validation folds.
    N = len(train_val_data)
    if group_by:
        folds = _build_group_folds(K, train_val_data, shuffle=shuffle, seed=seed)
        val_idx = folds[val_fold]
        train_idx = np.concatenate([folds[j] for j in range(K) if j != val_fold])
    else:
        idx = np.arange(N)

        if shuffle:
            rng = np.random.RandomState(seed)
            idx = rng.permutation(idx)

        folds = np.array_split(idx, K)
        val_idx = folds[val_fold]
        train_idx = np.concatenate([folds[j] for j in range(K) if j != val_fold])

    # Keep sample dictionaries in lists to preserve object identity.
    X_val   = [train_val_data[int(i)] for i in val_idx]
    X_train = [train_val_data[int(i)] for i in train_idx]

    assert set(map(id, X_train)).isdisjoint(set(map(id, X_val)))
    assert set(map(id, X_train)).isdisjoint(set(map(id, X_test)))
    assert set(map(id, X_val)).isdisjoint(set(map(id, X_test)))
    if group_by:
        assert_group_disjoint(X_train, X_val, "Train", "Val")
        assert_group_disjoint(X_train, X_test, "Train", "Test")
        assert_group_disjoint(X_val, X_test, "Val", "Test")

    return X_train, X_val, X_test
