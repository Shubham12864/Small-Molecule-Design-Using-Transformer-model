import json
import os
import random

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False


def canonicalize_smiles(smiles):
    if not HAS_RDKIT:
        return smiles

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def get_scaffold(smiles):
    if not HAS_RDKIT:
        return ""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def preprocess_smiles(smiles_list, dedup=True):
    processed = []
    invalid = 0

    for smiles in smiles_list:
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            invalid += 1
            continue
        processed.append(canonical)

    if dedup:
        processed = list(dict.fromkeys(processed))

    return processed, invalid


def _resolve_split_sizes(total, val_split, test_split):
    if total < 3:
        raise ValueError("Need at least 3 valid molecules for train/val/test splitting.")
    if val_split < 0 or test_split < 0:
        raise ValueError("val_split and test_split must be non-negative.")
    if (val_split + test_split) >= 1.0:
        raise ValueError("val_split + test_split must be less than 1.0.")

    val_size = max(1, int(total * val_split)) if val_split > 0 else 0
    test_size = max(1, int(total * test_split)) if test_split > 0 else 0

    while (val_size + test_size) >= total:
        if val_size >= test_size and val_size > 0:
            val_size -= 1
        elif test_size > 0:
            test_size -= 1
        else:
            break

    train_size = total - val_size - test_size
    if train_size < 1:
        raise ValueError("Split sizes left no samples for training.")

    return {"train": train_size, "val": val_size, "test": test_size}


def _split_random(processed, targets, seed):
    shuffled = list(processed)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    val_end = targets["val"]
    test_end = val_end + targets["test"]

    return {
        "val": shuffled[:val_end],
        "test": shuffled[val_end:test_end],
        "train": shuffled[test_end:],
    }


def _pick_target_split(group_len, counts, targets, rng):
    candidate_names = [name for name, target in targets.items() if target > 0]
    rng.shuffle(candidate_names)

    best_name = None
    best_score = None
    for name in candidate_names:
        remaining = targets[name] - counts[name]
        if remaining <= 0:
            continue

        score = remaining / max(targets[name], 1)
        if best_score is None or score > best_score:
            best_name = name
            best_score = score

    if best_name is not None:
        return best_name

    overflow_names = [name for name in candidate_names if name != "train"]
    rng.shuffle(overflow_names)

    if overflow_names:
        return min(
            overflow_names,
            key=lambda name: counts[name] - targets[name],
        )

    return "train"


def _split_scaffold(processed, targets, seed):
    rng = random.Random(seed)
    buckets = {}
    for smiles in processed:
        scaffold = get_scaffold(smiles)
        buckets.setdefault(scaffold, []).append(smiles)

    groups = list(buckets.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    split_data = {"train": [], "val": [], "test": []}
    counts = {name: 0 for name in split_data}

    for group in groups:
        target_name = _pick_target_split(len(group), counts, targets, rng)
        split_data[target_name].extend(group)
        counts[target_name] += len(group)

    return split_data


def validate_no_overlap(split_data):
    names = ["train", "val", "test"]
    overlaps = {}

    for i, left_name in enumerate(names):
        left = set(split_data.get(left_name, []))
        for right_name in names[i + 1 :]:
            right = set(split_data.get(right_name, []))
            key = f"{left_name}_{right_name}"
            overlaps[key] = len(left & right)
            if overlaps[key] > 0:
                raise RuntimeError(f"Leakage detected between {left_name} and {right_name}.")

    return overlaps


def build_dataset_splits(smiles_list, val_split, test_split, seed, split_method, dedup):
    processed, invalid = preprocess_smiles(smiles_list, dedup=dedup)
    if len(processed) < 3:
        raise ValueError("Not enough valid samples after preprocessing.")

    targets = _resolve_split_sizes(len(processed), val_split, test_split)
    effective_method = split_method

    if split_method == "scaffold" and HAS_RDKIT:
        split_data = _split_scaffold(processed, targets, seed)
    else:
        if split_method == "scaffold" and not HAS_RDKIT:
            effective_method = "random"
        split_data = _split_random(processed, targets, seed)

    overlaps = validate_no_overlap(split_data)
    metadata = {
        "requested_split_method": split_method,
        "effective_split_method": effective_method,
        "seed": seed,
        "dedup": dedup,
        "invalid_removed": invalid,
        "total_after_preprocessing": len(processed),
        "targets": targets,
        "counts": {name: len(values) for name, values in split_data.items()},
        "overlaps": overlaps,
    }

    return split_data, metadata


def save_split_artifacts(output_dir, split_data, metadata):
    os.makedirs(output_dir, exist_ok=True)

    for split_name, smiles_list in split_data.items():
        path = os.path.join(output_dir, f"{split_name}.txt")
        with open(path, "w", encoding="utf-8") as handle:
            for smiles in smiles_list:
                handle.write(smiles + "\n")

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def load_split_artifacts(split_dir):
    split_data = {}
    for split_name in ("train", "val", "test"):
        path = os.path.join(split_dir, f"{split_name}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing split file: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            split_data[split_name] = [line.strip() for line in handle if line.strip()]

    metadata_path = os.path.join(split_dir, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    validate_no_overlap(split_data)
    return split_data, metadata
