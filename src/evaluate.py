import argparse
import csv
import json
import os
from collections import OrderedDict

import torch

from analytics import compute_diversity_score
from generate import generate_smiles
from model import MoleculeTransformer
from property import compute_properties
from tokenizer import SmilesTokenizer
from utils import (
    get_checkpoint_path,
    get_data_path,
    load_model,
    load_smiles,
    set_seed,
)

try:
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated SMILES quality")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed_token", type=str, default="C")
    parser.add_argument(
        "--num_molecules",
        type=int,
        default=100,
        help="Number of generation attempts to evaluate.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=None,
        help="Generation length. Defaults to the checkpoint context window.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--min_new_tokens", type=int, default=8)
    parser.add_argument("--max_repeat_run", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train_data",
        type=str,
        default=None,
        help="Training SMILES file for novelty checking. Defaults to data/smiles.txt.",
    )
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def canonicalize_smiles(smiles):
    if not RDKIT_AVAILABLE:
        return smiles

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def make_output_paths(checkpoint_path, output_csv=None, output_json=None):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]

    csv_path = output_csv or os.path.join(checkpoint_dir, f"{checkpoint_name}_evaluation.csv")
    json_path = output_json or os.path.join(checkpoint_dir, f"{checkpoint_name}_evaluation.json")
    return csv_path, json_path


def load_training_reference(train_data_path):
    if train_data_path is None:
        train_data_path = get_data_path("smiles.txt")

    if not os.path.exists(train_data_path):
        print(f"[Eval] Training data not found at: {train_data_path}")
        print("[Eval] Novelty metrics will be skipped.")
        return None

    training_smiles = load_smiles(train_data_path)
    if not RDKIT_AVAILABLE:
        return set(training_smiles)

    canonical_set = set()
    skipped = 0
    for smi in training_smiles:
        canonical = canonicalize_smiles(smi)
        if canonical is None:
            skipped += 1
            continue
        canonical_set.add(canonical)

    print(f"[Eval] Training reference molecules : {len(canonical_set):,}")
    if skipped:
        print(f"[Eval] Training reference skipped : {skipped:,}")
    return canonical_set


def summarise_rows(rows, training_reference=None):
    total = len(rows)
    valid_rows = [row for row in rows if row["valid"]]
    valid_canonical = [row["canonical_smiles"] for row in valid_rows if row["canonical_smiles"]]
    unique_valid_canonical = list(OrderedDict.fromkeys(valid_canonical))

    novelty_available = training_reference is not None
    novel_canonical = []
    if novelty_available:
        novel_canonical = [smi for smi in unique_valid_canonical if smi not in training_reference]

    lipinski_pass = sum(1 for row in valid_rows if row["lipinski"])

    def average(key, digits=3):
        values = [row[key] for row in valid_rows if row.get(key) is not None]
        if not values:
            return None
        return round(sum(values) / len(values), digits)

    summary = {
        "total_generated": total,
        "valid_count": len(valid_rows),
        "invalid_count": total - len(valid_rows),
        "validity_rate": round(100 * len(valid_rows) / max(total, 1), 1),
        "unique_valid_count": len(unique_valid_canonical),
        "uniqueness_rate": round(100 * len(unique_valid_canonical) / max(len(valid_rows), 1), 1),
        "duplicate_valid_count": len(valid_rows) - len(unique_valid_canonical),
        "novelty_available": novelty_available,
        "novel_unique_count": len(novel_canonical) if novelty_available else None,
        "novelty_rate": round(100 * len(novel_canonical) / max(len(unique_valid_canonical), 1), 1)
        if novelty_available
        else None,
        "diversity": round(compute_diversity_score(unique_valid_canonical), 3)
        if len(unique_valid_canonical) >= 2
        else 0.0,
        "lipinski_pass_count": lipinski_pass,
        "lipinski_pass_rate": round(100 * lipinski_pass / max(len(valid_rows), 1), 1),
        "avg_qed": average("qed", digits=3),
        "avg_mw": average("mw", digits=2),
        "avg_logp": average("logp", digits=2),
        "avg_sa_score": average("sa_score", digits=2),
        "avg_tpsa": average("tpsa", digits=2),
    }
    return summary


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "attempt",
        "smiles",
        "canonical_smiles",
        "valid",
        "unique_valid",
        "novel",
        "qed",
        "mw",
        "logp",
        "sa_score",
        "tpsa",
        "hbd",
        "hba",
        "rot_bonds",
        "rings",
        "lipinski",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(summary, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Using device: {device}")

    checkpoint_path = args.checkpoint or get_checkpoint_path("best_model.pt")
    print(f"[Eval] Loading model from: {checkpoint_path}")

    model, tokenizer_data = load_model(MoleculeTransformer, checkpoint_path)
    model = model.to(device)
    model.eval()

    tokenizer = SmilesTokenizer()
    tokenizer.stoi = tokenizer_data["stoi"]
    tokenizer.itos = {int(k): v for k, v in tokenizer_data["itos"].items()}
    tokenizer.vocab_size = tokenizer_data["vocab_size"]

    model_max_len = model.pos_encoder.pe.size(1)
    effective_max_len = model_max_len if args.max_len is None else min(args.max_len, model_max_len)
    if args.max_len is None:
        print(f"[Eval] Using checkpoint max_len={model_max_len}.")
    elif args.max_len > model_max_len:
        print(f"[Eval] Requested max_len={args.max_len} exceeds checkpoint limit {model_max_len}.")
        print(f"[Eval] Using max_len={model_max_len} instead.")

    training_reference = load_training_reference(args.train_data)

    print(
        f"[Eval] Sampling {args.num_molecules} molecules | "
        f"Seed='{args.seed_token}' | Temp={args.temperature}"
    )

    rows = []
    seen_valid = set()

    for attempt in range(1, args.num_molecules + 1):
        smiles = generate_smiles(
            model,
            tokenizer,
            seed=args.seed_token,
            max_len=effective_max_len,
            temperature=args.temperature,
            device=device,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            min_new_tokens=args.min_new_tokens,
            max_repeat_run=args.max_repeat_run,
        )

        props = compute_properties(smiles)
        canonical_smiles = canonicalize_smiles(smiles) if props["valid"] else None
        unique_valid = bool(props["valid"] and canonical_smiles not in seen_valid)
        novel = None

        if unique_valid and canonical_smiles is not None:
            seen_valid.add(canonical_smiles)

        if training_reference is not None and canonical_smiles is not None:
            novel = canonical_smiles not in training_reference

        row = {
            "attempt": attempt,
            "smiles": smiles,
            "canonical_smiles": canonical_smiles or "",
            "valid": props["valid"],
            "unique_valid": unique_valid,
            "novel": novel,
            "qed": props["qed"],
            "mw": props["mw"],
            "logp": props["logp"],
            "sa_score": props["sa_score"],
            "tpsa": props["tpsa"],
            "hbd": props["hbd"],
            "hba": props["hba"],
            "rot_bonds": props["rot_bonds"],
            "rings": props["rings"],
            "lipinski": props["lipinski"],
        }
        rows.append(row)

    summary = summarise_rows(rows, training_reference=training_reference)
    summary.update(
        {
            "checkpoint": checkpoint_path,
            "seed_token": args.seed_token,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "min_new_tokens": args.min_new_tokens,
            "max_repeat_run": args.max_repeat_run,
            "max_len": effective_max_len,
            "seed": args.seed,
        }
    )

    csv_path, json_path = make_output_paths(
        checkpoint_path, output_csv=args.output_csv, output_json=args.output_json
    )
    save_csv(rows, csv_path)
    save_json(summary, json_path)

    print("\n[Eval] Summary")
    print("-" * 40)
    print(f"  Total Generated : {summary['total_generated']}")
    print(f"  Valid           : {summary['valid_count']} ({summary['validity_rate']}%)")
    print(f"  Unique Valid    : {summary['unique_valid_count']} ({summary['uniqueness_rate']}%)")
    if summary["novelty_available"]:
        print(f"  Novel Unique    : {summary['novel_unique_count']} ({summary['novelty_rate']}%)")
    else:
        print("  Novel Unique    : N/A")
    print(f"  Diversity       : {summary['diversity']}")
    print(f"  Avg QED         : {summary['avg_qed']}")
    print(f"  Avg MW          : {summary['avg_mw']}")
    print(f"  Avg LogP        : {summary['avg_logp']}")
    print(f"  Avg SA Score    : {summary['avg_sa_score']}")
    print(f"  Lipinski Pass   : {summary['lipinski_pass_count']} ({summary['lipinski_pass_rate']}%)")
    print(f"  CSV Saved       : {csv_path}")
    print(f"  JSON Saved      : {json_path}")


if __name__ == "__main__":
    main()
