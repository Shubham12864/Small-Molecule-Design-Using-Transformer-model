import argparse
import warnings

import torch

from model import MoleculeTransformer
from property import compute_properties
from tokenizer import SMILES_PATTERN, SmilesTokenizer
from utils import get_checkpoint_path, load_model

warnings.filterwarnings("ignore")

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def check_validity(smiles):
    props = compute_properties(smiles)
    return bool(props and props.get("valid", False))


def _apply_top_k_top_p(logits, top_k=0, top_p=1.0):
    filtered = logits.clone()
    vocab_size = filtered.size(-1)

    if top_k > 0 and top_k < vocab_size:
        threshold = torch.topk(filtered, top_k).values[-1]
        filtered[filtered < threshold] = float("-inf")

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        filtered[indices_to_remove] = float("-inf")

    return filtered


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SMILES molecules")
    parser.add_argument("--seed_token", type=str, default="C")
    parser.add_argument("--num_molecules", type=int, default=10)
    parser.add_argument(
        "--max_len",
        type=int,
        default=None,
        help="Total sequence length. Defaults to the checkpoint context window.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--min_new_tokens", type=int, default=8)
    parser.add_argument("--max_repeat_run", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--allow_invalid", action="store_true")
    return parser.parse_args()


def generate_smiles(
    model,
    tokenizer,
    seed="C",
    max_len=None,
    temperature=1.0,
    device=None,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.15,
    min_new_tokens=8,
    max_repeat_run=4,
):
    model.eval()

    model_device = next(model.parameters()).device
    if device is None:
        device = model_device
    else:
        device = torch.device(device)
        if device != model_device:
            device = model_device

    model_ctx_len = int(model.pos_encoder.pe.size(1))
    effective_max_len = model_ctx_len if max_len is None else min(max_len, model_ctx_len)

    tokens = [tokenizer.sos_token_id]
    seed_tokens = SMILES_PATTERN.findall(seed)

    if not seed_tokens:
        print(f"  [Warn] Seed '{seed}' produced no valid tokens. Using empty seed.")
    else:
        for tok in seed_tokens:
            if tok in tokenizer.stoi:
                tokens.append(tokenizer.stoi[tok])
            else:
                print(f"  [Warn] Token '{tok}' not in vocabulary. Skipping.")

    if len(tokens) >= effective_max_len:
        return tokenizer.decode(tokens[:effective_max_len])

    initial_token_count = len(tokens)
    blocked_token_ids = {
        tokenizer.pad_token_id,
        tokenizer.sos_token_id,
        tokenizer.unk_token_id,
    }

    with torch.no_grad():
        max_new_tokens = effective_max_len - len(tokens)
        for _ in range(max_new_tokens):
            input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model(input_tensor)
            raw_next_logits = logits[0, -1, :].float()
            next_logits = raw_next_logits.clone()

            for tok_id in blocked_token_ids:
                if 0 <= tok_id < next_logits.numel():
                    next_logits[tok_id] = float("-inf")

            if repetition_penalty > 1.0:
                for tok_id in set(tokens):
                    if tok_id in blocked_token_ids:
                        continue
                    if 0 <= tok_id < next_logits.numel():
                        if next_logits[tok_id] > 0:
                            next_logits[tok_id] /= repetition_penalty
                        else:
                            next_logits[tok_id] *= repetition_penalty

            generated_so_far = len(tokens) - initial_token_count
            if generated_so_far < min_new_tokens and 0 <= tokenizer.eos_token_id < next_logits.numel():
                next_logits[tokenizer.eos_token_id] = float("-inf")

            if max_repeat_run > 0 and len(tokens) >= max_repeat_run:
                last_token = tokens[-1]
                if all(tok == last_token for tok in tokens[-max_repeat_run:]):
                    next_logits[last_token] = float("-inf")

            next_logits = _apply_top_k_top_p(next_logits, top_k=top_k, top_p=top_p)

            if torch.isneginf(next_logits).all():
                next_logits = raw_next_logits.clone()
                for tok_id in blocked_token_ids:
                    if 0 <= tok_id < next_logits.numel():
                        next_logits[tok_id] = float("-inf")

            if temperature <= 0:
                next_token_id = torch.argmax(next_logits).item()
            else:
                scaled_logits = next_logits / max(float(temperature), 1e-6)
                probs = torch.softmax(scaled_logits, dim=-1)
                if torch.isnan(probs).any() or float(probs.sum()) <= 0.0:
                    probs = torch.softmax(raw_next_logits / max(float(temperature), 1e-6), dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()

            if next_token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id):
                break

            tokens.append(next_token_id)

    return tokenizer.decode(tokens)


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Generate] Using device: {device}")

    checkpoint_path = args.checkpoint or get_checkpoint_path("best_model.pt")
    print(f"[Generate] Loading model from: {checkpoint_path}")

    model, tokenizer_data = load_model(MoleculeTransformer, checkpoint_path)
    model = model.to(device)
    model.eval()
    model_max_len = int(model.pos_encoder.pe.size(1))

    tokenizer = SmilesTokenizer()
    tokenizer.stoi = tokenizer_data["stoi"]
    tokenizer.itos = {int(k): v for k, v in tokenizer_data["itos"].items()}
    tokenizer.vocab_size = tokenizer_data["vocab_size"]

    effective_max_len = model_max_len if args.max_len is None else min(args.max_len, model_max_len)
    if args.max_len is None:
        print(f"[Generate] Using checkpoint max_len={model_max_len}.")
    elif args.max_len > model_max_len:
        print(
            f"[Generate] Requested max_len={args.max_len} exceeds "
            f"checkpoint limit {model_max_len}. Using {model_max_len}."
        )

    print(
        f"\n[Generate] Seed: '{args.seed_token}' | "
        f"Target: {args.num_molecules} molecules | "
        f"Temp: {args.temperature}\n"
    )

    results = []
    seen = set()
    attempts = 0
    max_attempts = args.num_molecules * 5

    while len(results) < args.num_molecules and attempts < max_attempts:
        attempts += 1
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
        is_valid = props["valid"]
        is_unique = smiles not in seen

        if not args.allow_invalid and not is_valid:
            continue

        seen.add(smiles)
        results.append(
            {
                "smiles": smiles,
                "valid": is_valid,
                "unique": is_unique,
                "props": props,
            }
        )

    print(f"{'#':<4} {'SMILES':<45} {'Valid':<6} {'QED':<6} {'MW':<8} {'LogP':<6} {'Lipinski'}")
    print("-" * 90)

    for i, r in enumerate(results, 1):
        p = r["props"]
        if p and p["valid"]:
            print(
                f"{i:<4} {r['smiles']:<45} {'YES':<6} "
                f"{p['qed']:<6} {p['mw']:<8} {p['logp']:<6} "
                f"{'PASS' if p['lipinski'] else 'FAIL'}"
            )
        else:
            print(f"{i:<4} {r['smiles']:<45} {'NO':<6} {'N/A':<6} {'N/A':<8} {'N/A':<6} N/A")

    if len(results) < args.num_molecules:
        print(f"\n[Generate] Warning: produced {len(results)} molecules after {attempts} attempts.")

    if RDKIT_AVAILABLE:
        valid_results = [r for r in results if r["valid"]]
        unique_results = [r for r in results if r["unique"]]

        validity = 100 * len(valid_results) / max(len(results), 1)
        uniqueness = 100 * len(unique_results) / max(len(valid_results), 1)
        avg_qed = sum(r["props"]["qed"] for r in valid_results if r["props"].get("qed")) / max(
            len(valid_results), 1
        )
        avg_mw = sum(r["props"]["mw"] for r in valid_results if r["props"].get("mw")) / max(
            len(valid_results), 1
        )
        lip_pass = sum(1 for r in valid_results if r["props"].get("lipinski"))

        print(f"\n{'=' * 40}")
        print(f"  Total Generated : {len(results)}")
        print(f"  Valid           : {len(valid_results)} ({validity:.1f}%)")
        print(f"  Unique          : {len(unique_results)} ({uniqueness:.1f}%)")
        print(f"  Avg QED         : {avg_qed:.3f}")
        print(f"  Avg MW          : {avg_mw:.1f}")
        print(f"  Lipinski Pass   : {lip_pass}/{len(valid_results)}")
        print(f"{'=' * 40}\n")


if __name__ == "__main__":
    main()
