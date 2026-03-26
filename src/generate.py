import argparse
import torch
import warnings
warnings.filterwarnings("ignore")

from tokenizer import SmilesTokenizer, SMILES_PATTERN
from model import MoleculeTransformer
from utils import load_model, get_checkpoint_path
from property import compute_properties

try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SMILES molecules")
    parser.add_argument("--seed_token",    type=str,   default="C")
    parser.add_argument("--num_molecules", type=int,   default=10)
    parser.add_argument("--max_len",       type=int,   default=100)
    parser.add_argument("--temperature",   type=float, default=0.8)
    parser.add_argument("--checkpoint",    type=str,   default=None)
    parser.add_argument("--allow_invalid", action="store_true")
    return parser.parse_args()


def generate_smiles(model, tokenizer, seed="C", max_len=100,
                    temperature=1.0, device="cpu"):
    model.eval()
    model_max_len = model.pos_encoder.pe.size(1)

    tokens      = [tokenizer.sos_token_id]
    seed_tokens = SMILES_PATTERN.findall(seed)

    if not seed_tokens:
        print(f"  [Warn] Seed '{seed}' produced no valid tokens. Using empty seed.")
    else:
        for tok in seed_tokens:
            if tok in tokenizer.stoi:
                tokens.append(tokenizer.stoi[tok])
            else:
                print(f"  [Warn] Token '{tok}' not in vocabulary. Skipping.")

    with torch.no_grad():
        for _ in range(max_len):
            if len(tokens) >= model_max_len:
                break
            input_tensor = torch.tensor(
                [tokens], dtype=torch.long, device=device
            )
            logits      = model(input_tensor)
            next_logits = logits[0, -1, :]

            if temperature <= 0:
                next_token_id = torch.argmax(next_logits).item()
            else:
                probs         = torch.softmax(next_logits / temperature, dim=-1)
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
    model_max_len = model.pos_encoder.pe.size(1)

    tokenizer            = SmilesTokenizer()
    tokenizer.stoi       = tokenizer_data["stoi"]
    tokenizer.itos       = {int(k): v for k, v in tokenizer_data["itos"].items()}
    tokenizer.vocab_size = tokenizer_data["vocab_size"]

    effective_max_len = min(args.max_len, model_max_len)
    if args.max_len > model_max_len:
        print(f"[Generate] Requested max_len={args.max_len} exceeds "
              f"checkpoint limit {model_max_len}. Using {model_max_len}.")

    print(f"\n[Generate] Seed: '{args.seed_token}' | "
          f"Target: {args.num_molecules} molecules | "
          f"Temp: {args.temperature}\n")

    results      = []
    seen         = set()
    attempts     = 0
    max_attempts = args.num_molecules * 5

    while len(results) < args.num_molecules and attempts < max_attempts:
        attempts += 1
        smiles = generate_smiles(
            model, tokenizer,
            seed        = args.seed_token,
            max_len     = effective_max_len,
            temperature = args.temperature,
            device      = device
        )

        props     = compute_properties(smiles)
        is_valid  = props["valid"]
        is_unique = smiles not in seen

        if not args.allow_invalid and not is_valid:
            continue

        seen.add(smiles)
        results.append({
            "smiles":   smiles,
            "valid":    is_valid,
            "unique":   is_unique,
            "props":    props
        })

    print(f"{'#':<4} {'SMILES':<45} {'Valid':<6} {'QED':<6} {'MW':<8} {'LogP':<6} {'Lipinski'}")
    print("-" * 90)

    for i, r in enumerate(results, 1):
        p = r["props"]
        if p and p["valid"]:
            print(f"{i:<4} {r['smiles']:<45} {'YES':<6} "
                  f"{p['qed']:<6} {p['mw']:<8} {p['logp']:<6} "
                  f"{'PASS' if p['lipinski'] else 'FAIL'}")
        else:
            print(f"{i:<4} {r['smiles']:<45} {'NO':<6} "
                  f"{'—':<6} {'—':<8} {'—':<6} —")

    if RDKIT_AVAILABLE:
        valid_results  = [r for r in results if r["valid"]]
        unique_results = [r for r in results if r["unique"]]

        validity   = 100 * len(valid_results)  / max(len(results), 1)
        uniqueness = 100 * len(unique_results) / max(len(valid_results), 1)
        avg_qed    = sum(r["props"]["qed"] for r in valid_results
                         if r["props"].get("qed")) / max(len(valid_results), 1)
        avg_mw     = sum(r["props"]["mw"]  for r in valid_results
                         if r["props"].get("mw"))  / max(len(valid_results), 1)
        lip_pass   = sum(1 for r in valid_results if r["props"].get("lipinski"))

        print(f"\n{'='*40}")
        print(f"  Total Generated : {len(results)}")
        print(f"  Valid           : {len(valid_results)} ({validity:.1f}%)")
        print(f"  Unique          : {len(unique_results)} ({uniqueness:.1f}%)")
        print(f"  Avg QED         : {avg_qed:.3f}")
        print(f"  Avg MW          : {avg_mw:.1f}")
        print(f"  Lipinski Pass   : {lip_pass}/{len(valid_results)}")
        print(f"{'='*40}\n")


if __name__ == "__main__":
    main()
