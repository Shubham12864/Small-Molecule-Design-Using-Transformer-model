import argparse
import torch
import warnings
warnings.filterwarnings("ignore")
from tokenizer import SmilesTokenizer
from model import MoleculeTransformer
from utils import load_model, get_checkpoint_path
try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
def parse_args():
    parser = argparse.ArgumentParser(description="Generate SMILES molecules")
    parser.add_argument("--seed_token", type=str, default="C", help="Starting character(s)")
    parser.add_argument("--num_molecules", type=int, default=10, help="Number of molecules to generate")
    parser.add_argument("--max_len", type=int, default=50, help="Max generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0=greedy, 1.0=balanced)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    return parser.parse_args()
def generate_smiles(model, tokenizer, seed="C", max_len=50, temperature=1.0):
    model.eval()
    tokens = [tokenizer.sos_token_id]
    for char in seed:
        if char in tokenizer.stoi:
            tokens.append(tokenizer.stoi[char])
    with torch.no_grad():
        for _ in range(max_len):
            input_tensor = torch.tensor([tokens], dtype=torch.long)
            logits = model(input_tensor)  
            next_token_logits = logits[0, -1, :]
            if temperature <= 0:
                next_token_id = torch.argmax(next_token_logits).item()
            else:
                scaled_logits = next_token_logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
            if next_token_id == tokenizer.eos_token_id:
                break
            if next_token_id == tokenizer.pad_token_id:
                break
            tokens.append(next_token_id)
    generated_smiles = tokenizer.decode(tokens)
    return generated_smiles
def check_validity(smiles):
    if not RDKIT_AVAILABLE:
        return None  
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None
def main():
    args = parse_args()
    checkpoint_path = args.checkpoint or get_checkpoint_path("model.pt")
    print(f"Loading model from: {checkpoint_path}")
    model, tokenizer_data = load_model(MoleculeTransformer, checkpoint_path)
    tokenizer = SmilesTokenizer()
    tokenizer.stoi = tokenizer_data["stoi"]
    tokenizer.itos = {int(k): v for k, v in tokenizer_data["itos"].items()}
    tokenizer.vocab_size = tokenizer_data["vocab_size"]
    print(f"\nGenerating {args.num_molecules} molecules with seed '{args.seed_token}'...\n")
    print(f"{'#':<4} {'Generated SMILES':<50} {'Valid?'}")
    print("-" * 65)
    valid_count = 0
    generated_smiles_list = []
    for i in range(1, args.num_molecules + 1):
        smiles = generate_smiles(model, tokenizer, seed=args.seed_token, max_len=args.max_len, temperature=args.temperature)
        is_valid = check_validity(smiles)
        if is_valid:
            valid_count += 1
        validity_str = "YES" if is_valid else ("NO" if is_valid is not None else "?")
        print(f"{i:<4} {smiles:<50} {validity_str}")
        generated_smiles_list.append(smiles)
    if RDKIT_AVAILABLE:
        print(f"\nValidity: {valid_count}/{args.num_molecules} "
              f"({100 * valid_count / args.num_molecules:.1f}%)")
    else:
        print("\n(Install RDKit to check molecule validity)")
    return generated_smiles_list
if __name__ == "__main__":
    main()
import argparse
import torch
import warnings
warnings.filterwarnings("ignore")

from tokenizer import SmilesTokenizer
from model import MoleculeTransformer
from utils import load_model, get_checkpoint_path

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, QED
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SMILES molecules")
    parser.add_argument("--seed_token",     type=str,   default="C")
    parser.add_argument("--num_molecules",  type=int,   default=10)
    parser.add_argument("--max_len",        type=int,   default=100)
    parser.add_argument("--temperature",    type=float, default=0.8)
    parser.add_argument("--checkpoint",     type=str,   default=None)
    parser.add_argument("--allow_invalid",  action="store_true",
                        help="Include invalid molecules in output")
    return parser.parse_args()


def generate_smiles(model, tokenizer, seed="C", max_len=100,
                    temperature=1.0, device="cpu"):
    model.eval()

    tokens = [tokenizer.sos_token_id]

    import re
    SMILES_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    seed_tokens = re.findall(SMILES_PATTERN, seed)

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
            input_tensor = torch.tensor(
                [tokens], dtype=torch.long, device=device
            )
            logits          = model(input_tensor)
            next_logits     = logits[0, -1, :]

            if temperature <= 0:
                next_token_id = torch.argmax(next_logits).item()
            else:
                probs         = torch.softmax(next_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()

            if next_token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id):
                break

            tokens.append(next_token_id)

    return tokenizer.decode(tokens)


def get_properties(smiles):
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw       = Descriptors.MolWt(mol)
    logp     = Descriptors.MolLogP(mol)
    qed      = QED.qed(mol)
    hbd      = Descriptors.NumHDonors(mol)
    hba      = Descriptors.NumHAcceptors(mol)
    lipinski = mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10
    return {
        "mw": round(mw, 2),
        "logp": round(logp, 2),
        "qed": round(qed, 3),
        "hbd": hbd,
        "hba": hba,
        "lipinski": lipinski
    }


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Generate] Using device: {device}")

    checkpoint_path = args.checkpoint or get_checkpoint_path("best_model.pt")
    print(f"[Generate] Loading model from: {checkpoint_path}")

    model, tokenizer_data = load_model(MoleculeTransformer, checkpoint_path)
    model = model.to(device)
    model.eval()

    tokenizer            = SmilesTokenizer()
    tokenizer.stoi       = tokenizer_data["stoi"]
    tokenizer.itos       = {int(k): v for k, v in tokenizer_data["itos"].items()}
    tokenizer.vocab_size = tokenizer_data["vocab_size"]

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
            max_len     = args.max_len,
            temperature = args.temperature,
            device      = device
        )

        props    = get_properties(smiles)
        is_valid = props is not None
        is_unique= smiles not in seen

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
        if p:
            print(f"{i:<4} {r['smiles']:<45} {'YES':<6} "
                  f"{p['qed']:<6} {p['mw']:<8} {p['logp']:<6} "
                  f"{'PASS' if p['lipinski'] else 'FAIL'}")
        else:
            print(f"{i:<4} {r['smiles']:<45} {'NO':<6} {'—':<6} {'—':<8} {'—':<6} —")

    if RDKIT_AVAILABLE:
        valid_results  = [r for r in results if r["valid"]]
        unique_results = [r for r in results if r["unique"]]

        validity   = 100 * len(valid_results)  / max(len(results), 1)
        uniqueness = 100 * len(unique_results) / max(len(valid_results), 1)

        avg_qed = sum(r["props"]["qed"] for r in valid_results) / max(len(valid_results), 1)
        avg_mw  = sum(r["props"]["mw"]  for r in valid_results) / max(len(valid_results), 1)
        lip_pass= sum(1 for r in valid_results if r["props"]["lipinski"])

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