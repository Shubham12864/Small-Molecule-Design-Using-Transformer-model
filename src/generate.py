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
