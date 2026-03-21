import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizer import SmilesTokenizer
from dataset import SmilesDataset
from model import MoleculeTransformer
from utils import set_seed, load_smiles, save_model, get_data_path, get_checkpoint_path
def parse_args():
    parser = argparse.ArgumentParser(description="Train Molecule Transformer")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--d_model", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="Transformer layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()
def train():
    args = parse_args()
    set_seed(args.seed)
    data_path = get_data_path("smiles.txt")
    smiles_list = load_smiles(data_path)
    tokenizer = SmilesTokenizer()
    tokenizer.fit(smiles_list)
    dataset = SmilesDataset(smiles_list, tokenizer, max_len=args.max_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"\n[Train] Dataset size: {len(dataset)}")
    print(f"[Train] Batches per epoch: {len(dataloader)}")
    model = MoleculeTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * 2,
        max_len=args.max_len,
        pad_token_id=tokenizer.pad_token_id,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Model parameters: {total_params:,}\n")
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            logits = model(input_ids)  
            logits_flat = logits.view(-1, tokenizer.vocab_size)
            targets_flat = target_ids.view(-1)
            loss = criterion(logits_flat, targets_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch}/{args.epochs}]  Loss: {avg_loss:.4f}")
    checkpoint_path = get_checkpoint_path("model.pt")
    save_model(model, tokenizer, checkpoint_path)
    print("\n[OK] Training complete!")
    print(f"[OK] Model saved to: {checkpoint_path}")
if __name__ == "__main__":
    train()
