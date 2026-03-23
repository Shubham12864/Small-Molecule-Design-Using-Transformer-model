import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tokenizer import SmilesTokenizer
from dataset import SmilesDataset
from model import MoleculeTransformer
from utils import set_seed, load_smiles, save_model, get_data_path, get_checkpoint_path
import csv

def parse_args():
    parser = argparse.ArgumentParser(description="Train Molecule Transformer")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--max_len",    type=int,   default=100)
    parser.add_argument("--d_model",    type=int,   default=256)
    parser.add_argument("--nhead",      type=int,   default=8)
    parser.add_argument("--num_layers", type=int,   default=4)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--val_split",  type=float, default=0.1)
    parser.add_argument("--patience",   type=int,   default=5)
    return parser.parse_args()


def train():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")
    if torch.cuda.is_available():
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")

    smiles_list = load_smiles(get_data_path("smiles.txt"))
    tokenizer   = SmilesTokenizer()
    tokenizer.fit(smiles_list)

    full_dataset = SmilesDataset(smiles_list, tokenizer, max_len=args.max_len)
    val_size     = int(len(full_dataset) * args.val_split)
    train_size   = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True
    )
    val_loader   = DataLoader(
        val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    print(f"[Train] Train samples : {train_size}")
    print(f"[Train] Val   samples : {val_size}")
    print(f"[Train] Batches/epoch : {len(train_loader)}\n")

    model = MoleculeTransformer(
        vocab_size      = tokenizer.vocab_size,
        d_model         = args.d_model,
        nhead           = args.nhead,
        num_layers      = args.num_layers,
        dim_feedforward = args.d_model * 4,
        max_len         = args.max_len,
        pad_token_id    = tokenizer.pad_token_id,
    ).to(device)

    print(f"[Train] Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = args.lr,
        steps_per_epoch = len(train_loader),
        epochs          = args.epochs,
        pct_start       = 0.1
    )

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    history          = []
    best_val_loss    = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):

        model.train()
        train_loss = 0.0

        for input_ids, target_ids in train_loader:
            input_ids  = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(input_ids)
                loss   = criterion(
                    logits.view(-1, tokenizer.vocab_size),
                    target_ids.view(-1)
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids  = input_ids.to(device, non_blocking=True)
                target_ids = target_ids.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = model(input_ids)
                    loss   = criterion(
                        logits.view(-1, tokenizer.vocab_size),
                        target_ids.view(-1)
                    )
                val_loss += loss.item()

        avg_train  = train_loss / len(train_loader)
        avg_val    = val_loss   / len(val_loader)
        current_lr = scheduler.get_last_lr()[0]

        history.append({
            "epoch":      epoch,
            "train_loss": round(avg_train, 6),
            "val_loss":   round(avg_val,   6),
            "lr":         round(current_lr, 8)
        })

        print(f"Epoch [{epoch:02d}/{args.epochs}]  "
              f"Train Loss: {avg_train:.4f}  "
              f"Val Loss: {avg_val:.4f}  "
              f"LR: {current_lr:.6f}")

        if avg_val < best_val_loss:
            best_val_loss    = avg_val
            patience_counter = 0
            save_model(model, tokenizer, get_checkpoint_path("best_model.pt"))
            print(f"  ✓ Best model saved  (val_loss = {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ✗ No improvement  (patience {patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"\n[Early Stop] No improvement for {args.patience} epochs. Stopping.")
                break

    csv_path = get_checkpoint_path("loss_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        writer.writeheader()
        writer.writerows(history)

    print(f"\n[OK] Training complete!")
    print(f"[OK] Best val loss   : {best_val_loss:.4f}")
    print(f"[OK] Loss history    : {csv_path}")
    print(f"[OK] Best model      : {get_checkpoint_path('best_model.pt')}")


if __name__ == "__main__":
    train()