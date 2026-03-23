import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizer import SmilesTokenizer
from dataset import SmilesDataset
from model import MoleculeTransformer
from utils import set_seed, load_smiles, save_model, get_data_path, get_checkpoint_path
import csv

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False

def parse_args():
    parser = argparse.ArgumentParser(description="Train Molecule Transformer")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--max_len",    type=int,   default=100)
    parser.add_argument("--d_model",    type=int,   default=256)
    parser.add_argument("--nhead",      type=int,   default=8)
    parser.add_argument("--num_layers", type=int,   default=4)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--val_split",  type=float, default=0.1)
    parser.add_argument("--patience",   type=int,   default=8)
    parser.add_argument("--min_delta",  type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--dedup", action="store_true", default=True)
    parser.add_argument("--no_dedup", action="store_true")
    parser.add_argument(
        "--split_method",
        choices=["random", "scaffold"],
        default="scaffold",
        help="Use scaffold split to reduce structural leakage (requires RDKit).",
    )
    return parser.parse_args()


def canonicalize_smiles(smi):
    if not HAS_RDKIT:
        return smi
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def get_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def build_train_val_split(smiles_list, val_split, seed, split_method, dedup):
    processed = []
    invalid = 0

    for smi in smiles_list:
        canon = canonicalize_smiles(smi)
        if canon is None:
            invalid += 1
            continue
        processed.append(canon)

    if dedup:
        # dict preserves insertion order in modern Python.
        processed = list(dict.fromkeys(processed))

    if len(processed) < 2:
        raise ValueError("Not enough valid samples after preprocessing.")

    val_size = max(1, int(len(processed) * val_split))

    if split_method == "scaffold":
        if not HAS_RDKIT:
            print("[Split] RDKit not available, falling back to random split.")
        else:
            rng = random.Random(seed)
            buckets = {}
            for smi in processed:
                scaffold = get_scaffold(smi)
                buckets.setdefault(scaffold, []).append(smi)

            groups = list(buckets.values())
            rng.shuffle(groups)
            groups.sort(key=len, reverse=True)

            val_smiles = []
            train_smiles = []
            for group in groups:
                if len(val_smiles) + len(group) <= val_size:
                    val_smiles.extend(group)
                else:
                    train_smiles.extend(group)

            if not val_smiles:
                val_smiles = train_smiles[:val_size]
                train_smiles = train_smiles[val_size:]

            return train_smiles, val_smiles, invalid

    rng = random.Random(seed)
    rng.shuffle(processed)
    val_smiles = processed[:val_size]
    train_smiles = processed[val_size:]
    return train_smiles, val_smiles, invalid


def train():
    args = parse_args()
    if args.no_dedup:
        args.dedup = False

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")
    if torch.cuda.is_available():
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")

    smiles_list = load_smiles(get_data_path("smiles.txt"))
    train_smiles, val_smiles, invalid = build_train_val_split(
        smiles_list=smiles_list,
        val_split=args.val_split,
        seed=args.seed,
        split_method=args.split_method,
        dedup=args.dedup,
    )

    overlap = len(set(train_smiles) & set(val_smiles))
    print(f"[Split] Method           : {args.split_method}")
    print(f"[Split] Invalid removed  : {invalid:,}")
    print(f"[Split] Dedup enabled    : {args.dedup}")
    print(f"[Split] Train samples    : {len(train_smiles):,}")
    print(f"[Split] Val samples      : {len(val_smiles):,}")
    print(f"[Split] Train/Val overlap: {overlap:,}\n")
    if overlap > 0:
        raise RuntimeError("Leakage detected: overlap between train and validation sets.")

    tokenizer = SmilesTokenizer()
    tokenizer.fit(train_smiles)

    train_dataset = SmilesDataset(train_smiles, tokenizer, max_len=args.max_len)
    val_dataset   = SmilesDataset(val_smiles, tokenizer, max_len=args.max_len)
    train_size    = len(train_dataset)
    val_size      = len(val_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    print(f"[Train] Train samples : {train_size}")
    print(f"[Train] Val   samples : {val_size}")
    print(f"[Train] Batches/epoch : {len(train_loader)}\n")

    model_config = {
        "d_model":         args.d_model,
        "nhead":           args.nhead,
        "num_layers":      args.num_layers,
        "dim_feedforward": args.d_model * 4,
        "max_len":         args.max_len,
        "dropout":         args.dropout,
    }

    model = MoleculeTransformer(
        vocab_size  = tokenizer.vocab_size,
        **model_config
    ).to(device)

    print(f"[Train] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[Train] Model config    : {model_config}\n")

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id,
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = args.lr,
        steps_per_epoch = len(train_loader),
        epochs          = args.epochs,
        pct_start       = 0.1
    )

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
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

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
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
              f"Train: {avg_train:.4f}  "
              f"Val: {avg_val:.4f}  "
              f"LR: {current_lr:.6f}")

        if avg_val < (best_val_loss - args.min_delta):
            best_val_loss    = avg_val
            patience_counter = 0
            save_model(
                model, tokenizer,
                get_checkpoint_path("best_model.pt"),
                model_config
            )
            print(f"  ✓ Best model saved (val={best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ✗ No improvement (patience {patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"\n[Early Stop] Stopped at epoch {epoch}")
                break

    csv_path = get_checkpoint_path("loss_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        writer.writeheader()
        writer.writerows(history)

    print(f"\n[OK] Training complete!")
    print(f"[OK] Best val loss : {best_val_loss:.4f}")
    print(f"[OK] Loss history  : {csv_path}")
    print(f"[OK] Best model    : {get_checkpoint_path('best_model.pt')}")


if __name__ == "__main__":
    train()