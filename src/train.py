import argparse
import csv
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from .dataset import SmilesDataset
    from .model import MoleculeTransformer
    from .splits import build_dataset_splits, load_split_artifacts, save_split_artifacts
    from .tokenizer import SmilesTokenizer
    from .utils import get_data_path, get_project_root, load_smiles, save_model, set_seed
except ImportError:
    from dataset import SmilesDataset
    from model import MoleculeTransformer
    from splits import build_dataset_splits, load_split_artifacts, save_split_artifacts
    from tokenizer import SmilesTokenizer
    from utils import get_data_path, get_project_root, load_smiles, save_model, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train Molecule Transformer")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=60)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--dedup", action="store_true", default=True)
    parser.add_argument("--no_dedup", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--checkpoint_name", type=str, default="best_model.pt")
    parser.add_argument("--save_last", action="store_true")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for checkpoints and loss curves. Defaults to project checkpoints/.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Optional SMILES source file. Defaults to data/smiles.txt.",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default=None,
        help="Directory used to save or reuse train/val/test split artifacts.",
    )
    parser.add_argument(
        "--reuse_split",
        action="store_true",
        help="Load train/val/test splits from --split_dir instead of rebuilding them.",
    )
    parser.add_argument(
        "--split_method",
        choices=["random", "scaffold"],
        default="scaffold",
        help="Use scaffold split to reduce structural leakage (requires RDKit).",
    )
    return parser.parse_args()


def get_default_split_dir(args):
    split_tag = (
        f"{args.split_method}_seed{args.seed}"
        f"_val{int(args.val_split * 100)}_test{int(args.test_split * 100)}"
    )
    return os.path.join(get_project_root(), "data", "splits", split_tag)


def get_default_output_dir():
    return os.path.join(get_project_root(), "checkpoints")


def get_available_output_path(path):
    if not os.path.exists(path):
        return path

    directory = os.path.dirname(path)
    stem, ext = os.path.splitext(os.path.basename(path))
    index = 1

    while True:
        candidate = os.path.join(directory, f"{stem}_{index}{ext}")
        if not os.path.exists(candidate):
            return candidate
        index += 1


def resolve_run_paths(args):
    output_dir = os.path.abspath(args.output_dir or get_default_output_dir())
    os.makedirs(output_dir, exist_ok=True)

    requested_best_path = os.path.join(output_dir, args.checkpoint_name)
    best_model_path = get_available_output_path(requested_best_path)
    run_stem = os.path.splitext(os.path.basename(best_model_path))[0]

    return {
        "output_dir": output_dir,
        "requested_best_model": requested_best_path,
        "best_model": best_model_path,
        "last_model": os.path.join(output_dir, f"{run_stem}_last_model.pt"),
        "loss_history": os.path.join(output_dir, f"{run_stem}_loss_history.csv"),
    }


def train():
    args = parse_args()
    if args.no_dedup:
        args.dedup = False
    if args.no_pin_memory:
        args.pin_memory = False

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = args.pin_memory and device.type == "cuda"
    print(f"[Train] Using device: {device}")
    if torch.cuda.is_available():
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")

    split_dir = args.split_dir or get_default_split_dir(args)
    run_paths = resolve_run_paths(args)

    if args.reuse_split:
        split_data, split_metadata = load_split_artifacts(split_dir)
        print(f"[Split] Reusing split artifacts from: {split_dir}")
    else:
        smiles_path = args.data_path or get_data_path("smiles.txt")
        smiles_list = load_smiles(smiles_path)
        split_data, split_metadata = build_dataset_splits(
            smiles_list=smiles_list,
            val_split=args.val_split,
            test_split=args.test_split,
            seed=args.seed,
            split_method=args.split_method,
            dedup=args.dedup,
        )
        save_split_artifacts(split_dir, split_data, split_metadata)
        print(f"[Split] Saved split artifacts to: {split_dir}")

    train_smiles = split_data["train"]
    val_smiles = split_data["val"]
    test_smiles = split_data["test"]
    overlap_info = split_metadata.get("overlaps", {})

    print(f"[Split] Requested method  : {args.split_method}")
    print(
        f"[Split] Effective method  : "
        f"{split_metadata.get('effective_split_method', args.split_method)}"
    )
    print(f"[Split] Invalid removed   : {split_metadata.get('invalid_removed', 0):,}")
    print(f"[Split] Dedup enabled     : {args.dedup}")
    print(f"[Split] Train samples     : {len(train_smiles):,}")
    print(f"[Split] Val samples       : {len(val_smiles):,}")
    print(f"[Split] Test samples      : {len(test_smiles):,}")
    print(f"[Split] Train/Val overlap : {overlap_info.get('train_val', 0):,}")
    print(f"[Split] Train/Test overlap: {overlap_info.get('train_test', 0):,}")
    print(f"[Split] Val/Test overlap  : {overlap_info.get('val_test', 0):,}\n")

    tokenizer = SmilesTokenizer()
    tokenizer.fit(train_smiles)

    train_dataset = SmilesDataset(train_smiles, tokenizer, max_len=args.max_len)
    val_dataset = SmilesDataset(val_smiles, tokenizer, max_len=args.max_len)
    train_size = len(train_dataset)
    val_size = len(val_dataset)

    if train_size == 0:
        raise ValueError("Training split is empty. Check the split settings and input data.")
    if val_size == 0:
        raise ValueError("Validation split is empty. Use a positive --val_split for training.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    print(f"[Train] Train samples  : {train_size}")
    print(f"[Train] Val samples    : {val_size}")
    print(f"[Train] Num workers    : {args.num_workers}")
    print(f"[Train] Pin memory     : {pin_memory}")
    print(f"[Train] Batches/epoch  : {len(train_loader)}")
    print(f"[Train] Output dir     : {run_paths['output_dir']}")
    if run_paths["best_model"] != run_paths["requested_best_model"]:
        print(
            "[Train] Existing checkpoint detected; "
            f"using {os.path.basename(run_paths['best_model'])} instead."
        )
    print(f"[Train] Best model path : {run_paths['best_model']}")
    print(f"[Train] Loss history    : {run_paths['loss_history']}\n")

    model_config = {
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.d_model * 4,
        "max_len": args.max_len,
        "dropout": args.dropout,
        "pad_token_id": tokenizer.pad_token_id,
    }

    model = MoleculeTransformer(vocab_size=tokenizer.vocab_size, **model_config).to(device)

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
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for input_ids, target_ids in train_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(input_ids)
                loss = criterion(
                    logits.reshape(-1, tokenizer.vocab_size),
                    target_ids.reshape(-1),
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
                input_ids = input_ids.to(device, non_blocking=True)
                target_ids = target_ids.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(input_ids)
                    loss = criterion(
                        logits.reshape(-1, tokenizer.vocab_size),
                        target_ids.reshape(-1),
                    )
                val_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        current_lr = scheduler.get_last_lr()[0]

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(avg_train, 6),
                "val_loss": round(avg_val, 6),
                "lr": round(current_lr, 8),
            }
        )

        print(
            f"Epoch [{epoch:02d}/{args.epochs}]  "
            f"Train: {avg_train:.4f}  "
            f"Val: {avg_val:.4f}  "
            f"LR: {current_lr:.6f}"
        )

        if avg_val < (best_val_loss - args.min_delta):
            best_val_loss = avg_val
            patience_counter = 0
            save_model(model, tokenizer, run_paths["best_model"], model_config)
            print(f"  [OK] Best model saved (val={best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  [--] No improvement (patience {patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"\n[Early Stop] Stopped at epoch {epoch}")
                break

    if args.save_last:
        save_model(model, tokenizer, run_paths["last_model"], model_config)

    csv_path = run_paths["loss_history"]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        writer.writeheader()
        writer.writerows(history)

    print("\n[OK] Training complete!")
    print(f"[OK] Best val loss : {best_val_loss:.4f}")
    print(f"[OK] Loss history  : {csv_path}")
    print(f"[OK] Best model    : {run_paths['best_model']}")


if __name__ == "__main__":
    train()
