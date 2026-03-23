import os
import random
import torch
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    print(f"[Utils] Seed set to {seed}")


def load_smiles(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    smiles_list = []
    skipped     = 0
    with open(filepath, "r") as f:
        for line in f:
            smi = line.strip()
            if not smi:
                continue
            if smi.startswith("#"):
                skipped += 1
                continue
            if " " in smi or "\t" in smi:
                smi = smi.split()[0]
            smiles_list.append(smi)

    print(f"[Utils] Loaded   : {len(smiles_list):,} SMILES from {filepath}")
    if skipped:
        print(f"[Utils] Skipped  : {skipped:,} comment lines")
    return smiles_list


def save_model(model, tokenizer, path, model_config=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "stoi":             tokenizer.stoi,
        "itos":             tokenizer.itos,
        "vocab_size":       tokenizer.vocab_size,
        "model_config":     model_config or {},
    }

    torch.save(checkpoint, path)
    print(f"[Utils] Model saved to {path}")

    if model_config:
        print(f"[Utils] Config   : {model_config}")


def load_model(model_class, path, **model_kwargs):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    saved_config = checkpoint.get("model_config", {})

    merged_kwargs = {**saved_config, **model_kwargs}

    tokenizer_data = {
        "stoi":       checkpoint["stoi"],
        "itos":       checkpoint["itos"],
        "vocab_size": checkpoint["vocab_size"],
    }

    model = model_class(
        vocab_size=tokenizer_data["vocab_size"],
        **merged_kwargs
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"[Utils] Model loaded from {path}")
    if saved_config:
        print(f"[Utils] Config   : {saved_config}")

    return model, tokenizer_data


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_path(filename="smiles.txt"):
    return os.path.join(get_project_root(), "data", filename)


def get_checkpoint_path(filename="best_model.pt"):
    return os.path.join(get_project_root(), "checkpoints", filename)