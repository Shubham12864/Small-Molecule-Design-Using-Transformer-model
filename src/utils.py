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
    print(f"[Utils] Random seed set to {seed}")
def load_smiles(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    with open(filepath, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    print(f"[Utils] Loaded {len(smiles_list)} SMILES from {filepath}")
    return smiles_list
def save_model(model, tokenizer, path="checkpoints/model.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "stoi": tokenizer.stoi,
        "itos": tokenizer.itos,
        "vocab_size": tokenizer.vocab_size,
    }
    torch.save(checkpoint, path)
    print(f"[Utils] Model saved to {path}")
def load_model(model_class, path="checkpoints/model.pt", **model_kwargs):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    tokenizer_data = {
        "stoi": checkpoint["stoi"],
        "itos": checkpoint["itos"],
        "vocab_size": checkpoint["vocab_size"],
    }
    model = model_class(vocab_size=tokenizer_data["vocab_size"], **model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[Utils] Model loaded from {path}")
    return model, tokenizer_data
def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def get_data_path(filename="smiles.txt"):
    return os.path.join(get_project_root(), "data", filename)
def get_checkpoint_path(filename="model.pt"):
    return os.path.join(get_project_root(), "checkpoints", filename)
