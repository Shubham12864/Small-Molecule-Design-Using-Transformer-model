import torch
from torch.utils.data import Dataset
class SmilesDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        for smi in smiles_list:
            token_ids = tokenizer.encode(smi)
            if len(token_ids) > max_len + 1:
                token_ids = token_ids[: max_len + 1]
            while len(token_ids) < max_len + 1:
                token_ids.append(tokenizer.pad_token_id)
            self.data.append(token_ids)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        token_ids = self.data[idx]
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)
        return input_ids, target_ids
if __name__ == "__main__":
    from tokenizer import SmilesTokenizer
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    tok = SmilesTokenizer()
    tok.fit(smiles)
    ds = SmilesDataset(smiles, tok, max_len=20)
    print(f"Dataset size: {len(ds)}")
    inp, tgt = ds[0]
    print(f"Input shape:  {inp.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Input IDs:    {inp.tolist()}")
    print(f"Target IDs:   {tgt.tolist()}")
