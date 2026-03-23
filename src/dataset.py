import torch
from torch.utils.data import Dataset


class SmilesDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        pad_id         = tokenizer.pad_token_id

        sequences             = []
        skipped_empty         = 0
        skipped_too_short     = 0
        truncated             = 0

        for smi in smiles_list:
            smi = smi.strip()
            if not smi:
                skipped_empty += 1
                continue

            token_ids = tokenizer.encode(smi)

            if len(token_ids) < 3:
                skipped_too_short += 1
                continue

            if len(token_ids) > max_len + 1:
                token_ids = token_ids[:max_len + 1]
                truncated += 1

            padded      = [pad_id] * (max_len + 1)
            padded[:len(token_ids)] = token_ids
            sequences.append(padded)

        self.data = torch.tensor(sequences, dtype=torch.long)

        total = len(smiles_list)
        kept  = len(sequences)
        print(f"[Dataset] Total input      : {total:,}")
        print(f"[Dataset] Kept             : {kept:,}")
        print(f"[Dataset] Skipped (empty)  : {skipped_empty:,}")
        print(f"[Dataset] Skipped (short)  : {skipped_too_short:,}")
        print(f"[Dataset] Truncated        : {truncated:,}")
        lengths = [
            (self.data[i] != pad_id).sum().item()
            for i in range(min(1000, kept))
        ]
        if lengths:
            print(f"[Dataset] Avg seq length   : {sum(lengths)/len(lengths):.1f}")
            print(f"[Dataset] Max seq length   : {max(lengths)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_ids  = self.data[idx]
        input_ids  = token_ids[:-1]
        target_ids = token_ids[1:]
        return input_ids, target_ids


if __name__ == "__main__":
    from tokenizer import SmilesTokenizer

    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "ClC1=CC=CC=C1", "BrC1=CC=CC=C1", ""]
    tok    = SmilesTokenizer()
    tok.fit(smiles)

    ds = SmilesDataset(smiles, tok, max_len=20)
    print(f"\nDataset size : {len(ds)}")

    inp, tgt = ds[0]
    print(f"Input shape  : {inp.shape}")
    print(f"Target shape : {tgt.shape}")
    print(f"Input IDs    : {inp.tolist()}")
    print(f"Target IDs   : {tgt.tolist()}")