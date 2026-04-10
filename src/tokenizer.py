import json
import re

SMILES_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"
    r"|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$"
    r"|\%[0-9]{2}|[0-9])"
)


class SmilesTokenizer:
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    def __init__(self):
        self.special_tokens = [
            self.PAD_TOKEN,
            self.SOS_TOKEN,
            self.EOS_TOKEN,
            self.UNK_TOKEN,
        ]
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0

    def tokenize(self, smiles):
        tokens = SMILES_PATTERN.findall(smiles)
        if not tokens:
            return list(smiles)
        return tokens

    def fit(self, smiles_list):
        all_tokens = set()
        for smiles in smiles_list:
            all_tokens.update(self.tokenize(smiles))

        self.stoi = {}
        for idx, token in enumerate(self.special_tokens):
            self.stoi[token] = idx

        for idx, token in enumerate(sorted(all_tokens), start=len(self.special_tokens)):
            self.stoi[token] = idx

        self.itos = {idx: token for token, idx in self.stoi.items()}
        self.vocab_size = len(self.stoi)

        print(f"[Tokenizer] Vocabulary size : {self.vocab_size}")
        print(f"[Tokenizer] Unique tokens   : {sorted(all_tokens)}")

    def encode(self, smiles, warn_unknown=False):
        tokens = [self.sos_token_id]
        for token in self.tokenize(smiles):
            if token in self.stoi:
                tokens.append(self.stoi[token])
            else:
                if warn_unknown:
                    print(f"  [Tokenizer] Unknown token '{token}' -> <UNK>")
                tokens.append(self.unk_token_id)
        tokens.append(self.eos_token_id)
        return tokens

    def encode_batch(self, smiles_list, warn_unknown=False):
        return [self.encode(smiles, warn_unknown) for smiles in smiles_list]

    def decode(self, token_ids):
        skip = {self.PAD_TOKEN, self.SOS_TOKEN, self.UNK_TOKEN}
        tokens = []
        for token_id in token_ids:
            token = self.itos.get(token_id, "")
            if token == self.EOS_TOKEN:
                break
            if token in skip:
                continue
            tokens.append(token)
        return "".join(tokens)

    def save(self, path):
        data = {
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
            "vocab_size": self.vocab_size,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        print(f"[Tokenizer] Saved to {path}")

    def load(self, path):
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        self.stoi = data["stoi"]
        self.itos = {int(k): v for k, v in data["itos"].items()}
        self.vocab_size = data["vocab_size"]
        print(f"[Tokenizer] Loaded from {path} - vocab size: {self.vocab_size}")

    @property
    def pad_token_id(self):
        return self.stoi[self.PAD_TOKEN]

    @property
    def sos_token_id(self):
        return self.stoi[self.SOS_TOKEN]

    @property
    def eos_token_id(self):
        return self.stoi[self.EOS_TOKEN]

    @property
    def unk_token_id(self):
        return self.stoi[self.UNK_TOKEN]

    def __len__(self):
        return self.vocab_size

    def __repr__(self):
        return f"SmilesTokenizer(vocab_size={self.vocab_size})"


if __name__ == "__main__":
    tokenizer = SmilesTokenizer()
    tokenizer.fit(["CCO", "c1ccccc1", "CC(=O)O", "ClC1=CC=CC=C1", "BrC1=CC=CC=C1"])

    test_cases = ["CCO", "CC(=O)O", "ClC1=CC=CC=C1", "BrC1=CC=CC=C1"]
    for smiles in test_cases:
        tokens = tokenizer.tokenize(smiles)
        encoded = tokenizer.encode(smiles)
        decoded = tokenizer.decode(encoded)
        print(f"\nSMILES  : {smiles}")
        print(f"Tokens  : {tokens}")
        print(f"Encoded : {encoded}")
        print(f"Decoded : {decoded}")
        print(f"Match   : {smiles == decoded}")
