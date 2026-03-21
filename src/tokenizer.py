class SmilesTokenizer:
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    def __init__(self):
        self.special_tokens = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        self.stoi = {}  
        self.itos = {}  
        self.vocab_size = 0
    def fit(self, smiles_list):
        chars = set()
        for smi in smiles_list:
            chars.update(list(smi))
        chars = sorted(chars)
        self.stoi = {}
        for idx, token in enumerate(self.special_tokens):
            self.stoi[token] = idx
        for idx, char in enumerate(chars, start=len(self.special_tokens)):
            self.stoi[char] = idx
        self.itos = {idx: token for token, idx in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        print(f"[Tokenizer] Vocabulary size: {self.vocab_size}")
        print(f"[Tokenizer] Characters: {chars}")
    @property
    def pad_token_id(self):
        return self.stoi[self.PAD_TOKEN]
    @property
    def sos_token_id(self):
        return self.stoi[self.SOS_TOKEN]
    @property
    def eos_token_id(self):
        return self.stoi[self.EOS_TOKEN]
    def encode(self, smiles):
        tokens = [self.sos_token_id]
        for char in smiles:
            if char in self.stoi:
                tokens.append(self.stoi[char])
            else:
                pass
        tokens.append(self.eos_token_id)
        return tokens
    def decode(self, token_ids):
        chars = []
        for tid in token_ids:
            token = self.itos.get(tid, "")
            if token == self.EOS_TOKEN:
                break
            if token in (self.PAD_TOKEN, self.SOS_TOKEN):
                continue
            chars.append(token)
        return "".join(chars)
if __name__ == "__main__":
    tokenizer = SmilesTokenizer()
    tokenizer.fit(["CCO", "c1ccccc1", "CC(=O)O"])
    encoded = tokenizer.encode("CCO")
    print(f"Encoded 'CCO': {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded back:  '{decoded}'")
