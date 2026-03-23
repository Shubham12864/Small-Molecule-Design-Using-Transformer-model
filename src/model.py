import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MoleculeTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        max_len=128,
        dropout=0.1,
        pad_token_id=0,
    ):
        super().__init__()
        self.d_model      = d_model
        self.pad_token_id = pad_token_id
        self.vocab_size   = vocab_size

        self.embedding    = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.emb_dropout  = nn.Dropout(p=dropout)
        self.pos_encoder  = PositionalEncoding(d_model, max_len, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward= dim_feedforward,
            dropout        = dropout,
            batch_first    = True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        self.output_head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
        for name, p in self.transformer_decoder.named_parameters():
            if "weight" in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def _generate_causal_mask(self, seq_len, device):
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float("-inf"),
            diagonal=1,
        )

    def forward(self, x):
        seq_len      = x.size(1)
        device       = x.device
        causal_mask  = self._generate_causal_mask(seq_len, device)
        padding_mask = (x == self.pad_token_id)

        emb = self.embedding(x) * math.sqrt(self.d_model)
        emb = self.emb_dropout(emb)
        emb = self.pos_encoder(emb)

        out = self.transformer_decoder(
            tgt                    = emb,
            memory                 = emb,
            tgt_mask               = causal_mask,
            tgt_key_padding_mask   = padding_mask,
            memory_key_padding_mask= padding_mask,
        )

        return self.output_head(out)

    @torch.no_grad()
    def generate(self, tokenizer, seed_tokens, max_len=100, temperature=1.0, device="cpu"):
        self.eval()
        input_ids = torch.tensor([seed_tokens], dtype=torch.long, device=device)

        for _ in range(max_len - len(seed_tokens)):
            logits     = self.forward(input_ids)
            next_logits= logits[:, -1, :] / max(temperature, 1e-8)
            probs      = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token], dim=1)

        tokens = input_ids[0].tolist()
        return tokenizer.decode(tokens)


if __name__ == "__main__":
    vocab_size = 40
    model      = MoleculeTransformer(vocab_size=vocab_size)
    total      = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total:,}")
    dummy  = torch.randint(0, vocab_size, (4, 20))
    output = model(dummy)
    print(f"Input shape : {dummy.shape}")
    print(f"Output shape: {output.shape}")