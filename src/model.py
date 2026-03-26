import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
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
        max_len=512,
        dropout=0.1,
        pad_token_id=0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.emb_dropout = nn.Dropout(p=dropout)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        self.output_head = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].zero_()

        nn.init.normal_(self.output_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_head.bias)

        for name, p in self.transformer.named_parameters():
            if "weight" in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def _generate_causal_mask(self, seq_len, device):
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(self, x):
        seq_len = x.size(1)
        device = x.device
        causal_mask = self._generate_causal_mask(seq_len, device)
        padding_mask = (x == self.pad_token_id)

        emb = self.embedding(x) * math.sqrt(self.d_model)
        emb = self.emb_dropout(emb)
        emb = self.pos_encoder(emb)

        out = self.transformer(
            src=emb,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )

        return self.output_head(out)


if __name__ == "__main__":
    vocab_size = 40
    model      = MoleculeTransformer(vocab_size=vocab_size)
    total      = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total:,}")
    dummy  = torch.randint(0, vocab_size, (4, 20))
    output = model(dummy)
    print(f"Input shape : {dummy.shape}")
    print(f"Output shape: {output.shape}")
