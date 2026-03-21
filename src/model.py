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
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        max_len=128,
        dropout=0.1,
        pad_token_id=0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.output_head = nn.Linear(d_model, vocab_size)
        self._init_weights()
    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
    def _generate_causal_mask(self, seq_len, device):
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float("-inf"),
            diagonal=1,
        )
        return mask
    def forward(self, x):
        seq_len = x.size(1)
        device = x.device
        causal_mask = self._generate_causal_mask(seq_len, device)
        padding_mask = (x == self.pad_token_id)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(
            x, mask=causal_mask, src_key_padding_mask=padding_mask
        )
        logits = self.output_head(x)
        return logits
if __name__ == "__main__":
    vocab_size = 40
    model = MoleculeTransformer(vocab_size=vocab_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    dummy_input = torch.randint(0, vocab_size, (4, 20))  
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
