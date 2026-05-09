import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class LexicalBranch(nn.Module):
    def __init__(self, vocab_size=1000, d_model=512, nhead=8, num_layers=8, output_dim=32, pad_idx=0):
        super(LexicalBranch, self).__init__()
        self.pad_idx = pad_idx # Define which ID represents padding
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=512, 
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 1. Create the mask: (batch_size, seq_len)
        # True values tell the Transformer to ignore these positions
        src_key_padding_mask = (x == self.pad_idx)
        
        # 2. Embedding & Positional Encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        
        # 3. Transformer Analysis with Mask
        # The mask ensures padding tokens don't influence the attention scores
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # 4. Masked Global Average Pooling
        # Simple torch.mean(x, dim=1) still counts padding vectors. 
        # We should average only the non-padding tokens for a cleaner 32D representation.
        mask_for_mean = ~src_key_padding_mask.unsqueeze(-1) # (batch, seq_len, 1)
        x_masked = x * mask_for_mean
        sum_x = torch.sum(x_masked, dim=1)
        count_x = torch.clamp(mask_for_mean.sum(dim=1), min=1e-9) # Avoid div by zero
        x = sum_x / count_x
        
        # 5. Final Representation
        x = self.fc_out(x)
        x = self.relu(x)
        
        return x