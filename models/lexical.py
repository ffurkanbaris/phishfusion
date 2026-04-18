import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Transformer'ın sıra bilgisini (sequence order) anlaması için 
    vektörlere konum bilgisi ekleyen yardımcı sınıf.
    """
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
    """
    PhishFusion Projesi - Dal 2: Transformer Tabanlı Leksikal Analiz
    URL metnini karakter/token bazlı işleyerek 32D temsil üretir.
    vocab_size: SentencePiece url_unigram.model ile uyumlu (url_unigram.vocab satır sayısı, şu an 1000).
    """
    def __init__(self, vocab_size=1000, d_model=512, nhead=8, num_layers=8, output_dim=32):
        super(LexicalBranch, self).__init__()
        
        # 1. Gömülme Katmanı (Embedding)
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Konumsal Kodlama (Positional Encoding)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformer Kodlayıcı Katmanları (8 Katman, 8 Başlık)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=512, 
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Global Havuzlama (Sıralı veriyi tekil vektöre indirger)
        # Sequence Length boyunca ortalama alarak global bağlamı yakalar
        
        # 5. Çıkış Katmanı: 32 Boyutlu Temsil Vektörü
        self.fc_out = nn.Linear(d_model, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: (batch_size, seq_len) -> Tokenize edilmiş URL indisleri
        return: (batch_size, 32) -> Leksikal temsil vektörü
        """
        # Gömülme ve Konum Ekleme
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        
        # Transformer Analizi (8 Katmanlı Derin İşleme)
        x = self.transformer_encoder(x)
        
        # Global Average Pooling (Zaman boyutunu daraltma)
        x = torch.mean(x, dim=1)
        
        # Nihai Temsil (32D)
        x = self.fc_out(x)
        x = self.relu(x)
        
        return x

# Örnek Kullanım
if __name__ == "__main__":
    # Örnek URL token'ları (Batch: 4, URL Uzunluğu: 128 karakter)
    sample_url_tokens = torch.randint(0, 1000, (4, 128))
    
    model = LexicalBranch()
    output = model(sample_url_tokens)
    
    print(f"URL Giriş Boyutu: {sample_url_tokens.shape}") # [4, 128]
    print(f"Leksikal Temsil (32D): {output.shape}")      # [4, 32]