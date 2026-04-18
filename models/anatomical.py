import torch
import torch.nn as nn

class AnatomicalBranch(nn.Module):
    """
    PhishFusion Projesi - Dal 1: Anatomik Analiz Kanalı (QR Görsel Yapısı)
    QR kodun 24 yapısal özelliğini işleyerek 32D temsil üretir.
    """
    def __init__(self, input_dim=24, hidden_dim=64, output_dim=32, dropout_prob=0.2):
        super(AnatomicalBranch, self).__init__()
        
        # Blok 1: 24 Yapısal Özellik -> 64 Gizli Nöron
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        
        # Blok 2: Özellik Sentezi (64 -> 64)
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        
        # Blok 3: Temsil Vektörü Oluşturma (64 -> 32)
        self.block3 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )

    def forward(self, x):
        """
        x: (batch_size, 24) -> QR kodun yapısal öznitelikleri
        return: (batch_size, 32) -> Anatomik temsil vektörü
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

# Test Amaçlı Kullanım
if __name__ == "__main__":
    # Örnek Giriş (Batch: 16, Özellik: 24)
    sample_qr_features = torch.randn(16, 24)
    
    model = AnatomicalBranch()
    output = model(sample_qr_features)
    
    print(f"Anatomik Giriş Boyutu: {sample_qr_features.shape}") # [16, 24]
    print(f"Anatomik Temsil (32D): {output.shape}")             # [16, 32]