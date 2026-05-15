import torch
import torch.nn as nn

class AnatomicalBranch(nn.Module):
    """
    PhishFusion Projesi - Dal 1: Anatomik Analiz Kanalı (QR Görsel Yapısı)
    QR kodun 24 yapısal özelliğini işleyerek 32D temsil üretir.
    Residual bağlantılar ile sinyal kaybı minimize edilir.
    """
    def __init__(self, input_dim=24, hidden_dim=64, output_dim=32):
        super(AnatomicalBranch, self).__init__()
        
        # Blok 1: 24 -> 64 (Giriş projeksiyon)
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Blok 2: 64 -> 64 (Residual blok - giriş ile toplanacak)
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.relu2 = nn.ReLU()

        # Blok 3: 64 -> 32 (Temsil vektörü çıkışı)
        self.block3 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """
        x: (batch_size, 24) -> QR kodun yapısal öznitelikleri
        return: (batch_size, 32) -> Anatomik temsil vektörü
        """
        # Blok 1: Giriş projeksiyonu
        x = self.block1(x)
        
        # Blok 2: Residual bağlantı
        # Girdi ile blok çıkışını toplayarak sinyal kaybını önle
        residual = x
        x = self.relu2(self.block2(x) + residual)
        
        # Blok 3: 32D çıkış
        x = self.block3(x)
        return x
