import torch
import torch.nn as nn

class StatisticalBranch(nn.Module):
    """
    PhishFusion - Dal 3: İstatistiksel Analiz Kanalı
    Bu modül, URL'den çıkarılan 18 teknik özniteliği işleyerek 
    32 boyutlu bir temsil vektörü üretir.
    """
    def __init__(self, input_dim=18, hidden_dim=64, output_dim=32, dropout_prob=0.2):
        super(StatisticalBranch, self).__init__()
        
        # Blok 1: Giriş boyutunu (18) gizli boyuta (64) genişletir
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        
        # Blok 2: Özellikleri derinleştirir ve rafine eder (64 -> 64)
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        
        # Blok 3: Bilgiyi Meta-Learner için hedef boyuta (32) indirger
        self.block3 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )

    def forward(self, x):
        """
        x: (batch_size, 18) boyutunda teknik öznitelik vektörü
        return: (batch_size, 32) boyutunda temsil vektörü
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

# Test amaçlı kullanım örneği:
if __name__ == "__main__":
    # Örnek bir veri girişi (Batch size: 8, Özellik sayısı: 18)
    sample_input = torch.randn(8, 18)
    
    # Modeli oluştur
    model = StatisticalBranch()
    
    # İleri besleme
    output = model(sample_input)
    
    print(f"Giriş boyutu: {sample_input.shape}") # [8, 18]
    print(f"Çıktı (Temsil Vektörü) boyutu: {output.shape}") # [8, 32]