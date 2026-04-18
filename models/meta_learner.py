import torch
import torch.nn as nn
from .anatomical import AnatomicalBranch
from .lexical import LexicalBranch
from .statistical import StatisticalBranch

class PhishFusion(nn.Module):
    """
    PhishFusion Ana Mimari - Çok Modlu (Multimodal) Karar Merkezi
    3 Dal (32D+32D+32D) -> Late Fusion (96D) -> Meta-Learner -> Karar
    """
    def __init__(self):
        super(PhishFusion, self).__init__()
        
        # 1. Uzman Kanalların (Dalların) Tanımlanması
        self.anatomical_branch = AnatomicalBranch() # 24 -> 32
        self.lexical_branch = LexicalBranch()       # URL -> 32
        self.statistical_branch = StatisticalBranch() # 18 -> 32
        
        # 2. Meta-Öğrenici (Meta-Learner) Katmanları
        # 96 boyutlu birleşik vektörü işleyen karar mekanizması
        self.meta_learner = nn.Sequential(
            nn.Linear(96, 48),          # İlk sentez katmanı
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, 16),          # İkinci rafine katmanı
            nn.ReLU(),
            nn.Linear(16, 1),           # Nihai karar nöronu
            nn.Sigmoid()                # 0-1 arası Risk Skoru üretir
        )

    def forward(self, qr_features, url_tokens, stat_features):
        """
        qr_features: (batch, 24)
        url_tokens: (batch, seq_len)
        stat_features: (batch, 18)
        """
        # Adım 1: Her dalın kendi temsil vektörünü (32D) üretmesi
        v_anatomical = self.anatomical_branch(qr_features)
        v_lexical = self.lexical_branch(url_tokens)
        v_statistical = self.statistical_branch(stat_features)
        
        # Adım 2: Late Fusion (Geç Füzyon) - Vektörlerin birleştirilmesi
        # Boyut: (batch, 32+32+32) = (batch, 96)
        v_fusion = torch.cat((v_anatomical, v_lexical, v_statistical), dim=1)
        
        # Adım 3: Meta-Öğrenici ile karar verme
        risk_score = self.meta_learner(v_fusion)
        
        return risk_score

# Test ve Simülasyon
if __name__ == "__main__":
    # Örnek Girişler
    batch_size = 1
    dummy_qr = torch.randn(batch_size, 24)
    dummy_url = torch.randint(0, 100, (batch_size, 128))
    dummy_stat = torch.randn(batch_size, 18)
    
    # Modeli Başlat
    model = PhishFusion()
    
    # Tahmin Üret
    prediction = model(dummy_qr, dummy_url, dummy_stat)
    
    print("--- PhishFusion Analiz Sonucu ---")
    print(f"Meta-Öğrenici Çıktısı (Risk Skoru): {prediction.item():.4f}")
    
    if prediction.item() > 0.5:
        print("KARAR: [!] ZARARLI (Quishing Algılandı) - ÖNLEYİCİ MÜDAHALE AKTİF")
    else:
        print("KARAR: [✓] GÜVENLİ - YÖNLENDİRMEYE İZİN VERİLDİ")