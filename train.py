import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from models.meta_learner import PhishFusion
from utils.metrics import PhishMetrics # Yeni oluşturduğumuz dosya

class QuishingDataset(Dataset):
    def __init__(self, csv_path):
        # CSV dosyasını oku
        df = pd.read_csv(csv_path)
        
        # Sütunları ayır (Sıralama: URL, Lexical[128], Stat[18], Anatomical[24], Label)
        # Sütun indeksleri CSV yapına göre 1-2 kayabilir, kontrol et!
        self.url_data = torch.LongTensor(df.iloc[:, 1:129].values.copy())
        self.stat_data = torch.FloatTensor(df.iloc[:, 129:147].values.copy())
        self.qr_data = torch.FloatTensor(df.iloc[:, 147:171].values.copy())
        self.labels = torch.FloatTensor(df.iloc[:, -1].values.copy()).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.qr_data[idx], self.url_data[idx], self.stat_data[idx], self.labels[idx]

def train_model():
    # --- Hiperparametreler ---
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Veri Hazırlığı ---
    train_set = QuishingDataset("data/processed/train_features.csv")
    test_set = QuishingDataset("data/processed/test_features.csv")
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model, Loss, Optimizer ---
    model = PhishFusion().to(DEVICE)
    criterion = nn.BCELoss() # MetaLearner'da Sigmoid olduğu için BCELoss kalabilir
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Metrik takipçisi
    metrics = PhishMetrics()

    print(f"Eğitim Başlıyor... Cihaz: {DEVICE}")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for qr, url, stat, labels in train_loader:
            qr, url, stat, labels = qr.to(DEVICE), url.to(DEVICE), stat.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(qr, url, stat)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # --- Validation (Her epoch sonunda başarıyı ölç) ---
        model.eval()
        metrics.reset()
        with torch.no_grad():
            for qr, url, stat, labels in test_loader:
                qr, url, stat, labels = qr.to(DEVICE), url.to(DEVICE), stat.to(DEVICE), labels.to(DEVICE)
                outputs = model(qr, url, stat)
                metrics.update(outputs, labels)
        
        results = metrics.calculate()
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(train_loader):.4f} - Val Acc: {results['accuracy']:.2f}% - F1: {results['f1_score']:.2f}")

    # Ağırlıkları Kaydet
    torch.save(model.state_dict(), "weights/phishfusion_final.pth")
    print("Eğitim Tamamlandı. Model kaydedildi.")

if __name__ == "__main__":
    train_model()