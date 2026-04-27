from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable
from models.meta_learner import PhishFusion
from utils.metrics import PhishMetrics

# decoded_url + 128 lexical + 18 stat + 24 anatomical + label
_ANATOMICAL_DIM = 24
_EXPECTED_COLS = 1 + 128 + 18 + _ANATOMICAL_DIM + 1


class QuishingDataset(Dataset):
    def __init__(self, csv_path, stat_scaler: MinMaxScaler | None = None):
        df = pd.read_csv(csv_path)
        if df.shape[1] != _EXPECTED_COLS:
            raise ValueError(
                f"Beklenen sutun sayisi {_EXPECTED_COLS} (anatomik {_ANATOMICAL_DIM}), "
                f"CSV: {df.shape[1]}. Beklenen {_EXPECTED_COLS} sutun (172); url-processor ile uyumlu CSV kullanin."
            )

        self.url_data = torch.LongTensor(df.iloc[:, 1:129].values.copy())
        stat_np = df.iloc[:, 129:147].values.astype(np.float64, copy=True)
        if stat_scaler is not None:
            stat_np = stat_scaler.transform(stat_np)
        self.stat_data = torch.FloatTensor(stat_np.astype(np.float32))
        qr_end = 147 + _ANATOMICAL_DIM
        self.qr_data = torch.FloatTensor(df.iloc[:, 147:qr_end].values.copy())
        self.labels = torch.FloatTensor(df.iloc[:, -1].values.copy()).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.qr_data[idx], self.url_data[idx], self.stat_data[idx], self.labels[idx]

def train_model():
    # --- Hiperparametreler ---
    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Veri Hazırlığı: 18 istatistiksel özellik Min-Max [0,1], egitim istatistikleriyle ---
    train_csv = "data/processed/train_features.csv"
    train_df_head = pd.read_csv(train_csv)
    stat_scaler = MinMaxScaler()
    stat_scaler.fit(train_df_head.iloc[:, 129:147].values.astype(np.float64, copy=True))

    train_set = QuishingDataset(train_csv, stat_scaler=stat_scaler)
    test_set = QuishingDataset("data/processed/test_features.csv", stat_scaler=stat_scaler)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model, Loss, Optimizer ---
    model = PhishFusion().to(DEVICE)
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Metrik takipçisi
    metrics = PhishMetrics()

    n_train = len(train_set)
    n_batches = len(train_loader)
    print(f"Eğitim Başlıyor... Cihaz: {DEVICE}")
    print(f"Train: {n_train} örnek, batch={BATCH_SIZE}, epoch başına {n_batches} adım, {EPOCHS} epoch.")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for qr, url, stat, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
            leave=False,
        ):
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
            for qr, url, stat, labels in tqdm(
                test_loader,
                desc="Val",
                leave=False,
            ):
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