from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable
from models.meta_learner import PhishFusion
from utils.metrics import PhishMetrics

# decoded_url + 128 lexical + 22 stat + 24 anatomical + label
_ANATOMICAL_DIM = 24
_STAT_DIM = 22
_EXPECTED_COLS = 1 + 128 + _STAT_DIM + _ANATOMICAL_DIM + 1

# --- Early Stopping Sınıfı ---
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, path='weights/phishfusion_best.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_metrics = None

    def __call__(self, val_loss, model, metrics=None):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_metrics = metrics
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping sayacı: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_metrics = metrics
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        # Ağırlıkların kaydedileceği klasörün var olduğundan emin ol
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        print(f"Validation loss düştü. En iyi model kaydedildi: {self.path}")

class QuishingDataset(Dataset):
    def __init__(self, csv_path, stat_scaler: MinMaxScaler | None = None, qr_scaler: MinMaxScaler | None = None):
        df = pd.read_csv(csv_path)
        if df.shape[1] != _EXPECTED_COLS:
            raise ValueError(
                f"Beklenen sutun sayisi {_EXPECTED_COLS} (anatomik {_ANATOMICAL_DIM}), "
                f"CSV: {df.shape[1]}. Beklenen {_EXPECTED_COLS} sutun; url-processor ile uyumlu CSV kullanin."
            )

        self.url_data = torch.LongTensor(df.iloc[:, 1:129].values.copy())
        stat_start = 129
        stat_end = stat_start + _STAT_DIM
        stat_np = df.iloc[:, stat_start:stat_end].values.astype(np.float64, copy=True)
        if stat_scaler is not None:
            stat_np = stat_scaler.transform(stat_np)
        self.stat_data = torch.FloatTensor(stat_np.astype(np.float32))
        qr_start = stat_end
        qr_end = qr_start + _ANATOMICAL_DIM
        qr_np = df.iloc[:, qr_start:qr_end].values.astype(np.float64, copy=True)
        if qr_scaler is not None:
            qr_np = qr_scaler.transform(qr_np)
        self.qr_data = torch.FloatTensor(qr_np.astype(np.float32))
        self.labels = torch.FloatTensor(df.iloc[:, -1].values.copy()).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.qr_data[idx], self.url_data[idx], self.stat_data[idx], self.labels[idx]

def save_best_metrics_plot(metrics, save_path="weights/best_model_metrics.png"):
    plt.figure(figsize=(18, 5))
    
    # 1. Sol tarafa metrikleri metin olarak yazdir
    plt.subplot(1, 3, 1)
    plt.axis('off')
    text = (f"En Iyi Model Performansi (Epoch {metrics.get('epoch', '?')})\n\n"
            f"Train Loss: {metrics['train_loss']:.4f}\n"
            f"Validation Loss: {metrics['val_loss']:.4f}\n"
            f"Accuracy: {metrics['accuracy']:.2f}%\n"
            f"Precision: {metrics['precision']:.2f}%\n"
            f"F1 Score: {metrics['f1_score']:.2f}%\n"
            f"ROC-AUC: {metrics['roc_auc']:.2f}%\n"
            f"---\n"
            f"QR (Anatomik): Acc {metrics.get('acc_a',0):.1f}% | F1 {metrics.get('f1_a',0):.1f}% | AUC {metrics.get('roc_auc_a',0):.1f}%\n"
            f"URL (Lexical): Acc {metrics.get('acc_l',0):.1f}% | F1 {metrics.get('f1_l',0):.1f}% | AUC {metrics.get('roc_auc_l',0):.1f}%\n"
            f"İstatistiksel: Acc {metrics.get('acc_s',0):.1f}% | F1 {metrics.get('f1_s',0):.1f}% | AUC {metrics.get('roc_auc_s',0):.1f}%")
    plt.text(0.1, 0.5, text, fontsize=12, va='center', bbox=dict(boxstyle="round,pad=1", facecolor="aliceblue", edgecolor="steelblue"))
    
    # 2. Orta tarafa Karmasiklik Matrisini (Confusion Matrix) cizdir
    plt.subplot(1, 3, 2)
    sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Guvenli', 'Zararli'], 
                yticklabels=['Guvenli', 'Zararli'])
    plt.ylabel('Gercek Sinif', fontsize=12)
    plt.xlabel('Tahmin Edilen Sinif', fontsize=12)
    plt.title('Karmasiklik Matrisi (Confusion Matrix)', fontsize=14)
    
    # 3. Sag tarafa ROC Egrisi (ROC Curve) cizdir
    plt.subplot(1, 3, 3)
    fpr, tpr, _ = roc_curve(metrics['y_true'], metrics['y_probs'])
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC Egrisi (AUC = {metrics['roc_auc']/100:.3f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Egrisi', fontsize=14)
    plt.legend(loc="lower right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"En iyi model metrikleri gorsellestirilip kaydedildi: {save_path}")

def train_model():
    # --- Hiperparametreler ---
    EPOCHS = 5 # Early stopping eklendiği için epoch'u biraz daha yüksek tutabilirsin
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    PATIENCE = 3 # Kaç epoch boyunca iyileşme olmazsa durdurulacak
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Veri Hazırlığı: İstatistiksel ve Anatomik özellikler Min-Max [0,1], egitim istatistikleriyle ---
    train_csv = "data/processed/train2_features.csv"
    train_df_head = pd.read_csv(train_csv)
    
    stat_scaler = MinMaxScaler()
    stat_scaler.fit(train_df_head.iloc[:, 129:151].values.astype(np.float64, copy=True))

    qr_scaler = MinMaxScaler()
    qr_scaler.fit(train_df_head.iloc[:, 151:175].values.astype(np.float64, copy=True))

    train_set = QuishingDataset(train_csv, stat_scaler=stat_scaler, qr_scaler=qr_scaler)
    test_set = QuishingDataset("data/processed/test2_features.csv", stat_scaler=stat_scaler, qr_scaler=qr_scaler)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model, Loss, Optimizer ---
    model = PhishFusion().to(DEVICE)
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    # Metrik takipçisi ve Early Stopping Başlatımı
    metrics = PhishMetrics()
    early_stopping = EarlyStopping(patience=PATIENCE, path="weights/phishfusion_best.pth")

    n_train = len(train_set)
    n_batches = len(train_loader)
    print(f"Eğitim Başlıyor... Cihaz: {DEVICE}")
    print(f"Train: {n_train} örnek, batch={BATCH_SIZE}, epoch başına {n_batches} adım, {EPOCHS} epoch maksimum.")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for qr, url, stat, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
            leave=False,
        ):
            qr, url, stat, labels = qr.to(DEVICE), url.to(DEVICE), stat.to(DEVICE), labels.to(DEVICE)

            # --- Ana model adimi ---
            optimizer.zero_grad()
            outputs = model(qr, url, stat)
            loss_main = criterion(outputs, labels)
            loss_main.backward()
            optimizer.step()
            running_loss += loss_main.item()

        # --- Validation (Her epoch sonunda başarıyı ölç) ---
        model.eval()
        metrics.reset()
        val_loss = 0.0 # Val loss'u takip etmek için
        
        with torch.no_grad():
            for qr, url, stat, labels in tqdm(
                test_loader,
                desc="Val",
                leave=False,
            ):
                qr, url, stat, labels = qr.to(DEVICE), url.to(DEVICE), stat.to(DEVICE), labels.to(DEVICE)
                outputs = model(qr, url, stat)
                batch_val_loss = criterion(outputs, labels)
                val_loss += batch_val_loss.item()
                metrics.update(outputs, labels)
        
        # Ortalama loss değerleri
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        
        results = metrics.calculate()
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {results['accuracy']:.2f}% - F1: {results['f1_score']:.2f}% - ROC-AUC: {results['roc_auc']:.2f}%")
        #debug
        y_true = np.array(metrics.all_labels).reshape(-1).astype(int)
        y_pred = np.array(metrics.all_preds).reshape(-1).astype(int)
        pred_ones = int((y_pred == 1).sum())
        true_ones = int((y_true == 1).sum())
        total_eval = int(y_true.size)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        print(
            f"Val dağılımı -> pred_1: {pred_ones}/{total_eval}, true_1: {true_ones}/{total_eval} | "
            f"TN:{tn} FP:{fp} FN:{fn} TP:{tp}"
        )

        # Scheduler Adimi (Validation loss degerine gore lr dusurur)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < current_lr:
            print(f"Learning rate {current_lr} degerinden {new_lr} degerine dusuruldu.")

        # Early Stopping Kontrolü
        metrics_to_save = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'roc_auc': results['roc_auc'],
            'cm': cm,
            'y_true': np.array(metrics.all_labels),
            'y_probs': np.array(metrics.all_probs)
        }
        early_stopping(avg_val_loss, model, metrics_to_save)
        
        if early_stopping.early_stop:
            print("Early stopping devreye girdi! Model daha fazla öğrenemiyor, eğitim durduruluyor.")
            break

    print("Eğitim Süreci Tamamlandı.")
    
    if early_stopping.best_metrics is not None:
        save_best_metrics_plot(early_stopping.best_metrics, save_path="weights/best_model_metrics.png")

if __name__ == "__main__":
    train_model()