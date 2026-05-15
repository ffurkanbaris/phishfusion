"""
eval_branches.py
----------------
Her dalın (QR/URL/İstatistiksel) bireysel phishing tespit gücünü ölçer.
Her dal için ana modelden BAĞIMSIZ bir sınıflandırıcı (Branch + Linear head)
sıfırdan eğitilir.

Kullanım:
    python eval_branches.py
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_):
        return iterable

from models.anatomical import AnatomicalBranch
from models.lexical import LexicalBranch
from models.statistical import StatisticalBranch

# ─────────────────────────────────────────────
# Sabitler (train.py ile uyumlu)
# ─────────────────────────────────────────────
_ANATOMICAL_DIM = 24
_STAT_DIM       = 22
_LEX_DIM        = 128
_EXPECTED_COLS  = 1 + _LEX_DIM + _STAT_DIM + _ANATOMICAL_DIM + 1  # 176

TRAIN_CSV  = "data/processed/train2_features.csv"
TEST_CSV   = "data/processed/test2_features.csv"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS     = 5
LR         = 0.001


# ─────────────────────────────────────────────
# Dataset (her branch modu için ayrı ayrı kullanılır)
# ─────────────────────────────────────────────
class BranchDataset(Dataset):
    def __init__(self, csv_path, mode: str, stat_scaler: MinMaxScaler | None = None):
        """
        mode: 'lexical' | 'statistical' | 'anatomical'
        """
        df = pd.read_csv(csv_path)
        assert df.shape[1] == _EXPECTED_COLS, \
            f"Beklenmeyen sütun sayısı: {df.shape[1]} (beklenen: {_EXPECTED_COLS})"

        self.labels = torch.FloatTensor(df.iloc[:, -1].values.copy()).unsqueeze(1)

        if mode == "lexical":
            self.features = torch.LongTensor(df.iloc[:, 1:129].values.copy())

        elif mode == "statistical":
            stat_np = df.iloc[:, 129:151].values.astype(np.float64, copy=True)
            if stat_scaler is not None:
                stat_np = stat_scaler.transform(stat_np)
            self.features = torch.FloatTensor(stat_np.astype(np.float32))

        elif mode == "anatomical":
            self.features = torch.FloatTensor(
                df.iloc[:, 151:175].values.astype(np.float32, copy=True)
            )
        else:
            raise ValueError(f"Bilinmeyen mode: {mode}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ─────────────────────────────────────────────
# Tek-Dal Sınıflandırıcı Sarmalayıcısı
# ─────────────────────────────────────────────
class SingleBranchClassifier(nn.Module):
    def __init__(self, branch: nn.Module, branch_out_dim: int = 32):
        super().__init__()
        self.branch = branch
        self.head = nn.Sequential(
            nn.Linear(branch_out_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.head(self.branch(x))


# ─────────────────────────────────────────────
# Eğitim & Değerlendirme
# ─────────────────────────────────────────────
def train_and_eval(name: str, branch: nn.Module, train_loader, test_loader):
    print(f"\n{'='*55}")
    print(f"  {name} Dalı — Bağımsız Eğitim ({EPOCHS} Epoch)")
    print(f"{'='*55}")

    model = SingleBranchClassifier(branch).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    # ── Eğitim ──
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for features, labels in tqdm(train_loader, desc=f"  Epoch {epoch+1}/{EPOCHS}", leave=False):
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg = running_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{EPOCHS} — Train Loss: {avg:.4f}")

    # ── Değerlendirme ──
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for features, labels in test_loader:
            probs = model(features.to(DEVICE)).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.numpy().flatten().astype(int))

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_probs = np.array(all_probs)

    acc  = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, zero_division=0) * 100
    rec  = recall_score(y_true, y_pred, zero_division=0) * 100
    f1   = f1_score(y_true, y_pred, zero_division=0) * 100
    try:
        auc = roc_auc_score(y_true, y_probs) * 100
    except ValueError:
        auc = 0.0

    print(f"\n  ── {name} Sonuçları ──")
    print(f"  Accuracy  : {acc:.2f}%")
    print(f"  Precision : {prec:.2f}%")
    print(f"  Recall    : {rec:.2f}%")
    print(f"  F1 Score  : {f1:.2f}%")
    print(f"  ROC-AUC   : {auc:.2f}%")

    return {"name": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "roc_auc": auc}


# ─────────────────────────────────────────────
# Ana Akış
# ─────────────────────────────────────────────
def main():
    print(f"Cihaz: {DEVICE}")
    print(f"Eğitim CSV : {TRAIN_CSV}")
    print(f"Test CSV   : {TEST_CSV}")

    # MinMaxScaler sadece train setine fit edilir (veri sızıntısı yok)
    train_df = pd.read_csv(TRAIN_CSV)
    stat_scaler = MinMaxScaler()
    stat_scaler.fit(train_df.iloc[:, 129:151].values.astype(np.float64))

    results = []

    # ── 1. İstatistiksel Dal ──
    stat_train = BranchDataset(TRAIN_CSV, "statistical", stat_scaler)
    stat_test  = BranchDataset(TEST_CSV,  "statistical", stat_scaler)
    results.append(train_and_eval(
        "İstatistiksel",
        StatisticalBranch(),
        DataLoader(stat_train, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(stat_test,  batch_size=BATCH_SIZE, shuffle=False),
    ))

    # ── 2. Sözcüksel (URL) Dal ──
    lex_train = BranchDataset(TRAIN_CSV, "lexical")
    lex_test  = BranchDataset(TEST_CSV,  "lexical")
    results.append(train_and_eval(
        "Sözcüksel (URL)",
        LexicalBranch(),           # defaults: d_model=64, nhead=4, num_layers=2
        DataLoader(lex_train, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(lex_test,  batch_size=BATCH_SIZE, shuffle=False),
    ))

    # ── 3. Anatomik (QR) Dal ──
    anat_train = BranchDataset(TRAIN_CSV, "anatomical")
    anat_test  = BranchDataset(TEST_CSV,  "anatomical")
    results.append(train_and_eval(
        "Anatomik (QR)",
        AnatomicalBranch(),
        DataLoader(anat_train, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(anat_test,  batch_size=BATCH_SIZE, shuffle=False),
    ))

    # ── Özet Tablo ──
    print(f"\n{'='*65}")
    print("  ÖZET — Bireysel Dal Başarıları")
    print(f"{'='*65}")
    print(f"  {'Dal':<22} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8}")
    print(f"  {'-'*62}")
    for r in results:
        print(
            f"  {r['name']:<22} "
            f"{r['accuracy']:>7.2f}% "
            f"{r['precision']:>7.2f}% "
            f"{r['recall']:>7.2f}% "
            f"{r['f1']:>7.2f}% "
            f"{r['roc_auc']:>7.2f}%"
        )
    print()


if __name__ == "__main__":
    main()
