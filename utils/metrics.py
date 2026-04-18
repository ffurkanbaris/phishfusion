import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class PhishMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        """Her epoch başında istatistikleri sıfırlar."""
        self.all_labels = []
        self.all_preds = []

    def update(self, outputs, labels):
        """
        Model çıktılarını ve gerçek etiketleri toplar.
        outputs: Modelden gelen risk skorları (0-1 arası)
        labels: Gerçek sınıflar (0 veya 1)
        """
        # Skorları 0 veya 1'e çevir (Threshold: 0.5)
        preds = (torch.sigmoid(outputs) if outputs.max() > 1 else outputs) > 0.5
        
        self.all_labels.extend(labels.cpu().numpy())
        self.all_preds.extend(preds.cpu().numpy().astype(int))

    def calculate(self):
        """Toplanan verilerden temel metrikleri hesaplar."""
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_preds)

        acc = (y_true == y_pred).mean()
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        return {
            "accuracy": acc * 100,
            "precision": precision * 100,
            "recall": recall * 100,
            "f1_score": f1 * 100
        }

    def plot_confusion_matrix(self, epoch, save_path="logs/"):
        """Confusion Matrix görselleştirir ve kaydeder."""
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_preds)
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Güvenli', 'Zararlı'], 
                    yticklabels=['Güvenli', 'Zararlı'])
        plt.ylabel('Gerçek Sınıf')
        plt.xlabel('Tahmin Edilen Sınıf')
        plt.title(f'Epoch {epoch} - Karmaşıklık Matrisi')
        
        import os
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        plt.savefig(f"{save_path}/cm_epoch_{epoch}.png")
        plt.close()