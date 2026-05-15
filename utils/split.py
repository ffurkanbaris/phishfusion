import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

def split_qr_dataset(normal_dir, phishing_dir, output_dir, test_size=0.2, random_seed=42):
    """
    İki ayrı klasördeki QR resimlerini oranları koruyarak train ve test setlerine ayırır.
    """
    
    # 1. Dosya yollarını topla
    normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    phishing_files = [os.path.join(phishing_dir, f) for f in os.listdir(phishing_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    all_files = normal_files + phishing_files
    # Etiketler: Normal için 0, Phishing için 1
    labels = [0] * len(normal_files) + [1] * len(phishing_files)
    
    # 2. Train-Test ayrımını yap (Stratify ile oranları koru)
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, 
        labels, 
        test_size=test_size, 
        random_state=random_seed,
        stratify=labels
    )
    
    # 3. Klasör yapısını oluştur
    sets = ['train', 'test']
    categories = ['normal', 'phishing']
    
    for s in sets:
        for c in categories:
            Path(os.path.join(output_dir, s, c)).mkdir(parents=True, exist_ok=True)

    # 4. Dosyaları kopyalama fonksiyonu
    def copy_to_dest(file_list, label_list, set_name):
        for f, l in zip(file_list, label_list):
            category = 'normal' if l == 0 else 'phishing'
            # Kategori öneki ile benzersiz isim - Ayni isimli dosyalar birbirinin uzerine yazilmaz
            dest_filename = f"{category}_{os.path.basename(f)}"
            dest_path = os.path.join(output_dir, set_name, category, dest_filename)
            shutil.copy2(f, dest_path) # copy2 metadata'yi korur

    # İşlemi başlat
    print(f"Kopyalama işlemi başladı...")
    copy_to_dest(train_files, train_labels, 'train')
    copy_to_dest(test_files, test_labels, 'test')
    
    print(f"--- İşlem Tamamlandı ---")
    print(f"Eğitim Seti: {len(train_files)} adet ({train_labels.count(0)} Normal, {train_labels.count(1)} Phishing)")
    print(f"Test Seti:   {len(test_files)} adet ({test_labels.count(0)} Normal, {test_labels.count(1)} Phishing)")

# --- AYARLAR VE ÇALIŞTIRMA ---
if __name__ == "__main__":
    # Kendi klasör yollarını buraya yaz
    NORMAL_QR_FOLDER = "data/datasets/QR_All_benign/qrs"
    PHISHING_QR_FOLDER = "data/datasets/QR_All_Malicious/qrs"
    OUTPUT_BASE_FOLDER = "data/raw" # Train/Test buraya oluşacak

    split_qr_dataset(NORMAL_QR_FOLDER, PHISHING_QR_FOLDER, OUTPUT_BASE_FOLDER)