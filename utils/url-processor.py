import re
import csv
import ipaddress
import sys
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from utils.url_normalize import normalize_url_for_lexical
import cv2
from urllib.parse import urlparse
from scipy import stats
try:
    import sentencepiece as spm
except ImportError:
    spm = None


class PhishFusionProcessor:
    def __init__(self):
        # Parametre hatasını önlemek için boş bıraktık veya gerekli ayarları ekledik
        pass

    def extract_qr_anatomical_features(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros(24, dtype=np.float32)

        # Makaledeki akış: QR'ı hizala/crop et -> median+gaussian+CLAHE ile temizle
        # -> module-size tahmini ile binary module grid'e dönüştür.
        qr_only = self._extract_and_align_qr(img)
        cleaned_qr = self._clean_qr_image(qr_only)
        binary_matrix = self._convert_qr_to_binary_modules(cleaned_qr)
        h, w = binary_matrix.shape

        # --- A. Protocol-level (5): v/40, ECC, mask (0-7 skaler), align, remainder ---
        version_norm, ecc_level, mask_pattern, num_align_patterns, remainder_bits = self._extract_protocol_features(
            binary_matrix
        )

        # --- B. Statistical Features (19 Adet) ---
        num_white = np.sum(binary_matrix == 0)
        num_black = np.sum(binary_matrix == 1)
        bw_ratio = num_black / (num_white + 1e-5)
        qr_density = num_black / (h * w)
        qr_mean_density = np.mean(binary_matrix)
        
        row_sums = np.mean(binary_matrix, axis=1)
        col_sums = np.mean(binary_matrix, axis=0)
        qr_std_row = np.std(row_sums)
        qr_std_col = np.std(col_sums)
        
        row_transitions = np.sum(np.diff(binary_matrix, axis=1) != 0)
        col_transitions = np.sum(np.diff(binary_matrix, axis=0) != 0)
        
        flat_matrix = binary_matrix.flatten()
        counts = np.bincount(flat_matrix)
        probs = counts / len(flat_matrix)
        qr_entropy = stats.entropy(probs)
        
        qr_vert_asym = np.abs(binary_matrix[:h//2, :].mean() - binary_matrix[h//2:, :].mean())
        qr_horz_asym = np.abs(binary_matrix[:, :w//2].mean() - binary_matrix[:, w//2:].mean())
        
        tl = binary_matrix[:h//3, :w//3].mean()
        tr = binary_matrix[:h//3, 2*w//3:].mean()
        bl = binary_matrix[2*h//3:, :w//3].mean()
        br = binary_matrix[2*h//3:, 2*w//3:].mean()
        center = binary_matrix[h//3:2*h//3, w//3:2*w//3].mean()
        
        row_peaks = len(np.where(row_sums > row_sums.mean())[0])
        col_peaks = len(np.where(col_sums > col_sums.mean())[0])

        features = [
            version_norm,
            ecc_level,
            float(mask_pattern),
            num_align_patterns,
            remainder_bits,
            num_white, num_black, bw_ratio, qr_density, qr_mean_density,
            qr_std_row, qr_std_col, row_transitions, col_transitions, qr_entropy,
            qr_vert_asym, qr_horz_asym, tl, tr, bl, br, center,
            row_peaks, col_peaks
        ]
        return np.array(features, dtype=np.float32)

    def _extract_and_align_qr(self, gray_img):
        """
        OpenCV QRCodeDetector ile QR alanını çıkarıp hizalı QR üretir.
        detectAndDecode'dan dönen straight_qrcode önceliklidir.
        """
        detector = cv2.QRCodeDetector()
        _, points, straight_qrcode = detector.detectAndDecode(gray_img)

        if straight_qrcode is not None and straight_qrcode.size > 0:
            if straight_qrcode.dtype != np.uint8:
                straight_qrcode = np.clip(straight_qrcode * 255, 0, 255).astype(np.uint8)
            return straight_qrcode

        if points is not None and len(points) > 0:
            pts = points[0].astype(np.float32)
            side = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[1] - pts[2]),
                           np.linalg.norm(pts[2] - pts[3]), np.linalg.norm(pts[3] - pts[0])))
            side = max(side, 21)
            dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(pts, dst)
            return cv2.warpPerspective(gray_img, matrix, (side, side))

        return gray_img

    def _clean_qr_image(self, qr_gray):
        """
        Makalede tariflenen ön-işleme: median blur + gaussian blur + CLAHE + threshold.
        Threshold çıktısı module-size tahmini için, normalize edilmiş gri görüntü ise
        module ortalama yoğunluk hesabı için kullanılır.
        """
        median = cv2.medianBlur(qr_gray, 3)
        gaussian = cv2.GaussianBlur(median, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gaussian)

    def _estimate_module_size(self, cleaned_qr):
        _, qr_pixel = cv2.threshold(cleaned_qr, 189, 250, cv2.THRESH_BINARY)

        black_count = 0
        for row in qr_pixel:
            for col_idx in range(len(row)):
                bit = 1 if row[col_idx] == 0 else 0
                if bit == 1:
                    run = 0
                    for k in range(col_idx, len(row)):
                        bit_k = 1 if row[k] == 0 else 0
                        if bit_k == 1:
                            run += 1
                        else:
                            break
                    black_count = run
                    break
            if black_count > 0:
                break

        if black_count <= 0:
            return 1

        return max(1, int(np.ceil(black_count / 7.0)))

    def _convert_qr_to_binary_modules(self, cleaned_qr):
        """
        Makaledeki Algorithm-1 yaklaşımı:
        - module size tahmini
        - total modules = round(width / module_size)
        - her module'de mean intensity < 189 ise 1, aksi 0
        """
        qr_width = cleaned_qr.shape[1]
        module_size = self._estimate_module_size(cleaned_qr)
        total_modules = max(21, int(round(qr_width / float(module_size))))

        binary = np.zeros((total_modules, total_modules), dtype=np.uint8)
        for i in range(total_modules):
            for j in range(total_modules):
                y0 = i * module_size
                y1 = min((i + 1) * module_size, cleaned_qr.shape[0])
                x0 = j * module_size
                x1 = min((j + 1) * module_size, cleaned_qr.shape[1])
                block = cleaned_qr[y0:y1, x0:x1]
                if block.size == 0:
                    continue
                binary[i, j] = 1 if float(np.mean(block)) < 189 else 0

        return binary

    def _extract_protocol_features(self, binary_matrix):
        """
        5 skaler: version_norm (v/40), ECC 1-4, mask 0-7, alignment count, remainder bits.
        """
        n = int(min(binary_matrix.shape[0], binary_matrix.shape[1]))
        version = int(round((n - 21) / 4.0) + 1)
        version = max(1, min(40, version))
        version_norm = float(version) / 40.0
        ecc_level, mask_pattern = self._decode_format_info(binary_matrix)

        num_align_patterns = self._alignment_pattern_count(version)
        remainder_bits = self._remainder_bits(version)

        return version_norm, float(ecc_level), float(mask_pattern), float(num_align_patterns), float(remainder_bits)

    def _decode_format_info(self, binary_matrix):
        """Format bits'ten ECC seviyesi (L/M/Q/H -> 1/2/3/4) ve mask pattern (0-7) çıkarır."""
        mat = binary_matrix.astype(np.uint8)  # dark=1, light=0
        h, w = mat.shape[:2]
        if h < 21 or w < 21:
            return 0, 0

        # 15 format bitinin ilk kopyası (row/col indexleri, QR spec yerleşimi)
        coords = [
            (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5),
            (8, 7), (8, 8), (7, 8),
            (5, 8), (4, 8), (3, 8), (2, 8), (1, 8), (0, 8),
        ]

        # straight_qrcode küçükse index taşmasını önle
        for r, c in coords:
            if r >= h or c >= w:
                return 0, 0

        raw = 0
        for r, c in coords:
            raw = (raw << 1) | int(mat[r, c])

        # masked format info -> (ecc, mask)
        format_map = {
            0x5412: ("L", 0), 0x5125: ("L", 1), 0x5E7C: ("L", 2), 0x5B4B: ("L", 3),
            0x45F9: ("L", 4), 0x40CE: ("L", 5), 0x4F97: ("L", 6), 0x4AA0: ("L", 7),
            0x77C4: ("M", 0), 0x72F3: ("M", 1), 0x7DAA: ("M", 2), 0x789D: ("M", 3),
            0x662F: ("M", 4), 0x6318: ("M", 5), 0x6C41: ("M", 6), 0x6976: ("M", 7),
            0x1689: ("Q", 0), 0x13BE: ("Q", 1), 0x1CE7: ("Q", 2), 0x19D0: ("Q", 3),
            0x0762: ("Q", 4), 0x0255: ("Q", 5), 0x0D0C: ("Q", 6), 0x083B: ("Q", 7),
            0x355F: ("H", 0), 0x3068: ("H", 1), 0x3F31: ("H", 2), 0x3A06: ("H", 3),
            0x24B4: ("H", 4), 0x2183: ("H", 5), 0x2EDA: ("H", 6), 0x2BED: ("H", 7),
        }

        if raw in format_map:
            ecc_char, mask = format_map[raw]
        else:
            # Gürültüye dayanıklılık: en yakın format kelimesi (minimum Hamming distance)
            best_key = min(format_map.keys(), key=lambda k: (k ^ raw).bit_count())
            ecc_char, mask = format_map[best_key]

        ecc_to_num = {"L": 1, "M": 2, "Q": 3, "H": 4}
        return ecc_to_num[ecc_char], int(mask)

    def _alignment_pattern_count(self, version):
        if version <= 1:
            return 0
        centers = (version // 7) + 2
        return (centers * centers) - 3

    def _remainder_bits(self, version):
        remainder_table = {
            1: 0, 2: 7, 3: 7, 4: 7, 5: 7, 6: 7,
            7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0,
            14: 3, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3, 21: 4, 22: 4, 23: 4, 24: 4, 25: 4, 26: 4, 27: 4,
            28: 3, 29: 3, 30: 3, 31: 3, 32: 3, 33: 3, 34: 3,
            35: 0, 36: 0, 37: 0, 38: 0, 39: 0, 40: 0,
        }
        return remainder_table.get(int(version), 0)

class URLProcessor:
    def __init__(self, max_len=128, sp_model_path="models/sentencepiecemodel/url_unigram.model"):
        self.max_len = max_len
        self.char_to_idx = {char: i + 1 for i, char in enumerate("abcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;=")}
        self.vocab_size = len(self.char_to_idx) + 1
        # HATA DÜZELTİLDİ: Gereksiz parametre kaldırıldı
        self.qr_processor = PhishFusionProcessor()
        self.sp_model_path = str(sp_model_path)
        self.sp_processor = None
        self.known_legitimate_domains = [
            "google.com", "microsoft.com", "apple.com", "amazon.com", "facebook.com",
            "instagram.com", "linkedin.com", "paypal.com", "github.com", "netflix.com",
            "x.com", "youtube.com", "wikipedia.org", "yahoo.com", "dropbox.com"
        ]
        self.sensitive_path_keywords = [
            "login", "signin", "verify", "account", "password", "reset",
            "auth", "secure", "wallet", "billing", "payment", "confirm", "bank"
        ]

        if spm is not None and Path(self.sp_model_path).exists():
            self.sp_processor = spm.SentencePieceProcessor()
            self.sp_processor.load(self.sp_model_path)
            self.vocab_size = self.sp_processor.get_piece_size()

    def extract_statistical_features(self, url):
        """
        Tabloya uyumlu 18 URL tabanli ozellik:
        [Scheme, Has params, Has query, Has username, Has password, Has path,
         Has port, Has other chars, Is TLD in params, Is TLD in query, Is IP hostname,
         Has encoded chars, Has keyword login, Has keyword bank, Has repeated chars,
         Has sensitive path, Has many subdomains, URL similarity index]
        """
        parsed = urlparse(url)
        hostname = parsed.hostname if parsed.hostname else ""
        path = parsed.path if parsed.path else ""
        url_lower = url.lower()

        # Hostname'den TLD cikarma (son etiket)
        host_labels = [p for p in hostname.lower().split(".") if p]
        tld = host_labels[-1] if host_labels else ""

        # Has many subdomains: www hariç en az 2 subdomain
        subdomain_count = max(0, len(host_labels) - 2)
        if host_labels and host_labels[0] == "www":
            subdomain_count = max(0, subdomain_count - 1)

        # IP hostname kontrolu
        is_ip_hostname = 0
        try:
            if hostname:
                ipaddress.ip_address(hostname)
                is_ip_hostname = 1
        except ValueError:
            is_ip_hostname = 0

        # URL similarity index (0-1): hostname ile bilinen domainler arasinda max benzerlik
        similarity_score = 0.0
        host_for_similarity = hostname.lower().strip()
        if host_for_similarity:
            for domain in self.known_legitimate_domains:
                similarity_score = max(
                    similarity_score,
                    SequenceMatcher(None, host_for_similarity, domain).ratio(),
                )

        features = [
            1 if parsed.scheme in {"http", "https"} else 0,                         # 1. Scheme
            1 if bool(parsed.params) else 0,                                          # 2. Has params
            1 if bool(parsed.query) else 0,                                           # 3. Has query
            1 if bool(parsed.username) else 0,                                        # 4. Has username
            1 if bool(parsed.password) else 0,                                        # 5. Has password
            1 if bool(path and path != "/") else 0,                                   # 6. Has path
            1 if parsed.port else 0,                                                  # 7. Has port
            1 if re.search(r"[^a-zA-Z0-9:/?#\[\]@!$&'()*+,;=%._~-]", url) else 0,    # 8. Has other chars
            1 if (tld and tld in parsed.params.lower()) else 0,                       # 9. Is TLD in params
            1 if (tld and tld in parsed.query.lower()) else 0,                        # 10. Is TLD in query
            is_ip_hostname,                                                           # 11. Is IP hostname
            1 if re.search(r"%[0-9a-fA-F]{2}", url) else 0,                          # 12. Has encoded chars
            1 if "login" in url_lower else 0,                                         # 13. Has keyword login
            1 if "bank" in url_lower else 0,                                          # 14. Has keyword bank
            1 if re.search(r"(.)\1{2,}", url_lower) else 0,                           # 15. Has repeated chars
            1 if any(k in path.lower() for k in self.sensitive_path_keywords) else 0, # 16. Has sensitive path
            1 if subdomain_count > 1 else 0,                                          # 17. Has many subdomains
            float(similarity_score),                                                  # 18. URL similarity index
        ]
        return np.array(features, dtype=np.float32)

    def get_lexical_tokens(self, url):
        url_lex = normalize_url_for_lexical(url, strip_scheme=True)
        if self.sp_processor:
            tokens = self.sp_processor.encode(url_lex, out_type=int)
        else:
            tokens = [self.char_to_idx.get(c, 0) for c in url_lex]
            
        # PADDING (0 ID kullanımı SentencePiece'teki pad_id ile uyumlu olmalı)
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [0] * (self.max_len - len(tokens))
        return np.array(tokens, dtype=np.int64)

    def _tokenize_url(self, url):
        """
        SentencePiece Unigram modeli varsa onu kullanır.
        Model yoksa karakter-bazlı fallback tokenizasyona döner.
        """
        url_lex = normalize_url_for_lexical(url, strip_scheme=True)
        if self.sp_processor is not None:
            return self.sp_processor.encode(url_lex, out_type=int)
        return [self.char_to_idx.get(c, 0) for c in url_lex]

    def process(self, url, qr_image_path=None):
        stat_features = self.extract_statistical_features(url)
        lexical_tokens = self.get_lexical_tokens(url)
        qr_features = np.zeros(24, dtype=np.float32)

        if qr_image_path:
            qr_features = self.qr_processor.extract_qr_anatomical_features(qr_image_path)

        return stat_features, lexical_tokens, qr_features

    def process_from_qr(self, qr_image_path):
        """
        URL'yi doğrudan QR görselinden okuyup tüm feature'ları üretir.
        """
        decoded_url = self.extract_url_from_qr(qr_image_path)
        stat_features, lexical_tokens, qr_features = self.process(decoded_url, qr_image_path=qr_image_path)
        return decoded_url, stat_features, lexical_tokens, qr_features

    def extract_url_from_qr(self, qr_image_path):
        """
        QR görselinden URL/metin çözer.
        QR decode başarısız olursa ValueError fırlatır.
        """
        img = cv2.imread(qr_image_path)
        if img is None:
            raise ValueError(f"QR görseli okunamadi: {qr_image_path}")

        detector = cv2.QRCodeDetector()
        decoded_text, _, _ = detector.detectAndDecode(img)

        if not decoded_text:
            raise ValueError("QR icerigi decode edilemedi veya bos.")

        return decoded_text.strip()

    def _build_feature_header(self):
        lexical_headers = [f"lexical_token_{i}" for i in range(self.max_len)]
        statistical_headers = [f"stat_feature_{i}" for i in range(18)]
        anatomical_headers = [f"anatomical_feature_{i}" for i in range(24)]
        return ["decoded_url"] + lexical_headers + statistical_headers + anatomical_headers

    def _build_feature_header_with_label_last(self):
        return self._build_feature_header() + ["label"]

    def write_qr_features_to_csv(self, qr_image_path, output_csv_path="data/processed/qr_url_features.csv"):
        """
        Girdi olarak verilen QR görselinden:
        - URL decode eder
        - lexical token, statistical ve anatomical feature cikarir
        - hepsini tek satirda virgulle ayrilmis sekilde CSV'ye yazar
        """
        decoded_url, stat_features, lexical_tokens, anatomical_features = self.process_from_qr(qr_image_path)

        row = [decoded_url]
        row.extend(lexical_tokens.astype(np.int64).tolist())
        row.extend(stat_features.astype(np.float32).tolist())
        row.extend(anatomical_features.astype(np.float32).tolist())

        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        file_exists = output_path.exists()
        with output_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(self._build_feature_header())
            writer.writerow(row)

        return str(output_path), decoded_url

    def write_labeled_features_from_train_dirs(
        self,
        train_root="data/raw/train",
        output_csv_path="data/processed/train_features.csv",
    ):
        """
        data/raw/train altindaki sinif klasorlerinden ozellik cikarir.
        Beklenen klasorler: normal (label=0), phishing (label=1)
        CSV satir formati: decoded_url + lexical + statistical + anatomical + label
        """
        train_root_path = Path(train_root)
        if not train_root_path.is_dir():
            raise FileNotFoundError(f"Train klasoru bulunamadi: {train_root}")

        class_map = {
            "normal": 0,
            "phishing": 1,
        }
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_written = 0
        skipped = 0
        per_class_counts = {"normal": 0, "phishing": 0}

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self._build_feature_header_with_label_last())

            for class_name, label in class_map.items():
                class_dir = train_root_path / class_name
                if not class_dir.is_dir():
                    continue

                image_paths = sorted([p for p in class_dir.rglob("*") if p.suffix.lower() in valid_exts])
                for img_path in image_paths:
                    try:
                        decoded_url, stat_features, lexical_tokens, anatomical_features = self.process_from_qr(str(img_path))
                    except Exception:
                        skipped += 1
                        continue

                    row = [decoded_url]
                    row.extend(lexical_tokens.astype(np.int64).tolist())
                    row.extend(stat_features.astype(np.float32).tolist())
                    row.extend(anatomical_features.astype(np.float32).tolist())
                    row.append(label)  # label her zaman sonda
                    writer.writerow(row)

                    total_written += 1
                    per_class_counts[class_name] += 1

        return {
            "output_csv": str(output_path),
            "total_written": total_written,
            "normal_written": per_class_counts["normal"],
            "phishing_written": per_class_counts["phishing"],
            "skipped": skipped,
        }


if __name__ == "__main__":
    # Ornek:
    # python url-processor.py path/to/qr.png
    import sys

    if len(sys.argv) < 2:
        print("Kullanim:")
        print("  python url-processor.py <qr_image_path> [output_csv_path]")
        print("  python url-processor.py --from-train-dirs [train_root] [output_csv_path]")
        sys.exit(1)

    qr_image_path_arg = sys.argv[1]
    output_csv_arg = sys.argv[2] if len(sys.argv) > 2 else "data/processed/qr_url_features.csv"

    processor = URLProcessor(max_len=128)
    if qr_image_path_arg == "--outputcsv":
        train_root_arg = sys.argv[2] if len(sys.argv) > 2 else "data/raw/test"
        output_csv_labeled_arg = sys.argv[3] if len(sys.argv) > 3 else "data/processed/test_features.csv"
        result = processor.write_labeled_features_from_train_dirs(
            train_root=train_root_arg,
            output_csv_path=output_csv_labeled_arg,
        )
        print("Labeled feature CSV olusturuldu:")
        for k, v in result.items():
            print(f"{k}: {v}")
    else:
        csv_path, decoded_url = processor.write_qr_features_to_csv(qr_image_path_arg, output_csv_arg)
        print(f"QR'den cozulenen URL: {decoded_url}")
        print(f"CSV yazildi: {csv_path}")