from pathlib import Path
import argparse

import sentencepiece as spm


def train_sentencepiece_unigram(input_corpus_path, model_prefix="models/url_unigram", vocab_size=1000):
    """
    PhishFusion Makalesi Uyumlu:
    URL korpusu üzerinde SentencePiece Unigram modeli eğitir.
    """
    # Makalede belirtilen kritik ayırıcılar
    separator_symbols = ["/", ".", "@", ":", "?", "&", "=", "-", "_", "~", "#"]

    input_path = Path(input_corpus_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Korpus dosyasi bulunamadi: {input_corpus_path}")

    prefix_path = Path(model_prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(prefix_path),
        model_type="unigram",
        vocab_size=int(vocab_size),
        character_coverage=1.0,
        split_by_whitespace=False,
        normalization_rule_name="identity",
        # Karakter bazlı yedekleme: Bilinmeyen karakterleri byte seviyesinde kodlar
        byte_fallback=True, 
        # Delimiterları atomik token yapar
        user_defined_symbols=separator_symbols, 
        # ID Yapılandırması (Transformer uyumu için)
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        # Vocab limitini esnetmek bazen unigram için daha iyidir
        hard_vocab_limit=False,
    )

    return f"{prefix_path}.model", f"{prefix_path}.vocab"


def main():
    parser = argparse.ArgumentParser(description="URL korpusu ile SentencePiece Unigram modeli egit.")
    parser.add_argument("--input", required=True, help="Her satiri bir URL olacak korpus dosyasi")
    parser.add_argument("--model-prefix", default="models/sentencepiecemodel/url_unigram", help="Cikacak model dosya prefiksi")
    parser.add_argument("--vocab-size", type=int, default=512, help="SentencePiece vocab size")
    args = parser.parse_args()

    model_path, vocab_path = train_sentencepiece_unigram(
        input_corpus_path=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
    )

    print(f"Model olusturuldu: {model_path}")
    print(f"Vocab olusturuldu: {vocab_path}")


if __name__ == "__main__":
    main()
