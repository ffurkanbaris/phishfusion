"""Makale hizali URL on-isleme: kucuk harf + istege bagli http/https kirpma."""


def normalize_url_for_lexical(url: str, strip_scheme: bool = True) -> str:
    s = (url or "").strip().lower()
    if strip_scheme:
        if s.startswith("https://"):
            s = s[8:]
        elif s.startswith("http://"):
            s = s[7:]
    return s
