import re
import unicodedata

def normalization(text):
    # 1. Unicode normalize (NFKC)
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove text inside square brackets [...]
    text = re.sub(r"\[[^\]]*\]", " ", text)

    # 3. Remove text inside parentheses (...)
    text = re.sub(r"\([^)]*\)", " ", text)

    # 4. Replace markers, symbols, and punctuation with space
    # Unicode categories:
    # M = Mark, S = Symbol, P = Punctuation
    text = "".join(
        " " if unicodedata.category(ch)[0] in {"M", "S", "P"} else ch
        for ch in text
    )

    # 5. Lowercase
    text = text.lower()

    # 6. Replace successive whitespace with a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text.split(" ")

