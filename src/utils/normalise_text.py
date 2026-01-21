import re
import unicodedata


def normalization(text):
    # 1. Unicode normalize
    text = unicodedata.normalize("NFKD", text)

    # 2. Remove accents (diacritics)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) != "Mn"
    )

    # 3. Remove text inside square brackets
    text = re.sub(r"\[[^\]]*\]", " ", text)

    # 4. Remove text inside parentheses
    text = re.sub(r"\([^)]*\)", " ", text)

    # 5. Remove symbols & punctuation
    text = "".join(
        " " if unicodedata.category(ch)[0] in {"S", "P"} else ch
        for ch in text
    )

    # 6. Lowercase
    text = text.lower()

    # 7. Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text.split(" ")
def remove_words(trans):

    #respiration
    #externe
    #exterieur
    words = trans.split()

    clean_words = [w for w in words if not ("*" in w and w.count("*") >= 2)]

    res = " ".join(clean_words)
    return res