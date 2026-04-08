import re

from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Clean input text by lowercasing, removing special characters,
    extra spaces, and stopwords.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]

    return " ".join(words)
