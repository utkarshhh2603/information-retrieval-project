"""Text preprocessing: cleaning, tokenization, stop-word removal, stemming."""

import re
from functools import lru_cache

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

STOPWORDS = set(stopwords.words("english"))
_stemmer = PorterStemmer()

# Words joined by hyphens stay together ("omega-3", "beta-carotene"),
# and letter+digit tokens survive ("b12", "covid-19").
TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


@lru_cache(maxsize=None)
def stem(token):
    return _stemmer.stem(token)


def clean(text):
    """Lowercase and drop markup / odd characters."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)  # stray HTML tags
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    return text


def tokenize(text):
    return TOKEN_RE.findall(clean(text))


def _keep(tok):
    return tok not in STOPWORDS and not tok.isdigit() and len(tok) > 1


def preprocess(text):
    """Full pipeline used for both documents and queries.

    A hyphenated compound is kept whole and also split into its parts, so
    "cholesterol-rich" still matches a query for "cholesterol".
    """
    tokens = []
    for tok in tokenize(text):
        if "-" in tok:
            parts = [p for p in tok.split("-") if _keep(p)]
            if parts:
                tokens.append(stem(tok))
            tokens.extend(stem(p) for p in parts)
        elif _keep(tok):
            tokens.append(stem(tok))
    return tokens


if __name__ == "__main__":
    from src.data import doc_text, load

    docs, queries, qrels = load()
    qid = "PLAIN-2"
    did = next(iter(qrels[qid]))
    for label, text in [("Query", queries[qid]), ("Document", doc_text(docs[did])[:400])]:
        print(f"--- {label} (raw) ---\n{text}")
        print(f"--- tokens ---\n{tokenize(text)}")
        print(f"--- after stopwords + stemming ---\n{preprocess(text)}\n")
