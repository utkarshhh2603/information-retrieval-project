"""Hand-built inverted index over the preprocessed corpus."""

import os
import pickle
from collections import Counter, defaultdict

from src.data import DATA_DIR, doc_text
from src.preprocess import preprocess, stem, tokenize

INDEX_PATH = os.path.join(DATA_DIR, "index.pkl")


class InvertedIndex:
    def __init__(self):
        self.postings = {}  # term -> {doc_id: term frequency}
        self.df = {}  # term -> number of docs containing it
        self.doc_len = {}  # doc_id -> number of tokens
        self.N = 0
        self.avgdl = 0.0
        self.surface = {}  # stem -> most common original word (for display)

    def build(self, docs):
        postings = defaultdict(dict)
        forms = defaultdict(Counter)  # stem -> surface word counts, for display
        for doc_id, doc in docs.items():
            text = doc_text(doc)
            tokens = preprocess(text)
            self.doc_len[doc_id] = len(tokens)
            for term, tf in Counter(tokens).items():
                postings[term][doc_id] = tf
            for word in tokenize(text):
                for w in [word] + (word.split("-") if "-" in word else []):
                    forms[stem(w)][w] += 1
        self.postings = dict(postings)
        self.surface = {s: c.most_common(1)[0][0] for s, c in forms.items() if s in self.postings}
        self.df = {t: len(p) for t, p in self.postings.items()}
        self.N = len(docs)
        self.avgdl = sum(self.doc_len.values()) / self.N
        return self

    def save(self, path=INDEX_PATH):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load_or_build(cls, docs, path=INDEX_PATH):
        idx = cls()
        if os.path.exists(path):
            with open(path, "rb") as f:
                idx.__dict__.update(pickle.load(f))
        else:
            idx.build(docs).save(path)
        return idx

    def stats(self):
        sizes = sorted(self.df.values(), reverse=True)
        return {
            "documents (N)": self.N,
            "vocabulary size": len(self.postings),
            "total postings": sum(sizes),
            "avg doc length (tokens)": round(self.avgdl, 1),
            "avg postings list length": round(sum(sizes) / len(sizes), 1),
            "terms appearing in 1 doc": sum(1 for s in sizes if s == 1),
        }


if __name__ == "__main__":
    from src.data import load

    docs, _, _ = load()
    idx = InvertedIndex().build(docs)
    idx.save()
    for k, v in idx.stats().items():
        print(f"{k:28s} {v}")
    top = sorted(idx.df.items(), key=lambda x: -x[1])[:10]
    print("most frequent terms (df):", top)
    print("postings for 'statin' (first 5):", list(idx.postings.get("statin", {}).items())[:5])
