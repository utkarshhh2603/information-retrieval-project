"""Hand-built inverted index over the preprocessed corpus."""

import os
import pickle
from collections import Counter, defaultdict

from src.data import DATA_DIR, doc_text
from src.preprocess import preprocess

INDEX_PATH = os.path.join(DATA_DIR, "index.pkl")


class InvertedIndex:
    def __init__(self):
        self.postings = {}  # term -> {doc_id: term frequency}
        self.df = {}  # term -> number of docs containing it
        self.doc_len = {}  # doc_id -> number of tokens
        self.N = 0
        self.avgdl = 0.0

    def build(self, docs):
        postings = defaultdict(dict)
        for doc_id, doc in docs.items():
            tokens = preprocess(doc_text(doc))
            self.doc_len[doc_id] = len(tokens)
            for term, tf in Counter(tokens).items():
                postings[term][doc_id] = tf
        self.postings = dict(postings)
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
