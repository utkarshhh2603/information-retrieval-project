"""Ranking models: TF-IDF, BM25, dense Sentence Transformer, and Reciprocal Rank Fusion.

Every ranker has search(query, k) -> list of (doc_id, score), best first.
TF-IDF and BM25 also take a weighted term dict through search_terms(), which
query expansion uses.
"""

import math
import os
from collections import Counter, defaultdict

import numpy as np

from src.data import DATA_DIR, doc_text
from src.preprocess import preprocess

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _top_k(scores, k):
    return sorted(scores.items(), key=lambda x: -x[1])[:k]


class TFIDFRanker:
    """Vector space model with log tf * idf weights and cosine similarity."""

    name = "TF-IDF"

    def __init__(self, index):
        self.index = index
        self.idf = {t: math.log(index.N / df) for t, df in index.df.items()}
        norms = defaultdict(float)
        for term, plist in index.postings.items():
            idf = self.idf[term]
            for doc_id, tf in plist.items():
                norms[doc_id] += ((1 + math.log(tf)) * idf) ** 2
        self.doc_norm = {d: math.sqrt(v) for d, v in norms.items()}

    def search_terms(self, qweights, k=1000):
        q = {t: w * self.idf[t] for t, w in qweights.items() if t in self.idf}
        q_norm = math.sqrt(sum(v * v for v in q.values())) or 1.0
        scores = defaultdict(float)
        for term, qw in q.items():
            idf = self.idf[term]
            for doc_id, tf in self.index.postings[term].items():
                scores[doc_id] += qw * (1 + math.log(tf)) * idf
        for doc_id in scores:
            scores[doc_id] /= self.doc_norm[doc_id] * q_norm
        return _top_k(scores, k)

    def search(self, query, k=1000):
        tf = Counter(preprocess(query))
        return self.search_terms({t: 1 + math.log(c) for t, c in tf.items()}, k)


class BM25Ranker:
    """Okapi BM25 with term-frequency saturation (k1) and length normalisation (b)."""

    name = "BM25"

    def __init__(self, index, k1=1.2, b=0.75):
        self.index, self.k1, self.b = index, k1, b
        N = index.N
        self.idf = {t: math.log(1 + (N - df + 0.5) / (df + 0.5)) for t, df in index.df.items()}

    def search_terms(self, qweights, k=1000):
        idx, k1, b = self.index, self.k1, self.b
        scores = defaultdict(float)
        for term, qw in qweights.items():
            if term not in idx.postings:
                continue
            idf = self.idf[term]
            for doc_id, tf in idx.postings[term].items():
                norm = k1 * (1 - b + b * idx.doc_len[doc_id] / idx.avgdl)
                scores[doc_id] += qw * idf * tf * (k1 + 1) / (tf + norm)
        return _top_k(scores, k)

    def search(self, query, k=1000):
        return self.search_terms(dict(Counter(preprocess(query))), k)


class DenseRanker:
    """Sentence Transformer bi-encoder; cosine similarity over normalised embeddings."""

    name = "Sentence Transformer"

    def __init__(self, docs, model_name=DENSE_MODEL, cache=os.path.join(DATA_DIR, "doc_emb.npy")):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.doc_ids = list(docs)
        ids_path = cache.replace(".npy", "_ids.txt")
        if os.path.exists(cache) and os.path.exists(ids_path):
            with open(ids_path, encoding="utf-8") as f:
                cached_ids = f.read().split("\n")
            if cached_ids == self.doc_ids:
                self.emb = np.load(cache)
                return
        # Raw text, no stemming: the transformer expects natural language.
        texts = [doc_text(docs[d]) for d in self.doc_ids]
        self.emb = self.model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        np.save(cache, self.emb)
        with open(ids_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.doc_ids))

    def encode(self, texts):
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True)

    def search_vec(self, qvec, k=1000):
        sims = self.emb @ qvec
        top = np.argsort(-sims)[:k]
        return [(self.doc_ids[i], float(sims[i])) for i in top]

    def search(self, query, k=1000):
        return self.search_vec(self.encode([query])[0], k)


def rrf_fuse(ranked_lists, k=60, top=1000):
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank(d)).

    Uses only ranks (1-based), so no score normalisation across models is needed.
    """
    scores = defaultdict(float)
    for ranking in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return _top_k(scores, top)


class HybridRRF:
    name = "Hybrid RRF"

    def __init__(self, rankers, k=60, depth=1000):
        self.rankers, self.k, self.depth = rankers, k, depth

    def search(self, query, k=1000):
        lists = [r.search(query, self.depth) for r in self.rankers]
        return rrf_fuse(lists, self.k, k)


if __name__ == "__main__":
    from src.data import load
    from src.index import InvertedIndex

    docs, queries, qrels = load()
    index = InvertedIndex.load_or_build(docs)
    tfidf, bm25, dense = TFIDFRanker(index), BM25Ranker(index), DenseRanker(docs)
    models = [tfidf, bm25, dense, HybridRRF([tfidf, bm25, dense])]
    qid = "PLAIN-2"
    print("Query:", queries[qid])
    for m in models:
        print(f"\n== {m.name} ==")
        for doc_id, s in m.search(queries[qid], 5):
            mark = "REL" if doc_id in qrels[qid] else "   "
            print(f"  {mark} {doc_id:10s} {s:.4f}  {docs[doc_id]['title'][:70]}")
