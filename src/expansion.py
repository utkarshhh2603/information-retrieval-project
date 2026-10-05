"""Medical query expansion.

Two methods, both return (weighted query terms, list of added terms):

* PRFExpander - pseudo-relevance feedback in the style of RM3. The top BM25
  documents are assumed relevant, and the terms that carry the most weight in
  them get added to the query.
* EmbeddingExpander - each query word gets the corpus words closest to it in
  Sentence Transformer embedding space (synonyms, related medical terms).
"""

import math
import os
from collections import Counter, defaultdict

import numpy as np

from src.data import DATA_DIR, doc_text
from src.preprocess import STOPWORDS, preprocess, stem, tokenize


def _base_weights(query):
    return {t: float(c) for t, c in Counter(preprocess(query)).items()}


class PRFExpander:
    name = "PRF"

    def __init__(self, bm25, index, fb_docs=10, fb_terms=10, weight=0.5):
        self.bm25, self.index = bm25, index
        self.fb_docs, self.fb_terms, self.weight = fb_docs, fb_terms, weight
        # Each document's terms, so we don't have to re-scan the postings for every query.
        self.doc_terms = defaultdict(dict)
        for term, plist in index.postings.items():
            for doc_id, tf in plist.items():
                self.doc_terms[doc_id][term] = tf

    def expand(self, query):
        q = _base_weights(query)
        feedback = self.bm25.search_terms(q, self.fb_docs)
        if not feedback:
            return q, []
        total = sum(s for _, s in feedback)
        cand = defaultdict(float)
        for doc_id, score in feedback:
            dlen = self.index.doc_len[doc_id]
            for term, tf in self.doc_terms[doc_id].items():
                # P(t|d) * P(d|q), damped by idf so generic words ("studi") don't take over.
                idf = math.log(self.index.N / self.index.df[term])
                cand[term] += (tf / dlen) * (score / total) * idf
        for t in q:
            cand.pop(t, None)
        top = sorted(cand.items(), key=lambda x: -x[1])[: self.fb_terms]
        if not top:
            return q, []
        best = top[0][1]
        expanded = dict(q)
        added = []
        for term, w in top:
            expanded[term] = self.weight * w / best
            added.append((term, round(expanded[term], 3)))
        return expanded, added


class EmbeddingExpander:
    name = "Embedding"

    def __init__(self, dense, docs, min_df=3, per_word=3, threshold=0.6, weight=0.5):
        self.dense = dense
        self.per_word, self.threshold, self.weight = per_word, threshold, weight
        # Embed real words, not stems. Keep the most common surface form for each stem.
        surface = Counter()
        df = Counter()
        for doc in docs.values():
            toks = [t for t in tokenize(doc_text(doc)) if t not in STOPWORDS and not t.isdigit() and len(t) > 2]
            surface.update(toks)
            df.update(set(toks))
        best_form = {}
        for word, c in surface.most_common():
            if df[word] >= min_df:
                best_form.setdefault(stem(word), word)
        self.words = sorted(best_form.values())
        cache = os.path.join(DATA_DIR, "vocab_emb.npy")
        ids_path = os.path.join(DATA_DIR, "vocab_words.txt")
        if os.path.exists(cache) and os.path.exists(ids_path):
            with open(ids_path, encoding="utf-8") as f:
                if f.read().split("\n") == self.words:
                    self.emb = np.load(cache)
                    return
        self.emb = dense.encode(self.words)
        np.save(cache, self.emb)
        with open(ids_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.words))

    def neighbours(self, word, qvec):
        """Words close to `word` that also fit the query as a whole.

        Scoring against the full query embedding resolves ambiguity: for
        "heart attack prevention", "attack" should not bring in "battle".
        """
        vec = self.dense.encode([word])[0]
        sims = self.emb @ vec
        own = stem(word)
        out = []
        for i in np.argsort(-sims)[:30]:
            w = self.words[i]
            if sims[i] < self.threshold:
                break
            if stem(w) == own or w in word or word in w:
                continue
            q_sim = float(self.emb[i] @ qvec)
            if q_sim < self.context_threshold:
                continue
            out.append((w, round((float(sims[i]) + q_sim) / 2, 3)))
            if len(out) == self.per_word:
                break
        return out

    context_threshold = 0.35

    def expand(self, query):
        q = _base_weights(query)
        expanded = dict(q)
        added = []
        qvec = self.dense.encode([query])[0]
        words = [t for t in tokenize(query) if t not in STOPWORDS and not t.isdigit() and len(t) > 2]
        for word in words:
            for nb, score in self.neighbours(word, qvec):
                terms = [t for t in preprocess(nb) if t not in expanded]
                if not terms:
                    continue
                for term in terms:
                    expanded[term] = self.weight * score
                added.append((nb, score))
        return expanded, added


class ExpandedRanker:
    """Wraps a ranker that has search_terms() so it searches with an expanded query."""

    def __init__(self, ranker, expander):
        self.ranker, self.expander = ranker, expander
        self.name = f"{ranker.name} + {expander.name} expansion"

    def search(self, query, k=1000):
        weights, _ = self.expander.expand(query)
        return self.ranker.search_terms(weights, k)


if __name__ == "__main__":
    from src.data import load
    from src.index import InvertedIndex
    from src.rankers import BM25Ranker, DenseRanker

    docs, queries, qrels = load()
    index = InvertedIndex.load_or_build(docs)
    bm25, dense = BM25Ranker(index), DenseRanker(docs)
    prf, emb = PRFExpander(bm25, index), EmbeddingExpander(dense, docs)
    print("vocabulary embedded:", len(emb.words))
    for q in ["vitamin d deficiency", "Do Cholesterol Statin Drugs Cause Breast Cancer?", "heart attack prevention",
              "obesity in children"]:
        print(f"\nQuery: {q}")
        print("  PRF adds:      ", [t for t, _ in prf.expand(q)[1]])
        print("  Embedding adds:", emb.expand(q)[1])
