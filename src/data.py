"""Load the NFCorpus test collection (BEIR version) and cache it locally."""

import json
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA_DIR, "nfcorpus_test.json")


def _download():
    import ir_datasets

    ds = ir_datasets.load("beir/nfcorpus/test")
    docs = {}
    for d in ds.docs_iter():
        docs[d.doc_id] = {"title": d.title, "text": d.text, "url": getattr(d, "url", "")}
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    qrels = defaultdict(dict)
    for r in ds.qrels_iter():
        qrels[r.query_id][r.doc_id] = int(r.relevance)
    return docs, queries, dict(qrels)


def load():
    """Return (docs, queries, qrels).

    docs:    doc_id -> {"title", "text", "url"}
    queries: query_id -> query text
    qrels:   query_id -> {doc_id: grade}  (grades 1 and 2 are relevant)
    """
    if not os.path.exists(CACHE):
        os.makedirs(DATA_DIR, exist_ok=True)
        docs, queries, qrels = _download()
        with open(CACHE, "w", encoding="utf-8") as f:
            json.dump({"docs": docs, "queries": queries, "qrels": qrels}, f)
    with open(CACHE, encoding="utf-8") as f:
        blob = json.load(f)
    qrels = {q: {d: g for d, g in rel.items() if g > 0} for q, rel in blob["qrels"].items()}
    qrels = {q: rel for q, rel in qrels.items() if rel}
    queries = {q: t for q, t in blob["queries"].items() if q in qrels}
    return blob["docs"], queries, qrels


def doc_text(doc):
    """Text that gets indexed: title followed by abstract."""
    return f"{doc['title']}. {doc['text']}"


def stats(docs, queries, qrels):
    lengths = [len(doc_text(d).split()) for d in docs.values()]
    rel_per_q = [len(r) for r in qrels.values()]
    grades = Counter(g for r in qrels.values() for g in r.values())
    return {
        "documents": len(docs),
        "test queries": len(queries),
        "avg doc length (words)": round(sum(lengths) / len(lengths), 1),
        "avg query length (words)": round(sum(len(q.split()) for q in queries.values()) / len(queries), 1),
        "avg relevant docs / query": round(sum(rel_per_q) / len(rel_per_q), 1),
        "relevance judgments": sum(rel_per_q),
        "grade distribution": dict(sorted(grades.items())),
    }


if __name__ == "__main__":
    docs, queries, qrels = load()
    for k, v in stats(docs, queries, qrels).items():
        print(f"{k:28s} {v}")
    qid = next(iter(queries))
    print("\nExample query", qid, ":", queries[qid])
    did = next(iter(qrels[qid]))
    print("Relevant doc", did, ":", doc_text(docs[did])[:300], "...")
