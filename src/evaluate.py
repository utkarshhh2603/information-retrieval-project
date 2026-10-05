"""Evaluation metrics: Precision@k, Recall@k, Average Precision / MAP, nDCG@k, P-R curve."""

import math

import numpy as np

RECALL_LEVELS = np.linspace(0, 1, 11)


def precision_at_k(ranked, rel, k=10):
    return sum(1 for d in ranked[:k] if d in rel) / k


def recall_at_k(ranked, rel, k=100):
    return sum(1 for d in ranked[:k] if d in rel) / len(rel)


def average_precision(ranked, rel):
    hits, total = 0, 0.0
    for i, d in enumerate(ranked, start=1):
        if d in rel:
            hits += 1
            total += hits / i
    return total / len(rel)


def ndcg_at_k(ranked, rel, k=10):
    """Graded nDCG with gain 2^rel - 1 (NFCorpus grades: 1 = partly, 2 = highly relevant)."""
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def interpolated_pr(ranked, rel):
    """11-point interpolated precision at recall 0.0, 0.1, ..., 1.0."""
    precisions, recalls, hits = [], [], 0
    for i, d in enumerate(ranked, start=1):
        if d in rel:
            hits += 1
        precisions.append(hits / i)
        recalls.append(hits / len(rel))
    precisions, recalls = np.array(precisions), np.array(recalls)
    out = []
    for r in RECALL_LEVELS:
        mask = recalls >= r
        out.append(precisions[mask].max() if mask.any() else 0.0)
    return np.array(out)


def evaluate_run(run, qrels):
    """run: query_id -> ranked list of doc_ids. Returns (mean metrics, per-query rows, mean P-R curve)."""
    rows, curves = [], []
    for qid, rel in qrels.items():
        ranked = run.get(qid, [])
        rows.append({
            "query_id": qid,
            "P@10": precision_at_k(ranked, rel, 10),
            "Recall@100": recall_at_k(ranked, rel, 100),
            "AP": average_precision(ranked, rel),
            "nDCG@10": ndcg_at_k(ranked, rel, 10),
        })
        curves.append(interpolated_pr(ranked, rel))
    mean = {m: float(np.mean([r[m] for r in rows])) for m in ["P@10", "Recall@100", "AP", "nDCG@10"]}
    mean["MAP"] = mean.pop("AP")
    return mean, rows, np.mean(curves, axis=0)


if __name__ == "__main__":
    # Tiny worked example for checking the formulas by hand.
    ranked = ["d1", "d2", "d3", "d4", "d5"]
    rel = {"d1": 2, "d3": 1, "d9": 1}
    print("P@5     ", precision_at_k(ranked, rel, 5))  # 2/5 = 0.4
    print("R@5     ", recall_at_k(ranked, rel, 5))  # 2/3
    print("AP      ", average_precision(ranked, rel))  # (1/1 + 2/3) / 3 = 0.5556
    print("nDCG@5  ", round(ndcg_at_k(ranked, rel, 5), 4))
    print("PR curve", interpolated_pr(ranked, rel).round(3))
