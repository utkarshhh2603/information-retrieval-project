"""Per-query analysis of the evaluation run (reads results/per_query_ndcg10.csv, writes results/analysis.md)."""

import os

import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
B, D, T = "BM25", "Sentence Transformer", "TF-IDF"
R = "Hybrid RRF (TF-IDF + BM25 + Dense)"
RP = "Hybrid RRF (TF-IDF + BM25+PRF + Dense)"
P = "BM25 + PRF expansion"


def main():
    pq = pd.read_csv(os.path.join(OUT, "per_query_ndcg10.csv"), index_col=0)
    n = len(pq)
    lines = ["# Per-query analysis (nDCG@10, 323 NFCorpus test queries)", ""]

    lines += ["## Head-to-head", "",
              "| Comparison | Wins | Losses | Ties |", "|---|---|---|---|"]
    for a, b in [(D, B), (R, B), (R, D), (P, B), (RP, R)]:
        lines.append(f"| {a} vs {b} | {(pq[a] > pq[b]).sum()} | {(pq[a] < pq[b]).sum()} | {(pq[a] == pq[b]).sum()} |")
    both = ((pq[R] > pq[B]) & (pq[R] > pq[D])).sum()
    worse = ((pq[R] < pq[B]) & (pq[R] < pq[D])).sum()
    lines += ["", f"- RRF beats **both** BM25 and Dense on {both} queries and is worse than both on only {worse}.",
              f"- Queries with nDCG@10 = 0: BM25 {(pq[B] == 0).sum()}, Dense {(pq[D] == 0).sum()}, "
              f"RRF {(pq[R] == 0).sum()}, RRF+PRF {(pq[RP] == 0).sum()} (of {n}).", ""]

    pq["qlen"] = pq["query"].str.split().str.len()
    pq["length"] = pd.cut(pq.qlen, [0, 2, 4, 100], labels=["1-2 words", "3-4 words", "5+ words"])
    g = pq.groupby("length", observed=True)[[T, B, D, R, RP]].mean().round(4)
    g.insert(0, "queries", pq.groupby("length", observed=True).size())
    lines += ["## nDCG@10 by query length", "", g.to_markdown(), ""]

    cols = ["query", B, D, R]
    gap = pq[D] - pq[B]
    lines += ["## Dense beats BM25 the most (vocabulary mismatch)", "",
              pq.loc[gap.sort_values(ascending=False).index[:8], cols].round(3).to_markdown(), ""]
    lines += ["## BM25 beats Dense the most (rare exact terms)", "",
              pq.loc[gap.sort_values().index[:8], cols].round(3).to_markdown(), ""]
    gain = pq[R] - pq[[B, D]].max(axis=1)
    lines += ["## RRF beats both inputs the most (complementary evidence)", "",
              pq.loc[gain.sort_values(ascending=False).index[:8], cols].round(3).to_markdown(), ""]
    pgain = pq[P] - pq[B]
    lines += ["## PRF expansion: biggest gains and losses vs BM25", "",
              pq.loc[list(pgain.sort_values(ascending=False).index[:5]) + list(pgain.sort_values().index[:5]),
                     ["query", B, P]].round(3).to_markdown(), ""]

    with open(os.path.join(OUT, "analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[:16]))


if __name__ == "__main__":
    main()
