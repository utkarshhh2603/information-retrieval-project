"""Run every model over the NFCorpus test queries and write tables and graphs to results/."""

import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import ROOT, doc_text, load, stats
from src.evaluate import RECALL_LEVELS, evaluate_run
from src.expansion import EmbeddingExpander, ExpandedRanker, PRFExpander
from src.index import InvertedIndex
from src.rankers import BM25Ranker, DenseRanker, TFIDFRanker, rrf_fuse

OUT = os.path.join(ROOT, "results")
DEPTH = 1000  # documents retrieved per query per model
METRICS = ["P@10", "Recall@100", "MAP", "nDCG@10"]


def run_model(search, queries):
    t = time.time()
    run = {qid: search(text, DEPTH) for qid, text in queries.items()}
    return run, (time.time() - t) / len(queries) * 1000


def ids(run):
    return {q: [d for d, _ in lst] for q, lst in run.items()}


def fuse(runs, k=60):
    return {q: rrf_fuse([r[q] for r in runs], k, DEPTH) for q in runs[0]}


def dataset_charts(docs, queries, qrels):
    lengths = [len(doc_text(d).split()) for d in docs.values()]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(lengths, bins=50, color="#3b6ea5")
    ax.set(xlabel="Document length (words)", ylabel="Number of documents", title="NFCorpus document lengths")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dataset_doc_length.png"), dpi=150)
    plt.close(fig)

    rel = [len(r) for r in qrels.values()]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(rel, bins=50, color="#3b6ea5")
    ax.set(xlabel="Relevant documents per query", ylabel="Number of queries", title="Relevant documents per test query")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dataset_rel_per_query.png"), dpi=150)
    plt.close(fig)

    grades = pd.Series([g for r in qrels.values() for g in r.values()]).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.bar(["1 (partly relevant)", "2 (highly relevant)"], grades.values, color=["#8fb3d9", "#3b6ea5"])
    for i, v in enumerate(grades.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    ax.set(ylabel="Judgments", title="Relevance grade distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dataset_grades.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    docs, queries, qrels = load()
    print(f"{len(docs)} docs, {len(queries)} test queries")
    with open(os.path.join(OUT, "dataset_stats.json"), "w") as f:
        json.dump(stats(docs, queries, qrels), f, indent=2)
    dataset_charts(docs, queries, qrels)

    index = InvertedIndex.load_or_build(docs)
    tfidf, bm25, dense = TFIDFRanker(index), BM25Ranker(index), DenseRanker(docs)
    prf, emb = PRFExpander(bm25, index), EmbeddingExpander(dense, docs)

    runs, latency = {}, {}
    for name, search in [
        ("TF-IDF", tfidf.search),
        ("BM25", bm25.search),
        ("Sentence Transformer", dense.search),
        ("BM25 + PRF expansion", ExpandedRanker(bm25, prf).search),
        ("BM25 + Embedding expansion", ExpandedRanker(bm25, emb).search),
    ]:
        runs[name], latency[name] = run_model(search, queries)
        print(f"  ran {name} ({latency[name]:.1f} ms/query)")

    base = ["TF-IDF", "BM25", "Sentence Transformer"]
    runs["Hybrid RRF (TF-IDF + BM25 + Dense)"] = fuse([runs[m] for m in base])
    runs["Hybrid RRF (BM25 + Dense)"] = fuse([runs["BM25"], runs["Sentence Transformer"]])
    runs["Hybrid RRF (TF-IDF + BM25+PRF + Dense)"] = fuse(
        [runs["TF-IDF"], runs["BM25 + PRF expansion"], runs["Sentence Transformer"]])
    runs["Hybrid RRF (TF-IDF + BM25+Emb + Dense)"] = fuse(
        [runs["TF-IDF"], runs["BM25 + Embedding expansion"], runs["Sentence Transformer"]])

    core = base + ["Hybrid RRF (TF-IDF + BM25 + Dense)"]
    rows, per_query, curves = [], {}, {}
    for name, run in runs.items():
        mean, pq, curve = evaluate_run(ids(run), qrels)
        rows.append({"Model": name, "Group": "Core" if name in core else "Extension",
                     **{m: round(mean[m], 4) for m in METRICS}})
        per_query[name] = {r["query_id"]: r["nDCG@10"] for r in pq}
        curves[name] = curve

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, "metrics.csv"), index=False)
    with open(os.path.join(OUT, "metrics.md"), "w") as f:
        f.write(table.to_markdown(index=False, floatfmt=".4f"))
    print("\n" + table.to_string(index=False))

    pq = pd.DataFrame(per_query)
    pq.insert(0, "query", [queries[q] for q in pq.index])
    pq.index.name = "query_id"
    pq.round(4).to_csv(os.path.join(OUT, "per_query_ndcg10.csv"))

    # RRF smoothing constant sweep
    sweep = []
    for k in [1, 10, 30, 60, 100, 200]:
        mean, _, _ = evaluate_run(ids(fuse([runs[m] for m in base], k)), qrels)
        sweep.append({"k": k, **{m: round(mean[m], 4) for m in METRICS}})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(os.path.join(OUT, "rrf_k_sweep.csv"), index=False)
    print("\nRRF k sweep\n" + sweep.to_string(index=False))

    # Precision-Recall graph: core models
    styles = {"TF-IDF": ("#9e9e9e", "--"), "BM25": ("#3b6ea5", "-"),
              "Sentence Transformer": ("#e08a2c", "-"), "Hybrid RRF (TF-IDF + BM25 + Dense)": ("#2e8b57", "-")}
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, (color, ls) in styles.items():
        ax.plot(RECALL_LEVELS, curves[name], marker="o", ms=4, color=color, ls=ls, label=name)
    ax.set(xlabel="Recall", ylabel="Interpolated precision", title="11-point Precision-Recall curve (NFCorpus test)",
           xlim=(0, 1), ylim=(0, None))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pr_curve.png"), dpi=150)
    plt.close(fig)

    # Precision-Recall graph: expansion variants
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name in ["BM25", "BM25 + PRF expansion", "BM25 + Embedding expansion", "Hybrid RRF (TF-IDF + BM25 + Dense)",
                 "Hybrid RRF (TF-IDF + BM25+PRF + Dense)"]:
        ax.plot(RECALL_LEVELS, curves[name], marker="o", ms=4, label=name)
    ax.set(xlabel="Recall", ylabel="Interpolated precision", title="Effect of medical query expansion",
           xlim=(0, 1), ylim=(0, None))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pr_curve_expansion.png"), dpi=150)
    plt.close(fig)

    # Grouped bar chart of the core comparison
    core_tbl = table[table["Model"].isin(core)].set_index("Model")[METRICS]
    core_tbl.index = ["TF-IDF", "BM25", "Dense", "Hybrid RRF"]
    ax = core_tbl.T.plot.bar(figsize=(7, 4), rot=0, color=["#9e9e9e", "#3b6ea5", "#e08a2c", "#2e8b57"])
    ax.set(ylabel="Score", title="Core model comparison (NFCorpus test)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "metrics_bar.png"), dpi=150)
    plt.close()

    with open(os.path.join(OUT, "latency_ms_per_query.json"), "w") as f:
        json.dump({k: round(v, 2) for k, v in latency.items()}, f, indent=2)
    print("\nWrote results to", OUT)


if __name__ == "__main__":
    main()
