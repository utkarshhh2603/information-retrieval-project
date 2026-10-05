"""Draws the retrieval pipeline diagram used as Figure 1 in the report."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "pipeline.png")

fig, ax = plt.subplots(figsize=(10, 5.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis("off")

GREY, BLUE, ORANGE, GREEN, PURPLE = "#eeeeee", "#d6e4f2", "#fbe3c8", "#d4ecd9", "#e8defa"


def box(x, y, w, h, text, color, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=color, ec="#555555", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(x1, y1, x2, y2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
                                 color="#444444", lw=1.1, ls=ls))


# Offline indexing (top band)
ax.text(2, 57, "Offline indexing", fontsize=9, style="italic", color="#666666")
box(2, 47, 16, 7, "NFCorpus\n3,633 medical docs", GREY, True)
box(24, 47, 20, 7, "Preprocess: clean, tokenize,\nstop-words, Porter stem", BLUE)
box(50, 47, 18, 7, "Inverted index\nterm → {doc: tf}, df, |d|", BLUE)
box(74, 47, 23, 7, "MiniLM-L6 encoder\n384-d doc embeddings (cached)", ORANGE)
arrow(18.4, 50.5, 23.6, 50.5)
arrow(44.4, 50.5, 49.6, 50.5)
ax.annotate("", xy=(85.5, 54.4), xytext=(10, 54.4),
            arrowprops=dict(arrowstyle="-|>", color="#999999", lw=1, ls="--", connectionstyle="arc3,rad=-0.12"))
ax.text(47, 58.2, "raw text (no stemming)", fontsize=7.5, color="#888888", ha="center")

# Query time
ax.text(2, 41, "Query time", fontsize=9, style="italic", color="#666666")
box(2, 26, 14, 8, "User query", GREY, True)
box(21, 31, 19, 7, "Query preprocessing\n(same pipeline)", BLUE)
box(21, 18, 19, 8, "Medical term expansion\n(optional) PRF / embedding\nneighbours", PURPLE)
box(46, 34, 17, 6, "TF-IDF (inverted index)\nlog-tf·idf, cosine", BLUE)
box(46, 25, 17, 6, "BM25 (inverted index)\nk1 = 1.2, b = 0.75", BLUE)
box(46, 12, 17, 8, "Sentence Transformer\ncosine vs cached\ndoc embeddings", ORANGE)
box(70, 21, 16, 10, "Reciprocal Rank\nFusion\nΣ 1 / (60 + rank)", GREEN, True)
box(90, 22, 8.5, 8, "Ranked\nresults", GREY, True)

arrow(16.4, 31, 20.6, 34)
arrow(30.5, 30.6, 30.5, 26.4)
arrow(40.4, 35, 45.6, 37)
arrow(40.4, 34, 45.6, 29)
arrow(40.4, 23, 45.6, 27, ls="--")
ax.plot([9, 9, 30], [25.6, 12, 12], color="#444444", lw=1.1)
arrow(30, 12, 45.6, 15.5)
ax.text(12, 9.6, "raw query text (no stemming)", fontsize=7.5, color="#666666")
arrow(63.4, 37, 69.6, 29)
arrow(63.4, 28, 69.6, 26)
arrow(63.4, 16.5, 69.6, 23)
arrow(86.4, 26, 89.6, 26)

# Evaluation strip
box(62, 2, 36, 8, "Evaluation against graded qrels\nP@10 · Recall@100 · MAP · nDCG@10 · P-R curve", "#fff8d6")
arrow(94, 21.6, 90, 10.4, ls=":")

fig.tight_layout()
fig.savefig(OUT, dpi=180)
print("saved", os.path.abspath(OUT))
