"""Streamlit search interface for the NFCorpus medical search engine.

Run with:  streamlit run app.py
"""

import html
import json
import os
import re

import pandas as pd
import streamlit as st

from src.data import ROOT, doc_text, load
from src.evaluate import average_precision, ndcg_at_k, precision_at_k
from src.expansion import EmbeddingExpander, PRFExpander
from src.index import InvertedIndex
from src.preprocess import STOPWORDS, preprocess, stem
from src.rankers import BM25Ranker, DenseRanker, TFIDFRanker, rrf_fuse

RESULTS = os.path.join(ROOT, "results")
MODELS = ["TF-IDF", "BM25", "Sentence Transformer", "Hybrid RRF"]
DEPTH = 1000

st.set_page_config(page_title="NFCorpus Medical Search", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading index, models and embeddings...")
def load_engine():
    docs, queries, qrels = load()
    index = InvertedIndex.load_or_build(docs)
    tfidf, bm25, dense = TFIDFRanker(index), BM25Ranker(index), DenseRanker(docs)
    expanders = {"PRF (pseudo-relevance feedback)": PRFExpander(bm25, index),
                 "Embedding neighbours": EmbeddingExpander(dense, docs)}
    return docs, queries, qrels, index, tfidf, bm25, dense, expanders


docs, queries, qrels, index, tfidf, bm25, dense, expanders = load_engine()


def search_all(query, expander_name, rrf_k):
    """Run every model once; returns (results per model, added expansion terms)."""
    if expander_name == "Off":
        weights, added = None, []
    else:
        weights, added = expanders[expander_name].expand(query)
    if weights:
        tf_res, bm_res = tfidf.search_terms(weights, DEPTH), bm25.search_terms(weights, DEPTH)
    else:
        tf_res, bm_res = tfidf.search(query, DEPTH), bm25.search(query, DEPTH)
    dn_res = dense.search(query, DEPTH)
    out = {"TF-IDF": tf_res, "BM25": bm_res, "Sentence Transformer": dn_res,
           "Hybrid RRF": rrf_fuse([tf_res, bm_res, dn_res], rrf_k, DEPTH)}
    return out, added


EXP_MARK = "<mark style='background:#cfe3ff'>{}</mark>"


def _mark(word, qstems, estems):
    stems = preprocess(word)
    e = html.escape(word)
    if any(s in qstems for s in stems):
        return f"<mark>{e}</mark>"
    if any(s in estems for s in stems):
        return EXP_MARK.format(e)
    return e


def snippet(text, qstems, estems=frozenset(), width=45):
    """Pick the window of words with the most query-term hits and highlight the matches."""
    words = text.split()
    hits = [1 if any(s in qstems or s in estems for s in preprocess(w)) else 0 for w in words]
    best, best_i = -1, 0
    for i in range(0, max(1, len(words) - width + 1), 5):
        h = sum(hits[i:i + width])
        if h > best:
            best, best_i = h, i
    window = words[best_i:best_i + width]
    parts = [_mark(w, qstems, estems) for w in window]
    prefix = "… " if best_i > 0 else ""
    suffix = " …" if best_i + width < len(words) else ""
    return prefix + " ".join(parts) + suffix


def highlight_title(title, qstems, estems=frozenset()):
    return " ".join(_mark(w, qstems, estems) for w in title.split())


def render_results(results, qstems, rel, top_k, compact=False, estems=frozenset()):
    if not results:
        st.write("_No documents contain these terms._")
    for rank, (doc_id, score) in enumerate(results[:top_k], start=1):
        d = docs[doc_id]
        badge = ""
        if rel is not None:
            g = rel.get(doc_id, 0)
            badge = {2: " ✅✅", 1: " ✅"}.get(g, " ✖️")
        title = highlight_title(d["title"], qstems, estems)
        if compact:
            st.markdown(f"**{rank}.** {title}{badge}  \n<small>`{doc_id}` · {score:.4f}</small>",
                        unsafe_allow_html=True)
        else:
            st.markdown(
                f"**{rank}. {title}**{badge}  \n"
                f"<small>`{doc_id}` · score {score:.4f}</small><br>"
                f"<span style='font-size:0.92em'>{snippet(d['text'], qstems, estems)}</span>",
                unsafe_allow_html=True)


def query_metrics(results, rel):
    ranked = [d for d, _ in results]
    return {"P@10": precision_at_k(ranked, rel, 10), "AP": average_precision(ranked, rel),
            "nDCG@10": ndcg_at_k(ranked, rel, 10)}


# ---------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ Settings")
# Optional URL parameters (?q=...&mode=...&exp=...) so a search can be linked or bookmarked
params = st.query_params
mode_options = MODELS + ["Compare all 4"]
mode_idx = {"tfidf": 0, "bm25": 1, "dense": 2, "rrf": 3, "compare": 4}.get(params.get("mode", ""), 3)
mode = st.sidebar.radio("Ranking model", mode_options, index=mode_idx)
top_k = st.sidebar.slider("Results to show", 5, 30, 10)
exp_options = ["Off"] + list(expanders)
exp_idx = {"prf": 1, "emb": 2}.get(params.get("exp", ""), 0)
expander_name = st.sidebar.selectbox("Medical term expansion", exp_options, index=exp_idx)
rrf_k = st.sidebar.number_input("RRF smoothing constant k", 1, 500, 60)
st.sidebar.caption("Expansion changes the query sent to TF-IDF and BM25 (and therefore RRF). "
                   "The dense model always sees the original query.")
st.sidebar.markdown("---")
st.sidebar.caption(f"Corpus: {index.N:,} documents · {len(index.postings):,} index terms · "
                   f"{len(queries)} test queries")

# ---------------------------------------------------------------- main
st.title("🔎 NFCorpus Medical Search Engine")
st.caption("TF-IDF · BM25 · Sentence Transformer · Hybrid Reciprocal Rank Fusion. Information Retrieval, Project 6")

tab_search, tab_eval, tab_data = st.tabs(["Search", "Evaluation", "Dataset"])

with tab_search:
    test_q = {f"{t}  [{q}]": q for q, t in sorted(queries.items(), key=lambda x: x[1].lower())}
    c1, c2 = st.columns([3, 2])
    with c2:
        picked = st.selectbox("…or pick a test query (has relevance judgments)", ["—"] + list(test_q))
    default = queries[test_q[picked]] if picked != "—" else params.get("q", "")
    with c1:
        query = st.text_input("Search medical / nutrition literature", value=default,
                              placeholder="e.g. vitamin d deficiency and depression")
    examples = ["Do Cholesterol Statin Drugs Cause Breast Cancer?", "heart attack prevention",
                "obesity in children", "green tea and cancer", "gut bacteria"]
    st.caption("Examples: " + " · ".join(f"`{e}`" for e in examples))

    if query.strip():
        # Relevance labels are only known for test queries
        qid = next((q for q, t in queries.items() if t.strip().lower() == query.strip().lower()), None)
        rel = qrels.get(qid) if qid else None
        results, added = search_all(query, expander_name, rrf_k)
        qstems = set(preprocess(query)) - {stem(s) for s in STOPWORDS}
        estems = {t for w, _ in added for t in preprocess(w)} - qstems

        if expander_name != "Off":
            if added:
                st.info("**Expansion added** (highlighted in blue): " + ", ".join(f"{t} ({w:.2f})" for t, w in added))
            else:
                st.info("Expansion found no terms to add for this query.")
        if rel is not None:
            st.success(f"Test query `{qid}`: {len(rel)} judged relevant docs. ✅ partly relevant · ✅✅ highly relevant · ✖️ not judged relevant")

        if mode == "Compare all 4":
            if rel is not None:
                m = pd.DataFrame({name: query_metrics(res, rel) for name, res in results.items()}).T
                st.dataframe(m.style.format("{:.3f}").highlight_max(axis=0, color="#c8e6c9"),
                             width="stretch")
            cols = st.columns(4)
            for col, name in zip(cols, MODELS):
                with col:
                    st.subheader(name)
                    render_results(results[name], qstems, rel, top_k, compact=True, estems=estems)
        else:
            if rel is not None:
                m = query_metrics(results[mode], rel)
                a, b, c = st.columns(3)
                a.metric("P@10", f"{m['P@10']:.2f}")
                b.metric("Average Precision", f"{m['AP']:.3f}")
                c.metric("nDCG@10", f"{m['nDCG@10']:.3f}")
            st.subheader(f"{mode}: top {top_k}")
            render_results(results[mode], qstems, rel, top_k, estems=estems)

with tab_eval:
    st.header("Performance on all 323 NFCorpus test queries")
    mpath = os.path.join(RESULTS, "metrics.csv")
    if os.path.exists(mpath):
        table = pd.read_csv(mpath)
        metric_cols = ["P@10", "Recall@100", "MAP", "nDCG@10"]
        for group in ["Core", "Extension"]:
            st.subheader("Core models (assignment)" if group == "Core" else "Extensions (query expansion and RRF variants)")
            t = table[table["Group"] == group].drop(columns="Group").set_index("Model")
            st.dataframe(t.style.format("{:.4f}").highlight_max(axis=0, subset=metric_cols, color="#c8e6c9"),
                         width="stretch")
        a, b = st.columns(2)
        a.image(os.path.join(RESULTS, "pr_curve.png"), caption="Precision-Recall curve: core models")
        b.image(os.path.join(RESULTS, "metrics_bar.png"), caption="Metric comparison: core models")
        a, b = st.columns(2)
        a.image(os.path.join(RESULTS, "pr_curve_expansion.png"), caption="Effect of medical query expansion")
        with b:
            st.subheader("RRF k sensitivity")
            st.dataframe(pd.read_csv(os.path.join(RESULTS, "rrf_k_sweep.csv")).set_index("k"),
                         width="stretch")
    else:
        st.warning("Run `python run_eval.py` first to generate results.")

with tab_data:
    st.header("NFCorpus (BEIR test split)")
    spath = os.path.join(RESULTS, "dataset_stats.json")
    if os.path.exists(spath):
        with open(spath) as f:
            s = json.load(f)
        st.table(pd.DataFrame({"value": {k: str(v) for k, v in s.items()}}))
        a, b, c = st.columns(3)
        a.image(os.path.join(RESULTS, "dataset_doc_length.png"))
        b.image(os.path.join(RESULTS, "dataset_rel_per_query.png"))
        c.image(os.path.join(RESULTS, "dataset_grades.png"))
    st.subheader("Inverted index")
    st.table(pd.DataFrame({"value": {k: str(v) for k, v in index.stats().items()}}))
    term = st.text_input("Look up a term in the inverted index", "statin")
    if term:
        t = stem(term.lower())
        plist = index.postings.get(t, {})
        st.write(f"Stem `{t}` appears in **{len(plist)}** documents (df). Top postings by term frequency:")
        top = sorted(plist.items(), key=lambda x: -x[1])[:10]
        st.dataframe(pd.DataFrame([{"doc_id": d, "tf": tf, "title": docs[d]["title"]} for d, tf in top]),
                     width="stretch")
