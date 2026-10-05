# Project Plan: Hybrid Medical Search Engine on NFCorpus

**Course:** Information Retrieval, Project 6 (Medical Search using NFCorpus)
**Marks:** 20. Report: 10 (AI/similarity must be under 10%). Demo using code and report: 10.

## Overview

| Item | Details |
|---|---|
| Dataset | NFCorpus (BEIR): ~3.6k PubMed / NutritionFacts docs, 323 test queries, graded relevance (0/1/2) |
| Preprocessing | Cleaning, tokenization, stop-word removal, Porter stemming |
| Index | Hand-built inverted index |
| Ranking models | TF-IDF, BM25, Sentence Transformer (dense), Hybrid RRF (k = 60) |
| Extra feature | Medical term expansion (pseudo-relevance feedback + embedding neighbours) |
| Metrics | Precision@10, Recall@100, MAP, nDCG@10, Precision-Recall curve |
| Deliverables | Search interface (Streamlit), comparison table, P-R graph, report, demo |

RRF formula: `RRF(d) = Σ_m 1 / (k + r_m(d))`. Here k = 60 and r_m(d) is the rank of document d in model m.

## Planned repository layout

```
requirements.txt
src/
  data.py         # load + cache NFCorpus docs, queries, qrels
  preprocess.py   # cleaning, tokenization, stopwords, stemming
  index.py        # inverted index
  rankers.py      # TF-IDF, BM25, Dense, RRF
  expansion.py    # medical query expansion
  evaluate.py     # metrics + P-R curve
run_eval.py       # runs all models, writes results/
app.py            # Streamlit search interface
results/          # metrics tables, graphs
report/           # report draft
data/             # downloaded dataset + caches (not committed)
```

---

## Phase 0: Setup
1. Create the folders `src/`, `data/` (gitignored), `results/` and `report/`.
2. Write `requirements.txt` listing ir_datasets, nltk, scikit-learn, sentence-transformers, torch, pandas, numpy, matplotlib and streamlit.
3. Install `ir_datasets` and download the NLTK `stopwords` and `punkt` data.
4. Commit: "Project scaffold".

## Phase 1: Data (`src/data.py`)
5. Load `beir/nfcorpus/test` via ir_datasets: the corpus (doc_id, title, text), the test queries and the qrels.
6. Build each document's text as title + abstract. Cache everything to `data/` so later runs work offline.
7. Print dataset stats: doc count, query count, average doc length, relevant docs per query, grade distribution.
8. Commit.

## Phase 2: Preprocessing (`src/preprocess.py`)
9. Write `preprocess(text) -> list[str]`, which does the following in order:
   - lowercase the text
   - remove punctuation and numbers-only tokens, but keep medical tokens like "b12" and "omega-3"
   - tokenize
   - remove NLTK English stopwords
   - apply the Porter stemmer
10. Use the same function for documents and queries.
11. Print one document and one query before and after processing. These become the report's worked example.

## Phase 3: Inverted index (`src/index.py`)
12. Build the index by hand from a single pass over the corpus:
    - `postings: term -> {doc_id: term_freq}`
    - `df[term]` and `doc_len[doc_id]`
    - `avgdl` and `N`
13. Save it to `data/index.pkl`. Report the vocabulary size and posting statistics.
14. Commit.

## Phase 4: Ranking models (`src/rankers.py`)
Every ranker exposes `search(query, k) -> [(doc_id, score)]`.

15. **TF-IDF** (written by hand):
    - weight = (1 + log tf) · log(N / df)
    - cosine similarity between query and doc vectors
    - candidate docs come from the postings lists
16. **BM25** (written by hand): k1 = 1.2, b = 0.75, Robertson IDF.
17. **Dense:** `sentence-transformers/all-MiniLM-L6-v2`.
    - Encode all docs once and cache the vectors to `data/doc_emb.npy`.
    - Rank by cosine similarity.
    - Use raw, unstemmed text, because transformers need natural language.
18. **Hybrid RRF:** take the top 1000 from TF-IDF, BM25 and Dense, then `score(d) = Σ 1 / (60 + rank_m(d))` and sort.
19. Sanity check: compare the top 5 from each model for a sample query.
20. Commit.

## Phase 5: Medical term expansion (`src/expansion.py`)
21. **Pseudo-relevance feedback (RM3-style):**
    - run BM25 and take the top 10 docs
    - score candidate terms by tf·idf within them
    - add the top 10 terms with a weight of 0.5
    - rerun BM25
22. **Embedding-neighbour expansion:** embed the corpus vocabulary, then add the 2–3 nearest terms per query word (for example, "vitamin d" picks up "cholecalciferol" and "deficiency").
23. `expand(query) -> (expanded_terms, added_terms)` returns the added terms so the UI can display them.
24. Commit.

## Phase 6: Evaluation (`src/evaluate.py` + `run_eval.py`)
25. Metrics, averaged over the test queries that have at least one relevant doc:
    - P@10
    - Recall@100
    - AP / MAP
    - nDCG@10 (graded: gain = 2^rel − 1)
    - an 11-point interpolated Precision-Recall curve
26. Models evaluated over all 323 test queries:
    - **Core:** TF-IDF, BM25, Dense, RRF (all 3)
    - **Extensions:** BM25 + PRF, BM25 + embedding expansion, RRF (BM25 + Dense only), RRF with expanded BM25
27. Outputs:
    - `results/metrics.csv` and `results/metrics.md` (the comparison table)
    - `results/pr_curve.png` (all models on one graph)
    - `results/dataset_*.png` (doc-length histogram, relevant docs per query, grade distribution)
    - `results/per_query.csv` (for the failure analysis)
28. Check against the published BEIR baselines (BM25 nDCG@10 ≈ 0.32, MiniLM ≈ 0.31). A large gap means there's a bug to fix first.
29. Optional: a sweep over the RRF k parameter (10, 30, 60, 100).
30. Commit the results.

## Phase 7: Search interface (`app.py`, Streamlit)
31. Sidebar:
    - model selector (TF-IDF / BM25 / Dense / Hybrid RRF / Compare all)
    - top-k
    - "Medical term expansion" toggle
32. Main panel:
    - query box and example query buttons
    - results with rank, title, score and a snippet with query terms highlighted
    - the added expansion terms, when expansion is on
33. Compare mode: the four models side by side, with relevant docs marked ✓ for test-set queries.
34. Evaluation tab: the metrics table and the P-R graph.
35. Cache models with `st.cache_resource` so searches are instant.
36. Test with `streamlit run app.py` and take screenshots for the report.
37. Commit.

## Phase 8: README + push
38. Write a README with the summary, setup, how to run evaluation and the app, the results table and the folder structure.
39. Push to GitHub.

## Phase 9: Report (format: `IR_report formatt.docx`)
40. Sections, using our real numbers and figures:
    - Abstract
    - 1. Introduction (1 page)
    - 2. Dataset: graphs and a worked example (1 page)
    - 3. Methodology: pipeline diagram, preprocessing, index, the 4 models, the RRF formula, expansion (2 pages)
    - 4. Experiments: Table 1, P-R graph, extension results, per-query analysis (2 pages)
    - 5. Conclusion (1 page)
    - References: BM25 (Robertson & Zaragoza 2009), SBERT (Reimers & Gurevych 2019), RRF (Cormack et al. 2009), NFCorpus (Boteva et al. 2016), BEIR (Thakur et al. 2021)
41. **Similarity < 10%:** the technical draft must be rewritten in your own words. Our own results, per-query examples and the expansion analysis are unique content that keeps similarity low.
42. Fill in your name, registration number and email, then export to .docx / PDF.

## Phase 10: Demo prep
43. Demo script:
    - show the dataset
    - run 2–3 queries where BM25 and Dense disagree and show RRF combining them
    - toggle expansion
    - show the metrics tab
44. Viva questions to prepare: why RRF needs no score normalization, BM25 saturation vs TF-IDF, why the dense model gets unstemmed text, and what nDCG measures.

---

## Notes
- The numbers in the report template are placeholders. We report our actual results, and RRF is not guaranteed to win every metric.
- `data/` (the dataset and embedding caches) is not committed to GitHub.

## Final checklist
- [ ] `python run_eval.py` runs from a fresh clone, and the metrics are in line with BEIR baselines
- [ ] `streamlit run app.py` works for all models, and the expansion terms look sensible
- [ ] Code, results and README are on GitHub
- [ ] Report written in your own words, similarity < 10%
- [ ] Demo rehearsed
