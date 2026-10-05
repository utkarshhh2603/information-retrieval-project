# Demo Script and Viva Prep

The demo takes about 8 minutes.

## Before the demo (do this 10 min early)

```bash
cd IR
streamlit run app.py
```
- The first launch after a reboot takes about 20 seconds to load the model. After that every search is instant.
- Open http://localhost:8501 and keep these tabs ready:
  1. `?q=Do%20Cholesterol%20Statin%20Drugs%20Cause%20Breast%20Cancer%3F&mode=compare`
  2. `?q=kohlrabi&mode=compare`
  3. `?q=Vitamin%20D%3A%20Shedding%20some%20light%20on%20the%20new%20recommendations&mode=compare`
  4. `?q=heart%20attack%20prevention&mode=bm25&exp=emb`
- Keep a terminal open in the project folder, plus `report/IR_Report_NFCorpus.pdf`.

---

## 1. Problem and dataset (1 min). **Dataset** tab
- "This is medical search. The queries come from NutritionFacts.org and are written by ordinary people, but the documents are 3,633 PubMed abstracts written by researchers. The two groups use different words for the same thing: *heart attack* vs *myocardial infarction*."
- Point at the stats: 323 test queries, about 38 relevant docs per query, graded 1 or 2.
- **Inverted index lookup:** type `statin`. "This is our own index. The stem appears in N documents, and here are the postings with their term frequencies."

## 2. Pipeline (1 min). Show the code or Fig. 3 in the report
- `python -m src.preprocess`: show the query and a document before and after cleaning, tokenizing, stopword removal and stemming. Point out that `cholesterol-rich` is kept whole and also split into its parts.
- "TF-IDF and BM25 are written from scratch on this index (`src/rankers.py`). The dense model is MiniLM, and RRF fuses the three ranked lists."

## 3. Compare the 4 models (3 min). **Search** tab, Compare all 4
1. **Statin query (tab 1).** All four models do well. Point at the ✅ marks and the per-query metrics table.
2. **"kohlrabi" (tab 2).** BM25 = 1.0 and Dense = 0.0. "The dense model has never really learned this rare vegetable, so it drifts to other vegetables. BM25 matches the exact word. RRF keeps the 1.0."
3. **Vitamin D title (tab 3).** Dense = 0.82 and BM25 = 0.15. "BM25 matches on words like *light* and *new*. The dense model understands the meaning. RRF gets 0.70."
4. Punchline: "Each model wins on exactly 108 queries, and their errors are different. That's why fusion works."

## 4. Medical term expansion (1 min). Tab 4
- With Embedding expansion turned on for "heart attack prevention", it adds **cardiac, myocardial, cardiovascular**, shown highlighted in blue in the results.
- Switch to PRF and show that the added terms come from the top-ranked documents.
- "We check each candidate word against the whole query, so *attack* doesn't pull in *battle*."

## 5. Results (2 min). **Evaluation** tab
- Table: RRF improves nDCG@10 from 0.325 to 0.352 (+8%) and MAP from 0.150 to 0.179 (+20%). The best system is RRF with PRF at 0.361.
- Sanity check: "Our BM25 gives 0.325, which is exactly the published BEIR number, so the implementation is correct."
- P-R curve: RRF is above every single model at every recall level.
- Be honest about one result: "Dense alone has slightly higher Recall@100 than RRF. Two of the three lists RRF fuses are lexical."
- k sweep: the results hardly change between k = 10 and 200, so RRF is robust.

## 6. Close (30 s)
"Hybrid fusion is cheap (under 20 ms per query on a CPU), needs no training, and is the safest choice when you don't know what kind of query is coming. Next steps would be a biomedical encoder, a cross-encoder re-ranker, and MeSH-based expansion."

---

## Likely viva questions

**Why does RRF not need score normalization?**
It only uses ranks: 1/(k + rank). BM25 scores are unbounded (5–20 here) and cosine similarity is roughly 0–1, so adding the raw scores would let BM25 dominate. Ranks are on the same scale for every model.

**What does k = 60 do?**
It smooths the scores. With a small k, the #1 document gets a huge bonus over #2. With k = 60, the difference between ranks 1 and 10 is modest, so documents that appear fairly high in *several* lists win. Our sweep showed that k barely matters (nDCG@10 stays between 0.350 and 0.352).

**BM25 vs TF-IDF?**
BM25 adds two things. (1) Term-frequency saturation (k1): the 10th occurrence adds much less than the 1st. (2) Length normalization (b): long documents don't win just because they're long. TF-IDF with log-tf only partly saturates and uses cosine normalization instead. On NFCorpus the gap is small because the abstracts have similar lengths.

**Why not stem the text for the dense model?**
The transformer was trained on natural sentences. Stems like "caus" or "diseas" aren't real words to it, and the WordPiece tokenizer would split them badly. The dense model has its own learned notion of word similarity.

**What is nDCG and why use the graded version?**
DCG adds up gain/log2(rank+1), so relevant documents near the top count more. Dividing by the ideal DCG gives a number between 0 and 1. NFCorpus has grades 1 and 2, and with gain 2^rel − 1 a highly relevant document is worth 3× a partly relevant one.

**Why are the absolute scores low (MAP 0.18)?**
Each query has about 38 relevant documents, and many queries are one word. 98 queries get nDCG@10 = 0 even with BM25. These scores are typical for NFCorpus: the published BEIR numbers are in the same range.

**What is PRF and why can it hurt?**
It assumes the top 10 BM25 results are relevant and adds their most important terms to the query. When those top results are off-topic, the added terms make it worse: this is called query drift. "shelf life" dropped from 1.00 to 0.63, but overall PRF helped 111 queries and hurt 71.

**Why didn't embedding expansion help inside RRF?**
It overlaps with the dense model, which already handles synonyms. PRF adds terms taken from the corpus itself, such as *adenovirus-36*, which a dense model wouldn't suggest. That's why PRF adds something new to the fusion.

**How is the inverted index built?**
One pass over the corpus. Each document is preprocessed into tokens, the term frequencies are counted, and each `(doc, tf)` pair is appended to that term's postings list. We also store df and the document lengths. At query time we only touch the postings of the query terms.

**What are the limitations?**
- Only one dataset.
- MiniLM is a general-purpose model that wasn't fine-tuned on medical text, and it truncates input at 256 word pieces.
- RRF gives every model an equal vote: on "low-carb diets", two weak lexical lists outvoted the strong dense list.
- No tuning on the dev split. We used standard parameters to avoid overfitting the test set.
