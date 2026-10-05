# Per-query analysis (nDCG@10, 323 NFCorpus test queries)

## Head-to-head

| Comparison | Wins | Losses | Ties |
|---|---|---|---|
| Sentence Transformer vs BM25 | 108 | 108 | 107 |
| Hybrid RRF (TF-IDF + BM25 + Dense) vs BM25 | 119 | 54 | 150 |
| Hybrid RRF (TF-IDF + BM25 + Dense) vs Sentence Transformer | 121 | 85 | 117 |
| BM25 + PRF expansion vs BM25 | 111 | 71 | 141 |
| Hybrid RRF (TF-IDF + BM25+PRF + Dense) vs Hybrid RRF (TF-IDF + BM25 + Dense) | 82 | 65 | 176 |

- RRF beats **both** BM25 and Dense on 52 queries and is worse than both on only 9.
- Queries with nDCG@10 = 0: BM25 98, Dense 99, RRF 89, RRF+PRF 89 (of 323).

## nDCG@10 by query length

| length    |   queries |   TF-IDF |   BM25 |   Sentence Transformer |   Hybrid RRF (TF-IDF + BM25 + Dense) |   Hybrid RRF (TF-IDF + BM25+PRF + Dense) |
|:----------|----------:|---------:|-------:|-----------------------:|-------------------------------------:|-----------------------------------------:|
| 1-2 words |       172 |   0.3593 | 0.3607 |                 0.3269 |                               0.3858 |                                   0.3965 |
| 3-4 words |        55 |   0.2028 | 0.2283 |                 0.2483 |                               0.249  |                                   0.2642 |
| 5+ words  |        96 |   0.3056 | 0.3174 |                 0.3454 |                               0.3495 |                                   0.3513 |

## Dense beats BM25 the most (vocabulary mismatch)

| query_id   | query                                                     |   BM25 |   Sentence Transformer |   Hybrid RRF (TF-IDF + BM25 + Dense) |
|:-----------|:----------------------------------------------------------|-------:|-----------------------:|-------------------------------------:|
| PLAIN-307  | Vitamin D: Shedding some light on the new recommendations |  0.148 |                  0.819 |                                0.698 |
| PLAIN-3026 | Vitamin C-Enriched Bacon                                  |  0     |                  0.631 |                                0.387 |
| PLAIN-3085 | The Difficulty of Arriving at a Vitamin D Recommendation  |  0     |                  0.571 |                                0.204 |
| PLAIN-1537 | low-carb diets                                            |  0.095 |                  0.662 |                                0.274 |
| PLAIN-924  | cocaine                                                   |  0.22  |                  0.782 |                                0.782 |
| PLAIN-1635 | milk                                                      |  0.442 |                  1     |                                0.934 |
| PLAIN-882  | Chernobyl                                                 |  0.359 |                  0.863 |                                0.863 |
| PLAIN-1679 | myelopathy                                                |  0     |                  0.5   |                                0.5   |

## BM25 beats Dense the most (rare exact terms)

| query_id   | query                                |   BM25 |   Sentence Transformer |   Hybrid RRF (TF-IDF + BM25 + Dense) |
|:-----------|:-------------------------------------|-------:|-----------------------:|-------------------------------------:|
| PLAIN-1473 | kohlrabi                             |  1     |                  0     |                                1     |
| PLAIN-1288 | grapes                               |  0.905 |                  0.13  |                                0.78  |
| PLAIN-2081 | shelf life                           |  1     |                  0.301 |                                1     |
| PLAIN-792  | cadaverine                           |  0.645 |                  0.13  |                                0.645 |
| PLAIN-902  | chlorophyll                          |  0.637 |                  0.139 |                                0.637 |
| PLAIN-2019 | rickets                              |  0.469 |                  0     |                                0.469 |
| PLAIN-3452 | Bowel Movement Frequency             |  0.584 |                  0.121 |                                0.335 |
| PLAIN-2620 | Phytates for the Treatment of Cancer |  0.502 |                  0.053 |                                0.449 |

## RRF beats both inputs the most (complementary evidence)

| query_id   | query                                          |   BM25 |   Sentence Transformer |   Hybrid RRF (TF-IDF + BM25 + Dense) |
|:-----------|:-----------------------------------------------|-------:|-----------------------:|-------------------------------------:|
| PLAIN-499  | African-American                               |  0.139 |                  0.176 |                                0.378 |
| PLAIN-2540 | Does Cholesterol Size Matter?                  |  0.306 |                  0.428 |                                0.617 |
| PLAIN-1601 | memory                                         |  0.581 |                  0.676 |                                0.852 |
| PLAIN-78   | What Do Meat Purge and Cola Have in Common?    |  0.039 |                  0     |                                0.206 |
| PLAIN-3191 | Is Distilled Fish Oil Toxin-Free?              |  0.376 |                  0.236 |                                0.519 |
| PLAIN-2630 | Alkylphenol Endocrine Disruptors and Allergies |  0.547 |                  0.561 |                                0.703 |
| PLAIN-3141 | Relieving Yourself of Excess Estrogen          |  0.108 |                  0.098 |                                0.245 |
| PLAIN-1363 | Hiroshima                                      |  0.22  |                  0.222 |                                0.357 |

## PRF expansion: biggest gains and losses vs BM25

| query_id   | query                                          |   BM25 |   BM25 + PRF expansion |
|:-----------|:-----------------------------------------------|-------:|-----------------------:|
| PLAIN-583  | antinutrients                                  |  0.22  |                  0.934 |
| PLAIN-593  | apnea                                          |  0.469 |                  0.968 |
| PLAIN-2311 | veal                                           |  0     |                  0.482 |
| PLAIN-561  | aneurysm                                       |  0.359 |                  0.817 |
| PLAIN-1151 | factory farming practices                      |  0.571 |                  1     |
| PLAIN-2081 | shelf life                                     |  1     |                  0.631 |
| PLAIN-371  | Any update on the scary in vitro avocado data? |  0.376 |                  0.106 |
| PLAIN-418  | Fresh fruit versus frozen--which is better?    |  0.834 |                  0.566 |
| PLAIN-3302 | Fish Fog                                       |  0.249 |                  0     |
| PLAIN-934  | coffee                                         |  0.851 |                  0.64  |
