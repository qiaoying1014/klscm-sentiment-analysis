# Topic Discovery Model Selection: c1 Provisional Baseline

## Decision status

Candidate c1 is the **provisional**, not final, baseline for human topic-level interpretation. This is not a claim that c1 is objectively best. No further tuning, merging, reduction, splitting, stopword change, tokenizer change, embedding change, or new model run is authorized before interpretation.

## Existing comparison evidence

The completed local comparison used 13,743 unique, text-eligible records and made no paid API calls.

| Candidate | Topics | Outliers | Outlier share | Median topic | Largest topic | Decision |
|---|---:|---:|---:|---:|---:|---|
| c1 | 40 | 5,703 | 41.50% | 55.5 | 1,970 | Provisional baseline pending interpretation |
| c2 | 42 | 7,135 | 51.92% | 64.5 | 1,779 | Retained as comparison evidence |
| c3 | 4 | 91 | 0.66% | 101.5 | 13,395 | Rejected: severe under-clustering |
| c4 | 4 | 62 | 0.45% | 101.5 | 13,424 | Rejected: severe under-clustering |

c3 and c4 place more than 13,000 records in one cluster, so their low outlier shares are not evidence of useful thematic resolution. c2 has two more topics but a materially higher outlier share than c1. c1 offers 40 candidate topics with the lower of the two plausible outlier shares, making it the most useful candidate to inspect first. Its largest topics may still be under-separated, and its 41.50% outlier share is substantial.

## Required interpretation

The researcher reviews topics 0–39 using both c-TF-IDF/representative examples and fixed-seed random examples. The five largest topics receive expanded diagnostics. Topic -1 is inspected separately as outlier/unassigned content and is never called irrelevant. This is interpretation of discourse clusters, not another record-level relevance-validation exercise. A substantive topic may contain noisy individual records.
