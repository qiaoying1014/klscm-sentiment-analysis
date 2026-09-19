# Final Model-Selection Comparison

## Status

Recommendation: **recommend keep frozen c1**. ABSA remains blocked. No model rerun, API call, or record-level annotation is part of this review.

The compact evidence-driven review contains 28 refined topics; 28 are complete.

## Fixed objective comparison

| Evidence | Frozen c1 | c1-refined |
|---|---:|---:|
| Clustered coverage | 58.50% | 52.62% |
| Outliers | 41.50% | 47.38% |
| Topics | 40 | 44 |
| Additional outliers | — | 808 |

Frozen c1 has a completed full review: 24 coherent, 10 somewhat mixed, six highly mixed; 27 substantive topics cover 7,112 clustered documents.

## Compact comparative review

Interpretability counts are improved 15, similar 7, and worse 6. Refined-topic quality counts are coherent 14, somewhat mixed 8, and highly mixed 6.

## Conservative decision rule

c1-refined is recommended only after complete review when improved topics cover at least 60% of selected-topic documents, improved exceeds worse by at least 30 percentage points, at most 10% of selected topics mix unrelated themes, and at least 40% usefully merge, split, or recover outliers. Otherwise frozen c1 is recommended. The rule explicitly weighs interpretability against 808 additional outliers, lower coverage, and four additional topics; it never favors the newer model automatically.
