# Relevance v8 post-adjudication calibration report

## Evaluation protocol

No classifier was rerun and no API was called. The official 54-label adjudicated gold was overlaid by stable ID onto a new copy of the frozen 300-record calibration predictions. The other 246 gold labels remain unchanged. The obsolete review-to-exclude calculation is retained only in historical artifacts as `legacy_review_as_exclude`; it is not used here.

## Automatic-only classifier performance

- Decisions: 262 (87.33% coverage); include/exclude: 255 / 7
- Inclusion precision/recall: 0.9373 / 0.9917
- Exclusion precision: 0.7143
- Included/excluded F1: 0.9637 / 0.3571
- Binary macro-F1: 0.6604
- False inclusions/exclusions: 16 / 2
- Confusion matrix (gold rows, prediction columns; include, exclude): `[[239, 2], [16, 5]]`

## Routing performance

- Review: 38 (12.67%; 126.7 per 1,000)
- Relevant/irrelevant routed: 10 / 28
- False-exclusion prevention: 10
- Material disagreement/image-dependent/weak-connection reviews: 8 / 1 / 29
- Weak-empty automatic exclusions: 0

## SIMULATED HUMAN-ASSISTED PERFORMANCE

- Inclusion precision/recall: 0.9396 / 0.9920
- Included/excluded F1: 0.9651 / 0.7857
- Binary macro-F1: 0.8754
- Confusion matrix: `[[249, 2], [16, 33]]`

This view substitutes gold labels only for review-routed cases and is not classifier-only performance.

## Threshold analysis and decision

All candidates preserve the fixed routing invariants; violations are explicitly counted in the threshold table. At current 0.80/0.93, automatic inclusion precision=0.9373, inclusion recall=0.9917, macro-F1=0.6604, relevant automatically excluded=2, review rate=12.67%.

The priority-ranked grid's leading candidate is 0.65/0.99: inclusion precision=0.9373, inclusion recall=1.0000, macro-F1=0.6505, relevant automatically excluded=0, review rate=13.67%.

Recommendation: **B. Tune thresholds only.** No threshold or routing change has been implemented.

## Remaining errors

Every automatic error and every routed include/exclude is recorded in `data\processed\relevance_v8_calibration_results\relevance_v8_post_adjudication_error_analysis_v1.csv` with caption, gold, saved decision, evidence fields, category, and research interpretation. Routed cases are not counted as classifier errors. The observed automatic-error categories determine whether the remaining weakness is systematic; no model modification was made.

## Readiness statuses

- Classifier quality status: does not meet all approved automatic targets
- Annotation agreement status: low initial agreement on the disagreement-enriched subset
- Gold standard status: resolved and immutable v1
- Calibration status: complete using saved outputs
- Blind holdout readiness: NOT READY FOR BLIND HOLDOUT

# NOT READY FOR BLIND HOLDOUT

The current automatic operating point misses at least one approved target. The smallest justified next intervention is threshold tuning; do not revise routing, prompts, or policy unless threshold tuning cannot resolve the observed trade-off.
