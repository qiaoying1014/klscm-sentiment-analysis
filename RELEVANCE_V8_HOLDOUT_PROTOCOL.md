# Relevance v8 blind holdout protocol

## Frozen design

Holdout version: `relevance_v8_blind_holdout_v1`. Sample size 300 was declared before sampling because the eligible corpus is substantially larger, 300 matches the calibration scale, permits meaningful binary precision/recall estimates under expected imbalance, and remains feasible for two independent reviewers.

Membership was drawn with seed `91827` by proportional stratified probability sampling over source, event year, and independently generated OpenLID language metadata. Rare languages were pooled for allocation. No relevance prediction, confidence, routing, event connection, content type, historical label, or difficulty judgment was used.

All identifiable calibration, validation, audit, adjudication, review, policy-alignment, previous holdout, threshold-error inspection, and repeated-record IDs were excluded. Exact `text_hash` duplicates, project-marked duplicates, and conservative normalized near duplicates (case/punctuation/URL/hashtag/whitespace variation) were removed against development records and within the eligible pool.

## Annotation

Use the unchanged `RELEVANCE_V8_ANNOTATION_GUIDELINE.md`. Reviewer A and Reviewer B work independently from separately randomized files. Permitted fields are holdout ID, caption, source/year metadata, and independently generated language. Labels are only `include` or `exclude`, with evidence span, rationale, confidence, and optional comments. Reviewers must not inspect one another's decisions or any classifier output.

After both reviews: validate submissions; compute agreement; route only disagreements to a qualified supervisor; finalize immutable holdout gold. Do not score before gold finalization.

## Leakage and temporal rule

Normal v8 scoring rejects these source IDs. A future dedicated evaluation command may operate only after `gold_finalized_at` exists. Required order: `holdout_created_at < gold_finalized_at < classifier_first_scored_at`.

## Predeclared evaluation

Metrics and unchanged targets are frozen in `data\processed\relevance_v8_holdout\relevance_v8_holdout_predeclared_metrics_v1.json`. Automatic-only, routing, simulated human-assisted, and annotation-reliability views remain separate. Targets remain inclusion precision >=0.85, inclusion recall >=0.90, and binary macro-F1 >=0.80, with false-exclusion safety, exclusion precision, and workload also reported.

> The first evaluation of `relevance_v8_blind_holdout_v1` is the confirmatory evaluation of frozen `v8_op1`.

If it fails, do not tune and reuse it as blind evidence. Any modification creates a new system version; this holdout becomes development evidence, and another untouched holdout is required.
