# Relevance v8 final adjudication report

## Method and completion

Two blinded reviewers independently labeled the 54 disagreement-enriched records. They agreed on 23 and disagreed on 31. Supervisor `NBH` adjudicated every disagreement. Final labels use reviewer consensus where available and supervisor adjudication otherwise; model outputs and historical labels played no role in resolution.

Initial percent agreement remains **42.59%** and Cohen's kappa remains **0.1180**. This targeted sample was enriched for difficult historical-policy/model disagreements and is not a random representative sample; kappa must not be reported as corpus-wide inter-annotator reliability. It nevertheless demonstrates substantial ambiguity and different inclusion boundaries among difficult cases.

## Final results

- Gold version: `relevance_v8_adjudicated_gold_v1`
- Consensus: 23
- Supervisor adjudicated: 31
- Final include/exclude: 34 / 20
- Historical exclude to include: 33
- Historical include to exclude: 3
- Historical unchanged: 18
- Supervisor selected Reviewer A / B: 13 / 18
- Supervisor disagreement labels include/exclude: 18 / 13

Reviewer alignment is descriptive and is not evidence that either reviewer was poor.

## Status and provenance

- Initial annotation agreement status: low on a disagreement-enriched sample
- Adjudication completion status: complete (31/31)
- Gold-standard resolution status: complete and versioned

The final CSV preserves historical labels, both reviewer decisions and evidence, supervisor provenance for disagreements, resolution method, timestamps, and source-manifest hash. Consensus rows intentionally have blank supervisor fields. Hashes are recorded in `data\processed\relevance_v8_adjudication\final\relevance_v8_adjudicated_gold_v1.manifest.json`.

## Limitations

The 54 records are a purposive difficult-case subset. Adjudication resolves these records but does not estimate corpus-wide reliability or replace validation on a fresh blind holdout.
