# Relevance v8 Cost-Conservative Operational Protocol

## Purpose

Construct the final KLSCM topic-discovery corpus with controlled API cost while prioritizing protection against irreversible false exclusion. Relevance remains preprocessing for BERTopic and ABSA, not the thesis classifier contribution.

## Parent methodology

The semantic construct remains `v8_event_experience_binary`. The annotation guideline, v8 strict assessment schema, Luna initial prompt, Terra verification prompt, and thresholds 0.80/0.99 are unchanged. Historical `v8_op1` remains frozen evidence.

## Operational configuration

The additive version is `v8_single_researcher_cost_conservative_v1`. It descends from `v8_op1` development evidence but is **not `v8_op1`** because its Terra-eligibility and direct-human routing differ.

## Machine stages and Terra eligibility

Every eligible non-deterministic Instagram record receives or reuses a GPT-5.6 Luna initial assessment. Existing verification triggers remain: weak/none connection, excluded content family, confidence below 0.93, contradiction, image dependence, or model review recommendation.

A triggered record is Terra-eligible only when all three substantive-candidate conditions hold: event connection is `explicit` or `supported`; `meaningful_content_present` is true; and `primary_content_type` is in the existing included-content family. This deterministic rule does not consult gold, historical errors, or human labels.

Triggered non-candidates bypass Terra and enter unresolved human `REVIEW`. Skipping Terra can never create an automatic exclusion. Trigger reasons, skip reason, route source, initial assessment and configuration version remain auditable.

## Batch execution

Batch processing uses `/v1/responses` with the unchanged strict JSON schema. It is two-stage: create/submit/status/import Luna chunks, then derive and process only required Terra chunks. Creation is offline, excludes exact cache hits, uses deterministic custom IDs, configurable token limits, JSONL validation and hashed manifests. Submission requires `--run-api`; imports reject unknown, duplicate, missing, malformed or conflicting results. Successful results enter the existing deterministic cache identity so later synchronous work reuses them.

## Human review

Operational review is a speed-oriented binary include/exclude task; reason, evidence and notes are optional. The separate blinded 150-record quality audit remains stricter: 75 automatic includes and 75 automatic excludes, with rationale, evidence and confidence. Review-routed records never enter the automatic audit denominator.

## Acceptance criteria

- automatic-include confirmation >= 90%;
- automatic-exclude confirmation >= 90%;
- no more than 2 false exclusions among 75 audited automatic exclusions (2/75 ≈ 2.67%);
- every observed false exclusion receives qualitative inspection;
- every operational review row is resolved;
- zero unresolved records before topics.

The 2/75 statistic is not corpus-wide prevalence, and balanced-audit overall accuracy is not a prevalence estimate. Failure produces diagnostics and requires an explicit methodological decision; it does not create v9 automatically.

## Additive architecture-aligned audit v2 (2026-08-15)

The preceding 75/75 protocol is retained as historical `cost_conservative_audit_v1` evidence but is superseded for the finalized production architecture by `architecture_aligned_audit_v2`. Final production contains 10,532 automatic includes, 338 deterministic hashtag-only automatic exclusions, and 2,929 human-review routes. It contains no model-classifier automatic-exclude arm. Consequently, deterministic exclusions must not be described or evaluated as model classifier exclusions, and operational review must not enter either automatic-quality denominator.

The v2 blind random audit uses seed 104729 and two separately reported arms:

- 75 automatic includes, sampled proportionally across source/year/language/text-length strata where feasible;
- 50 of the 338 deterministic hashtag-only automatic exclusions, sampled using the same stratification method.

The deterministic sample covers approximately 14.8% of its finite population. With zero observed reversals, the simple rule-of-three upper 95% bound is approximately 6%; this is a diagnostic uncertainty statement, not corpus prevalence. The automatic-include confirmation rate and deterministic-exclusion confirmation rate are reported independently. No combined balanced-audit accuracy, model-exclusion precision, or model-exclusion false-negative rate is reported. Automatic acceptance requires both confirmation rates to be at least 0.90 and zero researcher reversals among deterministic exclusions; every reversal requires qualitative methodological review.

All 2,929 operational-review records remain a separate exhaustive annotation task. Topic unlock continues to require resolution of every operational-review row and acceptance of the v2 random audit. Package creation and finalization are local, versioned, overwrite-protected, and make no API calls.

## Limitations and reproducibility

Limitations include one researcher and no final inter-rater reliability estimate, model/API drift, pricing uncertainty, platform sampling bias, residual irrelevant content and residual false-exclusion risk. Reproducibility evidence includes seed 104729, source/output hashes, prompts, schema, models, configuration version, thresholds, Batch/file IDs, UTC timestamps, caches and human decisions.
