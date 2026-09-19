# ABSA V1 Validation Evidence Diagnostic and Safe Repair

## Status

Completed locally on 2026-08-19 without an API call. The validation Batch was not resubmitted, inference was not rerun, and the prompt, schema, ontology, frozen human gold, raw Batch results and imported cache entries were not changed. The first finalization failure is preserved as a methodological event: `ValueError: Every evidence_text must be an exact caption substring`.

## Diagnostic result

The diagnostic inspected all 80 imported validation documents and all 218 predicted mentions rather than stopping at the first invalid evidence string.

| Measure | Result |
|---|---:|
| Validation documents | 80 |
| Predicted mentions | 218 |
| Raw exact evidence matches | 204 (93.58%) |
| Raw mismatches | 14 (6.42%) |
| Deterministically recoverable mismatches | 1 |
| Final exact/recoverable spans | 205 (94.04%) |
| Unrecoverable grounding failures | 13 |
| Documents affected by a raw mismatch | 9 |

Raw mismatch categories are six `translated_or_paraphrased_evidence`, six `truncated_evidence`, one `whitespace_difference`, and one `emoji_alias_difference`. No model-provided offsets were present on the 14 failures. The one safe repair was a unique whitespace-normalized match. The repaired value is the corresponding original-caption substring, with final offsets; all unrecoverable values retain the raw model text in `original_model_evidence_text` and have no asserted final exact evidence.

## Repair and metric policy

Repair priority is valid and plausible model offsets, then a unique Unicode NFC/whitespace/trivial-punctuation normalized match, then a unique case-insensitive match. Offset ranges are checked. Normalized or case-insensitive recovery is rejected when more than one original span matches. Translation, paraphrase, semantic similarity, non-contiguous/truncated quotations, ambiguous matches and hallucinated evidence are not repaired.

The finalized prediction artifact retains every predicted mention. An unrecoverable evidence error does not remove the mention from aspect or sentiment evaluation: it remains eligible for aspect/sentiment matching where its controlled fields are valid, and it counts as a grounding failure. Consequently, the 13 unrecoverable mentions remain inside the 218-prediction denominator and contribute to false positives when their aspects do not match gold. Evidence metrics report both strict raw-model grounding (204/218) and final recoverable exact-span grounding (205/218); the repair is never reclassified as raw success.

## First frozen evaluation metrics

- Aspect detection: precision 0.3853, recall 0.8400, F1 0.5283 (TP 84, FP 134, FN 16).
- Matched sentiment: accuracy 0.8571; macro-F1 0.5346.
- Joint aspect-sentiment: precision 0.3303, recall 0.7200, F1 0.4528.
- Strict raw-model evidence grounding: 0.9358.
- Final recoverable exact-span grounding: 0.9404.
- Zero-mention document agreement: 0.8375.

These are the first frozen results. They were computed without changing the prompt or schema. Aspect F1 and sentiment macro-F1 are below the predeclared 0.70 usable-with-limitations band; grounding exceeds 0.90 but remains below the 0.95 strong band after safe recovery. Any decision about the single permitted controlled refinement must follow substantive error inspection rather than modifying this result.

## Integrity and artifacts

- Raw Batch SHA-256: `e5ac1d98c31de43cf616b39cf2d5370ebe774bf6c676fa81d3f44303fe8a52b1`.
- Frozen gold SHA-256: `5fbbe9dfd35cf717932866f3878205635fe6a3d206fc4d7e8dd1b071ffb6d381`.
- Frozen ontology SHA-256: `d09dc138472ab889ca60a4330639125742d9d52fcc5de1c609ac0acdc197c994`.
- Complete mention diagnostic: `data/processed/absa_v1/absa_validation_evidence_diagnostic_v1.csv`.
- Raw-failure/error subset: `data/processed/absa_v1/absa_validation_evidence_errors_v1.csv`.
- Diagnostic summary: `data/processed/absa_v1/absa_validation_evidence_diagnostic_summary_v1.json`.
- Safely finalized predictions: `data/processed/absa_v1/absa_validation_predictions_v1.csv`.
- Validation metrics: `data/processed/absa_v1/absa_validation_metrics_v1.json`.
- Idempotence and integrity manifest: `data/processed/absa_v1/absa_validation_finalization_manifest_v1.json`.

The finalizer is idempotent for the same raw Batch hash and refuses partial artifacts or a conflicting source hash. After adding the standalone evaluator, complete repository verification passed 230 tests with four unchanged sklearn warnings.

## Standalone frozen-gold evaluation command

`absa-v1-validation-evaluate` was added as a strictly local evaluation command. It reads the existing frozen 80-document sample/gold and finalized predictions, preserves the one-to-one document+aspect matching rule, and records input hashes. It writes `absa_validation_evaluation_v1.json`, `absa_validation_evaluation_summary_v1.csv`, and `absa_validation_evaluation_subgroups_v1.csv`. It is idempotent for identical inputs and refuses partial artifacts or changed source hashes.

The versioned evaluation records 80 total gold documents, 56 with mentions, 24 zero-mention documents and 100 gold mentions; predictions cover 65 documents with mentions, 15 predicted-zero documents and 218 mentions. Zero-mention confusion (gold rows, predicted columns) is 13 both zero, 11 gold-zero/predicted-mentions, two gold-mentions/predicted-zero and 54 both with mentions. Language-specific metrics are reported only at five or more documents: English (42), Malay (16), Chinese (5), Malay/Indonesian uncertain (6), and the combined non-English/uncertain group (38). Indonesian, insufficient-text and undetermined groups are retained with counts but suppressed as under five documents.
