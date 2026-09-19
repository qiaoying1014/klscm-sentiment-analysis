# Relevance v8 threshold selection report

## Frozen inputs and independent verification

Phase 8 used only the immutable adjudicated 300-record calibration artifact and saved model assessments. No classifier or API was run. Event connection, content type, confidence, model disagreement, image dependence, reason codes, human gold, material-disagreement routing, weak meaningful-content routing, and the deterministic no-content exception were held invariant.

Changing 0.80/0.93 to 0.80/0.99 changes exactly **3** records, all from automatic exclusion to review because their exclusion confidence is below 0.99. Two are gold include and one is gold exclude. No automatic inclusion and no other routing state changes.

## Candidate comparison

The complete focused comparison is in `data\processed\relevance_v8_calibration_results\threshold_selection\relevance_v8_threshold_selection_v1.csv`. At the selected 0.80/0.99 point: automatic decisions=259, coverage=86.33%, automatic include/exclude=255/4, review=41 (13.67%), inclusion precision=0.9373, inclusion recall=1.0000, exclusion precision=1.0000, false inclusions/exclusions=16/0, and automatic macro-F1=0.6505.

## Macro-F1 interpretation

The automatic confusion matrix at 0.80/0.99 is `[[239, 0], [16, 4]]` (gold rows, prediction columns; include then exclude). Automatic gold support is 239 include and 20 exclude. Excluded-class recall is 0.2000 and excluded-class F1 is 0.3333; included-class F1 is 0.9676. Their unweighted mean is 0.6505, independently reproduced from the confusion matrix and verified against sklearn.

The low macro-F1 is not an implementation error. It primarily reflects conservative routing and weak exclusion discrimination: only 4 records are predicted exclude, while 16 automatic includes are gold exclude. Class imbalance and reduced automatic excluded support contribute, but perfect exclusion precision alone cannot supply high excluded-class recall.

## False inclusions and exclusion safety

All 16 false inclusions were inspected. Category counts are `{'policy-boundary case': 8, 'short-caption ambiguity': 3, 'hashtag-supported over-inclusion': 3, 'event-information boundary': 2}`. They concentrate in short-caption, hashtag-supported, event-information, and policy-boundary cases. This is a systematic boundary limitation, but it does not justify prompt revision before a blind holdout: the primary exploratory-topic safety objective concerns irreversible false exclusion, inclusion precision remains 93.73%, and the errors are fully enumerated for prospective validation.

All 4 automatic exclusions under 0.80/0.99 are gold exclude. This calibration safety result must be confirmed on an untouched holdout.

## Operational cost

Moving to 0.80/0.99 adds 3 reviews per 300, 10.00 per 1,000, and a projected 140.00 per 14,000 records. Under the calibration distribution, 85.00% are automatically included, 1.33% automatically excluded, and 13.67% reviewed. The 14,000-record value is a workload projection, not a claim that calibration prevalence generalizes.

## Selection and target status

**FREEZE 0.80 / 0.99** as `v8_op1`.

- inclusion_precision_target_met: `true`
- inclusion_recall_target_met: `true`
- automatic_macro_f1_target_met: `false`
- false_exclusion_safety_target_met: `true`
- operating_point_accepted: `true`

Acceptance does not redefine the macro-F1 target. Exploratory topic discovery prioritizes preservation of relevant content; uncertain exclusions are intentionally routed to review; macro-F1 remains transparent; and generalization must be tested on an untouched blind holdout.

# READY TO FREEZE FOR BLIND HOLDOUT

Gold, prompts, classifier, routing, and operating thresholds are resolved and frozen; remaining false inclusions are understood, exclusion safety is acceptable on calibration, the additional review burden is small, and limitations are documented. This phase does not create or run the holdout.
