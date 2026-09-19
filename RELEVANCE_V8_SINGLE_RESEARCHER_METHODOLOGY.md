# KLSCM v8 Single-Researcher Relevance Methodology

## Purpose

Relevance filtering is a conservative preprocessing gate for exploratory topic discovery and ABSA, not a standalone production-classifier study. Its priority is to avoid irreversible removal of genuine KLSCM content while removing clearly irrelevant material.

## Construct and automated stage

The unchanged `v8_event_experience_binary` construct includes defensible KLSCM-linked participation, preparation, registration, logistics, race-kit/expo, route, aid stations, organization, weather, support, atmosphere, physical or emotional experience, results, achievement, safety, facilities, and event information. Generic running, another event as the subject, pure promotion, spam, incidental KLSCM mentions, hashtag stuffing, hashtag-only records, and records without a defensible KLSCM connection are excluded. `REVIEW` is routing, never a third gold class.

`v8_op1` freezes the v8 prompt, assessment schema, routing architecture, examples, and adjudicated development gold. It uses inclusion threshold 0.80 and exclusion threshold 0.99. Safe cases become automatic `include` or `exclude`; weak, image-dependent, contradictory, conflicting, or otherwise unsafe cases become `review`. Hashtag-only deterministic exclusions remain auditable and bypass paid inference. The paid production command refuses overwrite, retains provenance, and does not unlock topics.

## Human review

One researcher resolves every operational `review` row to `include` or `exclude` under the unchanged guideline. The interface requires a rationale, requires original-caption evidence for inclusion, autosaves atomically, creates one UTC session backup, resumes incomplete work, and offers optional cache-backed translation. Translation never replaces original evidence and may cost money only when deliberately requested on a cache miss.

## Random audit

After automatic decisions exist, seed **104729** draws 75 automatic includes and 75 automatic excludes. This differs from Phase 9 seeds 91827–91829. Sampling is proportional within each decision arm across source, event year, independent language metadata and text-length group where available. Gold labels and known errors are not inputs. Review-routed cases are excluded. The blind CSV hides the arm, confidence, routing, model output, historical labels and prior decisions; the hidden key preserves them and row hashes.

The balanced design directly estimates confirmation within each decision arm. Its unweighted combined accuracy does not estimate corpus prevalence. Include recall is reported only as a design-weighted estimate using recorded population/sample counts for each sampling stratum.

## Predeclared acceptance criteria

Before labels are inspected:

1. automatic-include confirmation is at least 0.90;
2. automatic-exclude confirmation is at least 0.90;
3. zero observed false exclusions passes the conservative safety check; any observed false exclusion triggers diagnostic and methodological review rather than a post-hoc tolerance;
4. every operational review row is manually resolved; and
5. no unresolved row can enter the final topic gate.

The historical macro-F1 development target remains documented but is not mandatory for this preprocessing audit. Failure creates an error audit and stops; it does not automatically initiate v9 tuning.

For `v8_single_researcher_cost_conservative_v1`, the versioned audit rule permits at most two false exclusions among the 75 audited automatic exclusions, while retaining both 0.90 confirmation thresholds. Two of 75 is approximately 2.67%; it is a sample acceptance limit, not an estimate of corpus-wide false-exclusion prevalence. Every observed false exclusion requires qualitative inspection. The earlier zero-observed-false-exclusion protocol remains historical and unchanged for its original workflow.

## Limitations

There is one researcher, so no independent inter-rater reliability estimate is available and no Cohen's kappa is claimed. Decisions remain subjective. The audit has sampling uncertainty and small subgroups may be unstable. API/model drift may change future outputs even with stable prompts, so model, prompt, schema, thresholds, timestamps and hashes are retained. Relevance errors can bias later topic and sentiment findings, alongside social-media sampling and representation biases.

## Reproducibility

Workflow: `v8_single_researcher_human_in_loop_v1`; operating point: `v8_op1`; thresholds: 0.80/0.99; seed: 104729. `relevance-v8-production --run-api` creates routes. `relevance-v8-single-audit-create` creates the blind audit, hidden key, separate operational queue/key and UTC/hash manifest. `relevance-v8-single-audit-finalize` validates identity/text/metadata and writes non-overwriting metrics, errors and provenance. Phase 9 status is preserved in `data/processed/relevance_v8_holdout/phase9_protocol_supersession_v1.json`.
