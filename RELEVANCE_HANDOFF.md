# Relevance Calibration Handoff

Updated: 2026-07-25 (Asia/Kuala_Lumpur)

## Current objective

Validate and calibrate the multilingual Instagram relevance classifier before
full-dataset classification or topic discovery. Do not lower quality targets or
run the full paid pipeline until the relevance gate is defensible.

## Current production state

- `data/processed/relevance.csv`: 1,182 classified records.
- Latest observed decisions: 821 relevant, 354 irrelevant, 7 ambiguous.
- Review status: 910 auto-resolved, 265 human-resolved, 7 pending.
- Existing human decisions must be preserved on reruns.
- The ordinary review queue is
  `data/processed/relevance_review_queue.csv`.
- The Streamlit labeler is `relevance_labeler.py`.

## Human-label artifacts

- Completed original blind labels:
  `data/processed/relevance_validation_sample_before_recalibration.csv`
- Matching original validation key:
  `data/processed/relevance_validation_key_before_recalibration.csv`
- A later replacement sample,
  `data/processed/relevance_validation_sample.csv`, is only partially labeled
  and should not be treated as complete gold data.
- The original sample has 300 primary records plus repeated records for
  intra-reviewer agreement.

## Validation history

### v2 baseline

- Holdout n: 150
- Relevant precision: 0.875
- Relevant recall: 0.5233644859813084
- Macro-F1: 0.36220113776504753
- Intra-reviewer Cohen's kappa: 0.6414342629482073
- Passed: false

The main v2 false negatives were `generic_running`, `hashtag_only`,
`weak_event_connection`, and `no_event_connection`.

### v3 recalibration

The v3 prompt treats a KLSCM identifier as event context when paired with an
event-journey cue and forces calibration-sensitive exclusion reasons through
stronger-model adjudication. Cache versions were advanced to:

- `relevance-v3`
- `relevance-adjudication-v3`
- `relevance-rule-v3`

Isolated v3 validation output:
`data/processed/relevance_v3_validation/`

Metrics:

- Holdout n: 150
- Relevant precision: 0.7591240875912408
- Relevant recall: 0.9719626168224299
- Macro-F1: 0.3158990372105126
- Intra-reviewer Cohen's kappa: 0.6414342629482073
- Passed: false

Holdout confusion matrix:

| Human label | Predicted relevant | Predicted ambiguous | Predicted irrelevant |
|---|---:|---:|---:|
| Relevant | 104 | 2 | 1 |
| Ambiguous | 4 | 0 | 0 |
| Irrelevant | 29 | 8 | 2 |

V3 recovered recall but overpredicted relevant. False positives were concentrated
in `event_preparation`, `event_participation`,
`event_experience_evaluation`, and `event_result_achievement`.

## Important annotation concern

Do not blindly tune against every current human label. Inspection found records
explicitly describing KLSCM preparation, participation, completion, results, or
experience that were labeled `irrelevant`, even though the documented policy
defines those categories as relevant. Other false positives are legitimate,
including:

- another event as the actual subject with an incidental KLSCM hashtag;
- commercial/product promotion around KLSCM;
- vague captions or photo posts with insufficient semantic connection;
- generic running content where KLSCM is only incidental metadata.

Reviewer agreement is moderate rather than strong (kappa 0.641), so the next
step must distinguish model false positives from inconsistent gold labels.

## Code changes already made

- `marathon_absa/openai_service.py`: calibrated v3 relevance and adjudication
  instructions.
- `marathon_absa/relevance.py`: mandatory adjudication for
  `generic_running`, `no_event_connection`, `weak_event_connection`,
  `hashtag_only`, and `conflicting_evidence`.
- `marathon_absa/pipeline.py`: reusable exact-document scorer and isolated
  validation rescoring.
- `marathon_absa/cli.py`: added:

  ```powershell
  python -m marathon_absa.cli relevance-rescore-validation --run-api
  ```

- `relevance_labeler.py`: defaults to the operational review queue and supports
  selecting another CSV with `RELEVANCE_LABEL_CSV`.
- Full test suite result after these changes: 55 passed.

## Required next work

1. Audit the annotation policy and current labels using only the calibration
   split first. Do not tune on the holdout.
2. Create a targeted disagreement report/queue containing calibration records
   where v3 predicts relevant and the human label is irrelevant or ambiguous.
3. Separate:
   - clear human-label inconsistencies;
   - incidental hashtag/another-event cases;
   - commercial promotion;
   - genuinely weak or image-dependent cases.
4. Present the disputed cases with a concise, explicit annotation rubric.
   Minimize further manual review; do not ask the user to relabel the full
   sample.
5. Correct labels only with user-confirmed policy or adjudication. Preserve an
   audit trail of original label, revised label, reason, and timestamp.
6. Calibrate v4 on the calibration split only. Likely policy refinements:
   - a KLSCM identifier plus a cue is relevant only when the cue is explicitly
     about KLSCM;
   - another named event as the main subject remains irrelevant;
   - product calls-to-action and pure promotion remain irrelevant;
   - vague text/emoji/photo captions should not become relevant merely because
     of a KLSCM hashtag;
   - explicit KLSCM preparation, participation, result, completion, experience,
     logistics, information, or support remains relevant.
7. Increment all prompt/cache versions to v4.
8. Add regression tests and run the complete pytest suite.
9. Rescore validation into a new isolated `relevance_v4_validation/` directory.
10. Do not overwrite production relevance outputs during validation.
11. Because the original holdout has now been inspected, clearly disclose that
    it is no longer a pristine final holdout. For a defensible final claim,
    prepare a fresh independent holdout after the rubric and v4 classifier are
    frozen.

## Quality gate

Current code requires:

- relevant precision >= 0.85
- relevant recall >= 0.90
- macro-F1 >= 0.80

Do not lower these thresholds merely to obtain a pass. If ternary macro-F1 is
methodologically inappropriate because `ambiguous` is a routing state rather
than a final substantive class, analyze and document that issue separately
before proposing any metric change.

## Safety and cost

- Commands using OpenAI require both `OPENAI_API_KEY` and `--run-api`.
- Always use a limited or exact validation set before a full run.
- Do not start duplicate relevance processes.
- Do not delete caches, labels, backups, or generated audit artifacts.
- Do not run `topics` while any production relevance record is pending or while
  the relevance validation gate remains unresolved.

## 2026-07-25 targeted calibration audit and v4 preparation

### Calibration-only annotation-policy audit

The completed archived labels were reused; no reviewer was asked to relabel the
full sample. The audit merged the 150-record calibration half with the isolated
v3 key and selected only records where v3 predicted `relevant` while the human
label was `irrelevant` or `ambiguous`. This produced 41 disputes: 36 against
human `irrelevant` and 5 against human `ambiguous`.

The auditable queue is
`data/processed/relevance_calibration_disagreement_audit_v4.csv`. It preserves
`human_relevance` unchanged and records `proposed_label`, category, reason,
timestamp, and `correction_status=proposed_not_applied`. The targeted policy
audit proposes:

- 18 clear human-label inconsistencies -> `relevant`;
- 12 incidental-hashtag or another-event cases -> `irrelevant`;
- 3 commercial-promotion cases -> `irrelevant`;
- 8 genuinely weak or image-dependent cases -> `ambiguous`.

Rubric: a journey cue is relevant only when the available text explicitly
attributes it to KLSCM. Another named event/activity as the subject, a broad
hashtag list, and pure product/photo-sales calls-to-action are irrelevant.
Vague captions whose meaning depends on an unseen image are ambiguous. Explicit
KLSCM preparation, registration, logistics, information, participation,
support, completion, result, achievement, or experience remains relevant even
when short. Proposed corrections have not been applied; original labels and all
production relevance outputs remain unchanged.

### v4 implementation prepared (not API-executed)

`marathon_absa/openai_service.py` now encodes the rubric above for both passes.
`marathon_absa/pipeline.py` uses cache/prompt versions `relevance-v4`,
`relevance-adjudication-v4`, and `relevance-rule-v4`. Validation rescoring now
defaults to the calibration split, accepts only `calibration` or `holdout`, and
writes split-specific isolated output (`relevance_v4_calibration/` for the next
run). `validation_report` accepts an explicit evaluation split and names its
sample-count metric accordingly. `marathon_absa/cli.py` exposes `--split`, with
`calibration` as the default. No v4 metrics exist yet because no paid call was
made.

Regression coverage checks the explicit-attribution/another-event/promotion
policy and split-isolated rescoring. The paid-call guard was tested: omitting
`--run-api` exited with code 2. Full verification on 2026-07-25:
`56 passed in 15.85s`.

The original holdout was already inspected during earlier work and is not a
pristine final holdout. It must not support the final quality claim. After the
rubric and v4 classifier are frozen on calibration evidence, create a fresh
independent holdout for the defensible final gate. Thresholds remain unchanged.
Do not run production relevance or topics.

### Exact next command (paid; calibration split only)

```powershell
.\.venv\Scripts\python.exe -m marathon_absa.cli relevance-rescore-validation --split calibration --run-api
```

This command requires `OPENAI_API_KEY`, makes paid OpenAI calls for the archived
calibration records only, and writes isolated v4 calibration artifacts. Review
its calibration metrics and disagreements before freezing v4; do not run the
holdout or production pipeline at this stage.
## 2026-07-25 v4 calibration execution and policy-audited analysis

The user executed the authorized calibration-only paid command. Raw v4 evidence
in `data/processed/relevance_v4_calibration/` is:

- n=150;
- relevant precision 0.8000;
- relevant recall 0.8571428571;
- ternary macro-F1 0.4875792407;
- intra-reviewer kappa 0.6414342629 (n=30);
- raw targets passed: false.

Raw confusion matrix (human rows, model columns in relevant/ambiguous/irrelevant
order): relevant 84/3/11; ambiguous 3/2/0; irrelevant 18/12/17.

A calibration-only v4 error audit found nine additional reverse policy
inconsistencies: records labeled human `relevant` whose explicit subjects were
another event, commercial promotion, lifestyle, or generic running and whose
KLSCM tag was incidental. The disagreement audit now contains 50 rows and is
`data/processed/relevance_calibration_disagreement_audit_v4.csv`. All rows retain
original labels and are marked `applied_to_policy_overlay`; the archived blind
sample itself was not edited. Of the 50 proposals, 36 actually change a label;
the remainder affirm the original label while documenting the dispute.

The overlay and metrics are:

- `data/processed/relevance_v4_calibration/relevance_policy_audited_labels.csv`;
- `relevance_policy_audited_metrics.json`;
- `relevance_policy_audited_confusion_matrix.csv`.

Policy-audited ternary results are precision 0.9523809524, recall 0.9345794393,
and macro-F1 0.6539363083. Ternary macro-F1 remains low principally because
`ambiguous` is a workflow routing state rather than a stable substantive class.
For the substantive binary analysis, gold-ambiguous records are excluded and a
predicted ambiguous is treated as not relevant/routed for review. On n=142 this
yields relevant precision 0.9615384615, relevant recall 0.9345794393, and binary
macro-F1 0.8985911835; all existing numeric targets pass under that explicitly
stated analysis.

This does not resolve the final quality gate. The passing evidence is from a
policy-audited calibration set used during development, not an independent
holdout. The ternary metric must not be silently replaced: the final protocol
must predeclare that `ambiguous` is routing, report routing/coverage separately,
and assess final relevant-vs-irrelevant decisions on a fresh independently
labeled holdout. The previously inspected holdout must not be used for the final
claim. No production relevance or topic command is authorized yet.

Final verification after recording the API artifacts: `56 passed in 17.88s`. The API run
changed artifacts, not code. There is no next paid classifier command yet. The
next implementation task is to add and test a deterministic fresh blind-holdout
builder that excludes every archived validation document and keeps its key
separate. Only after that sample is independently labeled should v4 be scored.
### Exact next command status

There is deliberately no authorized paid command at this point. Do not run the archived
holdout, production relevance, or topics. The next code task is the fresh blind-holdout
builder described above; after it is implemented and tested, its exact non-paid creation
command must be recorded here before execution.

## 2026-07-25 fresh independent holdout created

A deterministic non-paid builder is now implemented as
`relevance-fresh-holdout`. It excludes every nonblank `document_id` found in
both archived validation reviewer files, uses seed 1042 (project seed 42 plus
1000), keeps model fields out of the reviewer CSV, stores the key separately,
adds repeat rows, and refuses to overwrite any existing fresh-holdout artifact.
Regression coverage verifies exclusion, reproducibility, blind schema, split
identity, and repeats.

The command was executed once with `--size 150 --repeats 15`. Verified outputs:

- `data/processed/relevance_fresh_holdout_sample.csv`: 165 reviewer rows,
  comprising 150 base records and 15 repeats; all labels blank;
- `data/processed/relevance_fresh_holdout_key.csv` and `.parquet`: 150 unique
  base documents, all split `fresh_holdout`;
- zero document overlap with either prior validation reviewer file;
- no model relevance, confidence, or reason columns in the blind reviewer CSV;
- hidden-key sampling strata: 86 v3-production relevant, 59 irrelevant, and 5
  ambiguous. These strata are sampling metadata, not fresh v4 predictions.

Full repository verification after implementation and generation:
`57 passed in 15.23s`. Production relevance outputs were read for sampling but
not modified. No OpenAI call was made.

### Exact next command (non-paid human labeling)

```powershell
$env:RELEVANCE_LABEL_CSV='data/processed/relevance_fresh_holdout_sample.csv'; streamlit run relevance_labeler.py
```

Label the blind reviewer CSV only; do not open or use the key while labeling.
The 15 repeats measure intra-reviewer agreement. Do not run any paid v4 scoring,
production relevance, or topics until all 165 rows have valid labels. After
label completion, implement and test a fresh-holdout-specific v4 rescore command
that writes to a new isolated directory and never overwrites this reviewer file
or production relevance outputs.
## 2026-07-25 frozen v4 fresh-holdout scorer prepared

Implemented `relevance-rescore-fresh-holdout --run-api`. The command performs
all validation before the first API call: exactly 150 base rows and 15 repeats,
valid labels on all 165 rows, a 150-row `fresh_holdout` key, exact reviewer/key
review and document ID alignment, no overlap with either archived validation
sample, and complete prepared-document availability. Repeat rows are excluded
from scoring. The frozen cache identifiers remain `relevance-v4`,
`relevance-adjudication-v4`, and `relevance-rule-v4`.

The command refuses to run if `data/processed/relevance_v4_fresh_holdout/`
already exists. On success it writes only there: rescored CSV/Parquet key,
input hashes, ordinary ternary confusion/classification/subgroup artifacts, and
`relevance_fresh_holdout_metrics.json`. The final gate uses substantive binary
precision, recall, and macro-F1; gold ambiguous rows are excluded from that
binary calculation, predicted ambiguous is treated as not relevant/routed, and
gold/predicted ambiguous counts plus routing rate are reported separately.
Ternary macro-F1 remains descriptive. Repeat count, agreement rate, and kappa
are included.

Preflight evidence is stored at
`data/processed/relevance_fresh_holdout_preflight.json`:

- reviewer SHA-256: `31799c3c9f9466a84a028364cd7ed2cf62c110d87d42c8c192541cc10ba4bb4e`;
- key Parquet SHA-256: `29e4eaf736cfee867e11f5d1e33191595192fa746d7b785dbb5d667e100c8eb8`;
- key CSV SHA-256: `6c45799b2053fec9a21a811003d26771fd86c7b2c44a06d050a72f325a913e1b`;
- base labels: 34 relevant and 116 irrelevant;
- repeat agreement: 14/15 (0.9333333333); kappa 0.0 because the repeat subset
  has effectively no class prevalence variation.

The paid guard was verified: omission of `--run-api` exits with code 2. Tests
cover binary/routing metrics, repeat exclusion, isolated output, byte-level
preservation of reviewer and production relevance, and overwrite refusal.
Full suite: `59 passed in 16.07s` (four expected sklearn warnings from synthetic
single-class repeat fixtures). No API call was made and the result directory
does not yet exist.

### Exact next command (paid, one-time final holdout evaluation)

```powershell
.\.venv\Scripts\python.exe -m marathon_absa.cli relevance-rescore-fresh-holdout --run-api
```

Run this once only. Do not inspect errors and retune v4 afterward. If the
substantive binary gate fails, production relevance and topics remain blocked
and the result initiates a new development protocol requiring a different
future holdout. If it passes, document the final evidence before authorizing
production relevance classification.
## 2026-07-25 final independent v4 holdout result — gate failed

The user executed the frozen one-time command
`relevance-rescore-fresh-holdout --run-api`. Input hashes recorded by the run
exactly match the preflight hashes, confirming that neither the completed blind
labels nor hidden key changed between preflight and evaluation. Outputs are
isolated under `data/processed/relevance_v4_fresh_holdout/`. Individual errors
were not inspected and v4 was not retuned against this holdout.

Final independent metrics (n=150; 34 human relevant, 116 human irrelevant, no
gold ambiguous):

- relevant precision: 0.3469387755 (target >=0.85; failed);
- relevant recall: 1.0 (target >=0.90; passed);
- substantive binary macro-F1: 0.5670995671 (target >=0.80; failed);
- predicted ambiguous: 33, routing rate 0.22;
- descriptive ternary macro-F1: 0.2655443322;
- repeat agreement: 14/15 (0.9333333333), kappa 0.0 due extreme repeat-class
  prevalence;
- final gate: false.

Aggregate confusion matrix (human rows; model columns
relevant/ambiguous/irrelevant): relevant 34/0/0; irrelevant 64/33/19. Thus v4
retained every relevant record but substantially overclassified irrelevant
records as relevant. This is final independent evidence, not a calibration
result, and it supersedes the optimistic policy-audited calibration metrics for
release decisions.

The relevance quality gate is not resolved. Production relevance
classification, topic discovery, and ABSA remain blocked. Do not rerun this
holdout or change its labels, and do not tune v4 using its individual cases.
There is no authorized paid next command. Any continuation must be declared a
new development cycle (v5), use development/calibration data other than this
holdout, predeclare its policy and metrics, and reserve another unseen holdout
for final evaluation.
## 2026-07-25 v5 development prepared (no API execution)

V5 is a new development cycle using calibration evidence only. The spent v4
fresh holdout is excluded and its individual cases were not inspected. Every
initial relevant result now receives stronger-model adjudication, adjudicated
relevance requires confidence >=0.90, and the prompts require both actual KLSCM
subject identity and a concrete text-supported journey relationship. Generic
running/motivation, dates/distances, photo context, apparel, celebration,
hashtag proximity, another events, and promotion are explicitly insufficient.
Cache versions are `relevance-v5`, `relevance-adjudication-v5`, and
`relevance-rule-v5`; output is isolated at `relevance_v5_calibration/`.

Full suite: 60 passed in 15.37s. Paid guard: exit 2 without `--run-api`. No v5
API call has occurred and no v5 calibration directory exists.

Exact next paid command:

```powershell
.\.venv\Scripts\python.exe -m marathon_absa.cli relevance-rescore-validation --split calibration --run-api
```

Run calibration only. Do not run either old holdout, production relevance,
topics, or ABSA. Interpret raw metrics with the existing auditable policy-label
overlay before deciding whether v5 is frozen.
## 2026-07-25 policy alignment and v6 preparation

V5 is stopped. Root evidence is precision/recall oscillation plus incompatible
annotation prevalence between archived development and the independent holdout;
threshold tuning alone is insufficient.

Created the blind 60-record second-coder package:

- `relevance_policy_alignment_second_coder.csv` (share this);
- `relevance_policy_alignment_rubric.md` (share this);
- `relevance_policy_alignment_key.csv` (hidden; do not share).

The exact 20/20/10/5/5 strata were verified, all 60 required coder fields are
blank, IDs align, and no hidden label/model columns occur in the reviewer file.
V6 structured verification, binary/routing separation, weighted/bootstrap metric
utilities, and append-only policy finalization are implemented. Full suite: 65
passed in 15.63s. No API call was made; v6 calibration does not exist.

Next, obtain independent labels in the blind CSV using the rubric. Then run:

```powershell
.\.venv\Scripts\python.exe -m marathon_absa.cli relevance-policy-alignment-finalize
```

This is non-paid and will create the agreement report and disagreement
adjudication queue. Do not run v6 calibration, create a new holdout, or run
production relevance/topics/ABSA before adjudication is complete.
## 2026-07-26 policy-alignment Streamlit interface

Added `policy_alignment_labeler.py`, a dedicated blind UI for the 60-row
second-coder CSV. It collects every required field in one form, validates that
the evidence span occurs exactly in the caption, writes atomically with a
single session backup, tracks completion and advances to the next incomplete
record. It does not read or expose the hidden key.

Verification: 28 focused labeler tests passed; full suite 69 passed in 37.33s;
headless Streamlit health check returned `ok`. No API call occurred.

Launch:

```powershell
.\.venv\Scripts\python.exe -m streamlit run policy_alignment_labeler.py
```

After all 60 records are complete, stop the app and run:

```powershell
.\.venv\Scripts\python.exe -m marathon_absa.cli relevance-policy-alignment-finalize
```
## 2026-07-26 policy-labeler translation update

`policy_alignment_labeler.py` now automatically translates every non-empty
caption not recorded as English, Malay/Bahasa Melayu, Chinese or Mandarin.
Indonesian and all other languages are translated. It uses a separate cached
OpenAI translation stage and warns that the first translation may incur cost.
Failures do not block labeling. The rubric is displayed in the sidebar on every
record screen, ensuring it is continuously available without exposing the
hidden key.

Verification: 15 focused tests; full suite 80 passed in 136.47s; Streamlit
health `ok`. Launch and finalization commands are unchanged.