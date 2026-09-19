# KLSCM v8 formal human adjudication methodology

Workflow version: `v8_gold_adjudication_v1`  
Policy version: `v8_event_experience_binary`

## Purpose

This workflow creates an append-only, thesis-grade v8 gold-label overlay for the 54 records whose Phase 4 policy-audit proposal disagreed with the historical v7 label. It does not change classifier code, prompts, thresholds, routing, schemas, model outputs, historical labels, or calibration artifacts.

## Blinding and independence

The package generator reads the Phase 4 candidate audit but exports only:

- `record_id`
- caption
- recorded language
- event year
- blank reviewer fields

It hides the historical label, model prediction, confidence, explanation, routing, reason code, event connection, and model content type. Rows are reproducibly randomized with seed `8042`. The two reviewers receive separate physical CSV files and must not exchange files or decisions before both snapshots are imported.

Each reviewer records exactly one `include` or `exclude` decision plus an evidence span, rationale, confidence from 0 to 1, and optional comments. Reviewers answer only: “Under the official v8 relevance policy, should this caption be included in KLSCM topic discovery based on the available caption text and permitted metadata?”

## Files and provenance

`data/processed/relevance_v8_adjudication/` contains:

- `relevance_v8_blinded_annotation.csv`: frozen blank template.
- `relevance_v8_reviewer_a.csv`: reviewer A working copy.
- `relevance_v8_reviewer_b.csv`: reviewer B working copy.
- `adjudication_manifest.json`: workflow version, seed, source path/hash, output hashes, hidden-field declaration, and creation timestamp.
- `reviewer_imports/`: immutable completed-review snapshots and sidecar metadata.
- `agreement/`: statistics, confusion matrix, disagreement data, and adjudicator worksheet after two imports.
- `gold/`: immutable gold versions and provenance manifests after complete adjudication.

Every imported row stores reviewer identity, reviewer-file version, UTC import timestamp, original file path, and SHA-256 hash. Re-importing the same reviewer/version is refused. A gold-version filename is also non-overwriting.

## Reviewer workflow

1. Give `relevance_v8_reviewer_a.csv` to reviewer A and `relevance_v8_reviewer_b.csv` to reviewer B separately.
2. Each reviewer completes all labels, evidence spans, rationales, and confidence values without consulting the other reviewer or any model output.
3. Import reviewer A:

```powershell
.\.venv\Scripts\python.exe -m marathon_absa.cli relevance-v8-adjudication-import --file data/processed/relevance_v8_adjudication/relevance_v8_reviewer_a.csv --reviewer-id REVIEWER_A --reviewer-version v1
```

4. Import reviewer B with a different identity:

```powershell
.\.venv\Scripts\python.exe -m marathon_absa.cli relevance-v8-adjudication-import --file data/processed/relevance_v8_adjudication/relevance_v8_reviewer_b.csv --reviewer-id REVIEWER_B --reviewer-version v1
```

5. Run agreement analysis using the exact imported snapshot paths:

```powershell
.\.venv\Scripts\python.exe -m marathon_absa.cli relevance-v8-adjudication-agreement --reviewer-a data/processed/relevance_v8_adjudication/reviewer_imports/REVIEWER_A__v1.csv --reviewer-b data/processed/relevance_v8_adjudication/reviewer_imports/REVIEWER_B__v1.csv
```

6. Report Cohen’s kappa, percent agreement, and the 2×2 confusion matrix overall. Inspect subgroup agreement by language, event connection, and primary content type. Event connection and content type are joined only after reviewer decisions are frozen; reviewers never see them.
7. Give only `agreement/relevance_v8_adjudication_report.csv` to a qualified adjudicator. It contains disagreements, both decisions, evidence, rationales, comments, and blank adjudication fields.
8. The adjudicator completes `adjudicated_v8_label`, `adjudicator`, `adjudication_rationale`, and `adjudicated_at_utc` for every disagreement. Original reviewer judgments remain unchanged.
9. Finalize a new gold version:

```powershell
.\.venv\Scripts\python.exe -m marathon_absa.cli relevance-v8-adjudication-finalize --reviewer-a data/processed/relevance_v8_adjudication/reviewer_imports/REVIEWER_A__v1.csv --reviewer-b data/processed/relevance_v8_adjudication/reviewer_imports/REVIEWER_B__v1.csv --adjudication-report data/processed/relevance_v8_adjudication/agreement/relevance_v8_adjudication_report.csv --gold-version v1
```

Finalization is blocked if reviewer identities are not independent, record sets differ, any disagreement lacks a valid decision, or adjudicator provenance is incomplete.

## Agreement outputs

After both imports, the workflow produces:

- `agreement_statistics.csv`: overall and subgroup `n`, percent agreement, and Cohen’s kappa.
- `agreement_confusion_matrix.csv`: reviewer A rows versus reviewer B columns, ordered include/exclude.
- `reviewer_disagreements.csv`: full provenance-rich disagreement table.
- `relevance_v8_adjudication_report.csv`: disagreements only, with blank adjudicator fields.
- `agreement_summary.json`: identities, record count, agreement count/rate, and overall kappa.

Undefined subgroup kappa should be reported as unavailable when a subgroup contains insufficient label variation; percent agreement and group size remain interpretable.

## Final gold data

The final append-only dataset contains:

- historical v7 label
- reviewer 1 and reviewer 2 labels
- final adjudicated v8 label
- reviewer identities and versions
- adjudicator, UTC timestamp, and rationale
- both evidence spans, rationales, and comments
- both original reviewer file paths and hashes

Reviewer consensus becomes the final label for agreements. Only disagreements use the adjudicator’s label. The companion manifest hashes every source and the final gold file.

## Use in the next calibration cycle

The completed 54-record v8 gold file is a versioned overlay, not a destructive rewrite. The next calibration cycle should join `adjudicated_v8_label` by `record_id` onto a copy of the 300-record calibration key while retaining historical labels in separate columns. Only after this join is validated should binary classifier, routing, subgroup, and threshold metrics be recomputed; the classifier must not be tuned against labels that remain unresolved.
