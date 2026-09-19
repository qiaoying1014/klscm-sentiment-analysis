# Integrated Participant Experience & Organizer Insight Synthesis

## Purpose and research boundary

This downstream stage answers: which experiences are praised, criticized or mixed in the analyzed KLSCM feedback, and what might organizers reasonably review? It does not modify frozen ABSA, BERTopic, ALTA, ontology, reviewed classifications, statistical findings, denominators or research marts. No upstream model or analysis is rerun. No API is called in Streamlit.

## Evidence audit (2026-09-17 implementation audit)

The repository is `C:/Users/user/Documents/Sentiment Analysis/1. Data Scraping`; the initiating ChatGPT project is an empty reference mirror. Existing read-only loaders and integrity checks verify 7,704 Instagram documents, 15,486 mentions, 20 aspects, 101 ALTA source clusters, 64 finalized reviewed themes across 16 aspects, 25 blog parent reviews, 56 inference chunks, 301 blog mentions, 47 supported Instagram reference themes and 35 finalized emergent blog themes. Twenty-four emergent themes have singleton parent support. The existing 390-file baseline passes. Four aspects lack Instagram theme clustering support: facilities, transport/access, race pack/expo and event information, using their existing ontology IDs.

Inputs are the reviewed-theme summary, sentiment, taxonomy and representative evidence under `data/processed/absa_v1/aspect_level_themes_review_v1`; blog ABSA mentions, reviewed assignments, parent summaries, emergent taxonomy/lineage/decisions under `data/processed/blog_analysis_v1`; finalized comparisons under `data/processed/cross_source_analysis_v1`; and ontology/display labels in the existing ABSA dashboard mart. Known-author redaction also depends on the existing raw blog author inventory. Every dependency is hashed in the new evidence manifest, alongside all existing frozen-baseline paths. These are read-only inputs.

The reviewed Instagram sentiment summary contains valid theme-document counts. Blog assignments retain mention sentiment and stable mention/parent/chunk IDs, so a deterministic join to finalized theme lineage is possible. The existing blog theme-sentiment summary drops duplicate review/theme rows and therefore retains a first mention's sentiment; this stage does not use that summary to infer a review's overall opinion. It preserves that frozen artifact unchanged and supplies assigned-mention label counts alongside separate unique parent support. No new parent-theme sentiment label is assigned.

## Integration and interpretation rules

The output has 99 theme rows: 64 unchanged Instagram reference themes and 35 unchanged finalized emergent themes. Blog matched evidence attaches only through finalized same-aspect theme IDs. Coverage is `cross_source` (47), `instagram_only` (17), or `blog_only_emergent` (35). Similar labels are never automatically merged. Source absence is not disagreement.

Instagram support is documents; blog support is unique parent reviews. Blog chunks are inference units only. Source totals remain 7,704 documents and 25 reviews, including when filtering. Sentiment tables explicitly distinguish Instagram theme documents from blog assigned mentions. Neither a combined numerator nor a combined denominator, prevalence, satisfaction score, importance score or confidence percentage is produced.

Classification is a conservative presence rule over the source-specific frozen sentiment evidence: any mixed label, or both positive and negative labels, yields Mixed / contested experience. Otherwise any positive yields Positive experience driver; otherwise any negative yields Participant pain point; otherwise Primarily descriptive / insufficient evaluative evidence. Neutral labels do not erase positive or negative evidence. This is an interpretation category, not dominance, majority opinion or statistical estimation. It may classify many strongly positive themes as mixed; the UI retains separate positive and negative summaries instead of hiding minority criticism. A mixed label alone does not establish exactly what was liked or disliked; those details require excerpts.

Descriptors use unique source support: both sources with at least two units each = Cross-source recurring evidence; both with a singleton = Cross-source evidence with limited within-source repetition; Instagram alone with at least two documents = Repeated Instagram evidence; one document = Singleton observation; emergent blogs with at least two parents = Multiple blog-review support; one parent = Singleton blog-only emergent observation. The cutoff of two means literal repetition, not a scientific sufficiency threshold. All blog singleton rows explicitly warn that multiple mentions are not independent repetition.

## Artifact schemas and lifecycle

All new outputs live under `data/processed/participant_experience_v1`. Preparation refuses an existing output directory; subsequent stage outputs use exclusive creation and refuse overwrites. A new version should use a new `--root`. Never edit a finalized release in place.

* `evidence.json`: version, timestamp, input hashes, ontology/display labels, source-specific units, verified corpus counts, model limitation and theme rows. A theme stores original ID/label/aspect/origin, coverage, separate supports, separate sentiment units/counts, deterministic classification/descriptor, scope note and evidence records. Evidence carries stable source/evidence/parent IDs, original privacy-redacted text, separately labeled existing English gloss and frozen sentiment. Instagram starts from five frozen representative records per theme; nine blank excerpts are recorded as unavailable evidence IDs, leaving 311 nonblank Instagram examples. Blogs provide all 292 retained assigned mentions; nine existing theme-comparison exclusions remain excluded. Examples are not a sample proving every supporting document's contents.
* `evidence_manifest.json`: evidence SHA-256 and input hashes.
* `requests.jsonl`, `sample_request.json`, `request_manifest.json`: one structured Responses request per theme, schema, model, prompt version/hash, evidence hash, request hash/count/bytes/character count, timestamp and ready status. Default model follows the repository's `DEFAULT_MODEL`; no price is assumed. Parent identifiers are omitted from AI input; stable evidence IDs remain.
* `submission_started.json`, `upload.json`, `submission.json`, `responses.jsonl`: optional explicit API submission/collection records. A started submission cannot be silently repeated after an ambiguous failure; inspect the recorded remote upload/batch first.
* `candidates.json`: validated model responses, response hash, request-manifest hash, model, import timestamp and pending review status. Each claim has text and evidence IDs.
* `review.json`: one entry per theme, candidate hash, editable insight, decision, reviewer and reason. Initially every row is PENDING. No AI output is auto-approved.
* `finalized.json`, `finalized_manifest.json`: complete approved insight rows and hashes of evidence, requests, candidates, reviews and final text, plus finalizer identity/time/status. Streamlit requires this manifest and rejects missing, stale, incomplete or unreviewed data.

The seven claim fields are participant_experience_summary, positive_experience_summary, negative_experience_summary, mixed_experience_summary, organizer_insight, organizer_implication and evidence_scope_note. Every substantive claim requires resolving citations; unavailable direction summaries use exactly `Insufficient evidence at this level.` with no citations. Positive/negative cited summaries must include appropriate positive/negative or mixed-label evidence. Implications require citations and start with `Organizers may consider`. Blog singleton scope notes must explicitly identify singleton evidence.

Machine checks establish structural lineage, not semantic truth. Reviewers must read each cited excerpt, verify claimed direction and scope, distinguish generic race experience from an operational issue, preserve source differences, reject invented event facts or causal/representative claims, and ensure implications are proportionate. Review reasons must record substantive checks, especially singleton, comparator-event and mixed-setting evidence. No author identities or URLs should be introduced in synthesis. AI-assisted upstream theme review is not independent human validation.

## Commands and review procedure

Run from the repository root using its virtual environment. The dedicated CLI is also available through `python -m marathon_absa.cli participant-experience ...`.

```powershell
.\.venv\Scripts\python.exe -m marathon_absa.participant_experience prepare
.\.venv\Scripts\python.exe -m marathon_absa.participant_experience verify
```

Preparation is an offline dry run producing the full request artifact and a sample. Inspect these before production approval. No API key is read or exposed during preparation. Explicit production submission, only after user approval:

```powershell
.\.venv\Scripts\python.exe -m marathon_absa.participant_experience submit --run-api
```

Configure `OPENAI_API_KEY` in the invoking environment using the existing account setup. When the batch completes:

```powershell
.\.venv\Scripts\python.exe -m marathon_absa.participant_experience collect --run-api
```

Alternatively import a downloaded batch response without any API call:

```powershell
.\.venv\Scripts\python.exe -m marathon_absa.participant_experience import --responses PATH_TO_DOWNLOADED_JSONL
```

Edit only the downstream `review.json`: check all seven claim fields against evidence, revise text/citations where needed, set each valid row's decision to APPROVE, and supply reviewer and substantive reason. Leave uncertain rows PENDING. The finalizer rejects unresolved decisions, missing evidence, stale candidate hashes and incomplete theme coverage.

```powershell
.\.venv\Scripts\python.exe -m marathon_absa.participant_experience finalize --reviewer "Reviewer name"
.\.venv\Scripts\python.exe -m streamlit run absa_dashboard.py
```

The page loads finalized data only. Before generation/review it clearly reports unavailable organizer insights rather than substituting generated examples. Test-only lifecycle fixtures never become production artifacts.

## Dashboard and limitations

The additive page presents an alphabetical executive experience overview; praise, pain points, mixed, cross-source and long-form-only views; all 20 existing aspect filters; source coverage and category filters; separate source counts/sentiment units; original participant evidence; and claim-level provenance. Existing research pages remain available. No invented top-five ranking is used. Year and language filters are omitted because this insight release cannot faithfully reclassify reviewed synthesis along those dimensions.

The analyzed social/blog corpus is not representative of all race participants; blogs are few; source absence is not disagreement; model-estimated labels have known development limitations (approximately precision 0.513, recall 0.790, carried from the frozen manifest); the extraction precision target was not met. Cross-source agreement is descriptive triangulation, not statistical confirmation. Organizer interpretation is reviewed AI-assisted text, not new research evidence or a causal conclusion. Four Instagram aspects have no clustered themes; absence of a unified theme cannot be read as absence of aspect feedback. No new inferential tests are run.

## Verification

Tests cover actual frozen counts and hashes, schema/ontology/theme/evidence validation, separate denominator units, singleton caution, deterministic classification, no API calls during preparation/runtime, immutable filters, failed/missing/partial responses, pending review, approved release and hash tampering, plus Streamlit pending and finalized fixture states. Existing loaders and full-suite regressions are run before completion. The baseline run passed 62 tests and failed one existing blog request-preparation test because it attempted a frozen-path write despite a temporary output root; the write was blocked. Its test fixture is corrected to redirect both request paths to temporary storage, without changing production pipeline behavior. Executed results and readiness are recorded in the completion report; implemented generation/review must not be confused with executed generation/review.


### Verified implementation outcome (2026-09-17)

Offline preparation is complete: 99 requests, 749,630 UTF-8 bytes and approximately 497,581 input characters using the repository default model. No API submission, generated synthesis or review/finalization has occurred. Classification produces 70 mixed, 16 positive and 13 pain-point theme rows; these are not participant estimates. All 29 new tests pass, and the final complete suite passes 423 tests with four existing scikit-learn warnings in 213.01 seconds. Final checks confirm all 390 original baseline hashes and all 411 synthesis input hashes unchanged. Three legacy request-writing tests were isolated in temporary folders, the navigation assertion was updated, and the full suite used a writable temporary Numba cache after an earlier cache-import stall. The production page was visually checked in its correct pending-review state; finalized layout was exercised with synthetic test fixtures. See `PARTICIPANT_EXPERIENCE_IMPLEMENTATION_REPORT.md` for the exact file register, failure history, limitations and next approval command. Status: READY_FOR_SYNTHESIS_GENERATION.
