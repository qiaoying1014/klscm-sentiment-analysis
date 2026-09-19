# Blog/Review Integration Methodology V1

## Status and analytical boundary

As implemented and locally audited on 2026-09-03, this additive branch restores long-form online reviews as a separately processed source. It does not modify or rerun the frozen Instagram BERTopic, ABSA V3, ALTA, or reviewed-theme artifacts. Blog ABSA inference and researcher theme confirmation have not yet been executed; consequently, no blog analytical finding or cross-source result is claimed here.

The Instagram and long-form review corpora were processed separately because they differ substantially in document length, sampling frame and corpus size. Long-form reviews were divided into chunks solely for model inference; all prevalence measures were subsequently aggregated to the original review level. The blog corpus was not appended to the Instagram BERTopic fitting corpus. Instead, the frozen ABSA ontology and finalized researcher-reviewed aspect-level theme taxonomy derived from the Instagram analysis were used as a reference framework for cross-source comparison. Candidate theme alignments were researcher-verified, while themes not represented by the Instagram-derived taxonomy were retained separately as blog-emergent themes.

Cross-source comparisons are descriptive and triangulatory rather than population-level statistical comparisons. The small long-form review sample is intended to provide narrative depth and complementary evidence rather than equal statistical weight to the much larger Instagram corpus. Neither corpus is claimed to be representative of all KLSCM participants.

## Source audit

The authoritative input is `Online Review Blog/raw-data.json` (SHA-256 `2c1fd1fbeb54d2dd89303be4f59c1e3930f6f9b0a9791dfc277d79809e7e2932`). It contains 28 rows and all 28 contain non-empty review text. The historical `data/processed/documents.csv` preserves 28 blog parents. Three rows have `processing_status=duplicate` and explicit `duplicate_of` lineage, leaving 25 usable canonical parent reviews. The historical `data/processed/units.csv` preserves 56 chunks for those 25 usable parents. Ten reviews have multiple chunks; the observed range is 1–6 chunks per parent.

Usable-review years are 2018 (3), 2019 (4), 2020 (1), 2022 (2), 2023 (7), 2024 (6), and 2025 (2). Stored language metadata identifies 24 English reviews and one Japanese review. The three exclusions are exact duplicate descendants of canonical document IDs; no non-empty unique review is excluded.

The audit artifacts are under `data/processed/blog_analysis_v1/`: `blog_source_audit_v1.csv`, `blog_chunk_audit_v1.csv`, `blog_analysis_audit_v1.json`, and `blog_analysis_manifest_v1.json`. Existing SHA-256-derived `document_id` and `unit_id` values are reused as `review_id` and `chunk_id` lineage. Thus, the correct description is “25 blog reviews processed through 56 inference chunks,” never “56 blog documents.”

## ABSA reuse and validation

The request package reuses the frozen `absa_v1_instructions_3_aspect_ownership` instructions, prompt SHA-256 `196606110ecb2b78d56a3220c1bd8957e8e42f2f259c65d1094c037fa21f30d0`, model `gpt-5.6-luna`, `absa_mention_schema_v1`, and the frozen 20-aspect ontology. The isolated stage identity is `absa_v1_blog_mentions_v1`; it cannot collide with the Instagram production cache.

Finalization fails closed for missing/unknown/duplicate chunk IDs, malformed response payloads, uncontrolled aspect or sentiment labels, ungrounded evidence, and duplicate mention IDs. Zero-mention chunks remain explicit successful chunk results. Chunk artifacts are retained and then aggregated to their parent reviews.

For aspect prevalence, each `review_id + aspect` pair counts once. The denominator is all included parent reviews, never chunks or mentions. Opposing sentiment labels within one review-aspect become review-level `mixed`; repeated identical sentiment becomes one review observation. Mention counts remain separate secondary evidence. Every reported proportion must include raw `n/N`.

The branch inherits ABSA V3’s development limitation: aspect precision was approximately 0.513 and recall approximately 0.790. These results came from development evidence rather than independent confirmation. Researcher theme mapping does not validate or repair ABSA classification errors.

## Theme mapping and researcher confirmation

Candidate mapping uses the locally cached `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` model with local-files-only behavior. Representations use target, exact evidence, and English gloss when present. A blog mention is compared only with the finalized reviewed themes in its frozen ABSA aspect. ALTA is not rerun; its embeddings and the 64-theme taxonomy remain unchanged.

The top three cosine similarities are aids, not assignments. `blog_theme_review_app.py` requires one researcher decision per mention: `MATCH_EXISTING`, `EMERGENT_BLOG_THEME`, `UNCLEAR`, or `EXCLUDE_FROM_THEME_COMPARISON`. An existing match must be a reviewed theme in the same aspect. Emergent labels remain explicitly `blog_emergent` and do not alter the Instagram taxonomy. Finalization refuses unresolved rows. The UI uses UTF-8-SIG, an atomic replacement, one timestamped backup per session, immutable machine fields, filters, and resume from saved decisions.

## Cross-source synthesis and dashboard

`marathon_absa/cross_source_analysis.py` creates deterministic aspect and theme comparison marts only after blog ABSA and researcher mapping are finalized. Instagram document and blog review denominators remain separate. Statuses are descriptive (`observed_in_both`, `instagram_only_observed`, `blog_only_observed`). No pooled prevalence, combined sentiment percentage, chi-square test, Fisher test, or claim that one source “cares more” is implemented.

The dashboard adds a clearly separated **Cross-Source Analysis** page. Before downstream completion it displays corpus/audit counts and an explicit pending state, not premature findings. It does not expose author names or URLs and makes no API calls.

## Provenance limitation

The repository does not sufficiently document blog collection methodology. The audit therefore records `DOCUMENTATION_REQUIRED` for `source_platforms`, `search_strategy`, `collection_procedure`, `inclusion_criteria`, `exclusion_criteria`, `year_selection_rationale`, and `terms_ethics_considerations`. No platform or search method was inferred from URLs. This caution remains visible until supported documentation is supplied.

## Reproducible commands and execution state

Executed locally with zero network/API calls:

```powershell
.\.venv\Scripts\python.exe -m marathon_absa.cli blog-analysis-audit
.\.venv\Scripts\python.exe -m marathon_absa.cli blog-absa-create
```

The prepared package contains 56 requests, an estimated 200,846 input tokens and 14,560 output tokens, and an estimated Batch cost of USD 0.1983 under the inherited estimator assumptions. It is not submitted.

The exact paid next command is:

```powershell
.\.venv\Scripts\python.exe -m marathon_absa.cli blog-absa-submit --run-api
```

After completion, run the status and import commands with `--run-api`, then the offline stages:

```powershell
.\.venv\Scripts\python.exe -m marathon_absa.cli blog-absa-status --run-api
.\.venv\Scripts\python.exe -m marathon_absa.cli blog-absa-import --run-api
.\.venv\Scripts\python.exe -m marathon_absa.cli blog-absa-finalize
.\.venv\Scripts\python.exe -m marathon_absa.cli blog-theme-candidates
.\.venv\Scripts\streamlit.exe run blog_theme_review_app.py
.\.venv\Scripts\python.exe -m marathon_absa.cli blog-theme-finalize
.\.venv\Scripts\python.exe -m marathon_absa.cli cross-source-analysis
```

The last two commands remain blocked until all researcher mapping rows are resolved. All local audit, aggregation, mapping-finalization, comparison, and dashboard stages make zero API calls.

## Blog-emergent audit and consolidation workflow (2026-09-07)

This section supersedes earlier pending-stage descriptions for the already completed blog ABSA, original blog mapping, and cross-source marts. It does not imply that emergent consolidation is complete. The audit was executed locally on 2026-09-07: 67 researcher-marked emergent mention records map to 35 existing IDs across 17 frozen aspects, supported by 18 of the 25 usable parent reviews. All 35 original labels and all 67 evidence spans are recoverable. Twenty-four IDs have singleton support. Existing IDs were generated from the frozen aspect and the first eight hexadecimal characters of SHA-256 of the exact original label; repeated same-label observations already share an ID. No exact normalized same-aspect label duplicates were found. No further evidence-equivalent duplicate group has been established.

The original labels and mapping notes explicitly identify delegated AI-assisted review, not independent human validation. Their workflow status is not evidence of a separate researcher consolidation decision. All 35 consolidation decisions therefore remain UNCLEAR. Thirty-nine same-aspect candidate pairs are supplied for screening, ranked only by lexical label sequence similarity. This score is not an embedding cosine, semantic equivalence judgment, merge threshold, or final decision. No embedding model or API was run.

The completed evidence inspection identified scope questions, recorded in the audit: general course quality versus difficulty, race-village space versus ground conditions, non-completion versus injury/setback emotions, and an expo-volunteer observation mentioning GCM. The latter requires researcher checking of comparator-event scope. These observations do not justify an automatic merge or exclusion. Cross-aspect cooling, muddy-ground, and distance-marker concepts remain separate regardless of related wording.

### Consolidation method and reporting boundary

The following is the prescribed method; the past-tense consolidation wording is suitable for reporting only after researcher decisions and finalization have actually completed:

> Blog-emergent themes were identified only where researcher-reviewed blog evidence could not be aligned to the finalized Instagram-derived aspect-level theme taxonomy. Because multiple long-form observations could express semantically equivalent concepts, emergent observations were subsequently audited within their frozen ABSA aspect and conservatively consolidated where researcher review confirmed substantive equivalence. Long-form review support was counted at the parent-review level rather than the inference-chunk level. Low-support and singleton emergent themes were retained but interpreted cautiously. This audit did not modify the frozen Instagram-derived thematic taxonomy.

The blog-emergent taxonomy is complementary to, rather than an extension of, the frozen 64-theme Instagram taxonomy. The inherited labels were AI-assisted; no independent human confirmation is claimed by this audit. The Instagram taxonomy was not jointly derived from blogs.

`marathon_absa/blog_emergent_themes.py` implements audit, deterministic consolidation, complete mention lineage, and fail-closed finalization. A rename preserves the original ID. A merge hashes the aspect, normalized confirmed final label, and sorted source IDs. Merge chains/cycles and cross-aspect targets are rejected; use one retained target per merged group. The target must KEEP_SEPARATE or RENAME_ONLY, all incoming merges must confirm the same final label, and each merge needs an equivalence reason. EXCLUDE_AS_NON_THEME needs a reason and retains all source evidence in the lineage file. UNCLEAR blocks publication. Low support is explicitly `support_reviews < 2`; singleton is `support_reviews == 1`. Neither flag causes exclusion. Distinct mentions from repeated chunks in one parent contribute one review to prevalence and remain separately counted as mentions. The denominator must be exactly 25 usable parent reviews.

`blog_emergent_theme_review_app.py` presents all original labels, evidence, available English glosses, parent/chunk/mention IDs, original decision notes, and the surrounding inference-chunk text. All same-aspect candidates remain inspectable. Display copies redact known source author strings and URLs; original audit evidence remains intact. Saves use atomic replacement, content-addressed backups, a short-lived exclusive lock, and an optimistic file-version check; completed decisions survive reopening and audit reruns. A lock left by a crashed process must only be removed after checking that no review session is writing.

Audit artifacts under `data/processed/blog_analysis_v1/` are `blog_emergent_theme_audit_v1.csv`, its JSON companion, `blog_emergent_theme_candidates_v1.csv`, `blog_emergent_theme_decisions_v1.csv`, and `blog_emergent_frozen_baseline_v1.json`. The baseline records 390 existing files from topic discovery, Instagram ABSA/ALTA, blog outputs, and parent/chunk inputs before this stage; it must not be regenerated to accept a changed upstream state. Audit CSV notes preserve the specific scope questions above. Existing mappings, evidence, manifests, and cross-source outputs remain unchanged.

The future finalizer writes `blog_emergent_theme_taxonomy_v1.csv`, `blog_emergent_theme_lineage_v1.csv` (including exclusions), and `blog_emergent_theme_finalization_v1.json` only after validation passes. The manifest binds taxonomy, lineage, and decision-file hashes; a later edit invalidates consumption until finalization is rerun. Frozen counts, source hashes, prompt identity, usable-parent membership, source coverage, readable labels, and merge restrictions are checked before output. Cross-source regeneration then reads this separate taxonomy, redacts public evidence copies, and preserves the matched 64-theme comparison. It refuses changes to the observed 47/64 blog-supported matched themes or 20/20 shared aspects. Separate denominators remain 7,704 Instagram documents and 25 blog reviews, with descriptive triangulation, no pooling, no inferential tests, and zero API calls.

Commands from the repository root, in lifecycle order:

```powershell
.\.venv\Scripts\python.exe -m marathon_absa.cli blog-emergent-theme-audit
.\.venv\Scripts\python.exe -m streamlit run blog_emergent_theme_review_app.py
.\.venv\Scripts\python.exe -m marathon_absa.cli blog-emergent-theme-finalize
.\.venv\Scripts\python.exe -m marathon_absa.cli cross-source-analysis
```

Current lifecycle: audit and review preparation executed; 0 resolved and 35 unresolved consolidation decisions. Final taxonomy and cross-source regeneration were deliberately not executed. The dashboard adds only a pending-safe emergent table with label, aspect, parent support n/25, and low-support flags once finalized. Existing aspect/matched-theme findings are preserved. No analytical conclusion beyond the original reviewed artifacts is asserted. The small, differently sampled long-form corpus provides complementary descriptive evidence; percentages are not directly comparable population estimates. The ABSA development limitations (precision approximately 0.513, recall approximately 0.790) and incomplete collection provenance remain unchanged.

Validation outcome: 27 focused tests passed; the complete suite passed 387 tests with four existing sklearn warnings from relevance tests (single-label confusion matrices and invalid scalar division). Streamlit AppTest loaded the consolidation interface and navigated same-aspect evidence. A synthetic successful finalization and cross-source regeneration were tested only in isolated temporary directories. The real finalization command correctly blocked on unresolved researcher decisions; no real final taxonomy or regenerated mart was produced. All 390 upstream baseline hashes and the frozen-count/prompt checks passed again after testing.

## Delegated consolidation review completed (2026-09-07)

Following explicit user delegation, Codex completed the evidence review and finalized the emergent layer. This supersedes the previous 35-UNCLEAR state. Review provenance is AI-assisted interpretation performed on the user's behalf, not independent human coding, researcher validation, or inter-rater reliability. All 67 spans and available glosses were inspected with local source context, including full chunks for ambiguous references, and all 39 same-aspect pairs were considered. Existing identical-label grouping had already consolidated repeated concepts; no remaining pair justified a further merge. Twenty-eight labels were retained and seven corrected/narrowed. No exclusions were justified, and no unresolved decisions remain.

The resulting 35 themes retain all source IDs and mention lineage, cover 17 aspects and 18/25 parent reviews, and include 24 singleton/low-support observations. Low support was not an exclusion criterion. No claim is made that all 35 themes are strong recurring findings or narrow homogeneous constructs; heterogeneous inherited umbrella scopes are explicitly documented in the saved decision reasons.

Context corrections include an on-course mist arch previously labeled post-race, a toilet grouping containing two race-venue parents and one hotel parent, and a GCM promotional booth situated within the SCKLM expo. The new toilet label explicitly includes accommodation; its 3/25 support cannot be described as three accounts of race toilet provision. The expo-volunteer theme concerns contributing at a visiting-event booth, not receiving on-course support. Field strength is narrowed to female half-marathon runners. Cutoff allowance, average finishing times, and course-length/marker accuracy are labeled as perceived/reported observations, not verified measurements or population facts. General route appraisal (three parents) and challenge/difficulty (two) remain explicitly distinguishable within the inherited five-parent umbrella. Race-village space (two) and muddy-ground condition (one) likewise remain distinguishable within their three-parent umbrella. No cross-aspect merge or upstream reassignment was performed.

Authoritative files: `blog_emergent_theme_decisions_v1.csv`, `blog_emergent_theme_ai_review_evidence_v1.csv`, `blog_emergent_theme_pair_review_v1.csv`, `blog_emergent_theme_ai_review_manifest_v1.json`, `blog_emergent_theme_taxonomy_v1.csv`, `blog_emergent_theme_lineage_v1.csv`, and `blog_emergent_theme_finalization_v1.json`, all under `data/processed/blog_analysis_v1/`. The original audit CSV/JSON remains a historical pre-review snapshot. The explicit reproducible review is in `scripts/complete_blog_emergent_review_v1.py`; every decision reason preserves AI-assisted provenance.

Actual finalization and cross-source regeneration completed offline. The emergent summary now includes readable corrected labels, unique-parent n/25 support, evidence, and singleton/low-support indicators. Cross-source source hashes include the emergent taxonomy and finalization manifest. Original aspect and matched-theme comparison CSVs remain byte-identical; the 20/20 shared aspects, 47/64 supported reference themes and 17/64 without blog support are unchanged. All 390 baseline upstream hashes passed after regeneration. Comparison remains descriptive_triangulation with separate 7,704/25 denominators, no pooling, zero inferential tests and zero API calls. The blog-emergent taxonomy is complementary to, rather than an extension of, the frozen 64-theme Instagram taxonomy; it was not jointly used to derive that taxonomy.

Final validation after delegated review: 29 focused tests passed and the complete suite passed 389 tests. Four existing sklearn warnings in relevance tests were reported separately; no failures occurred. Post-suite verification reconfirmed all 390 frozen upstream hashes unchanged. Finalized taxonomy and regenerated human-readable emergent marts are now available; no consolidation decisions remain pending.

## Streamlit presentation of completed long-form results (2026-09-07)

The existing Cross-Source Analysis section now reads the finalized blog branch through `marathon_absa/blog_dashboard_data.py`. It displays 25 parent reviews, 301 ABSA mentions, 56 inference-only chunks, 20 shared aspects, 47/64 matched Instagram themes, 35 emergent themes and 24 singleton observations. Users can filter by aspect, inspect matched themes with or without blog support, inspect parent-review aspect sentiment, and read emergent evidence with saved interpretation caveats. Review-year and language tables describe only the 25 sampled parents. Filtering does not replace the prevalence denominator with chunk counts or an aspect-specific subpopulation.

The loader validates finalization/source hashes and reconciles parent, mention, chunk, sentiment and theme totals before displaying data. It rejects stale emergent marts. Public display fields omit author/URL and parent/chunk identifiers; known authors and URLs are redacted from evidence and scope notes. All blog review provenance is explicitly delegated AI-assisted interpretation, not independent human validation. The original Instagram chart/statistical pages remain separate and show a scope notice; the legacy preparation dashboard distinguishes source records from finalized research findings. No source populations, sentiment totals or percentages are pooled. No frozen artifact or finalized comparison mart was modified by this UI integration.

Validation of the Streamlit integration: 25 focused loader/UI tests and the 394-test complete suite passed, with four existing sklearn warnings and no failures. All 390 frozen-file hashes and the original aspect/matched-theme comparison CSVs remain unchanged. The dashboard reads the finalized long-form artifacts on rerun; no analytical pipeline rerun was required.
