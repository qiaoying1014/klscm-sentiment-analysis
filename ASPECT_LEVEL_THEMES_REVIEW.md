# ALTA Researcher Review and Consolidation v1

## Status and purpose

This additive stage converts frozen ALTA v1 machine-induced semantic clusters into a compact cluster-level researcher review. Machine clusters are not automatically validated qualitative themes. Review v1 introduces explicit researcher interpretation without reclustering, changing any mention, or manually relabeling 15,486 individual mentions.

The hierarchy is frozen ABSA aspect → frozen ALTA cluster → reviewed discussion theme → researcher-observed perception → social-media finding aid → candidate interview proposition. Reviewed themes remain downstream of model-estimated ABSA assignments and inherit upstream uncertainty (development aspect precision approximately 0.513; recall approximately 0.790). Researcher interpretation does not validate the classifier.

## Lifecycle

1. Run `python -m marathon_absa.cli absa-v1-alta-review-generate`.
2. Open `data/processed/absa_v1/aspect_level_themes_review_v1/cluster_review_workbook.csv` for context and edit only `cluster_review_mapping.csv`.
3. For every cluster, set `theme_quality` to `coherent`, `somewhat_mixed`, or `highly_mixed`; set `review_decision` to `KEEP`, `RENAME`, `MERGE`, `UNCLEAR_OTHER`, or `EXCLUDE_FROM_INTERPRETATION`; add the required label/merge target, notes, and set `review_status=reviewed`.
4. Run `python -m marathon_absa.cli absa-v1-alta-review-validate`. This reports pending work and rejects invalid completed rows.
5. Only after all 101 rows are reviewed, run `python -m marathon_absa.cli absa-v1-alta-review-finalize`. The command fails visibly while any row remains pending.

`KEEP` uses the provisional label when the researcher label is blank. `RENAME` requires a nonempty researcher label. `MERGE` requires an existing target cluster key in the same frozen aspect. Merge chains are resolved deterministically; cycles, self-merges, cross-aspect targets, missing targets, or targets excluded from interpretation are rejected. `UNCLEAR_OTHER` and `EXCLUDE_FROM_INTERPRETATION` remain in the mapping but do not become reviewed themes. Frozen cluster IDs remain lineage fields and are never overwritten.

## Review package

The workbook has one row for each of 101 non-noise clusters, ordered by aspect support and then cluster document support. It contains frozen cluster identity, provisional label, mention/document support, unique-document prevalence, document-level sentiment shares, years, targets/keyphrases, and five exact representative evidence examples with separate English glosses and document/mention lineage. Editable researcher fields are deliberately blank/pending. The mapping is the smaller authoritative edit surface.

Four insufficient-support aspects—event information, race pack/expo, transport/access, and facilities—are reported separately with counts and exact evidence. They are not given manufactured themes. Noise remains `theme_id=-1`, is never redistributed, and is summarized by aspect.

## Finalization and aggregation

Finalization maps one or more frozen same-aspect clusters to deterministic reviewed IDs such as `route_course__rt01`. It recomputes mention counts and unique-document prevalence directly from frozen ALTA assignments. It never sums percentages or cluster document counts. When a document contributes to multiple merged clusters, it is counted once. If its underlying frozen sentiments differ within the reviewed theme, its document sentiment is `mixed`; otherwise its sole sentiment is retained.

Year summaries always contain 2019, 2023, 2024, and 2025 with unique supporting documents, aspect documents in that year, and descriptive within-aspect prevalence. No significance testing or causal/temporal claim is made. Representative evidence is exact frozen evidence from distinct documents.

Finalization creates a perception sheet with conservative deterministic summaries and blank researcher perceptions initially marked `DRAFT`. Valid perception statuses are `DRAFT`, `APPROVED`, `REVISE`, and `DO_NOT_USE`. Only nonempty `APPROVED` perceptions can enter the interview proposition pool on a later re-finalization. Candidate interview relevance (`high`, `medium`, `low`) is a researcher judgment and is not calculated from prevalence. Candidate statuses are `PENDING`, `SHORTLIST`, `HOLD`, and `EXCLUDE`.

Interview statements use qualified social-media language and questions explicitly permit agreement, disagreement, or partial agreement. They are qualitative follow-up prompts, not proven factual claims. Social-media prevalence is descriptive of this corpus and not a probability estimate for all KLSCM participants.

## Integrity

All artifacts are isolated under `data/processed/absa_v1/aspect_level_themes_review_v1` as UTF-8-SIG CSV and Parquet. The manifest records every frozen ALTA input hash, ABSA source identity, review/finalization counts, mapping/output hashes, code identity, timestamps, and zero network/OpenAI/paid-inference activity. The public dashboard and frozen dashboard marts are unchanged. A later additive Aspect Explorer integration may display reviewed themes, document support, sentiment/year distributions, exact evidence, and approved perceptions while retaining the BERTopic Topic Explorer.

## Dashboard integration

After finalization to 64 reviewed themes, the existing Streamlit Aspect Explorer gained a read-only “What participants are talking about” section. It loads only the finalized taxonomy, summary, document-level sentiment summary, four-edition year summary, representative evidence and insufficient-support summary. The loader verifies the finalized manifest state, expected 101 reviewed source clusters, 64 final themes, schemas, output hashes and same-aspect lineage; it refuses pending or provisional packages.

For a selected sufficiently supported aspect, themes are ordered by frozen unique-document support and shown in a horizontal overview. Expandable theme details display frozen support, share of aspect-bearing documents, document-level positive/negative/mixed/neutral composition, descriptive 2019/2023/2024/2025 values, and five exact multilingual evidence spans with separate English glosses. Facilities, transport/access, race pack/expo and event information show their frozen insufficient-support counts instead of an empty or manufactured taxonomy. Noise is not displayed as a theme. The BERTopic Topic Explorer remains available with a concise construct distinction.

The frontend performs only display filtering, sorting, formatting and joining of frozen finalized tables. It introduces no prevalence definition, sentiment aggregation, temporal test, clustering, label, perception, recommendation or interview analysis. Researcher perceptions and participant interview propositions have not yet been developed.

## Streamlit Researcher Review Interface

The internal local interface is `alta_review_app.py`. Launch it from the repository root with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run alta_review_app.py
```

The application reads `cluster_review_workbook.csv` for frozen evidence and context and reads/writes only `cluster_review_mapping.csv` for researcher decisions. It does not modify the workbook, frozen ALTA artifacts, ABSA outputs, source corpus, dashboard marts, or perception/interview layers. Streamlit was already declared in `requirements.txt`, so no dependency was added.

Review is aspect-first: aspects are ordered by summed cluster document support and clusters by descending document support. The cluster page shows frozen support, prevalence, years, descriptive sentiment shares, machine lexical aids, five exact multilingual evidence examples and separate English glosses. A same-aspect comparison table and evidence previews support consolidation. The editable controls are limited to theme quality, decision, researcher label, same-aspect merge target and notes; successful saves derive `review_status=reviewed`.

`KEEP`, `RENAME`, `MERGE`, `UNCLEAR_OTHER`, and `EXCLUDE_FROM_INTERPRETATION` retain the established semantics above. Merge choices exclude the current cluster and every cluster outside the current aspect. The interface does not suggest labels, decisions, perceptions, or interview questions. Widget drafts are retained per cluster during the Streamlit session and visibly marked as unsaved until written.

Each save reloads the mapping from disk, changes one identified row, runs the existing `validate_review_mapping` logic, and atomically replaces the UTF-8-SIG CSV while preserving its column order and all other rows. The first successful save in an app session creates one timestamped recovery copy under `aspect_level_themes_review_v1/backups/`. A fail-closed resolved-path check prevents writes outside the review namespace.

The progress page reports workflow counts overall, by aspect, decision, and quality; these are not research findings. Insufficient-support aspects and noise have read-only pages and cannot be assigned to themes. The validation control calls the same `validate_review_package` function as the CLI. Finalization remains disabled until the existing complete-validation gate passes and the researcher checks the explicit confirmation box; it then calls the same `finalize_reviewed_taxonomy` function as the CLI. Validation never finalizes automatically.

Browser-level unload confirmation is not reliable in native Streamlit. The interface therefore retains per-cluster widget drafts in session state, displays an unsaved-change warning, and provides explicit `Save Review` and `Save & Next` actions. Only persisted CSV decisions survive closing the browser or restarting the app. This remains a single-researcher local workflow; the reload-before-save behavior reduces stale-row risk but is not multi-user locking.
