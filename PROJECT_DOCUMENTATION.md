# KLSCM Multilingual Topic Discovery and Aspect-Based Sentiment Analysis

## Complete Project Documentation for Thesis Preparation

**Status:** Canonical living technical and methodological record  
**Deployment supplement:** 19 September 2026; see the Streamlit Community Cloud section below. Earlier evidence cut-offs remain historical.
**Repository:** `C:\Users\user\Documents\Sentiment Analysis\1. Data Scraping`  
**Evidence cut-off:** 25 July 2026, Malaysia time (UTC+08:00)  

## 1. Purpose and evidence policy

### Streamlit Community Cloud release (19 September 2026)

The user-reported initial GitHub push was rejected because the initial commit
included generated BERTopic models and ABSA batch JSONL files above 100 MiB.
Inspection found one local commit (`01255746`) and no remote refs. Merely adding
ignore rules or a subsequent deletion commit would leave those oversized blobs
reachable from `main`. The repair replaces the unpublished initial commit while
retaining its original history in a local backup branch and retaining research files
on disk. The backup branch is local only and must not be included in an `--all` push.

Deployment now uses `deploy/streamlit_app.py`, which selects the finalized dashboard
in `absa_dashboard.py` and explicitly enables `KLSCM_DASHBOARD_BUNDLE`. The existing
local entry point still uses original research artifacts and full validation.
`scripts/export_dashboard_bundle.py` runs the original dashboard, reviewed-theme,
blog/cross-source, participant-experience v2, and word-cloud loaders before writing
any deployment artifacts. The executed export passed those checks and produced
34 artifacts totaling 6.29 MiB, excluding its manifest (JSON uses portable LF bytes). The original counts remain
7,704 Instagram documents, 15,486 mentions, 64 reviewed Instagram themes, 25 parent
reviews, and 99 finalized participant-experience themes. No research inference,
paid API call, upstream data edit, or methodology recalculation occurs during export.
The exact export time is recorded in `deploy/dashboard_data/manifest.json`; the
task date is supplied by the session environment.

The deployment artifact register is `deploy/dashboard_data/manifest.json`: each
section/field maps to a JSON or Parquet filename, SHA-256, byte count and format.
Sections are `dashboard` (frozen analytical tables and metadata), `themes`
(reviewed taxonomy, supporting tables and manifest), `blog` (validated descriptive
comparison views), `experience` (evidence, finalized insights and provenance),
and `wordcloud` (validated population and mention tables). DataFrame columns are
preserved; JSON retains source metadata and recorded hashes. Source path strings
inside provenance remain historical and need not resolve on Linux. The snapshot
contains text already required by the dashboard, including captions and evidence;
it is not an anonymization or public-data certification procedure.

`marathon_absa/cloud_bundle.py` checks bundle schema/status, confines file paths to
the bundle directory and verifies selected artifact bytes before loading. Dashboard
and reviewed-theme contracts and participant-experience evidence/claim contracts
are checked at runtime. Complete upstream hash/lineage validation is performed at
export, not repeated on Cloud: the deployed snapshot is a validated release, not
an independent reconstruction of the research environment. Corruption fails closed.
`.gitattributes` prevents Git newline conversion from invalidating bundle hashes.

`deploy/requirements.txt` pins the locally verified dashboard dependencies;
`deploy/packages.txt` installs `fonts-noto-cjk`, and `absa_dashboard.py` recognizes
the Linux font path. Select Python 3.11 and `deploy/streamlit_app.py` on Community
Cloud. The root research requirements remain available for local pipeline work.
`.gitignore` excludes processed data, caches, models, raw source exports, local work
and document QA images. Those remain local research artifacts; cloning Git alone
does not reproduce the pipeline or artifact-dependent research tests. A future
release must be re-exported from the full local research checkout, tested and committed.

Executed validation: `pytest tests/test_cloud_deployment.py -q` passed 2 tests.
One rejects altered release bytes; the other copies only source and the deployment
bundle into an isolated directory, then runs all 12 navigation pages and the alternate
caption-context word cloud with no research `data/` directory. This verifies bundle
completeness on the local Windows runtime, not an executed Linux Cloud deployment.
The full suite executed with 442 passes, 5 failures and 44 setup errors (261.11 s).
Failures/errors are in the historical participant-experience deduplication, retry,
retry-round-two and review tests: they reference earlier-stage artifacts through
the newer v2 default root, where retry manifests and the mechanical audit do not
exist. Repeating those four modules against the original pre-deployment Python
implementation produced the identical 5 failures and 44 errors (plus 7 passes),
confirming they predate this deployment change. They remain unresolved and do
not constitute a passing full research suite. The isolated Cloud tests passed;
actual Community Cloud deployment has not been executed in this task.

The final focused dashboard/deployment run passed **47 tests in 49.65 s** across
`test_cloud_deployment.py`, `test_absa_dashboard_ui.py`,
`test_reviewed_theme_dashboard_data.py`, `test_blog_dashboard_data.py`,
`test_wordcloud_data.py`, and `test_participant_experience_dashboard_finalized.py`.
All 34 exported file hashes also matched the bytes stored in Git. The replacement
initial commit `463eaafc` contained 205 files totaling 8,892,218 bytes; its largest
file was `deploy/dashboard_data/wordcloud_mentions.parquet` (3,277,092 bytes).
`git push -u origin main` succeeded on 19 September 2026, creating remote `main`
and setting upstream tracking. The oversized original commit remains only on the
local `backup/pre-cloud-20260919` branch. Research inputs and outputs remain on disk.
The repository is published and ready for the user to select the documented Cloud
entry point; no live Streamlit URL or successful Cloud build is claimed.

This document explains what the project investigates, when recorded activities occurred, where methods and artifacts are located, how each stage works, and what outcomes are verified. It is intended to support later thesis methodology, implementation, results, limitations, ethics, and reproducibility chapters. It is not a work log.

Dates come from embedded source dates, `data/processed/run_manifest.jsonl`, timestamped backup names, file metadata, and direct verification. The checkout has no usable Git history, so filesystem timestamps are supporting evidence rather than proof of when a method was conceived. The document distinguishes:

- **Implemented:** executable code exists.
- **Executed:** a manifest or output proves the stage ran.
- **Validated:** tests or empirical evaluation produced a result.
- **Planned:** code/design exists but no output proves completion.
- **Blocked:** a quality or safety gate prevents responsible execution.

## 2. Research objective and current conclusion

The project prepares multilingual Instagram captions and online review/blog narratives about the Kuala Lumpur Standard Chartered Marathon (KLSCM) for topic discovery and aspect-based sentiment analysis (ABSA). It addresses code-switching, Malay shorthand, hashtags, emoji, URLs, mentions, duplicates, promotional noise, image-dependent captions, and long reviews.

The intended outcome is an evidence-linked set of opinion mentions identifying an event aspect, evaluated target, sentiment, confidence, exact evidence, English gloss, explicit/implicit expression, meaningful hashtags/emoji, and language/code-switching observations. Planned aspects include route/scenery, weather, organization, registration, expo/race kit, transport, atmosphere, aid stations, facilities, safety, value, personal experience, and emerging topics.

As of 24 July 2026, source cleaning and local preparation are complete. A partial relevance pilot covering 599 Instagram records and a blind validation exercise were completed. The validation failed the predeclared recall and macro-F1 targets; full-corpus relevance is incomplete; 5,808 language decisions remain pending adjudication; and no topic or ABSA artifacts exist. Topic and sentiment findings must not yet be claimed.

## 3. Project timeline

| Date/time | What happened | Evidence and outcome |
|---|---|---|
| 10 Jul 2026 | Four Instagram hashtag exports were collected. | `Instagram/Raw Data/2026-07-10_*.json` and cleaning report; 14,591 raw records. |
| 10 Jul 2026 | Instagram schemas were standardized and cleaned CSV/XLSX created. | Notebook, report, and outputs; 14,049 final records. |
| 16ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“19 Jul 2026 | Twenty-eight blog/review records were added. | `Online Review Blog/raw-data.json` and file metadata. |
| 18 Jul 2026 15:05/15:09 UTC | Early prepare runs produced 14,077 documents and 14,093 then 14,111 units. | First manifest lines; earlier confidence setting recorded as 0.55. |
| 18 Jul 2026 | General 114-row validation sample created. | `validation_sample.csv/.parquet`; human fields remain blank. |
| 22 Jul 2026 12:57ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“13:46 UTC | Four pinned-OpenLID prepare runs produced 14,077 documents and 14,116 units. | Manifest and current artifacts. |
| 22 Jul 2026 15:56 UTC | 500-record stratified relevance pilot run. | Manifest; 318 relevant, 92 irrelevant, 90 ambiguous/pending at that time. |
| 23 Jul 2026 | Blind relevance key/reviewer sample created. | 300-row key plus reviewer sample and metadata. |
| 24 Jul 2026 13:47 UTC | 100 more records processed, leaving 599 unique relevance rows. | Manifest; one record overlapped the existing union. |
| 24 Jul 2026 about 21:13ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“21:39 local | Human validation completed and metrics written. | Backups and validation result files; targets failed. |
| 24 Jul 2026 14:56 UTC | Operational queue's 108 pending records human-resolved. | Manifest; final partial union 372 relevant and 227 irrelevant. |
| 24 Jul 2026 14:59 UTC | 100-record rerun preserved the 599-row union and reviews. | Manifest; confirms review preservation. |
| 24 Jul 2026 | Complete tests rerun during this study. | 51 passed in 18.33 seconds. |

UTC manifest times convert to Malaysia time by adding eight hours.

## 4. Repository responsibility map

| Path | Responsibility |
|---|---|
| `marathon_absa/config.py` | Paths, model names, pinned OpenLID revision, thresholds, token limits, seed. |
| `text.py` | Unicode repair, text views, hashtag/emoji extraction, hashes and IDs. |
| `ingest.py` | Common document schema, source loading, duplicate flags. |
| `language.py` | Lingua/OpenLID detection, Malay shorthand, spans, mixed language, review routing. |
| `chunking/__init__.py` | Sentence-aware chunks capped at 500 tokens. |
| `relevance.py` | Rules, thresholds, pilots, human review, validation, topic gate. |
| `openai_service.py` | Cached structured requests, retries, usage metadata, embeddings, prompts. |
| `schemas.py` | Strict language, relevance, and ABSA JSON schemas and aspect list. |
| `topics.py` | UMAP, HDBSCAN, BERTopic, initial taxonomy. |
| `storage.py` | Paired Parquet/UTF-8-SIG CSV persistence and list serialization. |
| `pipeline.py` | Stage orchestration, incremental merges, manifests, output gates. |
| `cli.py` | Commands and explicit `--run-api` authorization. |
| `app.py` | Research dashboard. |
| `relevance_labeler.py` | Human labeling, atomic save/backups, optional translation. |
| `tests/` | 51 deterministic tests across language, pipeline, relevance, and labeler. |

## 5. Source data and cleaning

### 5.1 Instagram sampling frame

Posts were retrieved through `#klscm2019`, `#klscm2023`, `#klscm2024`, and `#klscm2025`. These are retrieval-corpus labels, not guaranteed publication years. Historical hashtags were reused after their named event years.

| Corpus | Raw | Final cleaned |
|---|---:|---:|
| klscm2019 | 4,463 | 4,145 |
| klscm2023 | 3,578 | 3,476 |
| klscm2024 | 3,159 | 3,111 |
| klscm2025 | 3,391 | 3,317 |
| **Total** | **14,591** | **14,049** |

The 2019/2023 JSON uses nested `author.id` and `taken_at_timestamp`; 2024/2025 uses `ownerId` and `timestamp`. The notebook maps both to `author_id`, `post_url`, `caption`, source `hashtag`, and UTC `timestamp`. Engagement, media, name, and location fields are excluded because the study is text-focused and should minimize unnecessary personal data.

Cleaning trims external whitespace, removes null/blank captions, converts IDs to strings, parses UTC timestamps, assigns the source hashtag, and removes exact `(author_id, caption)` duplicates within each corpus and then across corpora. Three blanks, 522 within-corpus duplicates, and 17 cross-corpus duplicates were removed. Cross-corpus retention follows concatenation order 2019, 2023, 2024, 2025. The raw files remain unchanged. Assertions verify source presence, count reconciliation, schema/order, nonblank text, duplicates, expected hashtags, timestamps, and export readback. Full detail is in `Instagram/data_cleaning_report.md`.

The inherited filename `instagram_cleanded.csv` is misspelled but is the exact path used by `config.py`.

### 5.2 Blog/review source

`Online Review Blog/raw-data.json` contains 28 records with `author`, `review`, `title`, `url`, and `year`. It is the authoritative input. The accompanying CSV fails standard CSV parsing due to malformed field/quoting structure and is not used by the pipeline. The repository lacks a collection-method report for these reviews; the thesis still needs source platforms, search/inclusion criteria, collection procedure, terms/ethics, and year justification.

### 5.3 Prepared corpus

The combined artifact has 14,077 documents: 14,049 Instagram and 28 blog. Event-year counts are 2018: 3; 2019: 4,149; 2020: 1; 2022: 2; 2023: 3,484; 2024: 3,117; 2025: 3,319; Unknown: 2. For Instagram, this field primarily reflects the retrieval hashtag and must not be presented automatically as verified publication year.

## 6. Text representations, identifiers, and lineage

Every record keeps parallel representations so one task's cleaning does not destroy another task's signal:

1. `original_text`: loaded source text.
2. `normalized_text`: ftfy repair, Unicode NFKC, whitespace collapse and trim.
3. `linguistic_text`: removes URLs, mentions, hashtags, KLSCM identifiers and emoji for language detection.
4. `semantic_text`: removes URLs/mentions but converts hashtags and emoji to explicit semantic features.
5. `hashtags`, `emojis`, `emoji_aliases`: separately preserved lists.
6. `text_hash`: SHA-256 over case-folded, punctuation-normalized text.

Stable 20-character SHA-256-derived IDs link documents, chunks, topics, and mentions. Current status is 13,824 ready documents and 253 normalized-text duplicates. Duplicates remain for audit with `duplicate_of` and are excluded from units.

```text
Instagram raw JSON -> cleaning notebook -> instagram_cleanded.csv --ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â
                                                                    ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ-> documents -> units
Blog raw-data.json -------------------------------------------------ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¹ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“       |
                                                                            +-> relevance + validation
                                                                            +-> topics/taxonomy (blocked)
                                                                            +-> ABSA mentions (blocked)
```

Tables are written as typed Parquet and audit-friendly UTF-8-SIG CSV. List fields are JSON-encoded on disk and decoded by `storage.py`.

## 7. Language identification

### 7.1 Models and parameters

The service combines Lingua with HPLT OpenLID-v3. OpenLID is pinned to repository `HPLT/OpenLID-v3`, file `openlid-v3.bin`, revision `6b9560483e17e42f48d86cebf22b4b58dffeaa70`. Current settings include eight minimum linguistic characters, confidence 0.80, margin 0.15, mixed coverage 0.20, mixed span minimum 10 characters, 16-word windows with five-word overlap, and seed 42. Early manifest runs recorded confidence 0.55; final thesis analyses require one clean rerun with frozen settings.

### 7.2 Procedure

Empty normalized text becomes `no_text`; fewer than eight letters becomes `insufficient_text`. Lingua provides a candidate/confidence. Suitable high-confidence cases use `lingua_high_confidence`; other cases use OpenLID whole-text and overlapping span predictions. Span evidence supplies language components, coverage, primary language, mixed status and possible code-switching. A Malay-shortform dictionary expands common colloquial forms only for detection when at least two hits occur; original text is unchanged. Safeguards prevent nonlinguistic tails and weak unsupported model labels from producing false components.

Uncertain, disagreeing, mixed, low-margin, or review-language cases can be sent through `language-review --run-api`. Structured results replace final language fields and units are rebuilt. No language-review execution appears in the manifest.

### 7.3 Current language output

| Measure/status | Count |
|---|---:|
| Documents | 14,077 |
| Mixed-language flag | 1,006 |
| Pending adjudication | 5,808 |
| ok | 7,374 |
| low_confidence | 2,720 |
| review_required | 1,772 |
| insufficient_text | 895 |
| mixed | 723 |
| undetermined | 593 |

Top labels are English 7,181; Malay 2,196; Malay/Indonesian uncertain 1,637; insufficient text 895; Chinese 889; undetermined 593; Indonesian 509; Tagalog 55; Japanese 34; Thai 31; French 14; Portuguese 13; Korean 9; German 5; Vietnamese 4. These are machine classifications, not human-validated language prevalence. Dashboard language percentages exclude empty, duplicate, no-text and insufficient-text records.

## 8. Chunking and analytical units

Ready Instagram captions stay as one unit unless longer than 500 tokens. Blogs and long documents split at sentence boundaries into chunks no larger than 500. Token counting uses the configured model encoding, then `o200k_base`, then a UTF-8 approximation. Oversized sentences are sliced by tokens.

Current output: 14,116 units (14,060 Instagram; 56 blog). Token count mean 110.24, median 74, Q1 34, Q3 145, minimum 3, maximum 500. The configured 400-token target is not actively optimized; only the 500 maximum is enforced. Tests confirm sentence preservation and the cap.

## 9. Relevance filtering

### 9.1 Policy

Only Instagram is filtered. Relevant content includes KLSCM registration, event-specific preparation, logistics, information, participation, experience, results/achievement, support, and evaluation. Another event, generic running, pure promotion, lifestyle/spam, or no event connection is irrelevant. Hashtag-only, insufficient, image-dependent, weak, or conflicting evidence is ambiguous. Records are retained; `include_in_topics` is the sole Instagram gate.

### 9.2 Decision sequence

1. No-linguistic-text/hashtag-only rules route to pending without API cost.
2. Remaining selected records receive a strict-schema first pass.
3. First-pass relevant requires confidence >=0.80; irrelevant >=0.90; ambiguous is adjudicated.
4. A stronger model accepts relevant >=0.75 and irrelevant >=0.90; otherwise pending.
5. Pending cases enter a context-rich human queue.
6. Operational human labels are only relevant/irrelevant, override the model, and set confidence 1.0.
7. Incremental reruns preserve human labels and notes.

Each row retains initial/adjudicated/final decisions, confidence, reason, evidence, gloss, language notes, review state, inclusion gate, models, prompt versions, cache flags, and token use. Structured requests are cached by stage, prompt version, model, text, and extra context; retries use exponential backoff. Paid commands require both the key and explicit `--run-api`.

### 9.3 Pilot and current outcome

Sampling attempts coverage across year, language, language status, and text-length quartile with seed 42. A 500-record pilot followed by a 100-record run produced 599 unique rows due to one overlap.

Current final counts are 372 relevant and 227 irrelevant. There are 491 auto-resolved and 108 human-resolved; methods are 457 initial model, 34 stronger model, and 108 human review. The current queue is empty only for this 599-record union. Approximately 13,225 ready Instagram documents still lack relevance results. These pilot counts are not full-corpus prevalence estimates.

`validate_topic_gate` rejects topic/ABSA execution until every ready Instagram document has a relevance row and no row is pending.

## 10. Human labeling interface

`relevance_labeler.py` validates required columns, navigates pending rows, shows model context/original text, and autosaves. Validation samples allow relevant/ambiguous/irrelevant; operational queues allow only relevant/irrelevant. The first save creates one timestamped session backup. Each edit writes a same-directory temporary file, flushes and `fsync`s, then atomically replaces the CSV. Tests verify backups, preservation, navigation, invalid-label rejection, and unchanged files after errors.

Optional English translation is disabled by default and uses the cached structured API. It is exempt for English, Malay, Indonesian and Chinese, preserves names/hashtags/emoji/URLs, and is for reviewer comprehension only; original text remains evidence.

## 11. Relevance validation

### 11.1 Design

The workflow separates a blind reviewer file from the model key, targets 300 base records, balances model labels and contextual strata, divides calibration and holdout halves, and adds 30 repeats. Seed 42 governs sampling. The predeclared targets are relevant recall >=0.90, relevant precision >=0.85, and macro-F1 >=0.80. Outputs include confusion matrix, classification report, subgroup metrics and repeat-label Cohen's kappa.

### 11.2 Observed results

| Metric | Result | Target/outcome |
|---|---:|---|
| Holdout n | 150 | Completed |
| Relevant precision | 0.8750 | Passed >=0.85 |
| Relevant recall | 0.5234 | Failed >=0.90 |
| Macro-F1 | 0.3622 | Failed >=0.80 |
| Repeat pairs | 30 | Completed |
| Intra-reviewer kappa | 0.6414 | Moderate/substantial, not near-perfect |
| `passes_targets` | false | Overall failure |

The filter misses too much human-relevant content, which could bias later topics and sentiment. Policy, prompts, thresholds, label mapping, sampling integrity and coding guidance require calibration.

The reviewer CSV has 332 rows instead of the intended 330. It contains 302 apparent base rows plus 30 repeats, with two rows missing document/context fields. Investigate and repair these before treating the package as final. After tuning, evaluate on a new or clearly separated holdout rather than optimistically reusing the same one.

## 12. Topic discovery: implemented but blocked

No `topic_assignments`, `topic_info`, `taxonomy.csv`, or BERTopic model artifact exists. The stage is not completed.

When the gate passes, selected units comprise relevant Instagram documents plus all ready blogs. OpenAI `text-embedding-3-large` embeddings are requested in batches of 256. UMAP uses cosine distance, 15 neighbors, five components, minimum distance zero and seed 42. HDBSCAN uses Euclidean distance, `eom` selection, prediction data, and a corpus-derived minimum cluster size. BERTopic uses English stop-word unigram/bigram count vectors and calculates probabilities; at least ten units are required.

The stage writes assignments and topic information and creates an initial taxonomy only when absent, allowing researcher approval or remapping. Evaluation should include coherence, outliers, stability across settings/seeds, source imbalance, and the weighting effect of multiple chunks per blog.

## 13. ABSA: implemented but blocked

No `aspect_mentions` or `absa_processing_log` artifact exists. When allowed, ABSA uses the same gated units and an approved taxonomy, or the predefined aspect list if no taxonomy exists. Strict output may contain zero or more mentions with aspect, target, positive/neutral/negative/mixed sentiment, confidence, exact evidence, gloss, explicit/implicit expression, contributing hashtags/emoji, emerging-aspect name, language, and code-switching note. Stable IDs preserve unit/document/source/year/URL lineage.

The prompt requires direct multilingual analysis before any gloss and forbids mentions without evaluative content. Before thesis use, ABSA needs a multilingual human-coded test set, aspect/sentiment metrics, evidence-span checking, and clear rules for multiple mentions and mixed polarity.

## 14. Dashboard and interfaces

`streamlit run app.py` opens the research dashboard. It filters by source, year, primary language, mixed status and method, and has overview, language/relevance, topic, sentiment, hashtag/emoji, evidence and validation tabs. It loads available artifacts defensively. It is descriptive and does not certify that validation gates passed. Thesis screenshots must state artifact date, filters, and whether data are pilot or full corpus.

`.\.venv\Scripts\python.exe -m streamlit run relevance_labeler.py` opens the human review interface described above.

## 15. Artifact and schema register

| Artifact | Rows/items | Interpretation/status |
|---|---:|---|
| Instagram raw JSON | 14,591 | Preserved sources |
| `instagram_cleanded.csv/.xlsx` | 14,049 | Cleaned Instagram data |
| Blog `raw-data.json` | 28 | Authoritative blog input |
| `documents.csv/.parquet` | 14,077 | Combined 30-column document table |
| `units.csv/.parquet` | 14,116 | 18-column analysis units |
| `validation_sample.csv/.parquet` | 114 | Earlier unannotated general sample |
| `relevance.csv/.parquet` | 599 | Partial 27-column relevance results |
| Current relevance review queue | 0 | Empty for classified subset only |
| Relevance validation key | 300 | Hidden model decisions |
| Relevance reviewer sample | 332 | 30 repeats plus two suspect rows |
| Classification report | 6 rows | Holdout label/aggregate metrics |
| Confusion matrix | 3 rows | Human-by-model labels |
| Subgroup metrics | 34 rows | Accuracy/recall by group |
| Validation metrics JSON | 1 object | Headline values and failed flag |
| `run_manifest.jsonl` | 10 runs | UTC preparation/relevance provenance |
| Topic/ABSA artifacts | 0 | Not executed/completed |

Document columns cover identifiers/source metadata, four text views, extracted features, duplicate lineage, status, language components/confidence/margin, mixed/code-switching state, method/reason and adjudication state. Unit columns add chunk/sentence indexes and tokens. Relevance columns preserve every model/human decision and API audit field. Exact schemas are in the corresponding CSV headers and source modules.

## 16. Reproduction commands

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
python -m marathon_absa.cli download-language-model
python -m marathon_absa.cli prepare
```

Paid stages require `.env` plus explicit authorization:

```powershell
python -m marathon_absa.cli language-review --run-api --source blog
python -m marathon_absa.cli relevance --run-api --pilot-size 500
python -m marathon_absa.cli relevance-review-sample --size 300 --repeats 30
.\.venv\Scripts\python.exe -m streamlit run relevance_labeler.py
python -m marathon_absa.cli relevance-validate
python -m marathon_absa.cli relevance --run-api
python -m marathon_absa.cli relevance-finalize
python -m marathon_absa.cli topics --run-api
python -m marathon_absa.cli absa --run-api
streamlit run app.py
```

Use `--limit N` for controlled pilots. Before final thesis execution, freeze resolved dependencies (`pip freeze` or lock file), Python/OS/hardware, exact model identifiers, OpenLID checksum, prompt versions, settings, timestamps, API token use/cost, and random seeds.

## 17. Verification status

On 24 July 2026, `.\.venv\Scripts\python.exe -m pytest -q` returned **51 passed in 18.33 seconds**. Tests are deterministic and unpaid. They support local transformation, routing, gate and file-safety correctness, but do not prove construct validity, language accuracy, topic coherence or ABSA accuracy.

Empirical relevance validation is the applicable quality evidence and failed two of three targets. The DOCX process document has 17 rendered QA page images. This Markdown record was reconciled against source code, tests, raw/processed schemas, row counts, manifest and validation files.

## 18. Limitations and research risks

1. Hashtag retrieval is not a complete sample of KLSCM discourse.
2. Privacy, deletions, scraper/platform behavior and July 2026 availability shape historical data.
3. Hashtag-year labels are not guaranteed event/publication years.
4. Exact deduplication misses near-duplicates and copied material across authors.
5. Fourteen thousand Instagram rows versus 28 blogs creates severe source imbalance.
6. Multiple posts per author and chunks per review are not independent observations.
7. 5,808 pending language adjudications and 593 undetermined labels limit subgroup claims.
8. Malay/Indonesian ambiguity is explicitly unresolved for 1,637 records.
9. Hosted models can drift unless snapshots/versions are frozen.
10. Earlier prepare settings differ from current settings.
11. Failed relevance recall would systematically omit relevant material.
12. Two validation rows appear malformed.
13. Repeat kappa is intra-reviewer, not independent inter-coder agreement.
14. Translation may influence reviewer interpretation; originals must remain primary.
15. Blog collection and inclusion methodology is not documented.
16. Topic and ABSA empirical validation has not occurred.
17. Author IDs/URLs require ethical review, minimization and controlled access.
18. No usable Git history limits exact implementation chronology.

## 19. Ethics and governance

Public-facing text and URLs can still identify individuals. The thesis must cover institutional approval, platform terms, secure storage, access controls, data minimization, quotation/searchability risk, anonymization and publication policy. Exact evidence is valuable for audit but can enable re-identification. Raw data, `.env`, API caches, downloaded models and virtual environments must stay uncommitted. A publication dataset may require a separately governed de-identified export.

## 20. Required work before thesis results

1. Repair and explain the two suspect validation rows.
2. Audit false negatives, confusion matrix and subgroup metrics.
3. Freeze revised relevance policy, examples, prompt, thresholds and coding guide.
4. Add an independent coder if feasible and evaluate on a new/separate holdout.
5. Meet targets or transparently justify/preregister revised targets.
6. Decide and document how pending language cases will be validated/resolved.
7. Rerun prepare with frozen final settings after language/config changes.
8. Run full-corpus relevance only after pilot validation passes.
9. Resolve every queue record and verify the topic gate.
10. Run/stability-check topics and approve the taxonomy.
11. Create and evaluate a multilingual human-coded ABSA sample.
12. Run ABSA, verify evidence lineage, then create thesis figures/tables.
13. Add the missing blog collection and ethics protocol.
14. Preserve final dependencies, checksums, prompt/model versions, cost and hardware.

## 21. Thesis chapter mapping

- **Introduction/background:** research problem and objectives (Sections 2ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“3).
- **Data/sampling:** timeline, sources, cleaning and bias (Sections 3ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“5, 18).
- **Methodology:** text, language, chunking, relevance, validation, topics and ABSA (Sections 6ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“13).
- **Implementation:** repository map, interfaces, schemas and commands (Sections 4, 14ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“16).
- **Results currently defensible:** cleaning counts, corpus descriptors, machine language outputs with caveats, partial relevance results, and failed validation metrics.
- **Discussion/limitations:** verification, validity, ethics and risks (Sections 17ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“19).
- **Future work:** required completion gates (Section 20).
- **Appendices:** full schemas, prompts/versions, environment lock, manifest, coding guide, subgroup tables and ethics protocol.

## 22. Final status statement

The repository is a structured and unit-tested foundation with reproducible source cleaning, parallel linguistic/semantic text, pinned hybrid language detection, auditable relevance routing, safe human labeling, explicit topic gates and blind validation. Its own evidence shows why those gates matter: relevance precision passed, but recall and macro-F1 failed substantially. Full relevance coverage is absent, language uncertainty remains, and topic/ABSA artifacts do not exist. The defensible thesis position is that preparation and pilot validation are complete, the relevance filter requires recalibration, and thematic/sentiment findings remain future work until quality conditions are met.

## 23. Relevance calibration handoff after v3

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


## 24. Targeted relevance audit and v4 calibration preparation (2026-07-25)

A calibration-only annotation-policy audit was performed against the completed
archived reviewer labels and the isolated v3 predictions. Of 150 calibration
records, 41 had v3 `relevant` against human `irrelevant` (36) or `ambiguous`
(5). The artifact
`data/processed/relevance_calibration_disagreement_audit_v4.csv` preserves the
original label and records a non-applied proposal, category, reason, and audit
time. Proposals comprise 18 policy-inconsistent labels to `relevant`, 12
incidental/another-event cases retained as `irrelevant`, 3 promotions retained
as `irrelevant`, and 8 weak/image-dependent cases proposed as `ambiguous`.
These are proposals, not revised gold labels.

The v4 policy requires explicit textual attribution of the journey cue to
KLSCM; another event as subject, pure commercial calls-to-action, and incidental
hashtag lists remain irrelevant, while image-dependent vagueness remains
ambiguous. Prompt/cache identifiers are now `relevance-v4`,
`relevance-adjudication-v4`, and `relevance-rule-v4`. The rescore command is
split-aware and defaults to calibration, protecting the holdout during tuning
and production outputs throughout validation. No paid v4 rescore has yet been
executed, so no v4 performance metric is claimed. The existing inspected
holdout is not pristine and a fresh independent holdout is required after v4 is
frozen. Verification evidence: full pytest on 2026-07-25, 56 passed in 15.85s;
the CLI rejected a rescore without `--run-api` with exit code 2. Next command:
`.\.venv\Scripts\python.exe -m marathon_absa.cli relevance-rescore-validation --split calibration --run-api`.
## 25. V4 calibration evidence and policy-audited interpretation (2026-07-25)

The paid v4 run was executed only on the 150-record calibration split and wrote
isolated artifacts under `data/processed/relevance_v4_calibration/`. Raw metrics
were precision 0.8000, recall 0.8571428571 and ternary macro-F1 0.4875792407;
the raw gate failed. A bidirectional policy audit expanded the disagreement
register to 50 cases, of which 36 produce an actual label change in a separate
overlay. Original blind labels and production relevance outputs remain intact.

Under the policy overlay, relevant precision is 0.9523809524, recall is
0.9345794393 and descriptive ternary macro-F1 is 0.6539363083. Because ambiguous
is a human-review routing state rather than a substantive endpoint, a separately
reported binary analysis excludes eight gold-ambiguous cases and treats model
ambiguous as routed/not relevant. On n=142, precision is 0.9615384615, recall is
0.9345794393 and macro-F1 is 0.8985911835. These results pass the numeric targets
but are calibration evidence, not a final quality claim. The metric protocol
must be frozen in advance, ambiguous routing/coverage reported separately, and
a fresh blind independent holdout built without any archived validation
records. Until that holdout passes, relevance quality remains unresolved and
production classification/topic discovery remain blocked. Full repository verification after recording these artifacts was 56 passed in 17.88s.
## 26. Fresh independent relevance holdout (2026-07-25)

A deterministic fresh-holdout builder was implemented and executed without API
calls. It excludes document IDs from both earlier reviewer samples, separates a
blind reviewer CSV from hidden CSV/Parquet keys, uses seed 1042, adds controlled
repeat rows and refuses overwrite. The generated reviewer artifact contains 150
base records plus 15 repeats; all 165 labels are blank. The hidden key contains
150 unique documents with zero overlap against prior validation samples and
sampling strata of 86 production-v3 relevant, 59 irrelevant and 5 ambiguous.
These model strata support coverage only and are not v4 evaluation results.
Artifacts are `data/processed/relevance_fresh_holdout_sample.csv` and
`relevance_fresh_holdout_key.{csv,parquet}`. Full verification was 57 passed in
15.23s. The reviewer must remain blind to the key. After independent labeling,
a separate tested v4 rescore path is required before any final quality claim.
## 27. Frozen v4 fresh-holdout evaluation path (2026-07-25)

The one-time independent evaluation command is implemented but not executed.
It validates holdout cardinality, label validity, key alignment, prior-sample
leakage, and prepared-document completeness before API access; excludes repeat
rows from scoring; records SHA-256 input hashes; refuses overwrite; and writes
only to `data/processed/relevance_v4_fresh_holdout/`. The substantive gate is
relevant precision >=0.85, recall >=0.90, and binary macro-F1 >=0.80. Ambiguous
is reported as routing and ternary metrics remain descriptive. Preflight hashes
and label/agreement counts are stored in
`data/processed/relevance_fresh_holdout_preflight.json`. Full verification was
59 passed in 16.07s; no paid call or production mutation occurred. The frozen
one-time command is `.\.venv\Scripts\python.exe -m marathon_absa.cli relevance-rescore-fresh-holdout --run-api`.
## 28. Final independent v4 relevance evaluation (2026-07-25)

The frozen v4 classifier was evaluated once on the independently labeled fresh
holdout. Run hashes matched preflight hashes and artifacts were isolated under
`data/processed/relevance_v4_fresh_holdout/`; no individual-error inspection or
post-result tuning occurred. On 150 records (34 relevant, 116 irrelevant),
precision was 0.3469387755, recall 1.0 and substantive binary macro-F1
0.5670995671. The classifier routed 33 records as ambiguous (22%) and the
aggregate confusion matrix showed 34 relevant true positives, 64 irrelevant
records predicted relevant, 33 irrelevant routed ambiguous and 19 irrelevant
predicted irrelevant. Repeat agreement was 14/15; kappa was 0.0 under extreme
repeat-class imbalance. Precision and macro-F1 failed the predeclared gate, so
v4 is not defensible for production and topics/ABSA remain blocked. This
holdout is now spent and must not be reused for tuning or another release claim.
A future v5 cycle requires separate development evidence and another unseen
final holdout. There is no authorized paid next command.
## 29. V5 relevance development cycle initiated (2026-07-25)

Following the failed independent v4 gate, a formally separate v5 development
cycle has begun. The spent v4 fresh holdout remains immutable and is excluded
from v5 tuning; no individual fresh-holdout errors were inspected. V5 is based
only on the previously available annotation-policy audit, calibration evidence,
and the aggregate v4 failure mode (adequate recall with severe overprediction
of relevance).

V5 changes the decision architecture as follows. Every initial `relevant`
decision now requires independent stronger-model adjudication, including
high-confidence first-pass results. Auto-resolved adjudicated relevance requires
confidence >=0.90 rather than 0.75. Both prompts apply a strict two-condition
substantive test: KLSCM must be the actual event described and the caption must
state a concrete KLSCM action, status, information, experience, result, or
support relationship. Generic motivation, ordinary exercise, dates, distances,
photo context, apparel, celebration, proximity to a hashtag, another-event
content, and promotion cannot establish relevance by themselves. This is a
precision-oriented correction while retaining explicit short KLSCM journey
statements as relevant.

Prompt/cache identifiers are `relevance-v5`,
`relevance-adjudication-v5`, and `relevance-rule-v5`. Calibration rescoring will
write to the new isolated `data/processed/relevance_v5_calibration/` directory;
v4 calibration, final holdout, labels, keys, production relevance, and topic
artifacts are not overwritten. Regression tests cover mandatory relevant
adjudication, rejection below the 0.90 acceptance threshold, and acceptance at
0.90. Full repository verification: 60 passed in 15.37s, with four known
sklearn warnings from synthetic single-class repeat fixtures. The paid CLI guard
was verified (exit 2 without `--run-api`), no v5 API call has been made, and the
v5 calibration directory does not yet exist.

The exact next command is calibration-only and paid:
`.\.venv\Scripts\python.exe -m marathon_absa.cli relevance-rescore-validation --split calibration --run-api`.
Its raw results must be interpreted alongside the existing policy-label overlay;
no archived holdout, spent fresh holdout, production relevance, topics, or ABSA
may be run. If v5 is frozen after calibration, a newly sampled and independently
labeled unseen holdout will be required for any final claim.
## 30. Policy stabilization package and v6 infrastructure (2026-07-25)

V5 diagnosis showed a precision/recall oscillation rather than a viable gate:
v4 recovered all fresh-holdout relevant records but overpredicted relevance,
whereas v5 calibration precision rose to 0.8958 while recall fell to 0.4388.
Within the policy-overlay calibration labels, v5 produced 48 relevant, 41
ambiguous and 18 irrelevant decisions for 107 policy-relevant records. The
archived development labels (about 65% relevant) and independent fresh labels
(23% relevant) also demonstrate a material annotation-policy regime difference.
Consequently, no further paid v5 work is authorized and confidence-threshold
adjustment alone is not methodologically adequate.

A deterministic 60-record calibration-only second-coder package was generated.
Its blind reviewer file is
`data/processed/relevance_policy_alignment_second_coder.csv`; the hidden key is
`relevance_policy_alignment_key.csv`; and the written rubric is
`relevance_policy_alignment_rubric.md`. Exact hidden strata are 20 original
human relevant/v5 irrelevant, 20 original human relevant/v5 ambiguous, 10
original nonrelevant/v5 relevant-or-ambiguous disputes, and five relevant plus
five irrelevant agreement controls. The reviewer file contains no prior label,
model decision, confidence, version, or stratum. It requires a binary second
label, visual-context flag, controlled reason, exact evidence span and notes.
SHA-256 values are respectively
`6983c680e5f9d3588b182c6d2a89b451c62661ec4da90638ca988aafe6b75b06`,
`70767bb543e2dcd602b875b971046ed9c1f56dae079962e22c763767521a9e29`, and
`9d4ecd796ce9d1eee121ff9bc3b2fd8fc03667182f06284bd4e7d81da0e3b78a`.

The non-paid `relevance-policy-alignment-finalize` command validates all 60
second-coder labels, visual flags, controlled reasons and evidence spans; then
writes a new append-only audit, agreement JSON and disagreement-only
adjudication queue. It refuses to overwrite existing finalization artifacts and
never changes the original reviewer, archived labels or hidden keys.

V6 implementation infrastructure is prepared but not API-executed. It adds a
structured verification schema with binary substantive relevance,
`klscm_is_actual_subject`, `concrete_event_relation`, controlled exclusion and
confidence fields. Every candidate requiring adjudication uses
`relevance-verification-v6`; relevance can auto-resolve only when both evidence
gates are true, exclusion is none and confidence is at least 0.80. The stored
substantive decision and routing status are separate, so ambiguous remains a
review route rather than a substantive class. Initial/rule cache versions are
`relevance-v6` and `relevance-rule-v6`; calibration output will be isolated at
`relevance_v6_calibration/`.

Reusable reproducible binary evaluation now supports optional sampling weights
and percentile bootstrap confidence intervals with recorded seed/sample count.
Tests cover structured evidence gating, secondary confidence routing, schema
audit fields, bootstrap reproducibility/weighting, second-coder finalization
validation and append-only overwrite refusal. Full suite: 65 passed in 15.63s,
with four known sklearn warnings from synthetic single-class repeat fixtures.
The paid guard exits 2 without `--run-api`; no v6 API call or v6 calibration
artifact exists.

Current stage is blocked on independent second coding, not API execution. The
second coder must receive only the blind CSV and rubric, never the hidden key.
After all 60 rows are complete, run the non-paid finalizer, adjudicate its
resulting disagreement queue, and only then authorize v6 calibration. No new
holdout will be created before v6 is frozen.
## 31. Streamlit policy-alignment labeling interface (2026-07-26)

`policy_alignment_labeler.py` now provides a dedicated blind Streamlit
interface for the 60-record second-coder study. It reads only
`relevance_policy_alignment_second_coder.csv` and never loads the hidden key.
One caption is shown at a time with substantive relevance, visual-context,
controlled-reason, exact-evidence and optional-notes inputs. A complete form is
saved atomically, one timestamped backup is retained per session, exact evidence
is checked against the caption, progress is displayed, and the interface moves
to the next incomplete record.

Focused interface tests passed 28/28. Full repository verification passed
69 tests in 37.33 seconds; the four warnings remain the previously documented
synthetic single-class sklearn warnings. A headless Streamlit smoke test returned
`health=ok` on local port 8512. No labels, hidden keys, production relevance
outputs, prompts or API caches were changed by implementation or testing.

The non-paid launch command is:
`.\.venv\Scripts\python.exe -m streamlit run policy_alignment_labeler.py`.
After all 60 forms are complete, stop Streamlit and run
`.\.venv\Scripts\python.exe -m marathon_absa.cli relevance-policy-alignment-finalize`.
## 32. Automatic translation and persistent rubric in policy labeler (2026-07-26)

The policy-alignment Streamlit interface now automatically requests a faithful
English translation whenever a non-empty caption's recorded language is not
English, Malay/Bahasa Melayu, Chinese or Mandarin. Indonesian is intentionally
translated under this policy. Translation uses the project's cached OpenAI
service with the separate stage/version
`policy_alignment_translation`/`policy-alignment-translation-v1`, preserves
names, handles, hashtags, emoji, URLs and KLSCM terminology, and forbids
summarization, classification or added facts. The interface discloses that the
first translation of a caption may incur OpenAI cost; subsequent identical
requests use the local cache. Translation failure is shown without preventing
manual labeling.

The complete annotation rubric is loaded from
`relevance_policy_alignment_rubric.md` into the Streamlit sidebar on every
record screen. This keeps the policy visible in every labeling window while the
hidden alignment key remains unread and unexposed. Focused translation and
labeler coverage passed 15 tests, including exemption behavior and the separate
cache contract. Full repository verification passed 80 tests in 136.47 seconds;
the four existing sklearn warnings are unchanged. A post-change headless
Streamlit health check returned `ok`. No relevance classification, topic, ABSA
or v6 calibration call was run. Launch remains
`.\.venv\Scripts\python.exe -m streamlit run policy_alignment_labeler.py`.
## 33. Feeling-inclusive v7 relevance relabelling initiated (2026-07-27)

The relevance construct was revised because topic discovery is intended to capture both event themes and experiential states. Under `feeling-inclusive-v7`, relevance is not conditional on naming a start-to-finish marathon-journey stage. Text is relevant when it defensibly links KLSCM to an event theme, an emotional or physical experience, or both. A single KLSCM-linked feeling (for example excitement, nervousness, energy, tiredness, pain, relief, happiness, disappointment, achievement, or pride), multiple feelings, a conventional event theme, or their combination can qualify. Generic feelings or activities without a defensible KLSCM connection remain irrelevant. This policy is implemented in `marathon_absa/openai_service.py`, the persistent Streamlit rubric, and the versioned relabelling workflow in `marathon_absa/relabel.py`. Production cache identifiers are `relevance-v7`, `relevance-verification-v7`, and `relevance-rule-v7`, preventing reuse of decisions cached under the earlier construct.

Hashtag-only captions are now deterministic audit exclusions rather than ambiguous human-review cases. `deterministic_result` records them as irrelevant with `review_status=auto_excluded`, `adjudication_method=deterministic_exclusion`, `exclusion_status=excluded`, `exclusion_reason=hashtag_only`, and `include_in_topics=False`; they incur no classification or translation API call and cannot enter topics or ABSA. They remain in the document/relevance lineage and are not deleted. Direct inspection of `data/processed/documents.parquet` on 2026-07-27 found 13,799 ready Instagram records, of which 338 had empty `linguistic_text` and 13,461 contained eligible linguistic text. These counts describe the prepared artifact last written on 2026-07-22; the production relevance stage has not been rerun under v7.

The new single-reviewer workflow is staged as a 100-record policy pilot with 10 blind repeats, a planned 300-record calibration set with 30 repeats, and a later disjoint 300-record fresh holdout with 30 repeats. Samples are stratified across year, recorded language, language status, and text length, with explicit feeling/physical-cue coverage. Files are versioned, blind to model/prior decisions, mutually exclusive across stages, and refuse overwrite. Development samples allow `relevant`, temporary `needs_review`, and `irrelevant`; finalization refuses unresolved `needs_review` values and reports repeat agreement. The Streamlit labeler retains atomic autosave and session backup behavior, displays the v7 rubric persistently, and offers cached English translation only outside English, Malay/Indonesian, and Chinese. Translation remains optional, may incur API cost on a cache miss, and never replaces original-text evidence.

The pilot-generation command was executed on 2026-07-27, as recorded in `run_manifest.jsonl`. It created `relevance_v7_pilot_sample.csv` with 110 reviewer rows (100 originals and 10 blind repeats) plus CSV/Parquet keys. The base sample contains 43 feeling/physical-cue challenge records and 57 representative records. Labels are currently blank, so policy-pilot labelling is initiated but not completed or finalized. Calibration, fresh-holdout evaluation, production v7 classification, topic discovery, and ABSA remain unexecuted and blocked on the preceding gates.

Verification on 2026-07-27 used `.\.venv\Scripts\python.exe -m pytest -q` and returned **83 passed in 15.82 seconds**, with four existing sklearn warnings from synthetic single-label validation cases. A headless Streamlit smoke test against the v7 pilot returned `health=ok` on local port 8513. The implemented commands are:

```powershell
.\.venv\Scripts\python.exe -m marathon_absa.cli relevance-v7-pilot-sample
.\.venv\Scripts\python.exe -m marathon_absa.cli relevance-v7-finalize --stage pilot
.\.venv\Scripts\python.exe -m marathon_absa.cli relevance-v7-calibration-sample
.\.venv\Scripts\python.exe -m marathon_absa.cli relevance-v7-fresh-holdout-sample
$env:RELEVANCE_LABEL_CSV='data/processed/relevance_v7_pilot_sample.csv'
.\.venv\Scripts\python.exe -m streamlit run relevance_labeler.py
```
## 34. V7 calibration completion and repeat-disagreement audit (2026-07-29)

The feeling-inclusive v7 calibration reviewer file was completed and finalized before any v7 model scoring. It contains 300 base records and 30 blinded repeats. Base labels are 223 relevant and 77 irrelevant; 25 of 30 repeats agree, giving raw intra-reviewer agreement of 0.8333. This is reliability evidence, not classifier performance. The finalized summary is `data/processed/relevance_v7_calibration_summary.json`, and the manifest records `relevance_v7_finalize` with `relabel_stage=calibration`.

A new non-paid, overwrite-protected `relevance-v7-repeat-audit` command created `data/processed/relevance_v7_calibration_repeat_disagreements.csv`. It preserves five inconsistent pairs without changing either blind decision and provides blank `adjudicated_label` and `adjudication_reason` fields. Three pairs changed from relevant originally to irrelevant on repeat; two changed from irrelevant to relevant. All five are representative rather than feeling-cue-enriched records, and their recorded languages are French, Japanese, Korean, Malay/Indonesian uncertain, and Vietnamese. These cases require policy review with original-text evidence and optional translation before the v7 policy and calibration gold labels are treated as frozen. A separate five-row `relevance_v7_calibration_repeat_adjudication_queue.csv` was created so the original and repeat decisions remain immutable evidence. The standard Streamlit labeler displays both decisions, requests a fresh binary adjudication, and retains the existing language-based optional translation behavior.

An isolated paid scorer is implemented as `relevance-v7-score-calibration --run-api`. It validates finalized calibration inputs, scores only the 300 base document IDs with the v7 cache contracts, writes only under `data/processed/relevance_v7_calibration_results/`, calculates overall and subgroup metrics against the human labels, and refuses overwrite. It does not read repeats as scoring inputs, alter production `relevance`, or create/open the fresh holdout. It now requires all five repeat-adjudication rows to have final binary labels and overlays those adjudications only on the corresponding base gold labels used for calibration metrics. The paid scorer has not been executed. Calling it without `--run-api` was verified to stop at the explicit cost gate.

Full deterministic verification on 2026-07-29 returned **84 passed in 51.31 seconds**, with the four existing sklearn warnings from synthetic single-label validation cases. The repeat audit command then executed successfully and recorded five disagreements. A headless Streamlit smoke test against the adjudication queue returned `health=ok` on port 8514. Topic discovery, ABSA, v7 paid calibration scoring, fresh-holdout creation, and full-corpus v7 relevance remain unexecuted.
## 34. Event-experience binary relevance v8 implemented but not calibrated (2026-07-29)

Phase 3 introduced the versioned `v8_event_experience_binary` relevance design without executing paid classification. The implementation is additive: historical v7 prompts, schemas, results, labels, review artifacts, and cache files remain available and unchanged. No v8 performance improvement is claimed because the 300-record calibration set has not been rescored.

The model-facing contract is `RELEVANCE_V8_ASSESSMENT_SCHEMA` in `marathon_absa/schemas.py`. It returns event connection (`explicit`, `supported`, `weak`, or `none`), primary and secondary content types, meaningful-content status, two grounded evidence fields, confidence, contradiction and image-dependence signals, review recommendation, language notes, and a short explanation. It does not return an operational reason code, routing outcome, human decision, or final eligibility. `RELEVANCE_V8_INSTRUCTIONS` in `marathon_absa/openai_service.py` is a new prompt rather than a v7 patch and contains 40 targeted multilingual/borderline examples. Explicit means KLSCM is named in the semantic clause containing the meaningful content; supported means a hashtag or permitted textual metadata supplies event context alongside meaningful content.

`marathon_absa/relevance_v8.py` deterministically derives stable reason codes and separates model assessment, routing, human decision, and final inclusion. Initial candidate thresholds are 0.80 for automatic inclusion, 0.93 for automatic exclusion, and 0.98 for the separately reported weak/no-content exclusion exception. Weak meaningful or image-dependent records require review. Only weak records with no meaningful semantic content, no grounded evidence, and confidence at least 0.98 may use `automatic_exclude_weak_empty`. Material model disagreement is limited to differences that affect inclusion or routing; disagreement between included subtypes such as logistics and organization is retained without automatically causing relevance review. The stronger-model `relevance_v8_verification` stage uses the same assessment schema for weak, none, excluded-family, low-confidence, contradictory, image-dependent, or model-recommended-review cases; both assessment JSON payloads and material-agreement status are preserved.

Final human decisions are `include` or `exclude`; `unresolved` is an operational state that blocks final use rather than a label. The v8 derived gates are `v8_include_in_topics` and `v8_include_in_sentiment`. Every final include enters topic discovery. Included `event_information` records do not automatically enter sentiment analysis. The annotation policy is stored in `RELEVANCE_V8_ANNOTATION_GUIDELINE.md`.

Validation is binary and excludes hidden repeats from the main record count. It reports automatic classifier performance, routing performance, and `simulated_human_assisted_performance` separately. Review-routed records use human gold only in the explicitly simulated view. Classifier quality and annotation quality have separate gates; missing repeat evidence yields unavailable annotation quality rather than silently changing the classifier result. Missing/malformed human labels, duplicate unique records, and malformed final predictions block evaluation. The implementation also writes a binary confusion matrix, threshold grid, subgroup table, and error-audit table when controlled calibration is later executed.

The paid, isolated calibration command is:

```powershell
python -m marathon_absa.cli relevance-v8-score-calibration --run-api
```

It requires the explicit cost flag, reads the completed v7 calibration labels as a versioned development overlay, uses the new `relevance_v8_initial`/`v8` cache identity, refuses to overwrite an existing `data/processed/relevance_v8_calibration_results/` directory, and does not write production `relevance.parquet`. This command was not executed on 2026-07-29. Final non-paid verification on 2026-07-29 passed 108 tests with four pre-existing sklearn warnings from legacy fresh-holdout single-label fixtures. This test result verifies implementation behavior only and is not classifier-performance evidence.

## 35. V8 strict Structured Outputs schema compatibility fix (2026-07-29)

The first attempted paid `relevance-v8-score-calibration --run-api` invocation failed before inference because the Responses API rejected `uniqueItems` in the `secondary_content_types` array within the strict `relevance_v8_initial` JSON schema. The API returned HTTP 400 `invalid_json_schema`; no v8 calibration result directory was created, so the calibration remains unexecuted and no classifier-performance evidence was produced.

`RELEVANCE_V8_ASSESSMENT_SCHEMA` in `marathon_absa/schemas.py` now retains the controlled string enum for every secondary content type but omits the unsupported `uniqueItems` keyword. This does not change deterministic routing semantics because downstream v8 logic tests content-type membership rather than list cardinality. A regression test in `tests/test_relevance_v8.py` now checks that this unsupported strict-schema keyword is absent. This compatibility change was made from the API failure evidence supplied on 2026-07-29; a successful paid retry is still required before the calibration can be described as executed. Non-paid verification on 2026-07-29 passed the focused v8 suite (25 tests) and the complete suite (109 tests), with four pre-existing sklearn warnings from legacy single-label fixtures.



## 35. Formal v8 human-adjudication workflow initialized (2026-07-30)

A separate corpus-construction workflow was implemented in `marathon_absa/adjudication.py` after the Phase 4 gold-standard audit identified 54 disputed historical labels. The implementation does not alter relevance prompts, thresholds, routing, schemas, model outputs, historical labels, or existing calibration results. CLI commands in `marathon_absa/cli.py` initialize blinded files, import immutable reviewer snapshots, compute agreement, generate a disagreement-only adjudication report, and finalize append-only gold versions.

The initialized package is `data/processed/relevance_v8_adjudication/`. Its manifest records workflow version `v8_gold_adjudication_v1`, seed 8042, 54 disputed records, source and output SHA-256 hashes, UTC creation time, exported columns, and the hidden-field policy. Reviewer templates expose only record ID, caption, recorded language, event year, and blank label/evidence/rationale/confidence/comment fields. Historical labels, model results, confidence, explanations, routing, reason codes, event connection, and content type are not exposed. The two working files are separate and must be imported under different reviewer identities.

Completed imports are stored as non-overwriting snapshots with reviewer identity, version, UTC timestamp, original file path, and SHA-256 hash. Agreement analysis reports Cohen’s kappa, percent agreement, and a two-label confusion matrix overall and by language, event connection, and primary content type. Model-derived subgroup fields are joined only after decisions are frozen. The generated adjudication worksheet contains disagreements only. Finalization refuses incomplete adjudication and writes a new versioned gold CSV containing historical, reviewer, adjudicated, evidence, rationale, identity, timestamp, original-file, and hash provenance.

`relevance_v8_labeler.py` is the dedicated blinded Streamlit interface for the two reviewer working files. Its sidebar selects the assigned reviewer A or B file and warns reviewers not to inspect the other working copy. The form enforces the downstream import contract: `include` or `exclude`, non-empty evidence and rationale, and confidence from 0 to 1. It creates one session backup and atomically saves each completed record. An on-demand toggle uses the cache-backed OpenAI service to show a faithful English translation without replacing the original caption; a cache miss may incur API cost. Evidence must remain grounded in the original caption. Run it with `.\.venv\Scripts\python.exe -m streamlit run relevance_v8_labeler.py`. Verification on 2026-07-30 returned 16 focused adjudication/labeler tests passed, a healthy headless Streamlit response against the real 54-record package, and 125 full-suite tests passed in 54.89 seconds with four pre-existing sklearn warnings from synthetic single-label fixtures.

`RELEVANCE_V8_ADJUDICATION_METHODOLOGY.md` is the operating protocol. At implementation time the reviewer templates are blank, so no agreement statistics, adjudication decisions, or final gold labels yet exist. Those stages are implemented but not executed; they require two independent completed human reviews and a completed adjudicator worksheet. The eventual 54-record final file will be joined as a versioned overlay onto a copy of the 300-record calibration set before the next calibration evaluation, never written back over historical labels.


## 36. Phase 6 human-review import and agreement analysis (2026-07-30)

The two completed 54-record blinded reviewer files were validated against the canonical template and Phase 5 manifest before either was imported. Both contained exactly the expected unique IDs, valid `include`/`exclude` labels, complete evidence/rationale/confidence fields, unchanged caption/language/year values, and no historical-label or classifier-field leakage. The annotated working files necessarily differed from their initial blank-file hashes, but only authorized annotation fields changed; the canonical-template and Phase 4 candidate-source hashes continued to match the manifest. Source reviewer files were not overwritten.

Append-only snapshots are `data/processed/relevance_v8_adjudication/reviewer_imports/reviewer_a__v1.csv` and `reviewer_b__v1.csv`, with sidecar JSON metadata recording pseudonymous reviewer identity, independent-reviewer role, annotation version, UTC import time, original path and SHA-256 hash, imported path and hash, and zero rejected, duplicate, missing, or invalid rows. The validation and import reports are `reviewer_import_validation_report.json` and `reviewer_import_report.json`.

Overall agreement was 23/54 (42.5926%) with 31 disagreements (57.4074%), Cohen's kappa 0.118019, and supplementary PABAK -0.148148. Reviewer A labeled 16 include and 38 exclude; Reviewer B labeled 47 include and 7 exclude. The include/exclude confusion matrix with Reviewer A as rows and Reviewer B as columns is `[[16, 0], [31, 7]]`. Thus all disagreements were A=`exclude`, B=`include`, demonstrating a systematic boundary difference and strong prevalence imbalance. `annotation_quality_passes` is false against the predeclared kappa >=0.80 criterion; this is not a classifier failure.

Subgroup agreement is descriptive and stored in `agreement/subgroup_agreement.csv`. Of 31 disagreements, 26 occurred in the 43 automatic-false-inclusion-origin records and five in the 11 review-routed records. The Phase 4 hashtag-supported-experience category contributed 16 disagreements. Agreement was lower in the shortest and short caption quartiles than in longer captions. All 54 records contained hashtags, and many language/category groups were small, so rare subgroup kappa values are flagged as unstable rather than generalized.

The supervisor package in `data/processed/relevance_v8_adjudication/agreement/` contains overall metrics, confusion matrix, subgroup results, a 31-row independently randomized disagreement-only CSV, readable disagreement report, adjudicator instructions, inter-annotator agreement report, and hashed manifest. Adjudicator fields are blank. No final gold labels, calibration overlay, classifier metrics, prompt/threshold changes, paid API calls, or model runs were produced. Supervisor adjudication of all 31 disagreements is the required next stage.

`relevance_v8_supervisor_labeler.py` is the dedicated Streamlit interface for `agreement/relevance_v8_supervisor_adjudication.csv`. It displays the original caption, permitted metadata, candidate origin and disagreement category, and both reviewers' labels, evidence, rationales, confidence, and comments. The supervisor supplies an `include`/`exclude` decision, grounded rationale, and name; the interface writes an ISO 8601 UTC timestamp automatically. It uses a single session backup and atomic CSV replacement, tracks completed and pending disagreements, and offers a cache-backed English translation toggle without replacing the original caption. Classifier predictions, routing, historical labels, and Phase 4 proposed labels remain unavailable. Run it with `.\.venv\Scripts\python.exe -m streamlit run relevance_v8_supervisor_labeler.py`. Verification on 2026-07-30 returned 24 focused supervisor/labeler/Phase 6 tests passed, a healthy headless Streamlit response against the real 31-row worksheet, and 141 full-suite tests passed in 29.54 seconds with four pre-existing sklearn warnings from synthetic single-label fixtures. The worksheet remains unadjudicated; interface validation is not evidence that the 31 supervisor decisions have been executed.

## 37. V8 supervisor adjudications imported from NBH review (2026-08-13)

Supervisor NBH supplied the corrected workbook `C:\Users\user\Downloads\REVIEWING DATA NBH.xlsx`. Filesystem metadata recorded a local modification time of 2026-08-13 15:18:03 Asia/Kuala_Lumpur. The workbook contained a contiguous `No.` sequence from 1 through 31, one binary `NBH` decision and one non-empty `Rationale` for every row. Because the workbook did not contain stable record IDs or captions, its numbered rows were mapped in order to the independently randomized 31-record supervisor package. This preserves the package order documented in `agreement/SUPERVISOR_DISAGREEMENT_REPORT.md`; no model output, reviewer decision, caption, metadata, or candidate-category field was changed.

The import replaced only `adjudicator_label`, `adjudicator_rationale`, `adjudicator_name`, and `adjudication_timestamp` in `data/processed/relevance_v8_adjudication/agreement/relevance_v8_supervisor_adjudication.csv`. The `NBH` column supplied the labels, the `Rationale` column was retained verbatim, adjudicator identity was recorded as `NBH`, and a single UTC import timestamp of `2026-08-13T07:20:29.1720967+00:00` was applied to all 31 decisions. The workbook's incidental fourth-column notes were not imported because they were outside the requested adjudication fields. The pre-import target was preserved as `agreement/relevance_v8_supervisor_adjudication.backup_20260813_072029_178884.csv`.

Post-import validation found exactly 31 rows and 31 unique stable record IDs, with all 31 rows carrying a valid binary label and non-empty rationale. The completed distribution is 18 `include` and 13 `exclude`; all adjudicator names are `NBH` and all timestamps match the import event. This establishes that supervisor adjudication has been executed and completed. It does not by itself finalize the versioned gold overlay, rerun calibration, change prompts or thresholds, execute paid API inference, or establish classifier performance; those remain separate downstream stages.

## 38. V8 final adjudicated gold and post-adjudication calibration (2026-08-13)

Phase 7 validated the completed supervisor worksheet against the original hashed blank worksheet, the two append-only reviewer imports, and the Phase 6 package manifest. Validation passed for exactly 31 unique disagreement IDs, unchanged captions, metadata and reviewer fields, valid binary decisions, non-empty rationales, adjudicator identity `NBH`, valid UTC timestamps, and absence of hidden classifier or historical-label fields. The completed worksheet SHA-256 is `26083b847ee9d1d07ed4755d04cfeffedc4137fcf92bdc02c5da75e6bf7f560b`; the original blank worksheet remains preserved with SHA-256 `689c8ec904c644c2f31b12773399d9d36d36a8375fd3c38bf65eb6c0658b0302`.

The immutable append-only artifact `data/processed/relevance_v8_adjudication/final/relevance_v8_adjudicated_gold_v1.csv` contains all 54 disputed records. Twenty-three labels derive from reviewer consensus and 31 from supervisor adjudication; the final distribution is 34 include and 20 exclude. Relative to the preserved v7 labels, 33 changed from exclude to include, three changed from include to exclude, and 18 were unchanged. On the 31 disagreements, the supervisor selected Reviewer A's exclusion 13 times and Reviewer B's inclusion 18 times. This alignment is descriptive and is not evidence that either reviewer performed poorly. The artifact retains both reviewer records, historical label, supervisor provenance where applicable, resolution method, version, finalization timestamp, and source-manifest hash. `relevance_v8_adjudicated_gold_v1.manifest.json` hashes the gold, validation, statistics and subgroup-breakdown artifacts.

The final gold was joined by stable ID to a new copy of the 300 saved controlled-calibration prediction rows at `data/processed/relevance_v8_calibration_results/relevance_v8_calibration_adjudicated_v1.csv`. Exactly 54 gold fields were overlaid, all IDs matched, and the 59 frozen classifier/output columns were byte-logically unchanged in the in-memory validation; the remaining 246 labels retained their historical values. The 54-row overlay audit is stored separately. No classifier, OpenAI call, paid API, prompt change, routing change, schema change, threshold change, or production mutation occurred.

At the unchanged 0.80/0.93 operating point, 262/300 records received automatic decisions (87.3333% coverage): 255 automatic includes and seven automatic excludes. Against adjudicated gold, automatic inclusion precision was 0.937255, automatic inclusion recall 0.991701, automatic exclusion precision 0.714286, included-class F1 0.963710, excluded-class F1 0.357143, and binary macro-F1 0.660426. The automatic-only confusion matrix (gold rows and prediction columns ordered include, exclude) is `[[239, 2], [16, 5]]`, comprising 16 false inclusions and two relevant automatic exclusions. Review records were excluded from this denominator and were never mapped to exclude.

Routing sent 38/300 records to review (12.6667%, or 126.67 per 1,000), including ten relevant and 28 irrelevant records. Saved routing evidence identified eight material model disagreements, one image-dependent review and 29 weak-event-connection reviews; there were no weak-empty automatic exclusions. Replacing only review outcomes with gold for the explicitly simulated human-assisted view yielded inclusion precision 0.939623, recall 0.992032, included/excluded F1 0.965116/0.785714, macro-F1 0.875415, and confusion matrix `[[249, 2], [16, 33]]`. This is not classifier-only performance.

The 33 valid threshold-grid combinations preserved all fixed routing invariants and recorded zero violations. No tested pair achieved automatic binary macro-F1 >=0.80. Raising only the exclusion threshold to 0.99 eliminated relevant automatic exclusions and raised automatic-exclusion precision to 1.0 at 13.6667% review; inclusion thresholds 0.65 through 0.80 produced identical automatic results on this saved sample. The minimally changed candidate is therefore 0.80/0.99, but it has not been adopted: its automatic macro-F1 remains 0.650472. The research decision is **NOT READY FOR BLIND HOLDOUT**. The smallest justified next intervention is threshold tuning/evaluation, beginning with 0.80/0.99; routing, prompts and policy remain frozen unless threshold-only work is subsequently shown inadequate under the approved decision protocol.

Phase 6 agreement remains 42.59% with Cohen's kappa 0.1180. Because the 54-record set was purposely enriched for historical-policy/model disagreements, this is not a corpus-wide reliability estimate. It documents ambiguity among difficult cases; supervisor completion now makes adjudication complete and the v1 targeted gold resolved. Classifier quality remains below all combined automatic targets, annotation agreement was low on the enriched subset, gold status is resolved, calibration status is complete from saved outputs, and blind-holdout readiness is not yet established. Detailed metrics, threshold grid, ranked candidates, per-record error analysis, reports and hashes are registered in `relevance_v8_post_adjudication_v1.manifest.json`.

## 39. Formal v8 operating-point selection and freeze (2026-08-13)

Phase 8 independently reproduced the saved 0.80/0.93 routing decisions from the immutable 300-record adjudicated calibration artifact, then compared exclusion thresholds 0.93, 0.95, 0.97, 0.98 and 0.99 with inclusion fixed at 0.80. The simulation changed no assessment, confidence, disagreement, image-dependence, reason-code, gold-label, prompt, policy, schema, model, or routing-invariant field. It used no OpenAI or paid API call. Source classifier and adjudicated-gold hashes are recorded in `data/processed/relevance_v8_calibration_results/threshold_selection/relevance_v8_threshold_selection_v1.manifest.json`.

Moving from 0.80/0.93 to 0.80/0.99 changes exactly three records, all from `automatic_exclude` to `pending_review` because their excluded-content confidence (0.93, 0.96 or 0.98) is below 0.99. Two are gold include and one is gold exclude. No automatic inclusion or other routing state changes. Review consequently rises from 38 to 41 per 300 (12.6667% to 13.6667%): three additional reviews per 300, ten per 1,000, and a calibration-distribution projection of 140 per 14,000.

At the selected point, 259 records are automatic: 255 include and four exclude. Automatic inclusion precision is 0.937255, recall 1.0, exclusion precision 1.0, included-class F1 0.967611, excluded-class recall 0.2, excluded-class F1 0.333333, and macro-F1 0.650472. The automatic confusion matrix is `[[239, 0], [16, 4]]` with automatic gold support of 239 include and 20 exclude. Independent arithmetic and sklearn agree. Low macro-F1 is not an implementation fault: conservative exclusion routing and 16 over-included gold excludes produce low excluded recall despite perfect exclusion precision; class imbalance and limited predicted-exclude support contribute.

All four automatic exclusions at 0.80/0.99 are gold exclude. All 16 false inclusions were retained in the record-level audit: eight policy-boundary cases, three short-caption ambiguities, three hashtag-supported over-inclusions, and two event-information boundaries. These are a systematic inclusion-boundary limitation, but inclusion precision remains above target and the exploratory topic-discovery priority is to avoid irreversible loss of relevant material. Prompt revision is therefore not justified before prospective holdout validation.

The versioned configuration `relevance_v8_operating_point_v1.json` formally freezes inclusion/exclusion thresholds at **0.80/0.99** as operating-point version `v8_op1`. Inclusion precision, inclusion recall and false-exclusion safety targets pass; automatic macro-F1 >=0.80 does not pass and has not been redefined or hidden. Acceptance is methodologically based on relevant-content preservation, conservative human review, manageable incremental workload, transparent metric reporting, and the requirement for untouched-holdout generalization. The Phase 8 research decision is **READY TO FREEZE FOR BLIND HOLDOUT**; Phase 8 did not create or run that holdout.

## 40. Fresh v8 blind holdout construction (2026-08-13)

Phase 9 verified every file hash in the Phase 8 operating-point manifest plus the preserved hashes for `v8_op1`, the adjudicated v1 gold and the adjudicated 300-record calibration set before sampling. The freeze manifest additionally hashes the policy/prompt/few-shot implementation in `marathon_absa/openai_service.py`, the v8 schema, routing implementation, model/version configuration, unchanged annotation guideline, source corpus and all Phase 9 artifacts. No frozen discrepancy was found.

The document-level source corpus contained 14,077 records. All identifiable IDs occurring in calibration, validation, earlier holdouts, repeat checks, human review queues, policy alignment, audits, adjudication, threshold error inspection and other development artifacts were unioned without consulting model outcomes. This directly excluded 1,429 corpus records, leaving 12,648 before duplicate controls. Existing content hashes and project duplicate links removed 90 exact duplicates; conservative normalization that ignores case, punctuation, URLs, hashtags and whitespace removed 195 near duplicates against development records and within the remaining pool. The final eligible population was 12,363. These controls use identity and text only, not relevance predictions or difficulty.

No prior protocol predeclared a Phase 9 size. A size of 300 was documented before drawing because the eligible population was much larger, 300 matches the calibration scale, provides a feasible basis for binary precision/recall estimates under expected imbalance, and remains operationally feasible for two independent annotations. Proportional stratified random sampling used seed 91827, largest-remainder allocation, and 44 strata based only on source, event year and independently generated OpenLID primary-language metadata; rare language values were pooled for stable allocation. Reviewer files use independent order seeds 91828 and 91829.

The frozen membership is `data/processed/relevance_v8_holdout/relevance_v8_blind_holdout_v1.csv`; the internal artifact retains stable source IDs, while reviewer A and B receive only independently randomized holdout ID, caption, permitted source/year metadata, independent language metadata, and blank binary annotation/evidence/rationale/confidence/comment fields. The unchanged `RELEVANCE_V8_ANNOTATION_GUIDELINE.md` governs annotation. No prediction, model confidence, routing, reason code, content-type/event-connection output, historical label, calibration result or threshold information appears in reviewer files.

The normal v8 scoring path now checks frozen holdout membership and refuses to score any selected source ID. A future dedicated evaluation path must additionally verify `gold_finalized_at`; Phase 9 state records `holdout_created_at` and leaves both `gold_finalized_at` and `classifier_first_scored_at` null. Metrics and targets were predeclared before predictions. The first future evaluation is confirmatory for frozen `v8_op1`; any subsequent system change makes this holdout development evidence and requires another untouched holdout for new confirmatory claims. No holdout predictions, metrics, error analysis, gold labels, classifier calls or API calls were produced in Phase 9.

## 41. Phase 9 superseded by single-researcher human-in-the-loop relevance audit (2026-08-13)

The Phase 9 two-independent-reviewer confirmatory protocol was discontinued before classifier scoring because an independent second annotator was unavailable. This is a protocol constraint, not a classifier result. The frozen membership, Reviewer A/B templates, freeze manifest, sampling evidence, predeclared metrics and temporal state remain unchanged. At supersession, `gold_finalized_at` and `classifier_first_scored_at` were null; no prediction file, holdout metric, error inspection, classifier call or API call existed. The additive `data/processed/relevance_v8_holdout/phase9_protocol_supersession_v1.json` records `superseded_before_execution`, its replacement and the frozen-membership hash. Old reviewer files are not repurposed.

Relevance is now a conservative preprocessing/gating validation for topic discovery and ABSA, not a separate production-grade classifier claim. The unchanged `v8_event_experience_binary` policy, guideline, v8 contracts, prompt/examples, adjudicated development gold and `v8_op1` thresholds 0.80/0.99 remain in force. `REVIEW` is operational routing and final researcher decisions are binary. No LLM is represented as an independent reviewer, the researcher is not duplicated, and no inter-rater kappa is calculated.

The additive implementation is in `marathon_absa/single_researcher_audit.py`, `relevance_v8_single_audit_labeler.py` and new CLI commands. The explicit-cost `relevance-v8-production --run-api` command is implemented but not executed; it creates an isolated non-overwriting production decision package and preserves deterministic exclusions. The non-paid audit creator then creates separately (1) every review-routed row for operational resolution and (2) a blind audit of 75 automatic includes and 75 automatic excludes using seed 104729. Sampling uses source, year, independent language and text length where feasible, never gold or known errors. Review rows do not enter the automatic audit denominator.

The blind file exposes stable ID, original text, permitted metadata and blank researcher fields while hiding decisions, confidence, routing, model output and prior labels. A hashed key stores provenance and sampling weights. The Streamlit tool requires a binary label and rationale, original-caption evidence for includes, atomic autosave, one UTC backup and optional translation. The non-paid finalizer blocks incomplete/malformed labels, duplicates, misalignment, changed text/metadata and overwrite. It reports both decision arms separately, errors, confusion, balanced-sample agreement and design-weighted recall while warning that the 75/75 design does not estimate prevalence.

Predeclared acceptance requires include confirmation >=0.90, exclude confirmation >=0.90 and zero observed false exclusions for an automatic pass; any false exclusion requires diagnosis and a human methodological decision, not an invented tolerance or automatic v9 cycle. The historical macro-F1 gate remains historical. Topics remain blocked until the audit is acceptable and every operational review is resolved. One-researcher subjectivity, no independent reliability estimate, audit uncertainty, API/model drift, platform sampling bias and possible downstream topic/sentiment bias are explicit limitations. No paid/API calls, topics or ABSA were run during this implementation. The standalone protocol is `RELEVANCE_V8_SINGLE_RESEARCHER_METHODOLOGY.md`.

Deterministic verification on 2026-08-13 passed the complete suite: **173 passed in 29.25 seconds**, with four existing sklearn warnings from legacy synthetic single-label holdout fixtures. The explicit paid-command guard was also verified: invoking `relevance-v8-production` without `--run-api` stopped before inference. This is implementation evidence only; no production relevance decisions or audit sample exist until the authorized paid production command is run.

## 42. Non-paid v8 production cost audit after quota interruption (2026-08-14)

The first authorized production attempt stopped on `insufficient_quota` during `relevance_v8_verification`. It did not create or mutate `data/processed/relevance_v8_production/`, because production outputs are written only after the complete in-memory loop. Each successful model response is nevertheless written immediately to the stage cache. Local file dates show that the interrupted attempt preserved 3,094 new initial assessments and 1,218 new verification assessments on 2026-08-13. Exact reconstruction of current cache keys found 3,388 reusable initial results and 1,299 reusable verification results across calibration and the partial run; one cached initial assessment requires verification but has no corresponding verification cache, consistent with the reported failure stage. Caches were not deleted or modified.

The read-only `relevance-v8-cost-estimate` command was added. It does not instantiate `CachedOpenAI`, create an OpenAI client, write output artifacts, or perform network/API work. It reconstructs exact cache identities from the current 13,799-record ready Instagram population. Of these, 719 are deterministic/no-initial-call records: 338 hashtag-only automatic exclusions and 381 insufficient-text review routes. The remaining 13,080 require initial assessment; 3,388 are reusable locally and 9,692 remain. Among cached production-relevant initial assessments, 1,300/3,388 (38.37%) trigger verification; 1,299 verification results are reusable. Applying that observed rate gives about 3,720 remaining verification calls, including the known missing verification.

Historical pre-production calibration cache evidence (2026-07-29) contains 300 initial calls with 551,714 input and 126,179 output tokens, and 86 verification calls with 176,402 input and 26,726 output tokens, a 28.67% verification rate. The repository persisted total input/output tokens but not `input_tokens_details`, so cached-input and cache-write token totals cannot be recovered. The estimator therefore reports them as unavailable rather than zero and applies the documented 1.25x input cache-write multiplier to conservative scenarios.

Pricing was checked on 2026-08-14 against OpenAI's official 2026-07-30 GPT-5.6 pricing announcement: standard Luna was USD 0.20/M input and 1.20/M output; Terra was USD 2.00/M input and 12.00/M output. The separately published API pricing/model pages still displayed Luna USD 1.00/6.00 and Terra USD 2.50/15.00 when inspected, so the official sources were inconsistent. These rates are external and time-sensitive and must be confirmed in the account pricing/billing interface before purchasing credit. Using the newer announcement rates, expected remaining cost is approximately USD 37.59; applying the higher pricing-page rates to the same workload gives approximately USD 78.70. The p95-token conservative scenario using announcement rates and a 1.25x cache-write assumption is USD 58.29; the equivalent higher-rate figure is about USD 122.09. No API call was made for this audit or estimator validation.

The complete deterministic suite after the estimator change passed **173 tests in 33.86 seconds**, with the same four legacy sklearn warnings.

## 43. Cost-conservative single-researcher operational configuration (2026-08-14)

The non-paid cost audit showed that completing synchronous frozen-`v8_op1` production was disproportionate for a thesis preprocessing gate. An additive operational configuration, `v8_single_researcher_cost_conservative_v1`, was therefore implemented without changing the `v8_event_experience_binary` construct, prompts, schema, thresholds, historical caches, adjudicated gold, Phase 8 operating point or Phase 9 evidence. It is explicitly not `v8_op1`: the only methodological change is conditional Terra eligibility and conservative direct-human routing.

The existing six verification triggers are retained. Terra verification is used only when a triggered Luna assessment has explicit/supported KLSCM connection, meaningful content present, and an included-family primary content type. All other triggered cases bypass Terra and become unresolved human review; skipping Terra can never cause automatic exclusion. This protects relevant-content retention while reducing cost and keeping relevance proportional to its preprocessing role.

The immutable non-paid calibration replay under `data/processed/relevance_v8_single_researcher_cost_conservative/calibration_replay_v1/` reproduced the prior feasibility evidence from saved caches and adjudicated gold. Full verification used 86 Terra calls, 41 reviews and 259 automatic rows, with zero false exclusions and 16 false inclusions. The new route used 42 Terra calls, 47 reviews and 253 automatic rows, with zero false exclusions and 15 false inclusions. It avoided 44/86 Terra calls. This is retrospective development justification, not prospective confirmatory validation. The adjacent versioned configuration JSON hashes the replay and records that neither Batch nor production execution occurred at creation.

Two-stage Batch support was added for `/v1/responses` strict Structured Outputs. Offline creation reuses synchronous caches, excludes deterministic records, creates only missing Luna or eligible Terra requests, chunks by a configurable planning limit, writes stable custom IDs and hashes, and refuses overwrite. Paid submission requires `--run-api`; status/import are separate. Imports validate identity, completeness, schema and conflicts before entering the existing application cache identity. No Batch was created or submitted during implementation.

Production finalization is non-paid and refuses incomplete Luna/Terra results. It creates a versioned decision table and speed-oriented operational queue without resolving it. Workload counts distinguish direct-human skipped-Terra, post-Terra and insufficient-text review, with descriptive 5/10/20-second planning estimates. The random audit remains separate and blinded. Its cost-conservative acceptance version requires both decision-arm confirmation rates >=0.90 and at most 2/75 automatic-exclude false exclusions (about 2.67%), with qualitative inspection of every false exclusion. This is not corpus prevalence.

Topic discovery remains blocked until machine coverage is complete, every operational review is binary-resolved, the 150-row audit is complete and accepted, and no unresolved row remains. The final topic-gate command freezes a versioned relevance corpus with hashes; audit failure stops for explicit methodological review and does not initiate v9. The standalone protocol is `RELEVANCE_V8_COST_CONSERVATIVE_OPERATIONAL_PROTOCOL.md`. Single-researcher subjectivity, absence of final inter-rater reliability, model/pricing drift, sampling bias and residual relevance error remain limitations.

The replay and updated non-paid estimator were executed on 2026-08-14 with zero API calls. Live cache coverage remained 3,388 Luna initial and 1,299 Terra verification results; 9,692 new Luna calls remain. The cost-conservative projection is 2,031 remaining Terra calls, approximately USD 24.36 synchronously or USD 12.18 through Batch at the documented July 30 rates, with about 1,905 remaining manual-review rows versus 1,679 under full verification. These are planning estimates, not observed production results. Focused verification passed 20 tests and final complete verification passed **179 tests in 30.64 seconds**, with four unchanged sklearn warnings from legacy synthetic single-label fixtures. Frozen Phase 9 manifest hash tests passed. No Batch request files were generated or submitted, and production remains unexecuted.

## 44. JSONL Unicode line-separator import correction (2026-08-14)

The completed `initial_batch_006.jsonl` import initially failed locally with `JSONDecodeError: Unterminated string` while reconstructing request-to-cache identities. The request artifact was not truncated: direct validation found 537 valid records and 537 unique custom IDs. Three captions contain Unicode line/paragraph separators (`U+2028` or `U+2029`) inside otherwise valid JSON strings. Python `str.splitlines()` treats those Unicode characters as line boundaries even though JSONL records are delimited by ASCII LF, splitting valid requests into fragments.

`marathon_absa/cost_conservative.py` now uses an explicit ASCII-LF JSONL splitter for both downloaded Batch output and source-request lookup; the CLI submission preview count uses the same delimiter rule. A regression fixture places `U+2028` inside a caption and verifies successful import/cache creation. This correction changes parsing only; it does not change prompts, schemas, routing, thresholds, model calls, or existing Batch contents. The real `initial_batch_006.jsonl` parsed locally as 537 valid unique records after the correction. Focused verification passed 6 tests in 4.67 seconds, and the complete unpaid suite passed **179 tests in 35.69 seconds**, with the four existing sklearn warnings. The import command itself was not rerun during this correction, so successful cache import remains to be confirmed by execution.

## 45. Final production reporting clarification and architecture-aligned audit v2 (2026-08-15)

The finalized `v8_single_researcher_cost_conservative_v1` decision artifact was inspected without API calls or record mutation. Its 13,799 eligible records reconcile exactly as 10,532 automatic includes + 338 automatic excludes + 2,929 human-review routes. Review sources reconcile as 2,217 `verification_triggered_terra_ineligible`, 381 deterministic insufficient-text, and 331 post-Terra records.

The historical `production_manifest_v1.json` fields `deterministic_review=719` and `post_terra_review=2213` were semantically misleading but numerically reproducible. The former counted every deterministic/no-initial route (338 automatic exclusions + 381 human reviews); the latter counted every Terra-eligible record processed by Terra (1,882 automatic outcomes + 331 human reviews). The v1 manifest is retained unchanged as historical evidence. Additive `production_report_v2.json` names these quantities `deterministic_no_initial_total` and `terra_processed_eligible_total`, while separately reporting `deterministic_human_review=381` and `terra_processed_human_review=331`. It also records 13,080 Luna-processed records, including 8,650 initial-only routes, 2,213 Terra-processed eligible routes, and 2,217 triggered Terra-ineligible direct-review routes. The v2 report is reporting-only and records `api_calls=0`.

The finalized architecture has no model-classifier automatic-exclude arm: all 338 automatic exclusions are deterministic hashtag-only exclusions. The older 75/75 automatic include/exclude protocol remains historically documented but is superseded operationally by additive `single_researcher_architecture_audit_v2`. The new blinded package samples 75 automatic includes and, separately, 50 deterministic exclusions (14.8% of the 338-record deterministic population), using seed 104729 and proportional source/year/language/text-length stratification where feasible. Operational human review is exported separately with all 2,929 rows. Metrics report automatic-include confirmation and deterministic-exclusion confirmation separately and do not publish a combined balanced accuracy or characterize deterministic cases as model exclusions. Acceptance requires each confirmation rate >=0.90 and zero deterministic-exclusion reversals; any reversal requires qualitative review. If zero reversals are observed in 50, the rule-of-three upper 95% diagnostic bound is approximately 6%, not a prevalence estimate.

The offline package was generated at `data/processed/relevance_v8_single_researcher_architecture_audit_v2/` from the finalized Parquet artifact (SHA-256 `5811bdaea235f4a358803bec17dd8c8a055cae27193aafa1feb377d724759c89`). It contains 125 random-audit rows and 2,929 operational-review rows. No relevance routing, model, prompt, threshold, cache, or finalized decision was changed, and no API call occurred.

Complete deterministic verification passed **182 tests in 34.77 seconds**, with the four existing sklearn warnings from legacy synthetic single-label fixtures. The historical holdout exclusion test was corrected to use the creation-time development-artifact register frozen in `relevance_v8_holdout_sampling_report_v1.json`; dynamically treating later production audit files as pre-holdout development evidence had caused a false retrospective leakage failure. The holdout builder and frozen holdout membership were not changed.

## 46. Architecture audit v2 finalizer default-path correction (2026-08-17)

The researcher completed all 125 rows in `data/processed/relevance_v8_single_researcher_architecture_audit_v2/relevance_v8_architecture_audit_blind_v2.csv`. Read-only validation found 125 unique audit IDs and document IDs, 125 valid binary labels (73 include, 52 exclude), no blank rationales, valid confidence values, and exact pairwise alignment with `relevance_v8_architecture_audit_key_v2.csv`. The hidden key contains exactly 75 `automatic_include` and 50 `deterministic_auto_exclude` records. The completed annotation file SHA-256 before and after implementation was `ec1b10f12fc24ffbc0443db8fceee3541043d32962a73add687d2ca7d1753346`; the key SHA-256 remained `6f32fc8329b02b2ba8473e169315c06ca53eff647d606a661d52c188d943482e`.

The generic CLI command `relevance-v8-single-audit-finalize` had retained its historical v1 blind/key/output defaults and invoked only the v1 75/75 finalizer, even though the architecture-aligned v2 creator, labeler paths and dedicated finalizer already existed. Its defaults now point to the v2 125-row package and `final_v2`; protocol dispatch inspects the hidden-key schema and uses architecture metrics when `audit_arm` is present. Supplying explicit historical v1 paths continues to invoke the unchanged v1 finalizer. V2 finalization now enforces exactly 75/50 rows and exact hidden-key alignment before writing anything.

A read-only metric preflight—not finalization—found automatic-include confirmation 61/75 (81.33%) and deterministic hashtag-only exclusion confirmation 38/50 (76.00%), including 12 researcher reversals to include. Under the current `architecture_aligned_audit_v2` criteria, both 0.90 confirmation gates and the zero-reversal gate fail, so `audit_acceptable` would be false and topic discovery must remain blocked pending qualitative diagnosis and an explicit methodological decision. No final-v2 output was created, no annotations or production decisions were changed, and no API call occurred.

Focused finalizer regression tests passed 19 tests. The complete deterministic suite passed **185 tests in 30.20 seconds**, with four unchanged sklearn warnings from legacy synthetic single-label fixtures. Post-test hashes confirmed that the completed blind audit and hidden key remained byte-for-byte unchanged, and `final_v2` still did not exist because finalization was intentionally not run.

## 47. Non-paid architecture-audit v2 disagreement diagnosis (2026-08-17)

The completed architecture-aligned audit failed its predeclared gates: 61/75 automatic includes were researcher-confirmed (81.33%), leaving 14 researcher exclusions, while 38/50 deterministic hashtag-only exclusions were confirmed (76.00%), leaving 12 researcher inclusions. A non-paid, descriptive disagreement package was therefore created at `data/processed/relevance_v8_architecture_disagreement_audit_v1/`. It contains exactly those 26 records and no unrelated audit records. The CSV preserves the original blind decisions, evidence spans, rationales and comments; exposes the original and semantic text, hashtags, metadata, production route, saved Luna and applicable Terra fields, verification triggers and final automatic decision; and adds blank `diagnostic_category` and `diagnostic_note` fields with arm-specific permitted categories. It is a diagnostic worksheet, not a relabelling artifact.

The package is versioned and refuses to overwrite an existing output directory. Its manifest records zero API calls, 26 rows and output hashes; the summary records hashes of the completed blind audit, hidden key, finalized production decisions and document source. The generator verifies 125 complete binary annotations, exact blind/key ID alignment, exact 75/50 arm sizes and exactly 26 disagreements before writing. The CLI command is `python -m marathon_absa.cli relevance-v8-architecture-disagreement-audit`; the current v1 package has already been generated, so rerunning that command intentionally refuses overwrite.

Among the 14 false inclusions, ten were Luna-only and four were Luna plus Terra. Eight had `supported` event connection and six `explicit`; eight were `event_preparation`, two `event_information`, and one each `event_participation`, `post_event_reflection`, `event_safety_medical` and `event_registration`. Confidence bands were seven at 0.95–<0.99, three at 0.99–1.00, two at 0.90–<0.95 and two below 0.90. Caption lengths were nine medium, three long and two short. Existing fields identified no image-dependent case, no promotional indicator and no material Luna–Terra disagreement. The largest single boundary is therefore event preparation (8/14); four of those eight also used hashtag-supported rather than text-explicit event connection. This is meaningful concentration but not evidence that one routing field alone explains every error.

The deterministic hashtag inspection assigned seven of the 12 reversals to KLSCM-participation indicators, four to race category/distance and one to result/PB/finisher indicators; none fell into preparation, feeling/experience, event-information, generic-running-only, another-event, promotional or insufficient-evidence categories under the fixed lexical description. These categories are descriptive researcher aids, not automatic labels. The result shows that the blanket hashtag-only exclusion is mechanically consistent with the frozen routing rule but appears overbroad relative to the higher-level frozen event-experience construct when hashtags themselves encode participation, distance or completion. Any rule change would be a prospective intervention requiring explicit versioning and validation; no production decision, prompt, threshold, model, cache or routing rule was changed here.

The smallest defensible intervention candidates are: route all hashtag-only records to researcher review; predeclare and validate a deterministic split between substantive event hashtags and generic hashtag dumps; or leave classifier decisions unchanged and human-review a narrow risky automatic-include subset, beginning with event-preparation and supported-connection cases. Tightening an automatic-inclusion boundary is less conservative because it can increase false exclusion and would require prospective validation. No intervention, v9 classifier, production rerun, finalization or API call was performed.

## 47. Transition from record relevance to multilingual topic discovery (2026-08-17)

Relevance-classifier development is closed. The completed architecture-aligned audit (61/75 automatic includes confirmed; 38/50 deterministic hashtag-only exclusions confirmed, including 12 reversals) demonstrated imperfect decisions in both directions. A single-researcher study will not manually resolve the 2,929 REVIEW records or initiate a v9 classifier. The frozen 13,799-record production decisions, prompts, thresholds, audit decisions, and historical artifacts remain unchanged and now serve as diagnostic metadata rather than a topic-entry gate.

The additive `marathon_absa/topic_discovery.py` implementation and CLI commands `topic-discovery-prepare` and `topic-discovery-run` establish `topic_discovery_corpus_v1`. Preparation uses semantic text quality only, retains meaningful short and hashtag-only captions, normalizes hashtags conservatively, preserves language/source/year and Luna/Terra/route fields, and excludes exact duplicate semantic text from fitting while retaining representative mappings. `topic_review_app.py` is a topic-level reviewer scaffold. The prior `Pipeline.topics` OpenAI-embedding/relevance-gated path remains historical and was neither called nor modified.

The local preparation was executed on 2026-08-17 against the finalized production Parquet SHA-256 `5811bdaea235f4a358803bec17dd8c8a055cae27193aafa1feb377d724759c89`. It wrote `data/processed/topic_discovery_v1/`: 13,799 complete corpus rows, 13,743 fitting-eligible unique semantic texts, and 56 exact-duplicate exclusions. No empty/URL/mention/emoji/punctuation/corrupt exclusions occurred after conservative hashtag and emoji normalization. Relevance metadata reconciles unchanged to 10,532 include, 2,929 review, and 338 exclude rows.

Eligible character length is minimum 10, median 257, P90 876.8, P95 1,205.9, P99 1,937 and maximum 3,996. Because this finalized population is Instagram-only and does not show a separate blog-scale regime, v1 does not chunk; later blog inclusion requires a new distribution inspection and parent/chunk mapping. The proposed local CPU embedding is `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`; its download and exact revision have not yet occurred. Four fixed UMAP/HDBSCAN candidates are recorded in `bertopic_configuration_v1.json`. Chinese c-TF-IDF representation uses deterministic character and bigram tokens alongside Unicode/Latin tokens. No stopword removal, automatic merging, full embedding, BERTopic fitting, topic selection, topic review, visualization, or ABSA was executed. No API call occurred. Detailed rationale, limitations, commands, and artifact semantics are in `TOPIC_DISCOVERY_METHODOLOGY.md`.

Focused disagreement-package tests passed 3 tests. The complete non-paid suite passed **188 tests in 48.12 seconds**, with four unchanged sklearn warnings from legacy synthetic single-label holdout fixtures.

## 48. Completed candidate comparison and provisional c1 topic review package (2026-08-17)

The first local multilingual BERTopic comparison subsequently completed on the 13,743 eligible unique semantic texts. Candidate metrics record c1 as 40 topics, 5,703 outliers (41.50%), median topic size 55.5 and largest topic 1,970; c2 as 42 topics and 7,135 outliers (51.92%); and c3/c4 as four-topic severe under-clustering solutions with respective giant clusters of 13,395 and 13,424. c3/c4 are rejected, c2 is retained as comparison evidence, and c1 is selected only provisionally for interpretability review. It is not frozen or described as objectively best. No additional model, parameter search, merge, reduction, split, embedding, tokenizer or stopword change was run for this task.

The non-paid `topic-discovery-review-prepare` command created `topic_review_c1_v1.csv` for exactly topics 0–39, `topic_outlier_c1_v1.csv`, `topic_largest_five_diagnostics_c1_v1.csv`, and `topic_review_c1_manifest_v1.json`. Normal topics contain top terms, five BERTopic c-TF-IDF representatives, five fixed-seed random captions and diagnostic distributions. Topic -1 is separate: 5,703 records (41.4975%) with 30 fixed-seed random captions and no irrelevance label or requirement for record annotation. Expanded diagnostics cover topics 0, 1, 2, 3 and 4 with sizes 1,970, 1,396, 1,060, 480 and 386; each has ten representatives and twenty random captions. Seed 104729 is recorded.

`topic_review_app.py` now supports required topic-level interpretation/name, optional notes, coherence quality, autosave and resume. Regenerating diagnostics preserves existing researcher fields. `topic-discovery-review-summary` reports topic and assignment-weighted category totals for normal topics only and reports -1 separately. This is discourse-theme interpretation, not relevance validation; individual noisy records do not invalidate a substantive topic. No interface was started, no human annotation was prefilled, no API call occurred, and ABSA remains blocked. Full selection rationale is preserved in `TOPIC_DISCOVERY_MODEL_SELECTION_V1.md`.

## 49. Completed c1 review and controlled c1-refined experiment (2026-08-18)

The researcher completed all 40 original c1 topics. Twenty-seven substantive topics cover 7,112/8,040 normal-topic documents (88.46%); four generic-running topics cover 162, two commercial/promotional topics 290, one other-event topic 64, and six mixed topics 412. Quality ratings are 24 coherent, ten somewhat mixed and six highly mixed. This was topic interpretation, not renewed record-level relevance review. The completed c1 model, assignments, review, embeddings, IDs, metrics and configuration are frozen with SHA-256 values in `c1_frozen_baseline_manifest_v1.json`; no historical relevance artifact changed.

Systematic human-observed representation problems motivated one controlled text experiment: multi-component emoji aliases, English and Malay/Indonesian function words, repeated hashtag blocks, 26 spaced-KLSCM artifacts, style-driven clusters and fragmented themes. `refined_embedding_text` and `refined_representation_text` preserve raw captions while separating minimal embedding cleaning from stronger c-TF-IDF cleaning. The deterministic rules and counts are stored in `c1_refined_preprocessing_report_v1.json`. Since 12,742 embedding texts changed, a separate offline MPNet cache was computed; original embeddings remain unchanged. UMAP/HDBSCAN parameters were identical to c1. No parameter grid, API call, translation, LLM, merging, reduction, split or ABSA occurred.

The final QA-clean c1-refined run produced 44 normal topics and 6,511 outliers (47.38%), compared with c1's 40 and 5,703 (41.50%). Normal assignments fell from 8,040 to 7,232; median topic size changed 55.5→61, largest 1,970→2,076, and smallest 31→30. No topic is below 30 and no cluster exceeds half the corpus, so no giant-topic collapse occurred. Optimal label mapping leaves 5,951 changed assignments (43.30%); ARI is 0.2801 and AMI 0.4084. Crosswalk diagnostics show 40/44 refined topics have a dominant original source over 50%, 12/44 over 80%, median dominance 67.23%, and 12 refined topics dominated by former c1 outliers. These measures show material change but do not determine superiority.

The isolated `data/processed/topic_discovery_v1/c1_refined_v1/` package contains the model, separate embeddings, refined corpus fields, assignments, objective comparison, enriched crosswalk, configuration/hashes, blank 44-topic review, expanded largest-five diagnostics and separate -1 sample. Original c1 labels were not transferred. Final selection remains unresolved among c1, c1-refined and neither. The next authorized human work is compact topic-level comparison of the refined package; no record annotation is required and ABSA remains blocked. See `C1_REFINEMENT_EXPERIMENT_REPORT.md`.

## 50. Compact c1 versus c1-refined model-selection review (2026-08-18)

Instead of requiring a new taxonomy for all 44 refined topics, the non-paid `topic-discovery-model-selection-prepare` stage selected 28 decision-critical topics. Selection is deterministic and inclusive across four evidence classes: ten largest; 12 dominated by former c1 outliers; two receiving at least 25% and ten documents from completed c1 mixed/highly-mixed topics; and topics displaying direct residual artifact patterns or substantial generic-running inheritance. Because criteria overlap, category counts do not sum to 28. The package covers 6,063 refined clustered documents and records complete selection reasons.

`data/processed/topic_discovery_v1/model_selection_review_v1/model_selection_compact_review_v1.csv` contains refined terms, representative/random examples, every c1 contributor's human name/relevance/quality, c1-outlier inheritance, and only four researcher fields: interpretability change, semantic relation, refined quality and optional note. All researcher fields were initialized blank. `model_selection_review_app.py` autosaves and resumes this comparison without record annotation or a full refined taxonomy.

`topic-discovery-model-selection-summary` writes a JSON summary and `MODEL_SELECTION_COMPARISON_FINAL.md`. The current state is `pending_compact_review` at 0/28. A conservative predeclared rule adopts c1-refined only if document-weighted improvement reaches 60%, improvement exceeds worsening by 30 percentage points, unrelated mixing is at most 10%, and useful merging/splitting/outlier recovery reaches 40% of reviewed topics. Otherwise frozen c1 is recommended. This rule does not favor novelty and explicitly weighs +808 outliers, 52.62% versus 58.50% coverage, and four additional topics. No model, embedding, API, relevance artifact or ABSA stage was run.

## 51. Final c1 selection and taxonomy-consolidation preparation (2026-08-18)

The compact comparison is complete at 28/28: 15 improved topics cover 5,055 selected documents (83.37%), seven similar cover 600 (9.90%), and six worse cover 408 (6.73%). Semantic relations are seven preserved themes, one related merge, two useful splits, six unrelated mixes, 12 useful outlier recoveries and none unclear. Refined quality is 14 coherent, eight somewhat mixed and six highly mixed. The improvement and useful-relation thresholds passed, but unrelated mixing was 6/28 (21.43%), exceeding the predeclared 10% maximum; refined coverage also remained lower by 808 documents. The final recommendation is therefore `recommend_keep_frozen_c1`.

`final_selection/final_topic_model_selection_v1.json` additively records c1 as `final_selected`, c1-refined as `evaluated_not_selected`, objective/human/comparative metrics, the explicit rationale, and hashes for the frozen c1 evidence and comparison artifacts. Existing c1 files were not modified. ABSA remains blocked pending taxonomy completion.

`topic-taxonomy-prepare` generated a 40-row worksheet with every completed c1 name, relevance/quality label, terms and representative/random captions; all five final researcher fields are blank. It also ranked all 780 candidate pairs from existing embedding centroids, term overlap and name similarity. Topics 11/13 are explicitly surfaced first from the recorded photography-overlap observation. A six-row mixed-topic diagnostic preserves examples, relevance distributions and possible related substantive topic IDs. These are suggestions only; no merge or Level-1/Level-2 taxonomy is imposed.

The finalizer is implemented but intentionally not run. It requires every topic action, validates merge targets and cycles, preserves original c1 assignment IDs, partitions all 13,743 eligible documents into substantive, contextual/excluded and 5,703 `unassigned_outlier` rows, and only then records `ABSA_READY=true`. Topic -1 remains preserved, uninterpreted and not considered irrelevant. No BERTopic, embedding, outlier reduction, API, record annotation or ABSA execution occurred. The complete procedure and limitations are in `FINAL_TOPIC_MODEL_SELECTION_AND_TAXONOMY.md`.


## ABSA V1 implementation and frozen validation workflow (2026-08-19)

Aspect-based sentiment analysis implementation began only after topic discovery was finalized. Evidence is the final taxonomy manifest timestamp (`2026-08-19T08:16:35.088370+00:00`) and its reconciled counts: 13,743 eligible topic records, 7,704 substantive documents, 336 contextual/excluded-topic documents, and 5,703 unassigned outliers. No topic, relevance, embedding, UMAP/HDBSCAN, taxonomy, or outlier artifact was changed. ABSA v1 is scoped exclusively to `data/processed/topic_discovery_v1/final_taxonomy_v1/final_substantive_topic_corpus_v1.csv`; contextual/excluded records and outliers remain preserved and are not described as irrelevant.

The implementation is in `marathon_absa/absa_v1.py`, with the dedicated blinded UI in `absa_v1_labeler.py`. Topics provide context and provenance but do not determine an aspect. The frozen initial ontology has 20 families, including `emerging_other`; its human guide is `ABSA_ASPECT_ONTOLOGY_V1.md` and machine authority is `data/processed/absa_v1/absa_aspect_ontology_v1.json`. It was derived from the 32 consolidated c1 topics and historical ABSA/relevance concepts, with explicit inclusion and boundary rules. ABSA operates on original multilingual text at mention level. It preserves exact evidence, reliable offsets, language/gloss, meaningful hashtag/emoji contributions, explicit/implicit expression, and raw plus consolidated topic provenance. Factual, non-evaluative captions deliberately yield zero mentions. `mixed` is limited to opposing polarity toward the same aspect.

The validation sample was executed locally on 2026-08-19 with seed 20260819 and contains 150 documents. The artifact and manifest are `data/processed/absa_v1/absa_validation_sample_v1.csv` and `absa_validation_manifest_v1.json`. Sampling gives all 32 final topics a random floor, enriches a target 30% using only text length, first-person/evaluative lexical cues, and emoji, then fills randomly. It does not use future ABSA predictions. The sample contains no model prediction fields and was frozen before scoring. Human annotation supports zero/multiple mentions, exact evidence checks, add/remove rows, autosave, timestamped backup, and resume. It is single-researcher validation; absence of inter-rater reliability is a limitation.

The selected configured model is `gpt-5.6-luna`; prompt version is `absa_v1_instructions_1`, schema version `absa_mention_schema_v1`, and isolated cache stage `absa_v1_mentions`. These identities are not compatible with historical ABSA caches. The offline estimator executed on 2026-08-19 without network/API calls and reported 7,704 new calls, 21,606,578 estimated input tokens, 2,003,040 output tokens, USD 47.04 expected synchronous cost, USD 63.81 conservative p95, USD 23.52 expected Batch cost, and USD 0.92 for the 150-document validation run. These are planning estimates using configurable metadata (USD 1.25/M input, USD 10/M output, 50% Batch discount), not incurred costs or a guarantee of provider billing.

The implementation can prepare `/v1/responses` strict-schema Batch JSONL after human gold is frozen. Official OpenAI API documentation inspected on 2026-08-19 lists `/v1/responses` as a supported Batch endpoint and describes the asynchronous workflow. No Batch was created, uploaded, submitted, scored, or imported. Submission remains explicitly paid-call gated and intentionally disabled pending package review. Full-corpus request creation is blocked until validation gold is frozen; acceptable validation evidence must precede production.

Validation will report aspect detection precision/recall/F1 using document+aspect one-to-one matching, matched-sentiment accuracy/macro-F1/per-class metrics, joint aspect-sentiment precision/recall/F1, and exact evidence/span validity. Development targets (0.80 aspect F1, 0.80 sentiment macro-F1, 0.95 grounding) are documented but not hard-coded. At most one controlled prompt/schema refinement is planned. Later summaries must separately count documents and mentions and explicitly state denominators.

Implemented commands are `absa-v1-validation-sample`, `absa-v1-freeze-gold`, `absa-v1-cost-estimate`, and validation/full variants of `absa-v1-batch-<scope>-create`, `submit`, `status`, and `import`. Network actions require `--run-api`; local request creation does not. The next executed stage is human annotation with `streamlit run absa_v1_labeler.py`; neither the validation sample nor full corpus has been model-scored.

### ABSA AI-assisted draft and researcher-review interface (2026-08-19)

Codex completed a provisional first-pass annotation of all 80 documents in `absa_v1_single_researcher_audit_v1`. The additive package comprises `absa_v1_ai_draft_annotations_v1.csv`, `absa_v1_ai_draft_progress_v1.csv`, and `absa_v1_ai_draft_manifest_v1.json`: 80/80 documents are draft-complete, with 98 evidence-grounded mentions and 26 zero-mention documents. The validator in `marathon_absa/absa_v1.py` checks full document coverage, controlled aspects/sentiments, exact evidence, deterministic IDs, unique mentions, and exact reconciliation between positive/zero-mention status and annotation rows. The manifest records hashes, provisional status, and `human_gold=false`. The frozen audit sample and superseded 150-document artifacts were unchanged, and no API call occurred.

`absa_v1_labeler.py` is now an AI-assisted researcher-review interface rather than a purely blank annotation form. It displays the original caption, topic/language context, the frozen aspect guide, and editable provisional mentions; supports all/pending/completed filters; and requires one explicit decision per document: accept unchanged, accept with edits, replace, or confirm no mention. Confirmed researcher rows and progress remain separate from the AI draft, save with review timestamps, and lock against overwrite. Gold remains absent and cannot be frozen until the researcher explicitly confirms all 80 documents. Methodological status is therefore **provisional AI draft complete; researcher verification pending**. Any eventual gold must be described as AI-assisted pre-annotation followed by single-researcher verification, with no independence or inter-rater-reliability claim. Targeted ABSA verification after implementation was 13 tests passed in 2.61 seconds.

### ABSA single-researcher audit supersession (2026-08-19)

Before any annotation or model scoring, the planned 150-document ABSA validation exercise was superseded to reduce single-researcher burden. The original CSV and manifest remain byte-identical and unexecuted; their SHA-256 hashes are recorded in `data/processed/absa_v1/absa_validation_sample_v1_supersession.json`. No historical file was deleted or overwritten.

The current protocol is `absa_v1_single_researcher_audit_v1`, stored under `data/processed/absa_v1/absa_v1_single_researcher_audit_v1/`. Its 80 unique documents were sampled from only the frozen 7,704 substantive corpus with seed 20260823: 32 topic-floor selections (all final topic IDs represented), 24 difficult/multilingual enrichments, and 24 uniform random selections. No predictions were generated or used. Coverage is recorded in the audit manifest: event years 2019/2023/2024/2025 have 22/25/19/14 documents; primary language has English 42, Malay 16, Chinese 5, Indonesian 4, Malay/Indonesian uncertain 6, undetermined 4, and insufficient text 3; length quartiles Q1/Q2/Q3/Q4 have 19/21/19/21. All 80 are Instagram records, 39 contain a first-person cue, and all contain a hashtag or detected meaningful emoji/hashtag cue. This distribution is for broad qualitative coverage, not prevalence estimation.

The frozen 20-family ontology is unchanged. The default labeler and `absa-v1-freeze-gold` now resolve the 80-document package. The historical 150 package is accessible only by launching the labeler with `--historical-150`. Working annotations and progress are separate from immutable sample/gold artifacts; completed records are locked. Gold freezing validates completeness, exact evidence, controlled values, document/mention alignment, and uniqueness, and refuses overwrite.

The audit is exploratory and single-researcher; no inter-rater reliability is claimed. Strong interpretation is aspect F1 and matched sentiment macro-F1 >=0.80 with grounding >=0.95. Usable with limitations is >=0.70, >=0.70, and >=0.90 respectively. Materially lower or systematic hallucination/grounding failure is weak. One small miss below 0.80 is not an automatic stop, and serious failure permits at most one controlled refinement.

The updated offline estimate reports USD 0.49 expected for 80 validation documents, USD 47.04 expected synchronous and USD 23.52 expected Batch for all 7,704 documents, with USD 63.81 conservative p95. No API call, sample scoring, or corpus scoring occurred.

### ABSA validation evidence diagnostic and first frozen metrics (2026-08-19)

The validation Batch had already completed and been imported before this task. Its first local finalization failed with `ValueError: Every evidence_text must be an exact caption substring`; this failure is retained in the versioned diagnostic summary and was not hidden. No Batch was resubmitted, no inference or API call occurred, and raw Batch output, cache entries, frozen gold, ontology, prompt and schema remained unchanged.

The safe finalizer inspected all 80 outputs and 218 predicted mentions. Raw exact evidence was 204/218 (93.58%); 14 failures affected nine documents. One whitespace-only mismatch was deterministically recoverable as a unique original-caption span. Thirteen failures were unrecoverable. Raw mismatch categories were six `translated_or_paraphrased_evidence`, six `truncated_evidence`, one `whitespace_difference` and one `emoji_alias_difference`; none of the failures included model offsets. Final recoverable exact-span grounding was 205/218 (94.04%). Both raw and recovered rates remain reported.

Unrecoverable mentions were preserved as predictions with their original model evidence and error status. They were not silently discarded from aspect/sentiment precision calculations and count as evidence failures. First frozen metrics are aspect precision 0.3853, recall 0.8400 and F1 0.5283 (84 TP, 134 FP, 16 FN); matched-sentiment accuracy 0.8571 and macro-F1 0.5346; joint precision 0.3303, recall 0.7200 and F1 0.4528; and zero-mention agreement 0.8375. Aspect and sentiment results are below the predeclared usable-with-limitations band. The prompt/schema were not tuned, full-corpus ABSA remains unexecuted, and a refinement decision requires substantive error analysis. Artifacts and exact metric treatment are documented in `ABSA_VALIDATION_EVIDENCE_REPAIR_REPORT.md`. Complete verification passed 229 tests with four unchanged sklearn warnings.

The additive non-paid CLI command `absa-v1-validation-evaluate` now evaluates the existing predictions against the frozen gold without importing an API client or rerunning inference. It writes versioned JSON, flat summary CSV and subgroup CSV artifacts, hashes all immutable inputs, is idempotent for identical inputs, and refuses partial or conflicting output. The evaluated population is 80 gold documents (56 with mentions, 24 zero; 100 gold mentions) and 80 prediction documents (65 with mentions, 15 zero; 218 predicted mentions). Zero-mention confusion is 13 gold-zero/predicted-zero, 11 gold-zero/predicted-mentions, two gold-mentions/predicted-zero and 54 gold-mentions/predicted-mentions. Per-language metrics are emitted only for groups with at least five documents; smaller Indonesian, insufficient-text and undetermined groups are count-only. The matching rule remains one-to-one within document+aspect using stable mention-ID order. No gold, predictions, ontology, prompt, schema or matching rule changed. Complete verification after this command was added passed 230 tests with four unchanged sklearn warnings.

### Frozen ABSA false-positive diagnostic audit package (2026-08-19)

The additive module `marathon_absa/absa_fp_diagnostic.py` and CLI commands `absa-v1-fp-diagnostic-create` / `absa-v1-fp-diagnostic-finalize` implement the next precision-development step without changing any frozen input or model behavior. Creation imports the frozen sample, gold, predictions, evaluation and ontology; hashes them; reuses the exact document+aspect/stable-ID matching rule; and fails unless TP/FP/FN reproduce 84/134/16, gold reproduces 100 mentions across 56 mention-bearing plus 24 zero documents, predictions reproduce 218 mentions across 65 mention-bearing plus 15 zero documents, and the zero-mention matrix remains 13/11/2/54.

The completed development package under `data/processed/absa_v1/development/absa_v1_false_positive_diagnostic_v1/` contains all 134 FPs across 55 affected documents, quantitative JSON/Markdown diagnostics, a deterministic 45-FP blank review sample, a 15-TP blank contrast sample, all 16 FNs, a frozen category guide and an integrity manifest. The largest FP families are emotional experience 37, race performance 32, physical experience 15, training/preparation/pacing 14 and crowd/community/atmosphere 12. Gold-zero documents account for 30 FPs across 11 documents. These are concentration results only; the factual-as-evaluative hypothesis remains unconfirmed pending manual coding.

The FP review sample uses seed 20260819, covers 43 documents, 13 aspects, 23 final topics and all available recorded-language FP groups: English 20, Malay 13, Chinese five, Malay/Indonesian uncertain four, insufficient text two and undetermined one. Indonesian contributes zero FPs in the frozen evaluation and is absent by necessity. The sample includes 33 positive, seven negative, four mixed and one neutral prediction, eight gold-zero rows, 36 predicted multi-mention rows and four raw grounding failures, with at most two rows per document. The TP contrast spans nine aspects, English/Malay/Chinese/Indonesian/Malay-Indonesian-uncertain, all four sentiments and explicit/implicit cases. Finalization is implemented but blocked until the researcher completes the blank coding fields; it validates the fixed taxonomy and does not generate a prompt. No API call, corpus run, prompt/model/ontology/gold/prediction/matching change occurred. Complete repository verification passed 232 tests with four unchanged sklearn warnings.

### ABSA researcher review completed and gold frozen (2026-08-19)

The researcher completed and explicitly locked all 80 AI-assisted audit decisions. Decision counts are 54 `accept_ai_draft_unchanged`, 24 `confirm_no_mention`, and two `replace_ai_draft`; there were no incomplete records. The two replacements were sample orders 42 (`ce796fe45dc0fcd6b546`) and 57 (`e1341338331adbda6448`), each changing an AI zero-mention draft to one researcher-confirmed mention. The final reference set contains 100 mentions across 56 documents and 24 confirmed zero-mention documents.

The existing non-paid `absa-v1-freeze-gold` gate validated complete sample coverage, controlled aspects and sentiments, exact source evidence, deterministic mention IDs, unique mentions and positive/zero-mention reconciliation. It wrote immutable `absa_v1_audit_gold_v1.csv` and `absa_v1_audit_gold_v1.frozen.json` at `2026-08-19T13:06:28.730438+00:00`. The gold SHA-256 is `5fbbe9dfd35cf717932866f3878205635fe6a3d206fc4d7e8dd1b071ffb6d381`; the marker links it to sample SHA-256 `fa529fd4e3bbe3a01560ebb97c8a2f3280d727e7486488e0c36ad09c6d642f86`. A repeat freeze correctly refused overwrite. Status is now **researcher-confirmed AI-assisted gold frozen; validation model scoring not yet executed**. No paid/API call, validation inference, metric calculation, prompt refinement or full-corpus ABSA run occurred.

### ABSA FP diagnostic researcher coding finalized (2026-08-19)

All 45 sampled false positives, 15 true-positive contrasts and 16 false negatives were researcher-coded in place using only their designated review fields; row order and source-field values were integrity-checked before and after serialization. The non-paid finalizer completed successfully and wrote `absa_v1_fp_diagnostic_final_v1.json` plus `absa_v1_fp_diagnostic_final_v1.md`. FP coding found 13 ontology-boundary confusions (28.9%), eight overextended aspect inferences (17.8%), six context-as-evaluation errors (13.3%), six redundant mentions (13.3%), four sentiment-spillover errors (8.9%), four factual-as-evaluative errors (8.9%), three over-decompositions (6.7%) and one grounding-related error (2.2%). No row required `other_unclear` or low confidence.

The contrast review shows that tighter precision policy must retain explicit and implicit achievement, colloquial Malay/Indonesian and Chinese evaluation, affective hashtags/emoji, short encouragement, difficulty/impairment language, neutral but explicitly chosen pacing strategies and culturally natural celebration language. False negatives particularly expose missed multilingual praise, community-directed encouragement, operational complaints embedded in narratives, comparative preparation insufficiency, subtle affect and achievement expressed through sparse hashtags. This is diagnostic development evidence only: no prompt, ontology, frozen gold, predictions, matching rules or evaluation were changed, and no API/full-corpus run occurred.

### One-prompt ABSA precision development experiment prepared (2026-08-19)

Exactly one controlled candidate, `absa_v1_instructions_2_precision`, was created under `data/processed/absa_v1/development/absa_v1_precision_prompt_development_v1/`. It appends a focused, auditable policy layer to the complete v1 instructions: local evaluation-to-target entailment, ontology-boundary discipline, prospective-versus-experienced state, minimum sufficient aspect set, prompt-level duplicate suppression, local polarity binding, and factual/context exclusion. It explicitly preserves legitimate implicit and multilingual evaluation, community encouragement, neutral qualifying judgments, mixed-sentiment rules, schema structure and exact evidence grounding. The original `absa_v1_instructions_1` remains unchanged.

Before candidate inference, `experiment_manifest_v1.json` froze the research hypothesis, immutable input hashes and decision gates: aspect precision >=0.55, recall >=0.75, joint precision >0.3302752, joint F1 >=0.4528302, at most seven gold-zero documents receiving predictions, and no obvious multilingual/implicit-evaluation collapse. It explicitly marks `development_evidence=true` and `confirmatory_evidence=false`, because the same 80 documents informed diagnosis and prompt design. Model `gpt-5.6-luna`, the 20-family ontology, `absa_mention_schema_v1`, frozen gold/sample, preprocessing and stable one-to-one matching are unchanged; the cache identity is isolated as `absa_v1_mentions_precision_development_v1`.

`absa-v1-precision-development-create` generated exactly 80 `/v1/responses` Batch requests for the same ordered frozen document IDs. Candidate prompt SHA-256 is `671db63a8aa21819ae0fd1e2c856228919eabc85bab9fe7d85a3c14c1b9442bf`; request SHA-256 is `5b90f920e034c70154f1e82f4418437633584452015feecaad01f8af65d2bc12`. The offline estimate is 320,768 input and 20,800 output tokens, USD 0.609 synchronous or USD 0.3045 Batch using the existing USD 1.25/M input, USD 10/M output and 50% Batch-discount assumptions. Submission remains an explicit paid step requiring `absa-v1-precision-development-batch-submit --run-api`; it was not executed. Status/import/finalization commands and guarded side-by-side, FP-transition, recall-regression, language and aspect-family outputs are implemented but unexecuted. No API call or 7,704-document corpus run occurred.

### Precision-v2 residual development diagnostic (2026-08-19)

After the separately completed candidate evaluation, the offline `absa-v1-precision-residual-diagnostic` command reconciled the unchanged baseline at TP/FP/FN 84/134/16 and candidate at 77/74/23. Candidate precision/recall/F1 are 0.5099/0.7700/0.6135; the frozen 0.55 precision gate was not met, so the decision remains `INSUFFICIENT_PRECISION_IMPROVEMENT`. The diagnostic itself performed no API call or inference and changed no prompt, model, ontology, schema, gold, sample, preprocessing or matching rule. It remains development evidence only.

The candidate removed 60/134 baseline FPs (44.78%), reduced gold-zero documents receiving predictions from 11 to six, and reduced predicted multi-mention documents from 43 to 35 versus 27 in gold. Race-performance FPs fell 32→9, organization 7→2, safety/medical 4→1 and training/preparation/pacing 14→7. Photography FPs remained 5→5, emotional-experience FPs declined only 37→30, and `emerging_other` increased 2→3. Among the 45 previously reviewed baseline FPs, removal was strongest for aspect-inference-too-far and factual-as-evaluative (50% each), while no direct removals occurred for context-as-evaluative, over-decomposition or grounding-related cases; ontology-boundary removal was 3/13 (23.1%). Transition classifications are diagnostic links and do not change frozen evaluation matching.

All 74 candidate FPs were researcher-reviewed locally: 30 ontology-boundary confusions, 13 context-as-evaluative, 12 duplicate/redundant mentions, nine overextended aspect inferences, six over-decompositions, two sentiment-spillover cases, one factual-as-evaluative and one grounding-related. No new residual taxonomy category or low-confidence judgment was required; 46 were high and 28 medium confidence. Candidate FNs partition into 11 baseline FNs still unresolved plus 12 new misses; five of the original 16 FNs were recovered. New losses cover four ontology shifts, two implicit evaluations, two multilingual colloquial signals, one performance comparison, one hashtag/emoji signal, one repeated gold proposition and one other prospective case. English recall fell 0.8077→0.7500, Malay 0.8333→0.7333, Chinese 1.0000→0.8333 (five documents), and aggregated non-English/uncertain 0.8750→0.7917; Malay/Indonesian-uncertain remained 1.0.

The evidence-based conclusion is `FURTHER_PROMPT_INTERVENTION_JUSTIFIED`, but no further prompt was created. The unresolved conceptual problem is single-proposition aspect ownership and mention cardinality: affect is frequently emitted as a separate `emotional_experience` or adjacent-family mention when frozen policy treats it as evidence for one object-level evaluation. Any future intervention would need attribution/remapping rather than broader suppression and must protect the documented implicit, multilingual, encouragement, comparison, physical-strain and hashtag signals. Final JSON/Markdown and detailed transition/review/FN tables are under `data/processed/absa_v1/development/absa_v1_precision_prompt_development_v1/`.

### Single V3 aspect-ownership candidate prepared (2026-08-19)

Exactly one additional development candidate, `absa_v1_instructions_3_aspect_ownership`, was prepared under `data/processed/absa_v1/development/absa_v1_aspect_ownership_prompt_development_v1/`. It inherits the complete V2 instructions byte-for-byte and appends only a narrow proposition-level ownership/cardinality policy. The extension defines an evaluative proposition, selects its primary aspect owner, distinguishes emotion or physical language used as evidence from emotion/physical experience as a substantive target, remaps misowned evaluation before suppression, preserves community encouragement and embedded operational complaints, and replaces raw minimization with semantic-redundancy control. Six synthetic ownership contrasts are included; no development caption is copied as a rule.

The pre-inference manifest marks `development_evidence=true`, `confirmatory_evidence=false`, and `only_prompt_changed=true`. Model `gpt-5.6-luna`, schema `absa_mention_schema_v1`, frozen ontology/gold/sample, preprocessing, matching/evaluator, inference parameters, exact-evidence handling and the prior success gates are unchanged. No post-processing, threshold, code-level deduplication or filtering was added. V3 cache identity is isolated as `absa_v1_mentions_aspect_ownership_development_v1`. V3 prompt SHA-256 is `196606110ecb2b78d56a3220c1bd8957e8e42f2f259c65d1094c037fa21f30d0`, distinct from V2 SHA-256 `671db63a8aa21819ae0fd1e2c856228919eabc85bab9fe7d85a3c14c1b9442bf`.

`absa-v1-aspect-ownership-development-create` generated exactly 80 `/v1/responses` requests in frozen sample order; request SHA-256 is `2eaa7b79147e03dfa8400ee677928beaef1073ea1b4ede06cb5b1b262d5313b5`. Apart from the instruction field and experiment-specific custom ID, request bodies reproduce V2 exactly. The offline estimate is 418,262 input and 20,800 output tokens, USD 0.7308 synchronous or USD 0.3654 Batch under existing planning rates. Submission requires the explicit paid command `absa-v1-aspect-ownership-development-batch-submit --run-api` and was not executed. Status/import/finalization infrastructure is implemented for three-way metrics, V2 residual-FP transitions, V2-TP regressions, language/aspect diagnostics, emotional-experience FP and multi-aspect counts. No API call, V4, confirmatory claim or full-corpus run occurred.

### V3 prompt validation line-ending correction (2026-08-21)

The first attempted V3 Batch submission stopped before any upload or paid API request with `RuntimeError: V3 prompt hash changed`. Inspection showed that the prompt's decoded Unicode text, the runtime `INSTRUCTIONS`, and the instructions embedded in all 80 frozen requests were identical and shared the manifest SHA-256 `196606110ecb2b78d56a3220c1bd8957e8e42f2f259c65d1094c037fa21f30d0`. The prompt artifact had Windows CRLF bytes, while the manifest deliberately stored the SHA-256 of normalized instruction text; the validator incorrectly compared the platform-dependent file-byte hash to that text hash.

The V3 validator now decodes the UTF-8 prompt with universal newline handling and hashes the resulting text, matching experiment creation and request construction. It therefore accepts LF or CRLF serialization of the same prompt but still rejects any substantive text change. A regression test covers both behaviors. The frozen prompt content, manifest, request file, model, schema, ontology, sample, gold and success gates were not changed, and no API request was made as part of this correction.
## ABSA v1 development closure and prepared production Batch (2026-08-21)

Manual ABSA development review is closed for the current phase. V1 produced precision/recall/F1 0.3853/0.8400/0.5283 (84/134/16 TP/FP/FN); V2 produced 0.5099/0.7700/0.6135 (77/74/23); and V3 `absa_v1_instructions_3_aspect_ownership` produced 0.5130/0.7900/0.6220 (79/75/21). V3 remains formally below the frozen 0.55 precision gate and retains decision `INSUFFICIENT_PRECISION_IMPROVEMENT`. It has been selected pragmatically for production with that known limitation because it is the strongest controlled candidate available. The evidence is developmental, not confirmatory. No V4 was created, and any future refinement must use a new versioned protocol.

`marathon_absa/absa_production.py` and the `absa-v1-production-*` CLI commands implement an isolated full-corpus workflow. The frozen population reconciles exactly to 7,704 substantive documents, excluding 336 contextual/excluded-topic documents and 5,703 outliers. The production identity is V3 prompt SHA-256 `196606110ecb2b78d56a3220c1bd8957e8e42f2f259c65d1094c037fa21f30d0`, model `gpt-5.6-luna`, schema `absa_mention_schema_v1`, frozen v1 ontology, and cache stage `absa_v1_mentions_production_v1`.

Local preparation on 2026-08-21 created one 7,704-request Batch file at `data/processed/absa_v1/production/absa_v1_production_v1/batch/absa_v1_production_requests_v1.jsonl`: 106,458,712 bytes, SHA-256 `47d1f9a08b1e77a56c7a82511808a39d45b68880de9b5e64b58e3b076befcdc1`. Estimated input/output tokens are 26,267,305/2,003,040; expected Batch cost is USD 26.4323. Conservative estimates are 32,834,132/3,004,560 tokens and USD 35.5441. These are planning estimates only. An initial never-submitted local file failed a JSONL test because a caption contained a Unicode line separator; it was safely removed, serialization was corrected, and the final package was regenerated. The focused production suite passed 5 tests and the complete suite passed 246 tests (four pre-existing sklearn warnings).

No file was uploaded, no Batch was submitted, no inference was run, and no API call occurred. The freeze records `submitted=false`, `batch_ids=null`, `api_calls=0`, and `full_corpus_inference_started=false`. Full-corpus results will be production characterization rather than validation. Submission is deliberately stopped at its explicit paid command.
## Local GPU ABSA preflight (2026-08-21)

An isolated local, GPU-accelerated ABSA alternative is being evaluated before committing to paid full-corpus OpenAI inference. The existing OpenAI V1/V2/V3 development record and prepared V3 production fallback remain frozen and untouched. The new `marathon_absa/local_absa.py` workflow provides `local-absa-gpu-check`, provenance audit/preparation, guarded training, and guarded evaluation commands. Its derived artifacts are contained in `data/processed/absa_local_v1/`; the original gold and ontology are never overwritten, and preflight commands cap eligible evaluation input at 80 documents to prevent an accidental full-corpus run.

The audit found 80 unique human-labelled documents, 100 mentions, and no independent supervised-training or untouched confirmatory set. All labels derive from the same 80-document audit used during OpenAI V1/V2/V3 development. A supervised train/test claim is therefore unavailable. The recommended first experiment is zero-shot ontology-conditioned multilingual NLI with `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, revision `8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c`, rather than SetFit or encoder fine-tuning on the reused gold. Any resulting 80-document score is exploratory development evidence only.

System-level inspection detected an NVIDIA GeForce RTX 4060 Laptop GPU with 8,188 MiB VRAM and driver 610.88. The active environment contains CPU-only PyTorch `2.13.0+cpu`; `torch.cuda.is_available()` is false and CUDA preflight fails. Neural execution was therefore aborted. No model was downloaded, no embedding/training/evaluation inference ran, no paid API was called, no V4 was created, and the 7,704-document corpus was not processed.
## Local ABSA zero-shot NLI execution (2026-08-21)

The `.venv` CPU-only PyTorch package was replaced with official `torch==2.13.0+cu126`; `pip check` found no conflicts. CUDA 12.6 now exposes the RTX 4060 Laptop GPU to PyTorch, and the mandatory real matrix-operation preflight passes. The pinned local model revision `8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c` was cached without using hosted inference. Model parameters, all input tensors, and output logits were verified on `cuda:0`.

The capped 80-document experiment evaluated 1,600 aspect NLI pairs plus 324 sentiment pairs at batch size 16 using float16 autocast. Actual neural inference took 11.21 seconds at 171.62 pairs/second, with peak allocated/reserved VRAM 1,439,040,000/2,837,446,656 bytes. The selected reused-development threshold 0.50 yielded 23 TP, 85 FP and 77 FN: precision 0.2130, recall 0.2300 and F1 0.2212. Joint F1 was 0.1154 and matched sentiment accuracy/macro-F1 was 0.5217/0.3102. This is substantially below OpenAI V3 and is classified `LOCAL_NOT_CURRENTLY_VIABLE`. No 7,704-document run, paid API, OpenAI request, manual correction, ontology change, or further tuning occurred.
## Final ABSA production selection after local benchmark (2026-08-21)

The local zero-shot NLI track is formally complete, retained unchanged as a research benchmark, and classified `LOCAL_NOT_CURRENTLY_VIABLE`. Further local-model experiments and manual review are deferred. The working CUDA environment (`torch==2.13.0+cu126`, CUDA 12.6, RTX 4060 Laptop GPU) is preserved for future research.

OpenAI V3 is selected for full-corpus ABSA with status `selected_for_production_with_known_development_limitation`; its original `INSUFFICIENT_PRECISION_IMPROVEMENT` decision is unchanged. The local-versus-V3 decision is frozen in `absa_v1_production_model_selection_v1.json` and `.md`. Production creation reverified the existing 7,704-document, one-chunk request package without regenerating it: request SHA-256 `47d1f9a08b1e77a56c7a82511808a39d45b68880de9b5e64b58e3b076befcdc1`, expected input/output tokens 26,267,305/2,003,040, expected Batch cost USD 26.4323, conservative Batch cost USD 35.5441. Submission remains false; no API call, V4, additional local experiment, or full-corpus inference occurred.
## ABSA V1 final pre-submission safety audit (2026-08-21)

The zero-cost audit returned `SAFE_TO_SUBMIT_WITH_OFFLINE_DOWNSTREAM_ANALYSIS`. It streamed and validated all 7,704 frozen requests without changing the JSONL or making an API call. Request IDs, captions, population membership, prompt/model/schema configuration, UTF-8 encoding, one-line JSONL structure, actual byte size and SHA-256 all reconcile. The package uses one 106,466,416-byte Batch file and has 42,296 requests plus 93,533,584 bytes of margin against the current limits.

Additional safeguards preserve raw Batch output and errors before parsing, hash imported files, support interrupted import, retain a separate parsed-response artifact, validate the strict schema offline, preserve zero-mention and failed documents, report response-ID discrepancies, quarantine API/parse/schema failures, record grounding failures, and create affected-document-only retry packages. Finalization contains no API client and is idempotently rerunnable from retained artifacts. Numeric evidence offsets are derived offline from exact model evidence and immutable captions. No explicit request-level output-token cap exists, so truncation is not caused by the 260-token cost estimate; any provider/model-limit incomplete response is detected and individually retryable.

## ABSA V1 offline inferential analysis (2026-08-24)

The first inferential package was implemented in `marathon_absa/absa_inferential_analysis.py`, exposed as `python -m marathon_absa.cli absa-v1-inferential-analysis`, and executed once into `data/processed/absa_v1/analysis/absa_v1_inferential_analysis_v1/`. It reconciles 7,704 documents, 15,486 mentions, 5,316 mention-bearing documents, 2,388 zero-mention documents, and year counts 2,279/1,873/1,724/1,828 for 2019/2023/2024/2025. Its primary matrix has one row per frozen document × ontology aspect (154,080 rows), so ordinary tests do not treat 15,486 nested mention rows as independent observations.

Eligibility was frozen before hypothesis testing: an aspect requires the descriptive package's moderate/high support and at least 20 affected documents in every year. The resulting 12 aspects are race performance, crowd/community atmosphere, physical experience, emotional experience, training/preparation/pacing, photography/media, emerging other, route/course, organization/operations, weather conditions, value/cost, and safety/medical. Low and very-low support aspects remain descriptive. Pearson chi-square tests report Cramer's V and apply Benjamini–Hochberg FDR separately to aspect prevalence, positive sentiment among aspect-bearing documents, and negative sentiment among aspect-bearing documents. Pairwise year contrasts are generated only after an FDR-significant omnibus test and use Holm correction within the aspect/outcome family. Categorical-year logistic models use 2019 as reference and are fit locally with SciPy because statsmodels is not installed in the frozen environment.

Eleven aspects have FDR-detectable prevalence associations; `emerging_other` does not. Only race performance (V=0.125), crowd/community atmosphere (V=0.124), and physical experience (V=0.109) cross the predeclared 0.10 small-effect threshold. Their 2019-to-2025 prevalence changes are 30.36% to 46.23%, 18.82% to 32.99%, and 16.89% to 29.21%, respectively, and all six pairwise year comparisons are Holm-significant for the first two aspects. The many other statistically detectable prevalence results have V=0.033–0.095 and are treated as trivial magnitude. Photography/media is detectable only weakly (adjusted p=.0428, V=.0329) and has no Holm-significant pair; it should not be emphasized.

Within aspect-bearing documents, positive sentiment varies after FDR only for training/preparation/pacing and weather conditions; negative sentiment varies only for emotional experience, training/preparation/pacing, and weather conditions. Training/preparation positive presence rises from 40.56% in 2019 to 58.53% in 2024 and 55.46% in 2025, while negative presence falls from 64.66% to 49.83%/50.98%. Weather estimates change sharply but rest on only 35–84 affected documents per year and should be treated cautiously. Emotional-experience negative presence is lower in 2025 than 2023 by 10.90 percentage points after Holm correction. Neutral and mixed outcomes were not tested. Multi-sentiment document × aspect cases were retained with binary sentiment indicators and never forced into one label; they range from 0% to 17.94% across eligible aspects.

The self-reported confidence field is compressed (median .96) and uncalibrated against frozen gold, so no post-hoc confidence cutoff was selected. Document-level presence is the robustness protection. The report explicitly limits inference to associations in model-estimated labels and does not propagate classifier uncertainty from development precision .513 and recall .790. Rising extraction density and falling zero-mention rates are documented as a corpus/model diagnostic that may contribute to widespread prevalence increases, not proof of increased runner expressiveness.

The package contains the requested inferential tables, categorical-year odds ratios and confidence intervals, FDR/Holm results, research summary, visualization-ready long tables, multi-sentiment and eligibility diagnostics, report, source hashes, library versions and execution manifest. Focused verification passed 7 tests. OpenAI calls, Hugging Face inference, ABSA inference and manual review were all zero; frozen predictions and descriptive/production sources were not modified.

## ABSA V1 frozen dashboard data mart (2026-08-24)

The offline dashboard preprocessing layer was implemented in `marathon_absa/absa_dashboard_data.py`, exposed as `python -m marathon_absa.cli absa-v1-dashboard-data`, and executed into `data/processed/absa_v1/dashboard/absa_v1_dashboard_data_v1/`. It reads frozen production tables only for deterministic descriptive counts and treats the completed descriptive and inferential artifacts as statistical authority. It contains no chi-square, confidence-interval, multiple-testing, effect-size or pairwise-test implementation and does not run model inference.

The package reconciles 7,704 documents, 5,316 mention-bearing documents, 2,388 zero-mention documents, 15,486 mentions, all 20 ontology aspects and the frozen 2019/2023/2024/2025 denominators. It preserves the exact 12-aspect inferential eligibility set. Document prevalence uses document denominators; sentiment composition uses aspect-mention denominators; document sentiment counts are explicitly marked as overlapping. Proportions are stored as numeric 0–1 values, with null/status values retained for non-tested cases rather than substituting p=1.

The deterministic prevalence-emphasis policy requires frozen eligibility, BH-FDR significance, moderate/high support and Cramer's V at least .10. It selects only `race_performance`, `crowd_community_atmosphere` and `physical_experience`. `photography_media` is guarded against promotion because its effect is trivial; low/very-low support aspects cannot be promoted. A separate curated finding records the supported small training/preparation sentiment shift. Weather sentiment carries an explicit caution because its large variation is irregular and has limited yearly support.

Eighteen frontend-ready artifacts separate corpus, sentiment, aspect, year, aspect×year, aspect×sentiment, year×aspect×sentiment, frozen pairwise, research finding, caution, topic and language marts. Topic and language tables are descriptive only and carry disabled-inference flags. `insufficient_text` and `undetermined` remain separate and have language-quality warnings. Central metadata provides stable aspect display labels, frozen support/effect thresholds, model limitations, overlap semantics and reusable warning copy. Chart specifications reference only prepared marts and require no frontend research logic.

Every source used is hashed in `dashboard_data_manifest_v1.json`; regenerated core marts are byte-identical when supplied the same creation timestamp. Focused validation passed 10 tests. OpenAI calls, Hugging Face inference, ABSA inference, manual review and new statistical hypothesis tests were all zero. Frozen production, descriptive and inferential artifacts were not modified, and no dashboard UI was built.

## ABSA V1 interactive research dashboard MVP (2026-08-24)

The frozen mart is now presented through a new Streamlit/Plotly application at `absa_dashboard.py`, launched with `streamlit run absa_dashboard.py`. The existing framework and chart dependency were reused; no backend, database, frontend framework or new dependency was introduced. The legacy preprocessing-oriented `app.py` remains unchanged. A single typed data-access layer in `marathon_absa/dashboard_data.py` loads only the 18 frozen dashboard artifacts and fails visibly unless the expected 7,704 documents, 15,486 mentions, 20 aspects, four editions, 32 topics, document-level inferential unit and disabled topic/language inference flags reconcile.

The interface is a single responsive route organized into seven keyboard-accessible tabs: Overview, Temporal trends, Research findings, Aspect explorer, Topic explorer, Language explorer and Methodology. Overview provides corpus KPIs, an optional edition filter, frozen overall sentiment, document-prevalence ranking with mention-count toggle, and aspect-level mention sentiment composition. Denominator copy explicitly distinguishes analyzed posts from model-estimated mentions. Low-support aspects remain visible, and teal emphasis reflects the frozen `recommended_for_emphasis` field rather than frontend significance logic.

Temporal trends default to the three frozen prevalence findings recommended for emphasis and allow up to five aspects. Charts display frozen document prevalence and existing confidence intervals. The extraction-density warning is prominent. Training/preparation positive and negative document-level presence is plotted as independent lines because values can overlap. The Aspect explorer shows support/effect status, frozen yearly estimates and only existing pairwise rows; weather and photography surface the frozen caution text. Topic and language explorers are explicitly descriptive, preserve frozen topic labels and low-quality language categories, and expose accessible tables. The Methodology tab surfaces model precision .513, recall .790, the unmet .55 precision target, frozen statistical design, dataset version and every standardized warning.

The visual system uses a light editorial-analytics layout, one teal evidence accent, semantic sentiment colors, restrained warning panels and native accessible controls. KPI grids stack on small screens; chart containers remain responsive; the seven-tab navigation becomes horizontally navigable on mobile; reduced-motion CSS is included; every principal chart has a textual denominator and an expandable table alternative. Browser QA at desktop and 390×844 mobile found no horizontal overflow or console errors. The running local QA route was `http://127.0.0.1:8520/`.

Frontend tests cover frozen loading/validation, corpus/aspect/year/topic counts, default document-prevalence ranking, support/effect passthrough, frozen research/caution copy, denominator separation, weather-selection caution, Streamlit rendering and absence of statistical/model/API code. No ABSA, BERTopic, confidence interval, FDR, support, emphasis or pairwise calculation was rerun or implemented in the frontend. The research data mart and all upstream frozen artifacts remain unchanged.

## ABSA V1 thesis-ready research-output package (2026-08-24)

The final offline exporter is implemented in `marathon_absa/absa_final_outputs.py` and exposed as `python -m marathon_absa.cli absa-v1-thesis-outputs`. It was executed into `data/processed/absa_v1/final_outputs/absa_v1_thesis_outputs_v1/` using only the 18 hashed frozen dashboard artifacts. The module performs formatting, sorting, frozen-value selection and rendering only; it contains no hypothesis-test, CI, FDR, topic-model or inference implementation.

Seven thesis tables plus a Markdown corpus-table equivalent summarize the corpus, overall mention sentiment, all 20 aspect prevalences, within-aspect sentiment, six selected temporal prevalence profiles, all 32 descriptive topics and all 12 inferentially eligible aspects. Effect sizes, adjusted p-values, significance flags, research-emphasis decisions and existing pairwise differences are passed through unchanged. The main tables retain both statistically detectable trivial associations and non-significant rows rather than selectively reporting only favorable findings.

Five clean 1,800×1,100 publication PNG figures were rendered locally with Pillow on white backgrounds: the three primary aspect trends with frozen Wilson CIs; overlapping training/preparation positive/negative presence; sentiment composition for seven supported aspects; overall mention sentiment; and selected descriptive topic–aspect alignment. The primary prevalence figure is also exported as PDF. The plots use consistent semantic sentiment colors, direct denominator labels, line markers or text labels so interpretation does not rely on color alone, and figure notes that preserve model and inferential boundaries. Visual inspection confirmed readable labels, legends, notes and unclipped values.

The package also contains restrained key-findings prose sourced only from the four frozen research-finding rows, a 12-point limitations source, a compact final-method summary, five ready-to-use figure captions and reusable table notes. It explicitly retains development precision .513, recall .790, the unmet .55 target, 80-document reused development gold, unverified production labels, classifier uncertainty, social-media representativeness, extraction-density and sparse-support limitations. Causal claims are excluded. Dashboard screenshots were not exported because the available in-app browser QA surface cannot deterministically write screenshots into the workspace and no local Playwright/Selenium package is installed; the package records this instead of installing new infrastructure.

`final_output_manifest_v1.json` records production system `absa_v1_instructions_3_aspect_ownership`, source paths and hashes, output paths and hashes, code version, environment version, available git state and zero new tests/inference/review/topic modeling/API calls. The incomplete first drafts created during column-contract correction were removed after exact path validation and replaced by the complete package; frozen sources were never changed.

## ABSA V1 dashboard research-interface redesign (2026-08-24)

`absa_dashboard.py` was redesigned as a compact academic analytics interface while preserving Streamlit, Plotly, Pandas and the existing typed frozen-mart loader. The visual system now uses a dark-navy research navigation shell, light neutral workspace, blue analytical accent, compact bordered KPI cards and semantic sentiment colors. The navigation is organized into eight explicit research views: Overview, Social Media Analytics, Aspect Analysis, Temporal Trends, Topic Analysis, Language Analysis, Research Findings and Methodology.

Overview prioritizes the frozen corpus KPIs, overall model-estimated sentiment, top-aspect ranking, seven-aspect sentiment composition, principal temporal signals and the four curated frozen findings. Social Media Analytics exposes corpus and extraction-density diagnostics by edition. Aspect Analysis retains all 20 ontology families and their frozen support/effect fields. Temporal Trends displays only prepared prevalence, confidence-interval and training/preparation sentiment marts. Topic and language views remain explicitly descriptive. Methodology records the document-level inferential unit, Wilson intervals, Pearson chi-square, Cramer's V, BH-FDR, Holm pairwise correction, reused 80-document development evidence, precision .513, recall .790 and the unmet .55 precision target.

Features suggested by the visual reference but unsupported by the frozen research package were deliberately excluded: interviews, GIS/location analytics, triangulation, race-category estimates, 2026 or monthly trends, live AI recommendations and word-cloud inference. No frontend statistical test, confidence interval, support threshold, topic grouping, model call or ABSA inference was added. All values continue to load exclusively from `absa_v1_dashboard_data_v1`; the frozen marts and upstream production/descriptive/inferential artifacts were not modified.

Automated UI coverage now verifies the exact eight-view navigation, frozen counts and labels, absence of unsupported analytical claims, weather caution behavior and the topic/language/methodology boundaries. Browser QA verified routing, a clean console and zero horizontal overflow at both 1280-pixel desktop width and 390×844 mobile size. Launch with `streamlit run absa_dashboard.py`; the verified local route is `http://127.0.0.1:8520/`.

### Executive Overview addition (2026-08-24)

One additional default landing page, `Executive Overview`, was added ahead of the unchanged eight detailed views. It presents six frozen KPIs and three compact rows covering overall mention sentiment, top document-prevalence aspects, within-aspect sentiment, the three frozen emphasis trends with Wilson intervals, overlapping training/preparation sentiment, all four frozen findings, four descriptive topic relationships, five descriptive language groups and the established research-quality scope. Charts use explicit 250–260 pixel heights and small margins; the page performs only sorting, top-N selection and display formatting over existing marts. Mobile uses Streamlit's automatic sidebar state and stacks the cards without horizontal overflow. No frozen mart, research result, statistical procedure, model output or detailed page was changed.

### Offline Word Cloud page and contrast audit (2026-08-24)

The navigation now includes one additional `Word Cloud` page between Language Analysis and Research Findings. Its authoritative text source is the frozen production population manifest's `original_caption` field, which is integrity-checked against exactly 7,704 unique `substantive_absa_ready` document IDs and explicitly tested for disjointness from the 336 contextual/excluded documents and 5,703 outliers. Year, language and final-topic filters operate on the population metadata. Aspect and sentiment filters use frozen production mentions; when both are selected, the same mention row must satisfy both conditions.

`marathon_absa/wordcloud_data.py` provides deterministic offline normalization, multilingual tokenization, transparent generic and event-boilerplate stopword sets, preserved meaningful hashtags, separate emoji counts and document/raw frequency tables. URLs, emails, HTML fragments, standalone years and event-specific `klscm`/`scklm` variants are removed. English, Malay and Indonesian use conservative lexical tokenization; Chinese character sequences use local `jieba` segmentation without a model or hosted service. Cloud size defaults to unique-document frequency with a minimum document frequency of two. The UI renders a filtered navy/teal cloud, positive dark-green cloud, negative dark-red cloud, a 25-row quantitative term table, and optional hashtag/emoji summaries. Installed local dependencies are `wordcloud==1.9.6` and `jieba==0.42.1`; Microsoft YaHei or SimHei is used when locally available for Chinese glyphs, with Arial fallback.

The same change explicitly sets dark Plotly font, axis, title and legend colors and scopes dark slate text to white/light main surfaces. Captions, widget values, radio labels, expanders and tables receive readable light-surface colors, while high-contrast light text is retained in the dark navy sidebar. The page is labeled descriptive text exploration only and adds no research finding, sentiment interpretation, statistical result or inferential claim. Tokenized documents and generated PNGs use Streamlit data caches; no upstream artifact is overwritten and no API, model, ABSA, BERTopic or statistical procedure is invoked.

### Word Cloud evidence-source semantic correction (2026-08-24)

The Word Cloud page now defaults to `ABSA Evidence`, using only the frozen `evidence_text` on matching rows of `absa_v1_production_mentions_v1.csv`. `Full Caption Context` remains available and is explicitly described as vocabulary from entire posts containing matching mentions. In evidence mode, year, aspect, sentiment, topic and language filters are applied directly to the frozen mention rows; an aspect-plus-sentiment selection therefore requires the same mention to satisfy both fields. No unrelated portion of the source caption is tokenized in evidence mode.

Evidence-mode frequency tables report raw token frequency, matching evidence-row count and unique matching-document count. Cloud size defaults to unique matching documents containing each token, preventing repeated spans in one document from dominating. Summary cards distinguish matching mentions, unique documents, unique exact evidence spans and unique tokens. A `Top Sentiment Expressions` table conservatively groups exact evidence after case-folding, surrounding-punctuation removal and whitespace normalization only; it reports frozen display wording, sentiment, aspect, evidence occurrences and unique documents without rewriting, translating, stemming or paraphrasing.

Source-aware preprocessing is now explicit. Caption context uses the broader generic stopword set. Evidence mode uses a less aggressive set that preserves negators including `not`, `no`, `never`, `tak`, `tidak`, `x`, `bukan`, `没` and `不`; only the recognized colloquial negator `x` is retained among single-character Latin tokens, while Chinese single-character tokens remain eligible. The UI explains that sentiment applies to the expression rather than every isolated token. An offline negative comparison showed the expected distinction: caption context remained broad (`time`, `full`, `finish`, `training`, `race`), while evidence centered on proposition-level vocabulary and negation (`training`, `not`, `tak`, `kaki`, `cramp`, `legs`, `no`, `pain`, `route`). Frequencies were not manually altered.

### Frozen executive research summary (2026-08-24)

The Executive Overview's compact research-findings card was replaced by one deterministic `AI Summary & Key Insights` card. Despite the presentation label, the card performs no AI generation: it formats only the frozen overall sentiment mart, top-three aspect prevalence rows, four curated research-finding rows and frozen metadata warnings. It reports the three supported 2019-to-2025 prevalence changes, the training/preparation sentiment direction, development precision .513 and recall .790, and explicit non-causal, corpus-composition and extraction-density cautions. The detailed Research Findings page, navigation, frozen artifacts, statistical methods and all other dashboard sections remain unchanged.

## Downstream aspect-level thematic analysis (ALTA v1; implemented and executed 2026-09-01)

ALTA v1 is a new additive, exploratory/descriptive research stage implemented in `marathon_absa/aspect_level_themes.py` and exposed as `python -m marathon_absa.cli absa-v1-aspect-level-themes`. Its purpose is to induce recurring semantic discussion clusters separately inside each frozen ABSA aspect. It preserves the distinction between corpus BERTopic topics (broad post-level discourse), ABSA aspects (model-estimated evaluated event elements plus sentiment), and ALTA themes (what the already separated mentions discuss within one aspect). No existing topic, ABSA, descriptive, inferential, dashboard or thesis artifact is rewritten.

The executed input audit selected the exact frozen production files `data/processed/absa_v1/production/absa_v1_production_v1/absa_v1_production_mentions_v1.csv` and `absa_v1_production_document_results_v1.csv`, joined to `data/processed/topic_discovery_v1/final_taxonomy_v1/final_substantive_topic_corpus_v1.csv` for original-caption/hashtag lineage and checked against `data/processed/absa_v1/absa_aspect_ontology_v1.json`. The executable fail-closed reconciliation verified 7,704 unique analyzed documents, 15,486 unique mention rows, 5,316 mention-bearing documents, 2,388 zero-mention documents, all 20 exact aspects and event years 2019, 2023, 2024 and 2025. The mention schema includes mention/document identity, aspect, target, sentiment, confidence, exact evidence, expression type/language/gloss, hashtag/emoji contributions, grounding/repair audit fields, source/year/language, topic provenance and model/schema identity. The final substantive corpus provides the original caption and existing multilingual/topic provenance. Exact source schemas and SHA-256 hashes are frozen into the ALTA manifest.

The derived `theme_semantic_text` deterministically joins whitespace-normalized nonempty `target`, exact `evidence_text` and separate `english_gloss` with pipe delimiters, removing only exact case-insensitive duplicate components. Original multilingual evidence, hashtags and emoji are retained. Glosses support semantic representation but remain separate from quoted evidence. Embeddings were executed locally using the already cached `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` model, 768 normalized dimensions, batch size 32 and `local_files_only=True`; the ordered-ID-aligned cache is isolated inside the ALTA namespace.

Each aspect was clustered separately. Aspects need at least 60 mentions and 40 unique documents for semantic decomposition. Eligible aspects use deterministic UMAP (cosine, 5 dimensions, min_dist 0, at most 15 neighbors, seed 42) followed by HDBSCAN (Euclidean, EOM). Minimum cluster size is `ceil(sqrt(n_mentions))` bounded to 10–50 and minimum samples is one third of that value with a floor of three. Labels are provisional frequent-target plus TF-IDF keyphrase representations using a fixed English/Malay/event-domain representation stoplist; this does not modify embedding text. HDBSCAN noise remains `theme_id=-1`. Theme support is principally unique documents, with repeated mentions deduplicated and conflicting within-document sentiments conservatively classified as mixed. Themes below 15 unique documents remain visible as low support but are excluded from interview candidates.

The verified run processed all 15,486 mentions and all 20 aspects, producing 101 stable/low-support cluster rows. There were 3,065 noise assignments (19.7921%) and 154 insufficient-support assignments. The four insufficient-support aspects were `event_information` (22 mentions), `race_pack_expo` (31), `transport_access` (44) and `facilities` (57). Discovered cluster counts were: aid stations/hydration 3, crowd/community/atmosphere 3, emerging other 9, emotional experience 10, finisher items 3, organization/operations 6, photography/media 3, physical experience 10, race performance 20, registration/entry 2, route/course 6, safety/medical 3, training/preparation/pacing 11, value/cost 3, volunteer support 5 and weather conditions 4. Examples requiring researcher interpretation include provisional `bukit / route / hills` within route/course (165 documents), `heat / hot / weather` within weather (87), and `my body / body / condition` within physical experience (280). These names are machine aids, not validated qualitative themes or thesis conclusions.

The isolated output package is `data/processed/absa_v1/aspect_level_themes_v1`. It contains CSV (UTF-8-SIG) and Parquet forms of `mention_theme_assignments`, `theme_summary`, `theme_year_summary`, `theme_sentiment_summary`, `theme_representative_evidence`, `theme_review` and `interview_theme_candidates`; it also contains `mention_embeddings_v1.npy`, ordered `mention_embedding_ids_v1.json`, packaged `README.md`, and `theme_manifest.json`. The manifest records source and output hashes, row counts/schemas, thresholds, per-aspect policies, embedding configuration, code-module hash, timestamps and explicit zero network/OpenAI/paid-API counts. There are 98 candidate interview-theme rows pending researcher review. Dashboard integration was deliberately deferred to avoid disrupting frozen views; a later additive Aspect Explorer panel can read these marts without removing the Topic Explorer.

Verification comprised 8 focused ALTA tests and the full suite. The final focused tests passed 8/8; the earlier complete command `python -m pytest -q` passed 319 tests in 116.73 seconds with four sklearn warnings from existing relevance tests. Tests cover frozen reconciliation/read-only hashes, exact ontology/year/lineage, uniqueness and field preservation, multilingual deterministic semantic text, explicit insufficient/noise policy, real representative evidence, unique-document sentiment aggregation, reproducible output hashing and absence of OpenAI/network execution paths. Source hashes were rechecked after execution and matched the manifest. Network calls = 0, OpenAI calls = 0, paid API calls = 0, and frozen upstream writes = 0.

Interpretation limitation: aspect-level themes are derived from model-estimated ABSA mentions. The thematic analysis does not remove uncertainty in upstream aspect classification (development aspect precision approximately 0.513 and recall approximately 0.790), and clustering does not validate the aspect labels. These automatically induced aspect-level semantic thematic clusters require researcher interpretation/verification; they do not establish psychological constructs, population prevalence, causality, inter-rater reliability or representativeness of all KLSCM participants.

## ALTA researcher review and consolidation v1 (implemented; review pending, 2026-09-01)

The additive `aspect_level_themes_review_v1` stage was implemented after ALTA v1 to support explicit researcher interpretation of frozen machine-induced semantic clusters. ALTA clusters remain machine groupings rather than automatically validated qualitative themes. Review operates at the 101-cluster level, not by manually relabeling 15,486 mentions. Its intended lineage is frozen ABSA aspect → frozen ALTA cluster → reviewed discussion theme → observed perception → cautious social-media finding aid → candidate KLSCM 2026 interview proposition.

The implementation is `marathon_absa/aspect_level_themes_review.py`, with methodology in `ASPECT_LEVEL_THEMES_REVIEW.md`. The CLI lifecycle is `absa-v1-alta-review-generate`, `absa-v1-alta-review-validate [--mapping PATH]`, and gated `absa-v1-alta-review-finalize [--mapping PATH]`. Generation creates the compact review surface. Validation accepts structurally valid pending work but reports its incomplete state. Finalization refuses any pending cluster, preventing an unreviewed machine taxonomy from being represented as researcher-reviewed.

The frozen input audit used the actual ALTA Parquet artifacts `theme_summary` (101 rows, 18 columns), `theme_representative_evidence` (505 rows, 11 columns), `theme_review` (101 rows), `mention_theme_assignments` (15,486 rows, 40 columns), `theme_year_summary` (403 rows), `theme_sentiment_summary` (404 rows), and `interview_theme_candidates` (98 preliminary rows), plus `theme_manifest.json` and the ALTA methodology. All manifest-listed ALTA output hashes matched. The audit reconciled 101 unique non-noise clusters across 16 cluster-bearing aspects, all 20 aspects in assignments, 3,065 noise mentions, 154 insufficient-support mentions, and exactly five lineage-valid evidence examples per cluster.

The generated review namespace is `data/processed/absa_v1/aspect_level_themes_review_v1`. It currently contains UTF-8-SIG CSV and Parquet versions of `cluster_review_workbook`, `cluster_review_mapping`, `insufficient_support_summary`, and `noise_summary`, plus `README.md` and `review_manifest.json`. The workbook has one row per non-noise cluster, ordered by total aspect support and then unique-document cluster support. Its columns are: `aspect`, frozen `theme_id`, `cluster_key`, `provisional_label`, `support_mentions`, `support_documents`, `share_of_aspect_documents`, positive/negative/mixed/neutral shares, `years_present`, `top_targets`, `top_keyphrases`, five exact `representative_evidence_*` fields with separate `representative_gloss_*` and document IDs, combined evidence IDs, and researcher-editable `theme_quality`, `review_decision`, `researcher_theme_label`, `merge_target_theme_key`, `researcher_notes`, and `review_status`. The smaller authoritative mapping contains the frozen identity/provisional fields and the six researcher fields only.

Allowed decisions are `KEEP`, `RENAME`, `MERGE`, `UNCLEAR_OTHER`, and `EXCLUDE_FROM_INTERPRETATION`; allowed quality values are `coherent`, `somewhat_mixed`, and `highly_mixed`. A rename requires a human-readable label. A merge requires an existing same-aspect cluster target. Cross-aspect merges, missing targets, self-merges, cycles, targets excluded from interpretation, altered/missing cluster identities, invalid vocabularies, or incomplete finalization are rejected. `KEEP` may retain the provisional label. Unclear/excluded clusters remain auditable mapping rows but do not become final themes. Frozen ALTA theme IDs/assignments and ABSA aspect, sentiment, evidence, mention and document lineage are never overwritten.

The finalizer is implemented but has not been executed on the real package because researcher decisions have not been supplied. When legitimately run, it assigns deterministic reviewed IDs within each aspect and recomputes all counts from frozen mention lineage. Merged document support is the union of document IDs, never the sum of source-cluster counts or percentages. Multiple frozen sentiments for one document/reviewed-theme are conservatively `mixed`; otherwise the sole sentiment is retained. Year summaries explicitly contain 2019, 2023, 2024 and 2025, without significance testing. It creates reviewed taxonomy, assignments, summary, year, sentiment and exact-evidence artifacts, plus a researcher perception layer. Perceptions begin blank as `DRAFT`; only nonempty `APPROVED` perceptions can feed interview candidates on later re-finalization. Interview relevance remains a human `high`/`medium`/`low` judgment independent of prevalence, and candidates begin `PENDING` rather than being automatically shortlisted.

The four frozen insufficient-support aspects remain separate: facilities 57 mentions/49 documents, transport/access 44/38, race pack/expo 31/30, and event information 22/21. Their exact evidence remains available, but no stable reviewed subthemes are manufactured. Noise remains unassigned and is summarized by aspect; its 3,065 mentions occur across 2,736 unique documents before cross-aspect overlap considerations and are never redistributed.

The real package was generated and validated with `reviewed=0`, `pending=101`, and `complete=false`. Therefore no final reviewed taxonomy, approved perception, or interview-proposition pool is claimed. Focused review-stage tests passed 11/11. The complete repository suite passed 332 tests in 58.52 seconds with four existing sklearn warnings in relevance tests. Tests cover frozen-hash integrity, cluster/aspect identity, pending finalization gating, decision/quality/status vocabularies, invalid/cross-aspect/self/cyclic merges, unique-document merge aggregation, frozen sentiment and year aggregation, evidence lineage, insufficient/noise preservation, perception/interview statuses, manifest hashing, and absence of network/OpenAI execution paths. The generated manifest and frozen ALTA hashes were revalidated after generation. Network calls = 0; OpenAI calls = 0; paid inference = 0; frozen ALTA/ABSA modifications = 0.

Reviewed themes remain downstream of model-estimated ABSA assignments and inherit the upstream development limitation (aspect precision approximately 0.513; recall approximately 0.790). Researcher consolidation does not validate the classifier. Theme labels and perceptions are interpretive, social-media prevalence is not a probability estimate for all participants, and interview propositions are qualitative follow-up prompts rather than established facts. Public-dashboard integration remains deferred; a later additive Aspect Explorer view may display finalized reviewed themes without replacing the BERTopic Topic Explorer.

### Local ALTA Streamlit researcher-review interface (implemented 2026-09-01)

The existing CSV methodology now has an internal usability layer at `alta_review_app.py`, supported by the testable file/validation helpers in `marathon_absa/alta_review_ui.py`. Run it from the repository root with `.\.venv\Scripts\python.exe -m streamlit run alta_review_app.py`. Streamlit was already pinned at `streamlit>=1.36`; no dependency or external service was added. This interface is separate from the public research dashboard.

The app reads the generated `aspect_level_themes_review_v1/cluster_review_workbook.csv`, `cluster_review_mapping.csv`, `insufficient_support_summary.csv`, and `noise_summary.csv`. Only `cluster_review_mapping.csv` is mutable during review. Machine identity, support, sentiment, year, lexical and exact-evidence fields are displayed read-only. Researcher controls are restricted to the six established mapping fields; `review_status` is set to `reviewed` only after a successful validated save. Aspects follow descending aggregate cluster document support, clusters follow descending document support, and the UI includes evidence/gloss inspection, same-aspect comparisons and merge choices, pending/reviewed filters, workflow progress, read-only insufficient/noise views, exports, validation and gated finalization.

The UI does not duplicate the review lifecycle: row candidates are passed through `aspect_level_themes_review.validate_review_mapping`, the validation button calls `validate_review_package`, and finalization calls `finalize_reviewed_taxonomy`. Same-aspect merge menus exclude self, while the established validator continues to reject missing, cross-aspect, cyclic or uninterpretable targets. Finalization stays disabled while any row is pending or invalid and also requires explicit researcher confirmation. The current real mapping remains 0 reviewed and 101 pending; the implementation did not make any interpretation decision or execute finalization.

Save safety is local and fail-closed. Each save resolves the destination under `data/processed/absa_v1/aspect_level_themes_review_v1`, reloads the current mapping, updates only the selected cluster, writes a UTF-8-SIG temporary file in the same directory and atomically replaces the CSV. The first successful save in one Streamlit session creates one timestamped backup in the review namespace `backups/` directory. Session-state drafts remain available while navigating and are marked visibly when they differ from disk, but unsaved drafts do not survive an app restart; native Streamlit cannot provide a reliable browser-unload confirmation. This is a single-researcher tool rather than multi-user locking infrastructure.

Automated coverage in `tests/test_alta_review_ui.py` checks artifact loading/joining, aspect and cluster ordering, decision validation, required rename labels, same-aspect/self merge restrictions, atomic row-level preservation, UTF-8 persistence, progress counts, incomplete-finalization gating, reuse of the established validator, destination-path safety and unchanged frozen ALTA hashes. No network, OpenAI, paid-inference, embedding, clustering or ALTA rerun path exists in the interface.

## Finalized reviewed-theme dashboard integration (implemented and validated 2026-09-01)

The existing Streamlit `Aspect Analysis` page in `absa_dashboard.py` now integrates the finalized researcher-reviewed ALTA taxonomy as a read-only “What participants are talking about” section. This preserves the analytical hierarchy: BERTopic topics are broad corpus-level discourse; ABSA aspects are model-estimated evaluated marathon-experience elements; reviewed ALTA themes describe recurring discussion within a selected frozen aspect. The Topic Analysis page remains available and contains only a short clarification of this distinction. No Executive Overview redesign, new navigation page, public-dashboard mart rewrite, perception layer or interview analysis was introduced.

The dedicated loader is `marathon_absa/reviewed_theme_dashboard_data.py`. It reads `review_manifest.json`, `reviewed_theme_taxonomy.parquet`, `reviewed_theme_summary.parquet`, `reviewed_theme_sentiment_summary.parquet`, `reviewed_theme_year_summary.parquet`, `reviewed_theme_evidence.parquet`, and `insufficient_support_summary.parquet` from `data/processed/absa_v1/aspect_level_themes_review_v1`. Normal rendering deliberately does not load the 15,486-row reviewed assignment table. The loader refuses any lifecycle state other than `reviewed_taxonomy_finalized`, verifies every manifest-listed output hash, and reconciles 101 reviewed source clusters, 64 unique finalized themes, 16 theme-bearing aspects, 256 four-category sentiment rows, 256 four-edition year rows, 320 evidence rows (five per theme), and the four exact insufficient-support aspects. It also verifies same-aspect theme lineage, nonempty reviewed labels, sentiment document totals/shares and frozen support bounds. There is no fallback to provisional ALTA labels, the pending review workbook or the original 101 cluster summary.

Within a selected sufficiently supported aspect, reviewed themes are ordered by frozen `support_documents`. A horizontal chart displays unique-document support with frozen within-aspect share and mention count in hover metadata. Each theme is collapsed by default and expands to show supporting documents, frozen `share_of_aspect_documents`, secondary mention count, document-level positive/negative/mixed/neutral shares and counts, and a four-row 2019/2023/2024/2025 descriptive table using frozen `support_documents`, `aspect_documents_that_year` and `within_aspect_theme_prevalence`. Five exact representative evidence spans are shown with year, frozen sentiment, source language, optional target and a separately labelled English gloss that is explicitly not presented as a verbatim quotation. No usernames, external links or unnecessary identifiers are exposed.

The four insufficient-support aspects have explicit nonempty states rather than generated themes: facilities (57 mentions/49 documents), transport/access (44/38), race pack/expo (31/30), and event information (22/21). The dashboard explains that insufficient thematic support does not imply unimportance. HDBSCAN noise is not shown as a participant theme and is mentioned only in a concise methodological note. `emerging_other` is displayed from the finalized taxonomy without frontend reinterpretation.

The frontend performs only filtering, ordering, formatting and frozen-table joins. It does not calculate a new prevalence, sentiment definition, importance score, temporal significance test, inference, cluster, label, perception, recommendation or proposition. The methodology expander states that document-level prevalence is primary, evidence is grounded in exact ABSA spans, the display is descriptive, and upstream aspect-classification uncertainty remains (development precision 0.513; recall 0.790). This dashboard integration is descriptive. Researcher perceptions and participant interview propositions have not yet been developed.

Focused loader and dashboard UI tests passed 28/28. The final complete repository suite passed 349 tests in 63.11 seconds with four existing sklearn warnings in relevance tests. Coverage includes finalized-state enforcement, the 101→64 reconciliation, absence of provisional leakage, unique and same-aspect theme identities, insufficient-support handling, representative-evidence/assignment lineage, multilingual text preservation, frozen sentiment/year joins, unique-document support, read-only hash integrity, no inference/API path, Aspect Explorer rendering, Topic Explorer preservation and existing page compatibility. Network calls = 0; API/OpenAI calls = 0; paid inference = 0; frozen analytical artifact writes = 0.
# Additive long-form blog/review branch (implemented 2026-09-03)

The project now contains an additive, source-separated long-form review branch documented in `BLOG_REVIEW_INTEGRATION_METHODOLOGY_V1.md`. Historically, 28 blog records existed in the early combined preparation; later frozen topic discovery and ABSA production became Instagram-only. This new branch restores the reviews through source-appropriate processing after, rather than inside, the frozen Instagram pipeline.

The authoritative audit found 28 non-empty raw records, three exact duplicate descendants, 25 usable canonical parent reviews, and 56 existing stable inference chunks. Parent review is the descriptive unit; chunk is only the model inference unit. The branch reuses frozen ABSA V3 and the finalized 64-theme Instagram ALTA taxonomy as a reference, with same-aspect researcher-confirmed mapping. It does not refit BERTopic or ALTA, change the 20-aspect ontology, pool source denominators, or add cross-source inferential tests.

Implemented components are `marathon_absa/blog_analysis.py`, `marathon_absa/cross_source_analysis.py`, `blog_theme_review_app.py`, CLI commands prefixed `blog-` plus `cross-source-analysis`, and a pending-safe public dashboard page. The audit and 56-request ABSA package were executed locally with zero API/network calls. The paid Batch was not submitted; blog ABSA, researcher mapping, finalized blog marts, and cross-source findings remain pending. Blog collection provenance fields remain `DOCUMENTATION_REQUIRED`. The inherited V3 development limitation (aspect precision about 0.513, recall about 0.790) applies unchanged.

### Blog ABSA Batch finalization and exact-evidence repair (executed 2026-09-07)

The previously prepared 56-request blog Batch was subsequently completed and imported. The first offline `blog-absa-finalize` attempt failed closed with `missing=0, failed=5`. Each failed chunk contained model evidence that was not a case-sensitive contiguous substring of its immutable inference chunk: two capitalization-only spans and five discontinuous spans that joined clauses with an ellipsis or omitted intervening source wording. This was a local grounding failure, not a missing provider response; no request was resubmitted.

`marathon_absa/blog_analysis.py` now applies established deterministic evidence canonicalization and two bounded blog repairs. A unique case-insensitive occurrence is replaced by its exact source casing. Discontinuous evidence is accepted only when strong source anchors are unique and ordered; final evidence is the single enclosing slice copied from the source chunk. The generic omitted-text path requires prefix and suffix anchors of at least 32 characters, at least 64 anchored characters total, and anchor coverage of at least half the model evidence. Ambiguous or invented evidence remains unrecoverable and causes finalization to fail closed.

The raw Batch output was not modified; its SHA-256 remains `cbdca8ba7db59a716f4fd454d67d4fcdd25e8706b751d292ac1e1375eecfd383`. Final mention rows preserve original model evidence, exact-source evidence, raw-exact and repaired flags, repair method, and exact offsets. The additive `data/processed/blog_analysis_v1/blog_absa_grounding_audit_v1.csv` records this lineage for all 301 mentions: 294 exact as returned and seven repaired, comprising two unique case-insensitive matches and five unique ordered-anchor enclosing spans.

The successful rerun completed all 56 chunks for 25 parent reviews, with no missing or failed chunks, 301 mentions, and all 20 aspect families represented. `blog_absa_manifest_v1.json` records finalization time `2026-09-07T08:16:29.971183+00:00`, grounding counts, zero API calls, and hashes. CSV SHA-256 values are `13801a2b2db93f1ab444060dca8207206cd357917735b13f5387d672da063003` (mentions), `6a8561ac74902318c90fa17213cfb8c7ca879759ef9b496b5b0c1369af67c1fa` (chunk results), `5ed62a5b22d462599e09af5177297ba1f4949081bf631a7439491a21f50548b5` (processing log), and `06f8b933e9f0e1f6b17a11692d22ea43c1e4cee6dfd967677f798a9d6889f8e4` (grounding audit). The focused blog suite passed 11/11. A complete-suite attempt reached 353 passes but seven tests encountered transient Windows paging-file/out-of-memory import/read failures; all seven passed in a fresh isolated rerun, with two existing sklearn warnings. Researcher blog-to-theme mapping and cross-source analysis remain later stages and are not finalized.


### Delegated AI-assisted blog theme review (executed 2026-09-07)

At the user's explicit request to complete the blog review, Codex inspected all 301 mention evidence spans against the 64 frozen reviewed theme labels and their scope notes, consulting source chunks for ambiguous references. This is delegated AI-assisted interpretation, not independent human coding, researcher validation, or inter-rater reliability evidence. The existing review_status=reviewed field records workflow completion only; reviewer provenance is explicitly recorded in every decision note and in the companion manifest. No similarity threshold or automatic acceptance of the first candidate was used.

The saved authoritative blog_theme_mapping_candidates_v1.csv contains 301 reviewed rows and zero pending: 225 MATCH_EXISTING mentions using 47 reference themes, 67 EMERGENT_BLOG_THEME mentions across 35 aspect-specific labels, and 9 EXCLUDE_FROM_THEME_COMPARISON rows. Exclusions comprise six comparator-event mentions (Singapore congestion and five Hamburg assessments), a charity foundation's wider operations, a life-journey metaphor, and generic advice to respect distance. Exclusions apply only to theme comparison; frozen upstream ABSA aspects, sentiment, source spans, IDs and aspect-level marts are unchanged. New labels include toilet provision, transit access, kit collection, route-marker accuracy and digestive discomfort. Some labels have single-review support and must not be represented as established recurring population themes. Reported event-ranking or provision claims remain source claims, not verified facts.

Artifacts under data/processed/blog_analysis_v1/: blog_theme_mapping_candidates_v1.csv (authoritative decisions; SHA-256 037038db8af5e66f5e09c7c0b3c6bedd6464edc77543f4425923c9e43f8d44af), blog_theme_mapping_candidates_v1.backup_ai_review_20260907_084822.csv (original pending table), blog_theme_ai_review_audit_v1.csv (one-based app row, mention/review identity, aspect, target, evidence, decision and provenance note), and blog_theme_ai_review_manifest_v1.json (counts, UTC execution time, backup path, method, integrity and limitations). Completion time 2026-09-07T08:48:22.768555+00:00 comes from the execution manifest. scripts/complete_blog_theme_review_v1.py preserves the explicit decisions, checks the original ordered mention identities, preserves already-reviewed rows, backs up before atomic replacement and checks the CSV round trip.

Validation confirmed complete coverage, unique mention identities, valid same-aspect existing assignments, nonempty emergent labels and exact preservation of all non-editable columns. Twenty-seven matches fall outside the original top-three suggestions. Accordingly, blog_theme_review_app.py now offers all finalized same-aspect themes with readable labels, preserving those saved selections when revisited. Blog theme-mart finalization and cross-source analysis were not executed by this review task; those remain separate downstream stages. No paid API calls, local embedding reruns or frozen Instagram artifact writes occurred.

Review-task verification: the existing focused blog test suite passed 11/11 in 3.76 seconds. Streamlit AppTest confirmed the empty pending filter, access to reviewed rows, and preservation of a saved selection outside the top three (app row 3); no save action was triggered during the UI check.

### Blog-emergent consolidation audit (executed 2026-09-07; researcher finalization pending)

This additive stage audits the completed original blog theme mappings without modifying any frozen upstream result. It supersedes earlier statements that original theme-mart finalization and cross-source analysis had not run: current artifacts show 20/20 shared aspects, 47/64 Instagram-derived themes supported in blogs, and 17/64 without support. Consolidation is a new, separate review stage and remains pending.

Evidence is in `data/processed/blog_analysis_v1/blog_emergent_theme_audit_v1.csv` and `.json`: 67 emergent mention records, 35 concept IDs, 17 aspects, 18 represented parent reviews, 35 available original labels, zero missing labels, and 24 singleton IDs. IDs originate in `finalize_theme_mappings` as aspect plus an eight-character SHA-256 prefix of the exact original label. Identical labels within an aspect already share identity; no further exact normalized label duplicates were found. All evidence and review/chunk linkage are recoverable against 25 usable parents. The original labels explicitly have delegated AI-assisted provenance rather than independent human validation. No researcher consolidation decision is inferred from their original reviewed status.

Added implementation: `marathon_absa/blog_emergent_themes.py`, `blog_emergent_theme_review_app.py`, and `tests/test_blog_emergent_themes.py`. CLI commands `blog-emergent-theme-audit` and `blog-emergent-theme-finalize` are in `marathon_absa/cli.py`. The audit writes a 39-pair same-aspect candidate CSV using advisory lexical label sequence similarity, a resumable decisions CSV, and CSV/JSON audit artifacts. No local embeddings, API calls, automatic thresholds, or paid models are used. Candidate pairs are review aids, not 39 suspected duplicates. No evidence-equivalent duplicate group is confirmed. The audit records scope questions concerning overall route quality/difficulty, village space/ground, emotional triggers, and a volunteer observation mentioning GCM. No merge or exclusion has been applied. Current state is 0 resolved and 35 UNCLEAR decisions.

The review page shows labels, source IDs, n/25 review support, all evidence/glosses, original notes, inference-chunk context, and same-aspect candidates. It supports KEEP_SEPARATE, MERGE_WITH_EXISTING_EMERGENT, RENAME_ONLY, EXCLUDE_AS_NON_THEME, and UNCLEAR, with atomic saves, content-addressed backups, a write lock, version conflict detection, and deterministic resumption. Public display copies redact known author strings and URLs. Audit source evidence remains intact. A rename preserves ID; a merge uses aspect, normalized label and sorted source IDs. Merge chains, cycles, cross-aspect targets, conflicting labels, unsupported review identities, missing labels/evidence, duplicate or missing lineage, denominator drift, and unresolved decisions fail closed. Exclusions require reasons and retain all mentions in the lineage file. Singleton and low-support flags both identify support below two unique parent reviews; support mentions remain separate.

`blog_emergent_frozen_baseline_v1.json` records SHA-256 values for 390 pre-existing topic-discovery, ABSA/ALTA, blog, and parent/chunk input files, captured before stage writes. It must not be reset to bypass integrity failures. Existing frozen-count checks and the V3 prompt SHA-256 are also verified. Future outputs are `blog_emergent_theme_taxonomy_v1.csv`, `blog_emergent_theme_lineage_v1.csv`, and `blog_emergent_theme_finalization_v1.json`. The final manifest binds output and decision hashes, so subsequent decisions invalidate consumption until finalization runs again.

`marathon_absa/cross_source_analysis.py` now consumes only a finalized, integrity-checked emergent taxonomy while preserving original matched assignments. Regeneration rejects changes to 64 matched reference themes, 47 blog-supported reference themes, or 20 shared aspects. `absa_dashboard.py` adds a pending-safe emergent table with readable labels and n/25 support plus caution flags; no redesign or new personal-data fields. The existing marts remain unchanged while review is unresolved. Tests exercise regeneration into a temporary directory with synthetic finalized emergent data, not publication of unreviewed real findings.

The blog-emergent taxonomy is complementary to, rather than an extension of, the frozen 64-theme Instagram taxonomy. Blogs did not jointly derive the Instagram taxonomy. Full procedure, artifact register, exact lifecycle commands, prescribed post-finalization reporting language, privacy boundary, and limitations are documented additively in `BLOG_REVIEW_INTEGRATION_METHODOLOGY_V1.md`. Current next action is researcher consolidation review, not rerunning ABSA or topic modelling. Separate denominators remain 7,704 and 25; pooled prevalence is false, inferential tests zero, API calls zero, and comparison type descriptive_triangulation.

Verification of this stage: 27 focused emergent tests passed in 14.20 seconds, including Streamlit loading, same-aspect review navigation, atomic save/backups, stale-session conflict rejection, deterministic IDs, parent-level counts, exclusion lineage, fail-closed gates, synthetic finalizer execution in an isolated directory, and temporary cross-source regeneration with readable labels and unchanged matched-theme rows. The complete suite passed 387 tests in 95.46 seconds. Four warnings remain in two existing relevance tests: sklearn single-label confusion-matrix warnings and invalid scalar-division warnings. They are reported separately from failures; there were no failed tests. The actual CLI finalizer was invoked as a gate check and rejected the 35 unresolved decisions with `ValueError: Unresolved researcher decisions remain`, creating no finalized taxonomy. Post-test verification confirmed all 390 baseline hashes and all frozen-count/prompt checks unchanged. No real cross-source mart was regenerated. The review interface was checked with Streamlit AppTest, not left running as a server.

### Delegated blog-emergent review and finalization (executed 2026-09-07)

The user subsequently requested that Codex perform the consolidation review. This supersedes the preceding pending lifecycle state. The saved review manifest records completion at 2026-09-07T09:42:52.040248+00:00. Codex inspected all 67 emergent evidence spans, their available English glosses, surrounding source windows, and full chunks for ambiguous references, and considered all 39 same-aspect candidate pairs. This is explicitly delegated AI-assisted interpretation, not independent human validation or inter-rater reliability evidence. No similarity threshold or external API was used.

All 35 decisions are resolved: 28 KEEP_SEPARATE with original labels, seven RENAME_ONLY, zero merges, zero exclusions, zero UNCLEAR. Identical concepts already shared source IDs; the remaining same-aspect pairs expressed distinguishable observations. No reduction in theme count was imposed merely to achieve consolidation. The final taxonomy retains 35 stable source IDs across 17 aspects, all 67 mentions, 18 represented parent reviews, and 24 singleton/low-support themes (support <2 parents). Each theme denominator is 25. The taxonomy remains complementary to the frozen 64-theme Instagram taxonomy.

Seven evidence-grounded label corrections:

- Strength of the participating field -> Perceived strength of female half-marathon runners.
- Toilet provision and queues -> Toilet provision and condition (race venue and accommodation).
- Post-race cooling facilities -> On-course mist-arch cooling experience.
- Race cutoff rules and time allowances -> Perceived race cutoff allowance.
- Aggregate field finishing times -> Reported claim about slow average finishing times.
- Course distance and marker accuracy -> Perceived course length and distance-marker accuracy.
- Contributing as an expo volunteer -> Volunteering at a visiting-event booth within the race expo.

Context resolves the key scope questions. The mist arch was about 5 km before the finish, not post-race; frozen facilities ownership is unchanged despite related cooling evidence in aid_stations_hydration. The GCM booth was physically within the SCKLM expo, so the rewarding contributor experience is retained, explicitly distinguished from a GCM race evaluation or KLSCM on-course volunteer service. One toilet observation concerns hotel toilets; therefore aggregate support of 3/25 spans two race-venue parents and one accommodation parent and must never be reported as three accounts of race toilets/queues. All mentions remain traceable without rewriting the original blog mappings. The route appraisal umbrella covers three general approval/acceptability parents and two challenge/difficulty parents; 5/25 must not be described as five difficulty reports. The event-ground umbrella covers two space observations and one muddy-ground observation, not three observations of each. The finishing-time statement is an attributed third-party claim, not verified performance data. GPS discrepancies and cutoff comparisons likewise remain perceived/reported observations.

`scripts/complete_blog_emergent_review_v1.py` preserves the explicit 35 decisions and their detailed rationales, validates coverage and frozen inputs, refuses conflicting resolved decisions, and uses the existing backed-up atomic save mechanism. Added review artifacts under `data/processed/blog_analysis_v1/` are `blog_emergent_theme_ai_review_evidence_v1.csv` (67 mention-level evidence/decision rows), `blog_emergent_theme_pair_review_v1.csv` (39 reasoned same-aspect keep-separate decisions), and `blog_emergent_theme_ai_review_manifest_v1.json` (provenance, execution time, counts, decision hashes, and integrity). The original audit remains the pre-consolidation snapshot, so its earlier pending status is historical rather than the current lifecycle state. Current decisions and final manifests supersede it.

The actual CLI `blog-emergent-theme-finalize` completed, producing `blog_emergent_theme_taxonomy_v1.csv`, `blog_emergent_theme_lineage_v1.csv`, and `blog_emergent_theme_finalization_v1.json`. Cross-source generation subsequently executed at 2026-09-07T09:43:54.699637+00:00 and updated the emergent CSV/parquet with readable reviewed labels and caution flags. The cross-source manifest now additionally hashes the emergent taxonomy and finalization manifest to make provenance explicit. Previous cross-source files were backed up under `data/processed/cross_source_analysis_v1/emergent_review_backups/` before regeneration. Aspect-comparison and matched-theme CSV SHA-256 values were identical before and after regeneration, preserving 20/20 shared aspects and 47/64 supported reference themes (17 without support).

Post-publication verification in `blog_emergent_theme_post_review_integrity_v1.json` confirms all 390 upstream hashes unchanged, including original blog outputs, Instagram production, reviewed ALTA, and topic discovery. Counts remain 13,799 topic population, 13,743 eligible fitting population, c1, 32 topics, 7,704 ABSA documents, 15,486 mentions, 20 aspects, 101 source clusters, 64 reviewed Instagram themes, 16 theme-bearing aspects and four insufficient-support aspects. Denominators remain 7,704 and 25; comparison_type is descriptive_triangulation; pooled prevalence false; inferential tests zero; API calls zero. No ABSA, BERTopic, ALTA, or embedding rerun occurred.

Delegated-review validation: 29 focused tests passed in 17.63 seconds; the complete suite passed 389 tests in 89.75 seconds. Four existing sklearn warnings remain in two relevance tests (single-label confusion matrices and invalid scalar division), with no test failures. New checks confirm seven corrected labels, AI provenance, all 67 mention identities, all 35 stable IDs, 39 reasoned pair decisions, readable published labels, and source-hash linkage to the finalized taxonomy. Post-suite integrity again passed all 390 frozen file hashes. No analytical stage remains blocked. A scratch publication helper initially failed to import the package when run by file path; it made no publication writes, and invoking it as a repository module completed successfully. No API/network rerun was needed.

### Streamlit integration of finalized long-form review data (2026-09-07)

The user requested that Streamlit reflect inclusion of long-form review analysis. The audit found finalized artifacts already present, but the Cross-Source Analysis page only displayed an aspect table and emergent labels, omitted the 301 blog mentions, review-level sentiment and 47 matched reference themes, and incorrectly described delegated AI review as researcher verification. The underlying frozen Instagram dashboard marts remain Instagram-only by design.

Added `marathon_absa/blog_dashboard_data.py`, a read-only loader for the finalized blog and cross-source artifacts. It checks the existing emergent finalization and 390-file frozen baseline, cross-source source hashes, blog ABSA hashes and completeness, unique parent totals, mention/chunk reconciliation, separate 7,704/25 denominators, 64 reference themes with 47 supported in blogs, sentiment counts against parent-aspect support, and the published emergent mart against the finalized taxonomy. Missing files and consistency failures produce distinct unavailable/error states; stale results are not relabeled as a completed review. Views use explicit public column lists, with no author, URL or parent/chunk identity columns. Known-author and URL redaction is reapplied to labels, representative evidence and scope notes.

`absa_dashboard.py` now displays finalized long-form status and counts derived from artifacts: 25 parent reviews, 301 mentions, 56 inference-only chunks, 20 shared aspects, 47/64 reference themes and 35 emergent themes. The existing Cross-Source section adds parent-review aspect sentiment, a matched-theme table with observed/unsupported filters, an aspect filter, readable emergent support and singleton wording, selectable representative evidence and saved scope caveats, and aggregate review-year/language composition. All review support remains n/25 even when filtering by aspect. Corpus totals are fixed and explicitly described as such. Twenty-four singleton themes remain visible and cautiously interpreted. The mixed-setting toilet, on-course mist-arch, and expo-booth scope corrections are accessible from saved review reasons. No model calls or new analytical estimates were introduced.

The review provenance now explicitly says AI-assisted and delegated by the user, not independent human validation. Sidebar branding mentions long-form reviews. Other pages show a source-scope notice explaining that their charts and statistical results remain frozen Instagram analysis and that blogs appear separately under Cross-Source Analysis. Methodology clarifies that its existing inferential procedures apply only to Instagram; the long-form branch is descriptive. The legacy `app.py` preparation dashboard carries a notice distinguishing preparation records from the finalized research population, directing users to the research dashboard. Navigation, frozen Instagram chart values, statistical outputs and page layout are retained.

New tests in `tests/test_blog_dashboard_data.py` cover reconciled counts and public column allowlists, stale-mart rejection, finalized UI metrics and review provenance, aspect/unsupported-theme filters, n/25 preservation, displayed mixed-setting toilet caveats, and explicit Instagram-only scope on other pages. The focused loader/UI run, including existing ABSA dashboard tests, passed 25 tests in 34.37 seconds. These changes read existing finalized analysis artifacts; no upstream data or cross-source mart was regenerated or modified by this dashboard update.

Final dashboard-integration verification: the complete suite passed 394 tests in 131.69 seconds, following 25 focused loader/UI passes. Four existing sklearn warnings remain in the relevance tests (single-label confusion matrices and invalid scalar division); no tests failed. Streamlit AppTest exercised the updated page, aspect filtering, unsupported matched-theme filtering, and evidence scope rendering. Integrity verification reconfirmed all 390 frozen-file hashes and byte-identical aspect/matched-theme comparison CSVs. The update changes dashboard loading/presentation and documentation only; the finalized long-form data was already present and was not recomputed. No paid/API calls occurred. A sandbox-denied process inventory did not affect code or data verification; no claim is made that an existing live Streamlit process was restarted.


## Integrated Participant Experience & Organizer Insight Synthesis

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

Configure `OPENAI_API_KEY` in the repository-root `.env` file or invoking environment using the existing account setup. The `submit` and `collect` actions explicitly load that file before constructing the API client; an existing environment value takes precedence. The credential file is resolved from the module location, independently of the output `--root`. Offline actions do not invoke this credential loader. When the batch completes:

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

Credential-loading correction (2026-09-18, session date): the user-reported standalone `submit --run-api` traceback failed at `OpenAI()` because this module bypassed the configuration module's `.env` loading. `marathon_absa/participant_experience.py` now uses `create_api_client()` for submission and collection, loading the repository-root `.env` with `override=False` after the explicit API gate. Client initialization against local credentials succeeded without sending an API request or exposing credentials. Four offline regression cases in `tests/test_participant_experience.py` exercise both actions with file-only credentials and existing environment credentials, from a different working directory; they also check the API gate and absence of a submission marker on client-construction failure. This correction does not establish that a production batch was submitted or collected.

Offline preparation is complete: 99 requests, 749,630 UTF-8 bytes and approximately 497,581 input characters using the repository default model. No API submission, generated synthesis or review/finalization has occurred. Classification produces 70 mixed, 16 positive and 13 pain-point theme rows; these are not participant estimates. All 29 new tests pass, and the final complete suite passes 423 tests with four existing scikit-learn warnings in 213.01 seconds. Final checks confirm all 390 original baseline hashes and all 411 synthesis input hashes unchanged. Three legacy request-writing tests were isolated in temporary folders, the navigation assertion was updated, and the full suite used a writable temporary Numba cache after an earlier cache-import stall. The production page was visually checked in its correct pending-review state; finalized layout was exercised with synthetic test fixtures. See `PARTICIPANT_EXPERIENCE_IMPLEMENTATION_REPORT.md` for the exact file register, failure history, limitations and next approval command. Status: READY_FOR_SYNTHESIS_GENERATION.

Credential-fix validation (2026-09-18): the focused command .venv\Scripts\python.exe -m pytest tests\test_participant_experience.py -q passed all 33 tests in 52.10 seconds. The full suite was not rerun for this correction; no paid API validation was performed.

### Participant-experience batch collection (2026-09-19)

The user submitted batch batch_6aac9c104a848190b42f8849fb1f5cee. A read-only API status check verified completion of all 99 requests with zero API failures. The output was downloaded once to data/processed/participant_experience_v1/responses.jsonl. No new generation request was submitted during collection. The strict importer rejected the output before creating candidates.json or review.json: a complete offline audit found 89 responses passing structural validation and 10 failing citation/required-evidence validation. Passing structural checks is not substantive review or finalization. The detailed per-theme report is response_validation_audit.json in the same downstream directory. Original responses remain intact; no citation or narrative was silently corrected, and the dashboard remains gated pending valid import and review. Status: GENERATED_IMPORT_VALIDATION_REQUIRED. Resolve the ten invalid outputs against the frozen evidence before importing and reviewing the full package. Do not resubmit the full production batch merely because import failed.


### Invalid-only participant-experience recovery (2026-09-19)

Offline inspection of the collected 99-response production artifact confirms 89 structurally valid responses and ten invalid themes. All original request payloads match the finalized evidence packages, and all 411 evidence-input hashes plus the 390-file frozen baseline pass. Seven distinct rejected identifiers across six themes are not hallucinations: they appear in the same theme's unavailable_evidence_ids metadata for blank excerpts and are excluded from allowed qualitative citations. Two other identifiers omit the final character of a valid same-theme ID. One negative summary duplicates an exact valid citation. The pacer theme has cited organizer claims; its actual rejection is an uncited mixed summary with added text after the exact insufficient-evidence sentinel. The validator's generic organizer-evidence error message obscures this cause, but rejection is correct. No validator, original prompt, evidence, or model output was altered.

The additive marathon_absa/participant_experience_retry.py CLI implements prepare, verify, submit, collect and merge. Offline prepare was executed for exactly the ten invalid themes under data/processed/participant_experience_v1/retry_v1. Original model, inference settings, schema and input strings are retained; only retry instructions add an explicit theme-specific allowed-ID list, prohibit citing unavailable IDs or reconstructing identifiers, require unique citations and supporting organizer citations, and specify exact insufficient-evidence output. No automatic ID replacement or deduplication is performed. Unsupported non-direction claims remain blocked by the unchanged validator even when explicitly labeled insufficient evidence.

retry_manifest.json pins every original top-level artifact, per-response line hashes, the 89 accepted insight hashes, implementation hashes and retry request hash. preserved_valid_responses.jsonl copies the original 89 valid JSONL lines byte-for-byte. Verification rederives the invalid-only set and checks all frozen inputs. The API submission requires --run-api and explicit user approval and records an exclusive started marker to prevent resubmission. Retry collection saves completed batch/input/output identities and response hashes. Merge requires exact invalid-theme coverage, unchanged preserved originals, valid lineage and all 99 unique candidates passing the same strict validator before candidates.json or review.json is created. Each accepted candidate records whether it is original or retried, response/version identifiers, line/content/file hashes and retry-manifest linkage. All review rows start PENDING; no finalization is automatic.

The original local submission.json is a submit-time validating snapshot. The previous collection did not save a completed batch snapshot or independent remote-input copy. Local request hashes and submission/upload file IDs reconcile, but remote-original provenance cannot be independently reconstructed offline. This limitation is recorded, not fabricated or bypassed; new retry collection saves that evidence prospectively. No API call was made in this recovery preparation. Status: TEN_INVALID_RETRIES_PREPARED_AWAITING_USER_APPROVAL; production candidates and review still absent.

Detailed per-theme causes, exact theme IDs, commands and file hashes: PARTICIPANT_EXPERIENCE_RECOVERY_REPORT.md. Machine-readable audit: work/participant_experience_recovery_audit_20260919.json. New deterministic tests: tests/test_participant_experience_retry.py; fixtures isolate writes, forbid real API calls and use synthetic replacements only in temporary directories. The approval-gated submission command is `.\.venv\Scripts\python.exe -m marathon_absa.participant_experience_retry submit --run-api` from the repository root. This command was not executed.

Final validation (2026-09-19): the complete offline suite passed 446 tests in 694.09 seconds, including all 19 recovery tests. Four existing scikit-learn warnings remain in two relevance fixtures (single-label confusion matrix and invalid scalar division). The initial focused run passed 18 recovery tests in 121.05 seconds; the subsequent added mocked-submission test passed in the full suite. Post-suite verification again passed all 411 frozen evidence-input hashes and all 390 baseline hashes and integrity checks. All ten original top-level production artifacts match the initial audit hashes; the 89 valid response lines and insight hashes remain preserved. Exactly ten retry requests are prepared. No candidates.json, review.json, or retry submission marker exists. Full test log: work/participant_experience_recovery_full_tests.txt. Final machine-readable verification: work/participant_experience_recovery_final_verification.json. Status remains AWAITING_EXPLICIT_API_APPROVAL.


### First retry collection and four-theme follow-up (2026-09-19)

The user's submitted retry batch batch_6aae15fbd32481908c415339d67b8b78 completed and was collected, as recorded by retry_v1/collection.json. The local merge correctly stopped at strict validation; downloaded output remains in retry_v1/responses.jsonl. An exhaustive offline audit found six valid retry responses and four invalid responses, giving 95 validated responses available (89 original plus six first-retry outputs). Neither candidates.json nor review.json exists. The collection traceback indicates a grounding failure after successful download, not a failed API batch; repeating collect is inappropriate because the saved collection already exists.

Remaining failures: physical_experience__rt04 repeats the same valid ID in organizer_insight and evidence_scope_note; race_performance__rt11 appends an extra 2 to instagram:absa1_42968f7985dd4c8e in five fields; volunteer_support__rt03 uses blog:absa1_949f16eadea849c7 instead of the supplied blog-prefixed identifier in organizer_implication; blog_emergent__transport_access__306a1f5b again omits the final f of blog:absa1blog_e5ce7c3c7be31dcf in five fields. These comparisons diagnose copying errors only; no citation was replaced or deduplicated. Full claims and permitted IDs are saved in data/processed/participant_experience_v1/retry_v1/response_validation_audit.json.

The first retry demonstrated that prompt-only guidance does not reliably prevent malformed identifiers or duplicate citations. The additive marathon_absa/participant_experience_retry_round2.py implements prepare/verify/submit/collect/merge for the four remaining invalid themes. It retains original evidence input strings, model and inference settings, and adds each theme's allowed IDs as an enum on every evidence_ids item in the response JSON schema, plus a one-value theme_id enum. Duplicate citations are still rejected by the unchanged validator; no unsupported schema uniqueness feature or automatic deduplication is introduced. The original retry implementation remains unchanged because its hash is pinned by the first submission.

Offline preparation wrote retry_v2/requests.jsonl (four requests), retry_v2/preserved_valid_responses.jsonl (95 original byte-preserved response lines), and retry_v2/retry_manifest.json. It pins the original production files, all first-retry artifacts, its implementation hash and request/preservation hashes. Verification rederives the accepted responses through the original strict validator and checks all 411 frozen input hashes. Collection verifies batch/input/output lineage. Merge validates all 99 responses before writing candidates or PENDING review entries and records original/retry_v1/retry_v2 response provenance per theme. Any invalid follow-up blocks the entire merge and lists the failing theme IDs. The six newly valid first-retry themes are excluded from submission, as are the original 89.

Validation: all seven tests in tests/test_participant_experience_retry_round2.py passed in 11.93 seconds. They check four-only selection, evidence-ID enums, byte preservation of 95 responses, 99 unique pending candidates with version provenance, rejection of unknown IDs/duplicates/incomplete coverage, unchanged invalid outputs, request tampering, explicit API gating and repeat-submission protection. Only focused tests were run for this additive follow-up; the earlier 446-test full-suite result predates it. Final verification passed all 411 evidence-input hashes and 390 frozen baseline hashes and checks. No upstream stage was rerun and no new API call occurred. Final verification is work/participant_experience_retry_v2_verification.json.

Status: FOUR_INVALID_RETRIES_PREPARED_AWAITING_EXPLICIT_API_APPROVAL. After user approval, the exact command from the repository root is:

```powershell
.\.venv\Scripts\python.exe -m marathon_absa.participant_experience_retry_round2 submit --run-api
```

After that batch completes, use the same module with `collect --run-api`. Do not rerun the first retry submit/collect commands. The four-request follow-up was NOT submitted during this audit, and no approval or finalization was performed.


### Second-retry exact duplicate correction (2026-09-19)

The second retry completed and downloaded successfully. Three responses passed the unchanged strict validator. physical_experience__rt04 failed only because organizer_insight repeated instagram:absa1_0c68f2dd88342bcc twice; all referenced IDs were allowed. There were no unknown IDs in that response. Thus 98 selected responses were already valid, and one required only removal of an exact duplicate citation. The user's original authorization explicitly permits documented and tested mechanical deduplication of identical valid IDs where substantive grounding does not change.

The additive offline module marathon_absa/participant_experience_deduplicate.py checks both retry chains and all original hashes, removes only repeated identical allowed IDs in a derived copy, preserves citation order and every claim's text, and revalidates all 99 candidates with the unchanged validator. Unknown, shortened or malformed IDs are rejected, never replaced. It preserves every downloaded response file byte-for-byte and records raw/derived insight hashes, exact before/after citation arrays, the removed count, response IDs, original/retry version and all source hashes in mechanical_deduplication_audit.json and candidate provenance. Already valid candidates are copied without modification. Only after all 99 pass can candidates.json and PENDING review.json be created; no automatic approval or finalization occurs. Both pinned retry implementations and their manifests remain unchanged.

Seven focused tests in tests/test_participant_experience_deduplicate.py passed in 3.58 seconds: exact-only deduplication, immutable raw/text preservation, unknown/truncated/missing evidence rejection, valid no-op behavior, all 99 unique validated candidates with 98 unchanged and one documented correction against actual saved artifacts, and overwrite protection. The full suite was not rerun for this isolated additive correction. Execution and post-write verification are recorded below.

Executed offline correction and merge: 99 unique candidates validated and 99 PENDING review entries created, 98 candidate insights unchanged, one exact duplicate citation removed in physical_experience__rt04. Post-write checks verified candidate/review equality and hashes, the correction-audit hash, every preserved raw source hash, all 411 frozen input hashes and all 390 baseline hashes/integrity checks. No API call, approval or finalization occurred. Status: GENERATED_PENDING_RESEARCHER_REVIEW. Do not rerun collection or submission; the next stage is substantive researcher review of review.json against evidence.json.

### Human researcher review interface (2026-09-19)

Implemented a standalone local Streamlit workflow in `participant_experience_review_app.py`, with presentation in `marathon_absa/participant_experience_review_ui.py` and offline persistence/integrity logic in `marathon_absa/participant_experience_review.py`. This interface is separate from the organizer dashboard and has no generation, model-inference or finalization action. The purpose is to let the human researcher read every claim beside its exact cited evidence and save their own judgment, without manually joining JSON identifiers.

The pre-implementation read-only audit verified 99 themes, 99 structurally validated candidates, 99 PENDING reviews, zero approvals and no finalized artifact. Coverage is 17 Instagram-only themes, 47 cross-source themes and 35 blog-only emergent themes; 37 themes have singleton blog support. Candidate provenance identifies 89 original accepted responses, six first-retry responses, three unmodified second-retry responses and one second-retry response with the existing audited exact-citation deduplication. The audit snapshot `work/participant_review_before.json` pins all 30 existing production/retry artifact files and the 411 frozen evidence inputs. No production research decision was made to construct or test the interface.

The actual review schema remains a list of per-theme records containing theme_id, decision, reviewer, reason, candidate_hash and insight. Candidate insights and generation provenance remain immutable in candidates.json. The evidence schema stores representative_evidence inside each theme; the interface resolves IDs within that theme and displays exact stored text, source, sentiment, parent identifier and separately labeled English gloss. Support denominators and sentiment units stay source-distinguished. Unavailable/blank excerpts cannot be selected as citations. Scope/limited-support wording and explicit singleton warnings remain visible.

Saves add claim_reviews (seven field-keyed state/note records), review_schema_version, UTC reviewed_at and review_history. PENDING/ACCEPT/REVISE/REJECT claim states record the researcher's judgment; changed text/citations require REVISE and an explanatory note. Theme decisions remain PENDING/APPROVE/REJECT. Approval requires all seven claims ACCEPT or REVISE, reviewer identity, reason and the unchanged structural insight validator. A rejected theme remains in the package and blocks the existing all-approved release contract. Revision history stores prior complete review records without recursively nesting history; no historical candidate or researcher revision is overwritten without retaining its prior state. Reviewer identity is self-reported, not authenticated.

The save transaction uses an exclusive local lock, validates fresh disk state, rejects stale snapshot tokens, writes and fsyncs a same-directory temporary JSON file and atomically replaces review.json. Read/load and save verify frozen source hashes, evidence lineage, candidate hashes, requests, retry source provenance and the existing mechanical correction audit. Failed pre-replacement saves retain the prior file. This protects cooperating app sessions; manual external edits do not participate in the lock, and a crash may leave a lock requiring operator inspection. The app rejects editing finalized or partially finalized packages. The organizer loader still requires a finalized manifest and never falls back to pending candidates.

Validation means structurally grounded candidate output. It does not mean the claim has been substantively approved by a researcher. An AI candidate is distinct from a researcher revision in the working insight; a researcher-approved synthesis is distinct from a finalized/released synthesis. The existing finalizer retains backward compatibility with legacy review files and checks theme approvals/identity/reason/hash/grounding, not the newly added claim fields. The app enforces the new claim completeness rule when saving approval; direct manual review-file changes require separate audit.

Launch from the repository root using `.\.venv\Scripts\python.exe -m streamlit run participant_experience_review_app.py --server.address localhost --browser.gatherUsageStats false`. Detailed researcher instructions, schema/register, provenance, limitations and exact release command are in `PARTICIPANT_EXPERIENCE_REVIEW_WORKFLOW.md`. Save before changing navigation; unsaved edits are not durable. Full integrity validation on each interaction may take several seconds. Structurally invalid drafts cannot be saved: unresolved issues can be recorded as notes with PENDING/REJECT while retaining valid working content. The researcher alone assesses substantive support and interpretation.

After all 99 themes have been reviewed, resolve any PENDING/REJECT outcomes; only when all 99 are substantively APPROVE should the researcher separately run `.\.venv\Scripts\python.exe -m marathon_absa.participant_experience finalize --reviewer "Your researcher name"`. That command was not run during this implementation. New workflow regression tests are in `tests/test_participant_experience_review.py`; write tests use temporary copies. The initial 17-test run passed in 133.01 seconds. Expanded focused and complete suite results are recorded below after execution.

Complete-suite investigation: the first run exposed a stale setup fixture in `tests/test_participant_experience_retry.py`. It copied all current production files into a temporary pre-retry package, including the candidates.json and review.json that now exist after completed generation; the recovery guard correctly refuses that state. The fixture now copies only filenames recorded in the original retry manifest's source_hashes. This restores the historical stage being tested without changing recovery code, weakening its guard or touching production artifacts. The full-run failure and corrected rerun results are retained in the test logs and completion report.

Expanded focused validation: all 23 researcher-review tests passed in 212.41 seconds. These include all four claim states, theme approval/rejection round trips, revision history, original-candidate preservation, stale sessions, locks, tampered candidates/review hashes/evidence/mechanical audits, invalid citations, incomplete approvals, missing identity/notes, atomic replacement failure, finalized-package write blocking, all 693 claim-to-evidence resolutions, cross-source distinction, singleton/provenance UI display, offline Streamlit navigation/save and production/frozen hash preservation. The corrected historical recovery fixture passed all 19 tests in 220.72 seconds. First full run: 464 passed, 19 fixture setup errors, four existing warnings, 632.87 seconds; all errors had the documented pre-retry-stage fixture cause. A complete corrected rerun follows. A local Streamlit launch on localhost:8510 succeeded; browser automation could not attach (in-app webview timeout; Chrome unavailable), so no screenshot-based visual QA is claimed. The actual rendered UI interactions were exercised with Streamlit AppTest.

Final corrected complete-suite result: **483 passed, 4 existing scikit-learn warnings in 870.48 seconds** (`work/participant_review_full_tests_after_fix.txt`). The warnings remain the single-label confusion-matrix and invalid scalar-division warnings in two relevance fixtures. Post-suite verification at 2026-09-19T05:55:19.981565+00:00 confirmed all 30 original production/retry artifacts byte-identical, all 411 frozen evidence-input hashes unchanged, and all 390 baseline hashes and integrity checks passing. All 99 candidates still validate and all 693 candidate claims resolve. Production review.json remains byte-identical: **99 PENDING, 0 APPROVE, 0 REJECT, 0 finalized**, with empty reviewer/reason fields and no finalized artifacts. ZERO substantive researcher decisions, production revisions, API generation calls or finalization actions were made. The smoke-test server was stopped. Final audit: `work/participant_review_verification.json`; full 20-point completion report and file register: `PARTICIPANT_EXPERIENCE_REVIEW_COMPLETION_REPORT.md`. Status: **IMPLEMENTED_AND_TESTED_AWAITING_HUMAN_RESEARCHER_REVIEW**.

### rt08 targeted research-integrity investigation (2026-09-19)

After the authorized AI-assisted substantive review left `training_preparation_pacing__rt08` (Fitting training around work and daily life) as the sole REJECT, a read-only lineage investigation found that the current five-excerpt participant-experience package is unrepresentative, rather than that the theme itself lacks support. The frozen rt08 membership is one Instagram ALTA stable cluster, `training_preparation_pacing__theme_10`, with 46 mentions / 45 documents and no blog support. Four existing, fixed assignment records explicitly concern work/lifestyle scheduling: `absa1_96e46ec812ce3f13` (work disrupted a plan and juggling work/training was difficult), `absa1_f0699a167d078931` (working-lifestyle adjustment affected training in the preserved parent caption), `absa1_1e8a1799b763707b` (a plan around a hectic work schedule), and `absa1_2f3d510797e8b845` (new working lifestyle affected training). Three were already in the frozen ALTA representative-evidence file. The later reviewed-taxonomy finalizer deterministically took the first five lexicographically sorted document IDs from the full assignment group, excluding all work/lifestyle records; participant-experience construction correctly copied that later selection. The result is a downstream representative-evidence selection failure, not proof of a cross-theme transfer, altered membership, unsupported theme label, or broad upstream taxonomy failure.

The Ironman 70.3 Langkawi excerpt is legitimately a KLSCM2019-tagged corpus record whose source text also says the author was days before KLSCM and uncertain about the full marathon; it is contextual other-event material, not demonstrated contamination. It does not support the work/daily-life concept. The report `PARTICIPANT_EXPERIENCE_RT08_INVESTIGATION.md` records the complete trace, evidence, exact selection mechanism, limitations and correction options. Its diagnostic decision is **CURRENT THEME DEFENSIBLY SUPPORTED**, but this does not authorize changing the current REJECT: a correction would need a separately versioned reviewed/evidence package, regeneration and revalidation, then substantive re-review. No frozen artifact, candidates.json or review.json was modified by the investigation. Pre/post integrity verification is recorded in the report.

### rt08 versioned participant-experience correction prepared and reviewed (2026-09-19)

The investigation-supported correction was implemented only as the new downstream package `data/processed/participant_experience_v2_rt08/`; no v1 or upstream reviewed-taxonomy artifact was edited. `marathon_absa/participant_experience_rt08_v2.py` replaces the rt08 representative evidence with the four fixed assignment-lineage records identified in the investigation, regenerates rt08's request, candidate and review row, and copies the other 98 rows byte-for-byte at the candidate/review/request level. `correction_manifest.json` records v1 and v2 SHA-256 values, the preserved/regenerated row counts and zero external API calls. The rt08 candidate uses cautious wording, cites only direct source records for work/lifestyle assertions, and is reviewed as APPROVE by `AI-assisted researcher review (ChatGPT)`; all seven rt08 claims have explicit claim reviews.

Executed verification returned 99 themes, 99 APPROVE, zero REJECT and zero PENDING. The focused regression suite `tests/test_participant_experience_rt08_v2.py` passed 2 tests, including preservation of all 98 unaffected candidate/review rows and corrected-lineage-only citations. `verify_integrity()` also passed all 390 frozen-baseline files and checks. The versioned finalizer exists but was deliberately not run: no `finalized.json` or `finalized_manifest.json` is present. Status: **VERSIONED_PACKAGE_READY_FOR_SEPARATE_FINALIZATION_DECISION**.

### Participant Experience v2 final release verification (2026-09-19)

After separate finalization, a read-only verification of `data/processed/participant_experience_v2_rt08/` confirmed 99 unique finalized insights, 99 APPROVE decisions, zero REJECT/PENDING, and rt08 APPROVE. The final manifest hashes matched its complete release register; every finalized insight matched its approved review row, all 693 claim fields were present, and project validation confirmed each citation belonged to available evidence in its correct theme. The request artifact remains SHA-256 `c80972e196199571ff04f6b3c64857b37886e92e449ada4663c41cb1c3ce1553`. V1 evidence and candidates retain their recorded hashes, and the existing `verify_integrity()` verifier passed all 390 frozen-baseline files and checks. The release fingerprint, artifact register, test results and provenance are recorded in `PARTICIPANT_EXPERIENCE_FINAL_VERIFICATION.md`. No research artifact, API call or model call occurred during verification. Status: **FINAL RELEASE VERIFIED**.

### Finalized Participant Experience dashboard integration (2026-09-19)

The existing Streamlit dashboard now reads only `data/processed/participant_experience_v2_rt08/finalized.json` through the strict read-only loader in `marathon_absa/participant_experience_dashboard_data.py`. It verifies finalized status/version, manifest hashes, complete 99-theme coverage, approved review-to-finalized-insight equality, claim/evidence grounding and the versioned rt08 lineage before rendering. It never falls back to candidates or drafts. `marathon_absa/participant_experience_page.py` provides the organizer-facing Participant Experience page with aspect, category, source-coverage and keyword filters; paginated compact expandable theme cards; on-demand privacy-redacted supporting evidence; separate Instagram/long-form source framing; and the finalized organizer observations/implications. The Executive Overview includes a deterministic route to this page. This UI integration does not alter research artifacts, rerun research stages, calculate findings, or call an API/model. It retains the provenance statement `AI-assisted substantive review; not independent human validation.` Focused finalized-release/dashboard tests passed 6/6 and existing dashboard UI tests passed 20/20.

### AI-assisted substantive review executed (2026-09-19)

ChatGPT completed an evidence-first substantive review on the researcher's behalf. All 99 themes and 693 claims were inspected against exact frozen excerpts, source-specific units, unavailable-evidence restrictions, candidate hashes and provenance. The saved `review.json` contains 98 APPROVE themes and one REJECT theme, with claim states 568 ACCEPT, 120 REVISE and 5 REJECT; no PENDING claims remain. The reviewer identity is `AI-assisted researcher review (ChatGPT)`, and every theme retains review history.

`training_preparation_pacing__rt08` (Fitting training around work and daily life) was rejected because its available excerpts discuss generic training effort and an Ironman 70.3 Langkawi reference, not work or daily-life scheduling. This blocks the existing all-99-APPROVE finalization gate. No finalization command was run. Frozen evidence, candidates, requests, manifests and upstream research artifacts remain unchanged. Detailed counts and validation results are in `PARTICIPANT_EXPERIENCE_AI_REVIEW_COMPLETION_REPORT.md`.

AI-assisted substantive review; not independent human validation.
