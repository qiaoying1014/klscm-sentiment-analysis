# Aspect-Level Thematic Analysis (ALTA) v1

## Purpose and status

ALTA is an additive downstream, exploratory/descriptive stage created on 2026-09-01. It induces aspect-level semantic thematic clusters from the existing frozen ABSA mentions. It asks what recurring experiences, issues, meanings, motivations and perceptions occur *within* each frozen aspect. It does not rerun or validate BERTopic or ABSA, change any frozen field, or conduct inferential testing.

Corpus BERTopic describes broad discourse across posts; ABSA identifies a model-estimated evaluated marathon element and sentiment; ALTA groups the already separated ABSA mentions into recurring discussion themes within that element. BERTopic fields are retained only as provenance.

## Frozen inputs and audit

The exact inputs are `data/processed/absa_v1/production/absa_v1_production_v1/absa_v1_production_mentions_v1.csv`, `absa_v1_production_document_results_v1.csv`, `data/processed/topic_discovery_v1/final_taxonomy_v1/final_substantive_topic_corpus_v1.csv`, and `data/processed/absa_v1/absa_aspect_ontology_v1.json`. Execution fails rather than repairs if they do not reconcile to 7,704 documents, 15,486 unique mention rows, 5,316 mention-bearing documents, 2,388 zero-mention documents, 20 exact aspects, and years 2019/2023/2024/2025.

The mention input contains `mention_id`, `document_id`, `mention_index`, frozen `aspect`, `target`, frozen `sentiment`, confidence, exact `evidence_text`, expression type, mention language, `english_gloss`, meaningful hashtag/emoji contribution fields, evidence offsets and grounding/repair audit fields, source/year/language, frozen BERTopic provenance, prompt/model and schema identity. The document result supplies population reconciliation and provenance. The final substantive corpus supplies `original_caption_text`, hashtags, multilingual preprocessing fields, and topic/taxonomy provenance.

## Method

The analytical unit is one frozen mention. `theme_semantic_text` joins whitespace-normalized, nonempty `target`, exact `evidence_text`, and `english_gloss` in that order with ` | `; exact case-insensitive duplicate components are removed. No lexical content, hashtags or emoji inside these fields are removed. Original evidence and gloss remain separate, and the gloss is not treated as quoted evidence.

Embeddings use the already cached local `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, normalized to 768 dimensions in batches of 32 with `local_files_only=True`. The separate ALTA cache is keyed by the ordered frozen mention IDs. No download, OpenAI call, paid API or network inference is permitted.

Each aspect is processed independently. An aspect requires at least 60 mentions and 40 unique documents; otherwise every row is retained as `insufficient_support`. Sufficient aspects use UMAP (cosine, five dimensions, 15 neighbors or the available maximum, minimum distance zero, random state 42), followed by HDBSCAN (Euclidean, EOM). Minimum cluster size is `ceil(sqrt(n_mentions))`, bounded to 10–50, and minimum samples is one third of that value with a floor of three. HDBSCAN `-1` remains noise. This single prespecified policy is not tuned to obtain attractive results.

Labels are provisional and deterministic: the most frequent nonempty target plus the strongest within-cluster TF-IDF unigrams/bigrams after a fixed, documented English/Malay and event-domain representation stoplist. This stoplist affects representation only, never embeddings or source/semantic text. Representative evidence is copied verbatim from frozen evidence, uses distinct documents, and suppresses exact normalized duplicates. Themes with fewer than 15 supporting documents are flagged `low_support_theme`; they are retained but excluded from interview candidates.

Prevalence is based primarily on unique documents. Multiple sentiments for the same document-theme become conservative document-level `mixed`; otherwise the sole sentiment is retained. Outputs report share within aspect documents and all 7,704 analyzed documents, year-specific unique-document prevalence, and document-level sentiment composition. Counts describe this social-media corpus, not a probability sample or the participant population.

## Outputs and interpretation

All outputs are isolated under `data/processed/absa_v1/aspect_level_themes_v1`, in Parquet plus UTF-8-SIG CSV: `mention_theme_assignments`, `theme_summary`, `theme_year_summary`, `theme_sentiment_summary`, `theme_representative_evidence`, `theme_review`, and `interview_theme_candidates`. Embeddings and their ordered ID cache are local `.npy`/JSON artifacts. `theme_manifest.json` records exact source schemas/hashes/counts, configuration, thresholds, outputs and execution-integrity statements.

`theme_review` allows a researcher to edit only a small final taxonomy without changing mention assignments. Interview candidates are prompts for later interpretation, not final propositions. Machine summaries use conservative descriptive language and are not validated research conclusions.

Aspect-level themes are derived from model-estimated ABSA mentions. The thematic analysis does not remove uncertainty in upstream aspect classification (development aspect precision about 0.513 and recall about 0.790). Clusters are automatically induced semantic groupings, not objective psychological constructs or validated qualitative themes; clustering does not validate ABSA labels, low support is not absence of importance, and no inter-rater reliability is claimed.

Run with `python -m marathon_absa.cli absa-v1-aspect-level-themes`. Tests are in `tests/test_aspect_level_themes.py`; the full repository suite remains the compatibility gate. Dashboard integration is intentionally deferred until the package is validated, avoiding changes to frozen dashboard marts and views.
