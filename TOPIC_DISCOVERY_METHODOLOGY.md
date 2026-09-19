# KLSCM Multilingual Topic Discovery Methodology

## Status and scope

As of 2026-08-17, relevance-classifier development has ended. The implemented `topic_discovery_corpus_v1` stage prepares and profiles a local multilingual topic-modelling corpus. Corpus preparation has run; full sentence embedding and BERTopic candidate fitting have not run. ABSA remains blocked until a baseline is selected, topics are reviewed, and substantive-topic/refinement decisions are frozen. No OpenAI, Luna, Terra, translation, or other paid API is used by this stage.

## Why record-level relevance annotation was discontinued

The finalized cost-conservative population contains 13,799 records: 10,532 automatic includes, 338 deterministic hashtag-only exclusions, and 2,929 unresolved reviews. The completed architecture-aligned audit found only 61/75 automatic includes researcher-confirmed (81.33%) and 38/50 deterministic exclusions confirmed (76%); 12 deterministic exclusions were reversed. Thus neither automated arm is ground truth. Resolving thousands of uncertain records is also disproportionate for a single-researcher exploratory study.

Relevance is therefore retained as non-destructive diagnostic metadata. It neither admits nor excludes a record from topic discovery. The immediate gate is instead whether text supplies usable semantic information. This transition does not alter or overwrite any relevance decision, prompt, route, threshold, audit annotation, or historical artifact.

## Corpus construction and eligibility

`python -m marathon_absa.cli topic-discovery-prepare` joins the frozen production decisions to prepared document metadata by stable `document_id`. It emits one corpus row for every finalized record and separates fitting-eligible rows from data-quality exclusions. Transparent exclusions cover empty, URL-only, mention-only, emoji-only, punctuation-only, effectively empty, unusable/corrupt, and exact duplicate semantic text. There is no minimum word threshold: meaningful short text such as `PB 21KM` remains eligible.

URLs and mentions are removed from embedding text. Emoji are converted to textual aliases. Hashtags are preserved in their original field and converted into semantic tokens: separators, clear lower-to-upper CamelCase boundaries, and letter-number boundaries are split, while ambiguous lowercase strings remain intact. Consequently `#HalfMarathon` becomes `Half Marathon`, `#KLSCM2024` becomes `KLSCM 2024`, and an ambiguous `#klscm` remains `klscm`.

Exact duplicate semantic strings are represented once during fitting. Each duplicate row remains in the full corpus with `duplicate_representative_id`, source, year, language, and relevance metadata. No semantic near-duplicate removal is performed.

## Preflight evidence

The 2026-08-17 run read the finalized decision Parquet with SHA-256 `5811bdaea235f4a358803bec17dd8c8a055cae27193aafa1feb377d724759c89`. Of 13,799 rows, 13,743 are fitting-eligible and 56 are exact duplicate semantic texts. No other data-quality reason occurred because previously “insufficient” or hashtag-only captions still contained usable event/social semantic tokens after conservative normalization.

Eligible semantic-text character lengths are: minimum 10, Q1 122, median 257, Q3 500, P90 876.8, P95 1,205.9, P99 1,937, maximum 3,996, mean 385.68. The corpus is caption-scale rather than a mixed Instagram/blog population (the finalized relevance population contains Instagram rows). Version 1 therefore does not chunk. Before a later blog-inclusive run, documents beyond a predeclared distribution-based threshold should be inspected and, only if necessary, grouped into semantic chunks retaining `parent_document_id` and `chunk_id`; chunks must never inflate original-post counts.

## Multilingual representation

The proposed embedding model is `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768 dimensions), run locally on CPU with cached NumPy embeddings and an ordered record-ID sidecar. It supports multilingual and code-switched inputs without translation or destructive language filtering. Its exact downloaded revision should be added to the run manifest when first downloaded.

For c-TF-IDF, `multilingual_tokenizer_v1` extracts Unicode/Latin tokens and overlapping Chinese characters and bigrams, avoiding the false assumption that Chinese is whitespace-separated. This deterministic local strategy is intentionally simpler than linguistic segmentation and is a limitation. No stopwords are removed initially. English, Malay, Chinese, and domain-term frequencies must be inspected after raw topics exist; any later domain stopword list must be versioned and must not silently remove KLSCM/SCKLM, Kuala Lumpur, sponsor, marathon, or distance terms.

## BERTopic baseline and predefined comparison

The local architecture is multilingual sentence embeddings → seeded UMAP → HDBSCAN → BERTopic/c-TF-IDF. Fixed parameters are UMAP `n_components=5`, `min_dist=0.0`, cosine metric, seed 42; HDBSCAN Euclidean metric, EOM selection and prediction data; vectorizer 1–2 grams, `min_df=2`, maximum 50,000 features and the multilingual tokenizer; BERTopic 20 top words, no probabilities, no automatic topic reduction, and no LLM representation.

Only four candidates are predefined: `(n_neighbors, min_cluster_size, min_samples)` = `(15,30,10)`, `(30,30,10)`, `(15,50,15)`, and `(30,50,15)`. The run writes topic count, outlier count/proportion, median/largest/smallest topic size, assignments and topic information for each candidate. Selection must also inspect representative terms/documents, diversity, size distribution, stability where practical, multilingual behavior, and human interpretability; it must not rely on one coherence score.

BERTopic topic `-1` is retained and reported as an outlier assignment, not deleted or equated with irrelevance. No raw topics are automatically merged. Later merges/refinements require an explicit reproducible stage after topic-level review.

## Topic-level review

The next human annotation is at topic level through `topic_review_app.py`. Each selected-baseline topic-review row should contain size/share, 15–20 terms, representative captions/documents, source/year/language distributions, production relevance-route distribution, and available Luna confidence summaries. Permitted labels are `substantive_klsm_topic`, `generic_running`, `commercial_or_promotion`, `other_event`, `noise_or_uninterpretable`, `mixed_topic`, and `needs_topic_refinement`, plus a human topic name and notes. Topics are interpreted by meaning; a high REVIEW share is diagnostic only.

## Limitations

The study has a single researcher; residual irrelevant records and social-media sampling bias remain. Results are sensitive to embedding and clustering parameters, multilingual and Chinese tokenization is imperfect, the embedding model can flatten language/cultural nuance, HDBSCAN outliers can still be substantive, and topic naming/refinement is subjective. The failed relevance audit is evidence motivating the non-destructive design, not a result to conceal.

## Commands and artifacts

Preparation: `python -m marathon_absa.cli topic-discovery-prepare`.

First real local model run (do not execute until preflight approval): `python -m marathon_absa.cli topic-discovery-run`.

Preparation writes `data/processed/topic_discovery_v1/topic_discovery_corpus_v1.csv`, `topic_discovery_excluded_data_quality_v1.csv`, `topic_discovery_corpus_manifest_v1.json`, and `bertopic_configuration_v1.json`. The model command creates reusable embeddings, four raw candidate models, assignments/topic info, and `bertopic_candidate_metrics_v1.csv`. Selected-baseline naming, enriched review tables, and visualizations follow explicit human selection; they are not falsely claimed as executed now.

## Provisional c1 selection and topic-level interpretation (2026-08-17)

The first local comparison completed for c1–c4. c3 and c4 were rejected for severe under-clustering: each produced four topics and placed more than 13,000 records in one cluster. c2 remains comparison evidence with 42 topics and 7,135 outliers (51.92%). c1 is the provisional—not final—baseline because it yields 40 topics with a lower 5,703 outliers (41.50%). This pragmatic choice does not establish objective superiority; interpretability must be inspected before freezing any model. No new search, refit, parameter/tokenizer/stopword change, merge, reduction, or split is performed at this stage. See `TOPIC_DISCOVERY_MODEL_SELECTION_V1.md`.

`topic-discovery-review-prepare` reads the existing c1 model and assignments without refitting. Topics 0–39 receive 20 c-TF-IDF terms, five BERTopic c-TF-IDF representative captions, five random captions sampled with fixed seed 104729, and language/year/relevance/source distributions. The five largest topics receive ten representatives and twenty separate fixed-seed random captions to expose possible under-separation. Topic -1 is a separate outlier/unassigned diagnostic with 30 fixed-seed random captions and is excluded from normal review counts.

The Streamlit interface records one required discourse interpretation and human topic name per normal topic, plus optional notes and a coherence assessment. It autosaves and resumes without evidence spans or record-level decisions. A substantive cluster may contain noisy posts; this is thematic interpretation, not renewed relevance validation. Existing annotations are preserved whenever diagnostics are regenerated.

Review package: `python -m marathon_absa.cli topic-discovery-review-prepare`.

Interface: `streamlit run topic_review_app.py`.

Non-paid summary after review: `python -m marathon_absa.cli topic-discovery-review-summary`. It reports reviewed topics and assignment-weighted document counts/percentages for each category, while reporting -1 separately. ABSA remains blocked until interpretation and any explicit refinement decision are frozen.

## One controlled c1-refined experiment (2026-08-18)

The completed 40-topic review found 27 substantive topics covering 7,112/8,040 normal-topic documents, but also systematic emoji-alias, stopword, repeated-hashtag, formatting, style-cluster, and theme-fragmentation problems. This evidence triggered one controlled preprocessing experiment. It was not an open-ended parameter search and did not optimize against desired human labels. Original c1 and its review are frozen by artifact hashes.

Raw captions remain unchanged. Separate `refined_embedding_text` and `refined_representation_text` fields distinguish minimal semantic cleaning for multilingual embeddings from stronger function/style-token removal for c-TF-IDF. Multi-component aliases are derived deterministically from the installed `emoji` package and matched in underscored, collapsed, hyphen/underscore skin-tone forms; single-word aliases remain. Spaced KLSCM lettering is collapsed conservatively. Repeated hashtags are case-insensitively deduplicated, clear segmentation remains, and lists over 12 unique hashtags retain domain-important tokens first before deterministic truncation. Representation text additionally uses sklearn English stopwords and a conservative 68-item Malay/Indonesian list; Chinese and domain terms are preserved.

Because refined embedding text changed for 12,742 records, a separate offline embedding cache was mathematically necessary. The original embeddings were not overwritten. UMAP (`15`, `5`, cosine, seed 42, min-dist 0) and HDBSCAN (`30`, `10`, Euclidean, EOM) were unchanged.

c1-refined produced 44 topics, 6,511 outliers (47.38%), 7,232 normal assignments, median size 61, largest 2,076 and smallest 30. There was no giant-topic collapse and no topic below the fixed cluster minimum. Compared with c1, ARI is 0.2801, AMI 0.4084, and optimal label mapping leaves 5,951 assignment changes (43.30%). These are comparison diagnostics, not evidence of superiority. A blank 44-topic review package and c1→refined crosswalk were generated. Final choice remains original c1, c1-refined, or neither; ABSA is blocked. Full details are in `C1_REFINEMENT_EXPERIMENT_REPORT.md`.

## Compact model-selection review (2026-08-18)

A full second taxonomy review was judged unnecessary because original c1 already has a completed 40-topic interpretation and the refined crosswalk identifies where decision-critical change occurred. `topic-discovery-model-selection-prepare` deterministically selects refined topics meeting any review-risk condition: top ten by size; dominant contribution from c1 outliers; at least 25% and ten documents from a c1 mixed/highly-mixed topic; or a directly detected residual artifact pattern in refined top terms, including repeated-token, photography, multiword emoji-alias or stopword dominance. The same substantial-contribution threshold selects refined topics inheriting generic-running/incidental-KLSCM material. This operational definition keeps the package compact while directly targeting observed failure modes.

The resulting v1 package contains 28/44 refined topics covering 6,063 refined clustered documents. Overlapping reasons include ten largest topics, 12 outlier-dominated topics, two with substantial mixed/highly-mixed inheritance, ten direct artifact-pattern topics and four generic-running inheritance topics. Each row exposes the refined terms/examples, complete contributing-c1 topic names/relevance/quality, and c1-outlier share. It requests only interpretability change, semantic relation, refined quality and an optional note. No refined taxonomy label or record decision is requested.

The final non-paid summary is generated by `topic-discovery-model-selection-summary`. Until all 28 rows are complete, recommendation remains pending. The predeclared conservative rule recommends c1-refined only if improved topics cover at least 60% of selected-topic documents, improved exceeds worse by at least 30 percentage points, unrelated mixing affects at most 10% of selected topics, and at least 40% of topics show useful merging, splitting or outlier recovery. Otherwise it recommends frozen c1, explicitly weighing the refined model's 808 additional outliers, lower coverage and extra fragmentation. ABSA remains blocked.

## Final c1 selection and taxonomy consolidation (2026-08-18)

All 28 compact comparisons were completed: 15 improved topics cover 83.37% of selected-topic documents, seven similar cover 9.90%, and six worse cover 6.73%. Relations were seven preserved, one related merge, two useful splits, six unrelated mixes, 12 useful outlier recoveries and zero unclear. Although improvement was substantial, unrelated mixing affected 21.43% of reviewed topics and refined clustered coverage remained 52.62% versus c1's 58.50%. The predeclared conservative rule therefore selected frozen c1 finally; c1-refined remains evaluated evidence and is not substituted.

`topic-taxonomy-prepare` creates an additive selection manifest and a populated 40-topic worksheet from the completed c1 review. All 780 topic pairs are ranked without new modeling using existing embedding-centroid cosine, top-term Jaccard and human-name TF-IDF similarity. Topics 11/13 carry an explicit prior-observation flag as overlapping photography themes. Suggestions never mutate assignments or prefill researcher decisions.

The worksheet supports Level-1 `final_topic_group` and Level-2 topic names plus keep, merge, contextual-retain or substantive-exclude actions. Generic, commercial, other-event and mixed records remain traceable. Topic -1 remains 5,703 preserved `unassigned_outlier` documents, excluded provisionally from thematic prevalence/ABSA but never equated with irrelevance. `topic-taxonomy-finalize` validates complete actions, targets, cycles, stable original assignments and all-ID reconciliation before writing final taxonomy/mapping/corpus partitions and `ABSA_READY=true`; it never runs ABSA itself. Detailed rationale is in `FINAL_TOPIC_MODEL_SELECTION_AND_TAXONOMY.md`.
