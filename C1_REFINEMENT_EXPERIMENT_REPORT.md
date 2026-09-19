# c1 versus c1-refined Controlled Experiment

## Decision status

The final choice is deliberately unresolved: keep original c1, adopt c1-refined, or conclude that neither is adequate. c1-refined does not replace c1 automatically. ABSA remains blocked.

## Trigger and design

All 40 original c1 topics were reviewed at topic level. Although 27 topics covering 7,112 normal-topic documents were judged substantive, review found systematic c-TF-IDF and clustering artifacts: multiword emoji aliases, English and Malay/Indonesian function words, repeated hashtag blocks, spaced `K L S C M 2 0 1 9` formatting, style-driven clusters, and fragmented photography themes. These observations motivated one predefined text-representation experiment, not optimization against the human topic labels.

Original c1 is frozen by `c1_frozen_baseline_manifest_v1.json`, including SHA-256 hashes of its model, assignments, completed review, embeddings, IDs, metrics, and configuration. The experiment retained c1's UMAP and HDBSCAN parameters exactly.

## Deterministic preprocessing

Raw captions are unchanged. `refined_embedding_text` applies minimal semantic cleaning: URL/mention suppression; conservative `K L S C M [year]` collapse; emoji demojization followed by removal of installed emoji-library aliases having at least two components, including collapsed spellings and skin-tone variants; stable case-insensitive hashtag deduplication; clear CamelCase/letter-number splitting; and a maximum of 12 unique hashtag tokens with domain-important hashtags placed first. Single-word emoji aliases are retained to avoid blindly deleting ordinary words.

`refined_representation_text` starts from that embedding text and additionally removes sklearn's established English stopword set and a documented 68-item conservative Malay/Indonesian function-word list. Chinese tokens are not removed. Domain terms including KLSCM, marathon, race, runner/running, finish, training, pacer, clinic, route, water, and distance tokens are protected from hashtag truncation and absent from the custom stopword list.

Embedding input and representation input are intentionally different. Clustering needs minimally cleaned multilingual semantics; c-TF-IDF needs less stylistic/function-word noise. Because embedding text changed for 12,742/13,743 documents, the original mathematical vectors were not reused. A separate offline multilingual-MPNet cache was generated. The existing c1 embedding cache was untouched.

Preprocessing diagnostics found multiword emoji aliases in 8,225 original semantic texts, normalized 26 spaced-KLSCM cases, deduplicated hashtag repetitions in 566 documents, and capped more than 12 unique hashtags in 1,742 documents.

## Objective results

| Metric | Original c1 | c1-refined |
|---|---:|---:|
| Normal topics | 40 | 44 |
| Normal-topic documents | 8,040 (58.50%) | 7,232 (52.62%) |
| Outliers | 5,703 (41.50%) | 6,511 (47.38%) |
| Median topic size | 55.5 | 61 |
| Largest topic | 1,970 | 2,076 |
| Smallest topic | 31 | 30 |
| Topics below 30 | 0 | 0 |
| Giant topic over half the corpus | No | No |

There was no giant-topic collapse. The refined result has four more topics and 808 more outliers. After optimal one-to-one label mapping, 5,951 documents (43.30%) change assignment. ARI is 0.2801 and AMI is 0.4084, indicating material structural change rather than preservation under renumbering. These measures do not establish quality.

The crosswalk shows that 40/44 refined topics have one original c1 source contributing at least 50%, but only 12/44 reach 80%; median dominant contribution is 67.23%. Twelve refined topics are dominated by documents that were c1 outliers. This suggests both survival of recognizable clusters and substantial reallocation/recovery from the original outlier pool. The human-label-enriched crosswalk must be used to inspect whether those changes are substantively helpful.

## Required human decision

The compact refined review package contains 44 normal topics, each with five BERTopic representatives, five fixed-seed random captions, metadata, and original-c1 contribution crosswalk. The five largest refined topics (2,076; 844; 497; 356; 278) have ten representatives and twenty random captions. Topic -1 has a reproducible diagnostic sample only. Original c1 labels are not copied into refined topics.

Human topic-level comparison—not record annotation—is now required before choosing among original c1, c1-refined, or neither. The refined model's higher outlier rate is a cost; cleaner terms or better semantic coherence, if observed, must be weighed against it.
