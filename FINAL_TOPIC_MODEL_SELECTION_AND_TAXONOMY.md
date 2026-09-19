# Final Topic Model Selection and Taxonomy Consolidation

## Final model decision

Original c1 is the final selected topic-discovery model. The decision is recorded additively in `data/processed/topic_discovery_v1/final_selection/final_topic_model_selection_v1.json`; existing c1 model, assignments, embeddings and human annotations remain unchanged and hash-verified. c1-refined is `evaluated_not_selected`.

The initial comparison produced c1 with 40 topics and 41.50% outliers, c2 with 42 topics and 51.92% outliers, and severe four-topic under-clustering in c3/c4. Full c1 interpretation found 27 substantive topics covering 7,112 clustered documents, four generic-running, two commercial/promotional, one other-event and six mixed topics. Quality was 24 coherent, ten somewhat mixed and six highly mixed.

Human-observed representation artifacts motivated one controlled c1-refined experiment with unchanged clustering parameters. It improved selected representations but reduced clustered coverage from 58.50% to 52.62% and added 808 outliers. The compact comparison reviewed 28 decision-critical refined topics: 15 improved, seven similar and six worse; semantic relations included seven preserved, one related merge, two useful splits, six unrelated mixes and 12 useful outlier recoveries. Because unrelated mixing occurred in 6/28 topics, exceeding the predeclared 10% ceiling, and coverage declined, the conservative final rule retained c1. The decision does not deny improvements in c1-refined; it weighs them against corpus loss and semantic mixing.

## Consolidation procedure

The final worksheet contains all 40 c1 topics with their completed names, relevance/quality assessments, terms and representative/random captions. Researcher fields are limited to `keep`, `merge`, `exclude_from_substantive_taxonomy`, or `retain_as_contextual_topic`, an optional revised Level-2 name, a Level-1 group, merge target and note. The two-level structure is researcher-derived; no example taxonomy is imposed automatically.

Merge candidates are suggestions only. All 780 pairs are ranked using existing multilingual embedding-centroid cosine (55%), top-term Jaccard (20%), and human-name TF-IDF cosine (25%). Existing review evidence adds an explicit flag for topics 11/13, the race-photography overlap. No embedding or model is rerun. The finalizer validates complete actions, valid targets, absence of cycles, assignment identity, and complete document partitioning.

## Non-substantive and mixed topics

Generic-running, commercial/promotional, other-event and mixed assignments are preserved. Researchers may retain contextual/sponsor content separately or exclude it from substantive sentiment analysis without deleting records. Six mixed topics receive existing examples, relevance distributions and suggested related substantive topics; no record-level splitting or relabelling is required.

## Outliers and limitations

Topic -1 contains 5,703/13,743 documents (41.50%). These are preserved as `unassigned_outlier`, are not considered irrelevant, and are excluded from thematic prevalence and ABSA unless a later separately justified reassignment method is approved. No outlier reduction is performed here. This large unassigned share, parameter sensitivity, multilingual representation limits, social-media sampling bias and single-researcher interpretation remain material limitations.

## ABSA gate

ABSA remains blocked during worksheet preparation. Only successful taxonomy finalization—complete actions, resolved merges, frozen substantive/contextual treatment and a reconciled topic-to-document mapping—writes `ABSA_READY=true`. The finalizer does not run ABSA.
