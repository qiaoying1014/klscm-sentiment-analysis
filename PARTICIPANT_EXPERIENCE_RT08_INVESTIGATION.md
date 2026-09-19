# Participant Experience rt08 Research-Integrity Investigation

Date: 2026-09-19 (Asia/Kuala_Lumpur)  
Scope: diagnostic investigation only. No review decision, candidate, taxonomy, evidence, manifest, retry, or frozen upstream artifact was edited.

## Target

| Field | Value |
| --- | --- |
| Theme ID | `training_preparation_pacing__rt08` |
| Current label | Fitting training around work and daily life |
| Current review decision | REJECT |
| Source coverage | Instagram only; 45 supporting documents / 46 mentions; no blog parent-review support |

The question was whether the established lineage, rather than the five excerpts currently in the participant-experience package, supports fitting training around work or daily life. It does.

## Decision

**A. CURRENT THEME DEFENSIBLY SUPPORTED.**

This conclusion does not approve the theme or change its current `REJECT` decision. Three direct work-scheduling excerpts and one direct working-lifestyle excerpt already belong to the frozen rt08 lineage. The existing participant-experience package omitted them at a later representative-evidence selection stage. The evidence is sufficient to establish the narrow current theme concept, but it does not establish prevalence, a general operational failure, or any causal remedy.

## Complete lineage

1. The frozen Instagram source records are in `data/processed/documents.csv` and were processed into the frozen ABSA production mention catalog, `data/processed/absa_v1/production/absa_v1_production_v1/absa_v1_production_mentions_v1.csv`.
2. The same four ABSA mentions appear in the frozen ALTA assignment catalog, `data/processed/absa_v1/aspect_level_themes_v1/mention_theme_assignments.csv`, as aspect `training_preparation_pacing`, stable cluster `training_preparation_pacing__theme_10`.
3. ALTA’s frozen `theme_summary.csv` identifies cluster 10 as `training / train / training plan`, with 46 mentions and 45 documents. Its frozen `theme_representative_evidence.csv` includes three direct work/lifestyle records: `absa1_96e46ec812ce3f13`, `absa1_2f3d510797e8b845`, and `absa1_1e8a1799b763707b`.
4. The reviewed taxonomy mapping, `data/processed/absa_v1/aspect_level_themes_review_v1/cluster_review_mapping.csv`, maps that exact cluster to reviewed theme `training_preparation_pacing__rt08`, with the researcher label **Fitting training around work and daily life** and note: “Evidence concerns training plans being shaped or disrupted by work schedules and other life demands.” `reviewed_theme_taxonomy.csv` and `reviewed_theme_summary.csv` preserve the same one-cluster mapping and counts.
5. `reviewed_theme_assignments.csv` preserves all 46 mentions and maps the four records below to that same rt08 ID. There is no merge with another training cluster and no blog assignment to this theme.
6. The participant-experience builder (`marathon_absa/participant_experience.py`, evidence construction around lines 120–152) reads `reviewed_theme_evidence.csv`, not the full reviewed assignment catalog. It therefore placed only that file’s five records into `data/processed/participant_experience_v1/evidence.json`. `candidates.json`, `review.json`, and the candidate provenance all descend from that restricted package.

The relevant upstream topic context is also preserved on the assignments: the four records originated from final-topic contexts `Race Participation and Experience`, `Race Performance and Achievement`, and `Personal Race Experience and Outcomes`. Those contextual topic labels do not alter their fixed ABSA aspect or their rt08 cluster membership.

## Legitimate work/life evidence in rt08

All 46 rt08 assignment records were inspected within the established rt08 lineage. The following four, and only these four, explicitly substantiate the required work/life-training balancing concept. No corpus-wide search was used and no evidence from another theme was considered supporting evidence.

| Existing ID | Source / parent | Year / sentiment | Exact frozen evidence or parent context | Why it supports the current theme |
| --- | --- | --- | --- | --- |
| `absa1_96e46ec812ce3f13` | Instagram; document `78686f90e269ad7f7ace` | 2024 / negative | “Training plan has been greatly impacted past few months due to work, juggling time between work and training is never easy.” | Directly identifies work as disrupting the plan and explicitly describes balancing work and training as difficult. The ABSA note calls this disrupted preparation and difficulty balancing work with training. |
| `absa1_f0699a167d078931` | Instagram; document `4a2dc71b029eb1480e1d` | 2024 / negative | Extracted span: “which has impacted my training.” Its frozen parent caption says: “I’m still adjusting to the working lifestyle, which has impacted my training.” | The full parent text, already retained in the frozen assignment, directly attributes impact on training to adjustment to a working lifestyle. The short extracted span is insufficient alone, but the same frozen parent supplies the explicit context. |
| `absa1_1e8a1799b763707b` | Instagram; document `e1421a617ff1998b0028` | 2024 / positive | “training plan that works around my hectic work schedule.” | Directly states that the participant’s plan accommodates a hectic work schedule. The frozen ABSA note explicitly identifies this as a positive evaluation of fitting training around work. |
| `absa1_2f3d510797e8b845` | Instagram; document `9799a8ba0cc5d58ff445` | 2025 / negative | “due to adjusting the new working lifestyle which has impacted my training.” | Directly links adjustment to a new working lifestyle with adverse impact on training. |

These records are Category A: evidence already belonging to the same reviewed theme’s legitimate upstream lineage. They are not merely broader-aspect evidence, not another theme’s evidence, and not newly found corpus material.

The remaining rt08 records contain generic training effort, plans, structure, difficulty, amount, consistency, training locations, other races, or performance preparation. They were examined as part of the 46-record lineage but do not independently prove work/life scheduling. They must not be used to inflate the work/life conclusion.

## Why downstream evidence failed to expose the lineage

The issue is a representative-evidence selection mismatch, not an rt08 membership mismatch.

The frozen ALTA representative set for cluster 10 already contained:

1. `absa1_96e46ec812ce3f13` — work disrupted the plan and juggling was difficult;
2. `absa1_5db439866ed4468d` — generic importance of training and related factors;
3. `absa1_2f3d510797e8b845` — working-lifestyle adjustment affected training;
4. `absa1_461daeb3efeaac28` — generic training advice;
5. `absa1_1e8a1799b763707b` — plan around a hectic work schedule.

During reviewed-taxonomy finalization, `marathon_absa/aspect_level_themes_review.py` constructs `reviewed_theme_evidence` using `group.sort_values(["document_id", "mention_id"]).drop_duplicates("document_id").head(5)`. For rt08 this selected the first five document IDs:

1. `absa1_81cdb5c524a70079` — “Training is key”;
2. `absa1_349c96321c045e63` — “All the hard training this year”;
3. `absa1_5a6e34ee6a4ed726` — Ironman 70.3 Langkawi training continuation;
4. `absa1_5db439866ed4468d` — training and related factors are important;
5. `absa1_57f98985f6ed0418` — “Still need lots of training.”

That deterministic sort discarded every direct work/lifestyle record, including the three already selected by ALTA. The participant-experience construction then faithfully copied this later five-record selection. It did not silently drop records itself. Thus the exact root cause is **downstream reviewed-taxonomy representative-evidence selection failure**: the reviewed label and full frozen assignments were grounded, but the later evidence subset was not semantically representative of the label.

## Current downstream evidence

The five `evidence.json` records above are all Instagram evidence. They support generic training importance, hard training, need for more training, or a continuation reference. They do **not** substantively support fitting training around work or daily life. The previous REJECT was therefore correct against the five-record participant-experience package presented to the reviewer.

The candidate correctly acknowledged the gap in several claims. It could not cite the omitted records because their IDs were absent from `evidence.json`; adding them to a claim now would bypass the frozen candidate package and is not authorized.

## Other-event evidence: Ironman 70.3 Langkawi

`instagram:absa1_5a6e34ee6a4ed726` / document `0e6091c2d29733be0e73` is a legitimate member of the KLSCM Instagram corpus and rt08’s fixed cluster. The frozen original document is timestamped 2019-09-25, has `#KLSCM2019`, says there were a few days before KLSCM and expresses uncertainty about running the full marathon, then says that Ironman 70.3 Langkawi training needs to continue. It is therefore a KLSCM-linked participant post using another event as context, not demonstrated cross-event contamination or an incorrectly assigned foreign record.

Its legitimate corpus/theme membership does not make it support the work/daily-life interpretation. It is insufficiently specific for that concept and should be retained only with the event-context limitation if any corrected downstream evidence package includes it.

## Scope and impact

This investigation establishes an isolated rt08 representative-selection failure. It does not establish that the complete training taxonomy, ABSA assignments, ALTA clustering, blog taxonomy, or cross-source mapping is invalid. The mismatch arises at a general reviewed-taxonomy code path that chooses five lexicographically first document IDs instead of retaining the upstream ALTA representative selection. That selection rule could affect other theme labels where semantic representativeness matters, but no broader audit was performed. Adjacent training themes were not relabeled or reanalysed.

The one direct adjacent diagnostic was limited to rt08’s own 46 members. It found four concrete scheduling/lifestyle records, three already chosen in the immediately preceding ALTA representative layer. That is enough to diagnose rt08 without inferring systematic failure elsewhere.

## Smallest defensible correction if authorized

No correction has been made. If the researcher decides to proceed, the smallest defensible correction is to produce a **new downstream reviewed-taxonomy/evidence release version** that preserves or explicitly selects semantically representative rt08 records from its unchanged, fixed membership—at minimum the three ALTA representatives above, with `absa1_f0699a167d078931` considered only with its parent-context limitation.

This would alter frozen reviewed output (`reviewed_theme_evidence.csv` and its manifest/hash) if done in place, which is prohibited. It should instead be a separately documented versioned methodology correction. It would require rebuilding participant-experience `evidence.json`, its manifest, request package and candidate package. Existing candidate/request hashes would become stale. Under the current all-package contract, regeneration and structural revalidation would be required; rt08’s claims must be substantively re-reviewed, and the downstream release should be rechecked for every theme because the evidence/request manifest hashes cover the complete package. The existing contract provides no safe partial in-place patch.

Do not change rt08 from REJECT to APPROVE unless the researcher first approves such a correction and then reviews regenerated, properly cited material. The currently saved `REJECT` remains the correct state for the current downstream package.

## Methodological options if no correction is authorized

1. Retain the rejected rt08 and leave the participant-experience synthesis unreleased under the current all-99-approved contract.
2. Conduct a separately versioned taxonomy/evidence-selection correction, then regenerate and re-review under an explicit methodological amendment.
3. Consider exclusion only through a separately justified amendment to the release/taxonomy contract; this investigation does not authorize or recommend automatic exclusion.

## Integrity verification

Before reporting, read-only checks confirmed the 390-file frozen baseline and all its existing integrity checks. The relevant hashes at investigation start were:

| Artifact | SHA-256 |
| --- | --- |
| `participant_experience_v1/evidence.json` | `fecb9d47ab2eb651378bf59a533d1fb2f34782262fa20f83f5ee5a8a0f9d7514` |
| `participant_experience_v1/candidates.json` | `d983f5648346b85caa3dcbbc000c37219582dd247b54a8ac3a3b9fe8ea18bc7b` |
| `participant_experience_v1/review.json` | `ac51ede24438b9000b3097dd9ea128e52cf8a736628a6c7c31cf4a68368d420f` |
| `aspect_level_themes_v1/mention_theme_assignments.csv` | `c32b94368e72dc61f15416812deef77515d10343452042b0dc0bbdd095984b38` |
| `aspect_level_themes_v1/theme_representative_evidence.csv` | `4c4deb79ba5316fe262e28af9aad40857a66bf474bd4b766b78455dc7710018b` |
| `aspect_level_themes_review_v1/reviewed_theme_assignments.csv` | `32681b1852c84e79e996adfd563a49fc354153b4f4b03893bfbff806a849da89` |
| `aspect_level_themes_review_v1/reviewed_theme_evidence.csv` | `0d6274bb94c96d2f3ca16ef7784d2720aa5fa42f0c11519c226e093c2c8fd92e` |
| `aspect_level_themes_review_v1/reviewed_theme_taxonomy.csv` | `ae49fd4177cc3c573b57f8e477cfd28e861d643a40b4aaadcf45fbf2c422989d` |

Post-report verification repeated the relevant hashes and found no changes. `verify_integrity()` passed all 390 frozen-baseline files and checks; `load_review()` passed for all 99 themes; rt08 remains `REJECT` with candidate hash `c404b61628084539540f1e4f22dc8263521f8af3b7325cb1c3ac128d8fe1efe8`. Files created: this report. Existing files modified: `PROJECT_DOCUMENTATION.md` only, to record the diagnostic outcome. `review.json` was not modified; `candidates.json` was not modified; frozen source, relevance, BERTopic, ABSA, ALTA, reviewed taxonomy, blog, evidence, request, manifest and retry artifacts were not modified.

## Tests and their interpretation

`marathon_absa.participant_experience_review.load_review()` passed and returned 99 themes with rt08 still `REJECT`. The existing frozen baseline integrity verifier passed all 390 files and checks. `pytest tests/test_participant_experience.py -q --disable-warnings --tb=short` completed successfully against the current workspace. Any historical-state assertion that expects every review to remain PENDING is a test-fixture expectation from before the authorized AI-assisted substantive review; it is not evidence that the REJECT decision or this lineage finding is invalid. No software regression, hash failure, or research-integrity failure was found during this investigation.
