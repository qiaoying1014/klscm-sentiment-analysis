# Participant Experience invalid-response recovery audit

As of 2026-09-19. Offline preparation only. No API call, approval, finalization, or upstream analytical rerun was performed.

## Findings

All 99 original local request payloads match their finalized evidence packages after the original documented removal of internal parent/source identifiers. Original request/evidence hashes and submission/upload file IDs reconcile. The unchanged strict validator accepts 89 responses and rejects 10.

| Theme to retry | Root cause |
| --- | --- |
| `emotional_experience__rt01` | Scope note cites a blank-excerpt ID supplied in unavailable_evidence_ids; not an allowed citation. |
| `emotional_experience__rt05` | Scope note cites a blank-excerpt ID supplied in unavailable_evidence_ids; not an allowed citation. |
| `physical_experience__rt04` | Negative summary repeats instagram:absa1_0c68f2dd88342bcc twice. This is model output duplication, not parser insertion. No deduplication is performed; retry the theme. |
| `race_performance__rt02` | Organizer insight truncates instagram:absa1_cb0baa2d8fe2f56d by omitting its final d. The real excerpt expresses satisfaction with performance; comparison establishes the likely copying failure, not an authorized replacement. |
| `race_performance__rt11` | Scope note cites a blank-excerpt ID supplied in unavailable_evidence_ids. |
| `race_performance__rt12` | Scope note cites two blank-excerpt IDs supplied in unavailable_evidence_ids. |
| `volunteer_support__rt01` | Scope note cites a blank-excerpt ID supplied in unavailable_evidence_ids. |
| `volunteer_support__rt02` | Mixed summary has [] but appends a sentence after the exact insufficient-evidence sentinel. Both organizer fields already contain valid evidence IDs. The generic validator error misleadingly names organizer claims. |
| `volunteer_support__rt03` | Participant summary, implication and scope note cite the same blank-excerpt ID supplied in unavailable_evidence_ids. |
| `blog_emergent__transport_access__306a1f5b` | Organizer insight truncates blog:absa1blog_e5ce7c3c7be31dcf by omitting its final f. The real excerpt supports the short wait for a complimentary ride; no replacement was made. |

Seven distinct rejected IDs across six themes are real IDs in the SAME theme package, explicitly listed as unavailable because their frozen excerpts are blank. They do not occur in any available representative_evidence list elsewhere. This is an additional case beyond the proposed A/B/C categories: A-like provenance-only evidence exclusion, not cross-theme borrowing or hallucination. Two other distinct IDs are B, final-character truncations of available same-theme evidence. There is no demonstrated C hallucination or evidence-serialization corruption.

The original request construction exposed unavailable identifiers without a dedicated allowed-ID list. That prompt-contract weakness plausibly encouraged scope-note citations of unavailable records (D-related prompt ambiguity, not corrupted evidence). The parser preserves citation arrays exactly. The generic validator diagnostic is misleading for the pacer failure but its rejection is correct. Neither validator nor original prompt, parser, request, or response artifacts was changed. Retry instructions resolve the ambiguity without loosening validation.

Full claim-level diagnostics, exact unknown IDs, allowed IDs, original hashes, and truncation comparison excerpts are in `work/participant_experience_recovery_audit_20260919.json`. String comparison was used only during this audit; runtime recovery does not search for or replace similar IDs.

## Recovery and provenance

Implementation: `marathon_absa/participant_experience_retry.py`. Tests: `tests/test_participant_experience_retry.py`. Original evidence, request input strings, model, response schema, inference settings, and all original responses remain unchanged. Only retry instructions append the stricter citation contract and a theme-specific allowed-ID list.

The prepared `data/processed/participant_experience_v1/retry_v1/` contains 10 requests, `retry_manifest.json`, and `preserved_valid_responses.jsonl` with the original 89 valid JSONL lines including original line-ending bytes. It contains no submitted batch or regenerated responses. The original responses.jsonl includes all 99 original records unchanged.

Preparation and verification independently rederive the invalid-only selection. Before submission and merge, checks verify the original files, all 411 frozen input hashes, per-response hashes, preserved insight hashes, exact preserved response bytes, retry request bytes, and implementation hashes. Submission has an exclusive started marker and refuses existing review artifacts. The retry cannot include an originally valid theme. No semantic correction or mechanical deduplication is implemented.

Collection records the completed batch ID, input/output file IDs, downloaded response hash, and retry-manifest hash. Merge requires that lineage, the exact ten-theme response set, no duplicate themes, all responses completed, and validation of every candidate with the unchanged validator. Only then are exactly 99 candidates written and review.json initialized entirely PENDING. Provenance records original/retry version, response ID, batch request ID, response-line hash, insight hash, source-file hash and retry manifest/collection hashes. No automatic review approval or finalization occurs. An invalid retry stops before either candidate or review artifact is created.

An unsupported direction summary must use the exact sentinel and []. Other fields retain the strict citation requirement. If no supported limited observation can be made, an explicit uncited insufficient-evidence output remains invalid for researcher attention; no fallback is silently accepted.

The original submission.json is a submit-time validating snapshot. No completed original batch snapshot, collection manifest, or independently downloaded remote input payload was saved. Local request hashes and the submission/upload link reconcile, but this audit does not claim independent remote attestation. No API was called to reconstruct missing historical records. New retry collection records close this provenance gap prospectively.

## Preservation and integrity

- Original response file SHA-256: `c818cc9d8ca9921e236c1b916b430df03276d9aeb47a4aead65c57385c99be54`.
- Original evidence SHA-256: `fecb9d47ab2eb651378bf59a533d1fb2f34782262fa20f83f5ee5a8a0f9d7514`.
- Original requests SHA-256: `8d011dba6f5ea2c7248da264132397709e9d610467e5de90229ae9297701c346`.
- Preserved 89-line response file SHA-256: `928e2e0d597dc1c7bd678642c54d61455614a6320e77938ab0a6eea34a160a9e`.
- Prepared retry request SHA-256: `6d6584b227581f3318b1ea24859bd16ddcf76f74683b9098a396fd0661ddaa7b`.
- All 411 evidence-input hashes pass; all 390 frozen baseline files and the existing upstream integrity checks pass.

## Execution

From `C:\Users\user\Documents\Sentiment Analysis\1. Data Scraping`, AFTER explicit user approval:

```powershell
.\.venv\Scripts\python.exe -m marathon_absa.participant_experience_retry submit --run-api
```

After completion, authorized collection uses `participant_experience_retry collect --run-api`; offline revalidation/merge after a downloaded recorded collection uses `participant_experience_retry merge`. Read-only preflight is `participant_experience_retry verify`. A second prepare on the same directory or a repeated submission is rejected. Do not rerun the original 99-request submit command.

## Validation results

Recorded below after test completion. Production retry has NOT been executed.

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
