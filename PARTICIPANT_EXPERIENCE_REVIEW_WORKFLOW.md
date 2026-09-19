# Participant experience researcher review

Implemented 2026-09-19. This is a local human research workflow for the 99 existing SESA / KLSCM themes. It makes no API calls, performs no model inference and cannot finalize a release.

> Validation means structurally grounded candidate output. It does not mean the claim has been substantively approved by a researcher.

## Launch

From PowerShell:

```powershell
Set-Location 'C:\Users\user\Documents\Sentiment Analysis\1. Data Scraping'
.\.venv\Scripts\python.exe -m streamlit run participant_experience_review_app.py --server.address localhost --browser.gatherUsageStats false
```

Keep the app local. No additional dependencies are needed beyond the existing environment.

## Review procedure

1. Enter your reviewer identity (your name or stable researcher identifier). Filter by decision or aspect, or search by label/ID. Every theme is available through the theme selector.
2. Read the theme scope and separate Instagram document / blog parent-review support. Singleton warnings are explicit. Sentiment counts retain their source-specific units: Instagram theme documents versus blog assigned mentions.
3. Review each of the seven claims. The original AI candidate is read-only. Exact cited evidence appears beside the working text with source, sentiment, evidence ID and parent ID. Existing English glosses are labeled separately. Evidence is the exact stored privacy-redacted excerpt, not a newly generated translation or a full raw source document.
4. Choose PENDING, ACCEPT, REVISE or REJECT for each claim. Working text and citations are editable; citation selection is restricted to that theme's available evidence. Edited claims must be marked REVISE. REVISE and REJECT require notes. Removed original citations remain inspectable. All theme evidence is also available in an expander.
5. Record the theme decision (PENDING, APPROVE or REJECT) and a reason. Save explicitly. Every save requires reviewer identity and a reason; a PENDING save is a researcher draft. APPROVE requires all seven claims to be ACCEPT or REVISE and the complete edited insight to pass existing structural grounding validation. REJECT keeps the theme and blocks release.
6. Save before navigating. Unsaved edits are not durable, and changing theme or filters can discard them. A successful save displays confirmation. Saved identity, time and history can be inspected below the save button.

The review unit is one existing theme, with seven separately recorded claim judgments. Accepting one claim never automatically approves the theme. The interface makes no substantive judgments, recommendations to accept, or automatic claim revisions.

## Existing schema and additive review fields

The audited `evidence.json` is an object containing `themes`, `source_units`, `aspects`, frozen `input_hashes` and methodology metadata. Each theme owns `representative_evidence`, not a separate global evidence-ID table. Evidence records contain evidence_id, source, source_evidence_id, parent_id, sentiment, text and english_gloss. Blank/unavailable evidence IDs remain separate and are not valid citation choices.

`candidates.json` is an object with an `insights` list and generation/retry metadata. Each insight contains theme_id and seven named claims, each with text and evidence_ids. `review.json` is a list, initially containing theme_id, decision, reviewer, reason, candidate_hash and insight. All 99 entries were PENDING, with 0 approved and no finalized artifacts before implementation.

The workflow retains the existing finalization contract: `insight` is the current researcher working/approved content; `candidate_hash` always identifies the immutable original insight in candidates.json. Saves add `claim_reviews` (field -> state and note), `reviewed_at` (UTC), `review_schema_version: 1`, and `review_history`. Every history entry records the save timestamp and the entire prior review entry excluding its nested history, preserving original and intermediate researcher text, citations, decisions and identities. Existing unknown fields are preserved. Old review entries require no migration or write when opened.

- **AI candidate:** immutable generated text and citations in candidates.json; structural validity only.
- **Researcher revision:** manually edited text/citations in review.json insight, with REVISE and a note; may remain PENDING.
- **Researcher-approved synthesis:** a named researcher's APPROVE decision with all seven claim judgments complete; still unreleased.
- **Finalized/released synthesis:** separately created finalized.json and finalized_manifest.json after the existing release checks pass.

## Provenance and save safety

The provenance expander shows request model/prompt/version/hashes, generation timestamp, candidate hash, and the theme's accepted original/retry version, response IDs, response file/line hashes, raw/derived insight hashes and exact mechanical-correction before/after citation arrays. Mechanical deduplication is pre-existing generation provenance, never described as a researcher revision. It is neither rerun nor altered by this app.

Loading verifies all frozen evidence inputs and original source lineage, request/evidence hashes, candidate/review coverage and candidate hashes, retry source hashes, accepted insight hashes and the mechanical audit hash. Claim citations resolve exclusively inside the selected theme, preserving Instagram/blog distinctions. Invalid evidence references fail validation before save, including approval. Candidate, evidence or provenance changes invalidate saving.

Saving takes an exclusive local lock, reloads and validates disk state, compares an optimistic snapshot token, serializes to a temporary file in the same directory, flushes and fsyncs it, checks artifact hashes again and calls os.replace for atomic replacement of review.json. Other review entries remain intact. A failure before replacement leaves the original file untouched; temporary files and the lock are cleaned up in normal error handling. Stale sessions fail instead of overwriting another saved review. A crashed process may leave `.researcher_review.lock`: first establish that no review save process is running, then remove only that lock. Do not manually edit artifacts concurrently; external editors do not honor this lock. This is local cooperative concurrency, not a transactional multi-user database or tamper-proof identity service.

## Finalization and exact next step

The review UI has no finalize action and refuses writes to finalized or partially finalized packages. The organizer page continues to require a verified finalized manifest; it does not expose candidates or researcher drafts.

After all 99 themes have been reviewed, first resolve any PENDING or REJECT decisions. The current release contract requires **all 99 APPROVE**; rejection is a research finding that blocks release, not permission to drop a theme. Any change to this release policy is a separate methodology decision. When all 99 are substantively approved, inspect the saved decisions/history and run the existing separate command with the actual finalizer identity:

```powershell
.\.venv\Scripts\python.exe -m marathon_absa.participant_experience finalize --reviewer "Your researcher name"
```

This command was NOT executed during implementation. The existing finalizer validates theme-level approval, named reviewer, reason, candidate hash and insight grounding; it does not enforce the new claim-review fields for legacy files. New claim completeness is enforced by the review app's save boundary. Direct manual file editing can bypass UI rules and must be separately audited.

## Limitations

Structural checks cannot establish substantive accuracy, sufficient support, causality, representativeness or methodological approval. The researcher remains responsible for those judgments. No pooled denominator or importance ranking is introduced. The existing validator requires cited organizer observations/implications, the cautious implication prefix, the prescribed uncited insufficient-evidence directional wording and singleton scope wording. Drafts that violate those structural rules cannot be saved; record an unresolved issue using a claim note and PENDING/REJECT while retaining structurally valid working content. Identity is self-reported and not authenticated. Local disk hash validation on every interaction favors integrity over responsiveness. Navigation uses a searchable theme selector rather than automatic advancement. Revision history is in the review artifact and is not a cryptographically signed audit log.

## Implementation register

- `participant_experience_review_app.py`: standalone Streamlit entry point.
- `marathon_absa/participant_experience_review_ui.py`: claims/evidence layout, navigation, provenance and explicit save controls.
- `marathon_absa/participant_experience_review.py`: read/verify, exact evidence resolution and atomic review transaction.
- `tests/test_participant_experience_review.py`: isolated persistence, safety and Streamlit regressions.
- `work/participant_review_before.json`: pre-implementation production/frozen hash snapshot.
- `work/participant_review_verification.json`: post-test verification (produced at completion).

Test results and final production state are recorded in PROJECT_DOCUMENTATION.md and the completion report. No production research decisions are made by the implementation tests.
