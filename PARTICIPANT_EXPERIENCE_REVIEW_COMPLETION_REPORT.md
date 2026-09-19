# Researcher review workflow completion report

Completed 2026-09-19. Final integrity verification: 2026-09-19T05:55:19.981565+00:00. Implementation and testing are complete; substantive researcher review has not begun.

1. **Repository/schema audit.** `review.json` is a list of theme records with theme_id, decision, reviewer, reason, candidate_hash and insight. `candidates.json` contains an insights list and generation/retry provenance. Each insight has seven named text/evidence_ids claims. `evidence.json` owns representative_evidence within each theme, with source and parent lineage; it is not a flat global citation table. The existing finalizer requires complete coverage and APPROVE, reviewer, reason, matching candidate hash and valid insight for every theme. Coverage: 17 Instagram-only, 47 cross-source and 35 blog-only themes; 37 themes have singleton blog support.

2. **Exact production state before.** 99 themes, 99 validated candidates, 99 PENDING review entries, 0 APPROVE, 0 REJECT, 0 finalized themes; no finalized artifact. `work/participant_review_before.json` records the original production and frozen hashes.

3. **Files created.** `participant_experience_review_app.py`; `marathon_absa/participant_experience_review.py`; `marathon_absa/participant_experience_review_ui.py`; `tests/test_participant_experience_review.py`; `PARTICIPANT_EXPERIENCE_REVIEW_WORKFLOW.md`; this report. Audit/test outputs: `work/participant_review_before.json`, `work/participant_review_collection.txt`, `work/participant_review_focused_tests.txt`, `work/participant_review_full_tests.txt`, `work/participant_review_full_tests_after_fix.txt`, `work/participant_review_retry_fixture_tests.txt`, `work/participant_review_verification.json`. Normal interpreter/test caches are not research outputs.

4. **Existing files modified.** `PROJECT_DOCUMENTATION.md` received the methodology, implementation, schema, interface, limitations, commands and validation record. `tests/test_participant_experience_retry.py` received a fixture-only correction: recreate the original pre-retry stage using the original manifest's filenames, instead of copying later candidate/review outputs. No existing production pipeline, finalizer, dashboard or pinned generation/retry implementation was modified.

5. **UI architecture.** A standalone local Streamlit entry point calls a presentation module; a separate offline module loads/verifies evidence and handles saves. The app provides decision/aspect filters, label/ID search, a selector covering all 99 themes, support/scope context, seven two-column claim/evidence sections, reviewer identity, explicit save, provenance and revision history. It imports no generation, inference or finalization entry point.

6. **Claim-to-evidence resolution.** Each claim's evidence_ids is joined against its own theme's representative_evidence. Exact stored excerpts appear beside working text, with Instagram/blog source, sentiment, evidence ID and parent ID. Glosses are separately labeled. Removed original citations remain inspectable, and all theme evidence is available for selecting revised citations. All 693 candidate claims were tested for resolution. Unknown, cross-theme, duplicate or otherwise invalid citations cannot pass saving/approval.

7. **Original candidate protection.** Original text and citations are displayed read-only from candidates.json. Researcher saves never write that file. Each review retains its original candidate_hash; loading and saving validate that hash and accepted generation provenance. Stale/tampered candidates block the workflow.

8. **Researcher revision storage.** Working/approved content stays in the existing review record's insight. Additive claim_reviews records each claim's PENDING/ACCEPT/REVISE/REJECT state and note. Changed claims require REVISE and a note. Saves add UTC reviewed_at, review_schema_version and review_history snapshots retaining prior text, citations, identities and decisions. Opening the app does not migrate or write production records.

9. **Generation/retry/mechanical provenance.** The expander shows request model/prompt/version/hashes, generation timestamp, candidate hash, accepted generation version, response identifiers and file/line hashes, raw/derived insight hashes and exact mechanical before/after citation arrays. Actual accepted versions are 89 original, six retry_v1, three retry_v2 and one retry_v2_exact_dedup. Existing mechanical correction remains generation provenance, never a researcher revision. All original artifacts remain byte-identical.

10. **Reviewer identity.** The researcher enters a name or stable identifier. Every save, including a draft, requires this identity and a theme reason. Revision/rejection claim notes are required. Saved identity/time/history are inspectable. Identity is self-reported rather than authenticated.

11. **Atomic save and stale-state protection.** An exclusive local lock serializes cooperating writers. Saving reloads/validates artifacts and compares an optimistic snapshot token. JSON is written to a same-directory temporary file, flushed and fsynced; hashes are checked again before os.replace atomically replaces review.json. Failed pre-replacement saves leave the original intact. Tests cover stale tokens, existing locks, tampering and simulated replacement failure. A process crash can leave a lock requiring operator inspection; external editors do not honor the lock.

12. **Separate finalization.** APPROVE is a researcher decision, not a release. Every claim must be ACCEPT or REVISE before app approval. REJECT retains the theme and blocks release. The app cannot finalize, and refuses writes once either finalization artifact exists. The unchanged organizer loader never falls back to pending candidates. After all 99 themes are reviewed, resolve PENDING/REJECT outcomes; only after all 99 are substantively approved should the researcher separately run `python -m marathon_absa.participant_experience finalize --reviewer "Your researcher name"` with the repository's virtual-environment Python. This command was not run. The legacy finalizer validates theme-level review but does not enforce the new claim fields on manually edited legacy files.

13. **Focused results.** All 23 new workflow tests passed in 212.41 seconds. The corrected recovery fixture's 19 tests passed in 220.72 seconds. Write tests use temporary copies only. Streamlit AppTest exercised navigation, saving, singleton warnings and provenance with network/client construction blocked. Static tests exclude generation/inference imports. A local Streamlit launch succeeded. Screenshot QA could not run because the in-app webview attachment timed out and Chrome was unavailable; no visual screenshot verification is claimed.

14. **Complete suite results.** Final corrected run: **483 passed, 4 warnings, 870.48 seconds**. The four warnings are existing scikit-learn single-label confusion-matrix/invalid-division warnings in two relevance fixtures. The first full run had 464 passes and 19 setup errors, all caused by the old recovery fixture copying already-created candidate/review files into a pre-retry test. Both logs are retained. The production recovery guard was correct and was not weakened.

15. **Frozen integrity verification.** Post-suite verification confirmed **all 30 existing production/retry artifact files byte-identical**, **all 411 frozen evidence-input hashes unchanged**, and **all 390 baseline hashes and associated integrity checks passing**, including the frozen prompt hash. No changed artifacts or frozen inputs were found. Machine-readable evidence: `work/participant_review_verification.json`.

16. **Exact production state after.** 99 themes, 99 validated candidates, 693 resolvable candidate claims, 99 PENDING reviews, 0 APPROVE, 0 REJECT and 0 finalized themes. Reviewer/reason fields remain empty. Neither finalized.json nor finalized_manifest.json exists. Production review.json is byte-identical to its pre-implementation state.

17. **Exact launch command.** Run the following in PowerShell. The temporary smoke-test server has been stopped.

```powershell
Set-Location 'C:\Users\user\Documents\Sentiment Analysis\1. Data Scraping'
.\.venv\Scripts\python.exe -m streamlit run participant_experience_review_app.py --server.address localhost --browser.gatherUsageStats false
```

18. **Methodological/UX limitations.** Structural validity cannot establish substantive support or research approval. Evidence is the frozen privacy-redacted excerpt, not the full raw source. Sources/denominators are never pooled. Every interaction rechecks integrity and may take several seconds. Save before navigating; unsaved edits are not durable. Structurally invalid drafts cannot be saved; use notes with PENDING/REJECT while retaining structurally valid working content. There is no automatic advancement, authenticated identity, signed audit log or multi-user database. Manual file edits can bypass UI rules. The existing all-approved release policy requires separate methodological action if a theme remains rejected.

19. **Zero researcher decisions.** I made **ZERO substantive researcher review decisions**: no production approval, rejection or claim revision, no finalization, no new API generation and no model inference from the review app. Synthetic judgments used only for temporary test fixtures are not research decisions.

20. **All production themes remain PENDING.** Explicitly verified after the complete suite: **99 of 99 PENDING**. The next substantive step belongs to the human researcher using this interface.

Validation means structurally grounded candidate output. It does not mean the claim has been substantively approved by a researcher.

The workflow distinguishes AI candidate, researcher revision, researcher-approved synthesis and finalized/released synthesis. Detailed operating instructions and the exact separate finalization command are in `PARTICIPANT_EXPERIENCE_REVIEW_WORKFLOW.md`.
