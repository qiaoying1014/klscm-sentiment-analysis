# KLSCM Multilingual Topic Discovery and ABSA

Research pipeline and Streamlit dashboard combining KLSCM Instagram captions and online reviews. It preserves hashtags and emoji as analysis features while excluding them from language detection, uses pinned OpenLID-v3 plus span-level mixed-language analysis, filters irrelevant captions, chunks reviews, discovers BERTopic themes, and performs OpenAI ABSA.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Keep `OPENAI_API_KEY` in `.env`. Paid API stages never run without `--run-api`.

## Workflow

```powershell
# One-time 1.2 GB pinned OpenLID-v3 download
python -m marathon_absa.cli download-language-model

# Local ingestion, cleaning, language detection, and chunking
python -m marathon_absa.cli prepare

# Paid: adjudicate only uncertain, disagreeing, or mixed-language candidates
python -m marathon_absa.cli language-review --run-api --source blog
# Remove --source blog later to adjudicate all pending records

# Paid relevance pilot: stratified across year, language, status, and length
python -m marathon_absa.cli relevance --run-api --pilot-size 500

# Create a blind 300-record validation file plus 30 repeat records
python -m marathon_absa.cli relevance-review-sample --size 300 --repeats 30
# Complete human_relevance in data/processed/relevance_validation_sample.csv
python -m marathon_absa.cli relevance-validate

# After calibration, classify the complete Instagram corpus
python -m marathon_absa.cli relevance --run-api
# Complete human_relevance in data/processed/relevance_review_queue.csv
python -m marathon_absa.cli relevance-finalize

# Topic discovery is blocked until relevance coverage is complete and reviews are resolved
python -m marathon_absa.cli topics --run-api
python -m marathon_absa.cli absa --run-api

streamlit run app.py
```

Use `--limit N` for pilots. Outputs are under `data/processed`; API responses are cached under `data/cache`.
## Relevance filtering

Relevance covers the full KLSCM event journey: registration, event-specific preparation, logistics,
participant information, participation, experience, results, achievement, support, and evaluation.
Generic running, unrelated events, pure promotion, lifestyle/spam, and hashtag-only connections are
not admitted to topic discovery. Initial borderline decisions receive a stronger-model pass; unresolved
records enter `relevance_review_queue.csv`. Records are never deleted. The final `include_in_topics`
field is the only Instagram inclusion gate used by topic discovery and ABSA.

Validation outputs include holdout precision/recall/F1, a confusion matrix, subgroup metrics, and
intra-reviewer agreement. Required targets are relevant recall >= 0.90, relevant precision >= 0.85,
and macro F1 >= 0.80.

## Language semantics

- `no_text`: normalized source text is empty.
- `insufficient_text`: fewer than eight linguistic characters after removing metadata noise.
- `undetermined`: usable text exists but cannot be classified reliably.
- `mixed`: at least two languages have substantial spans.
- `review_required`/`low_confidence`: queued for optional OpenAI adjudication.
- `language_method` states whether OpenLID, masked span analysis, Lingua fallback, or OpenAI produced the final result.

Empty and duplicate records remain in data-quality counts but are excluded from language percentages, topics, relevance, and ABSA.



## Participant Experience & Organizer Insights

The additive organizer page combines finalized Instagram and blog themes interpretively, with separate source units and traceable evidence. Offline generation requires explicit API approval; the dashboard shows insights only after review and freezing. See [workflow, schemas and commands](PARTICIPANT_EXPERIENCE_SYNTHESIS.md).

## Streamlit Community Cloud deployment

Deploy repository `qiaoying1014/klscm-sentiment-analysis`, branch `main`, with
**main file path `deploy/streamlit_app.py`**. Python 3.11 is supported, and the
requirements also support the Python 3.14 runtime used by the current Cloud app.
This opens the finalized research dashboard. `app.py` is the older preparation
dashboard and needs local intermediate data.

The Cloud entry point uses the SHA-256-verified snapshot in `deploy/dashboard_data/`.
It needs no OpenAI key, downloaded model, or API calls. The adjacent
`deploy/requirements.txt` installs only dashboard dependencies; root `packages.txt`
provides a Linux CJK font. Streamlit selects dependency files next to the entry point
before root-level files (see the [official dependency documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)).

The multiple-requirements warning is expected: Cloud uses the file beside the
entry point, while root `requirements.txt` remains the local research environment.
NumPy is pinned to 2.3.5 on Python 3.11+ for prebuilt Python 3.14 wheels, and to
2.2.6 on the existing local Python 3.10 environment. After a dependency update,
Cloud automatically reinstalls dependencies. If the old import error remains after
the build finishes, use **Manage app → Reboot app**. Check the build log for the
selected Python version and successful installation of Plotly, not just resolution.

To preview the exact Cloud entry point locally:

```powershell
.\.venv\Scripts\python.exe -m streamlit run deploy/streamlit_app.py
```

To refresh the release after changing finalized research outputs, run from a full
local research checkout, with `KLSCM_DASHBOARD_BUNDLE` unset:

```powershell
.\.venv\Scripts\python.exe scripts/export_dashboard_bundle.py
.\.venv\Scripts\python.exe -m pytest tests/test_cloud_deployment.py -q
git add deploy/dashboard_data
git commit -m "Refresh validated dashboard release"
git push origin main
```

Export first runs all five original data loaders and their source/provenance checks.
Cloud verifies exported artifact hashes and relevant schema contracts; it does not
repeat the complete upstream research audit. The snapshot contains the caption and
evidence text required by the existing dashboard, plus reviewed insights and lineage.
Keep original inputs, models, caches and intermediate outputs in local/backed-up
research storage: Git ignores these directories. A fresh Git clone runs the frozen
Cloud dashboard but cannot rerun the full research pipeline without those inputs.
