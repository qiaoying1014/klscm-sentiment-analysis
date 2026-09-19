# Repository Guidelines

## Project Structure & Module Organization

`marathon_absa/` contains the Python package. Keep pipeline orchestration in
`pipeline.py`, command-line entry points in `cli.py`, configuration in `config.py`,
and focused processing logic in modules such as `language.py`, `relevance.py`, and
`topics.py`. `app.py` is the Streamlit dashboard entry point. Tests live in
`tests/` and mirror package concerns (`test_language.py`, `test_pipeline.py`, and
`test_relevance.py`).

Source datasets are under `Instagram/` and `Online Review Blog/`. Treat
`data/processed/`, `data/cache/`, and `models/` as generated local artifacts; they
are intentionally ignored by Git. Do not commit `.env`, virtual environments,
API responses, or downloaded models.

## Setup, Test, and Development Commands

Run commands from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
python -m marathon_absa.cli prepare
streamlit run app.py
```

`pytest` runs the complete unit suite. `prepare` performs local ingestion,
cleaning, language detection, and chunking. Use `pytest tests/test_language.py`
or `pytest -k mixed_language` for focused checks. Download OpenLID once with
`python -m marathon_absa.cli download-language-model`.

Commands that incur OpenAI cost require both a configured `OPENAI_API_KEY` and
the explicit `--run-api` flag. Use `--limit N` for pilot runs.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation. Use `snake_case` for modules,
functions, variables, and DataFrame columns; use `PascalCase` for classes and
`UPPER_CASE` for constants. Add type hints to public functions and keep imports
grouped as standard library, third-party, then local modules. Prefer small,
single-purpose functions and preserve audit fields rather than deleting records.
No formatter or linter is currently configured, so keep changes consistent with
the surrounding file.

## Testing Guidelines

Write pytest functions named `test_<behavior>` in `tests/test_<area>.py`. Tests
must be deterministic, must not call paid APIs, and should use `tmp_path`,
`monkeypatch`, or lightweight fakes for files, models, and services. Cover both
successful processing and safety gates, especially review routing and topic
inclusion. Run `pytest` before opening a pull request.

## Commit & Pull Request Guidelines

Usable Git history is not available in this checkout. Use concise, imperative
commit subjects such as `Add mixed-language span validation`, keeping each
commit focused. Pull requests should explain the research or pipeline impact,
list validation commands, identify schema or generated-output changes, and link
the relevant issue. Include screenshots for dashboard changes and disclose any
API-backed validation performed.

## Security & Configuration

Keep secrets only in `.env`. Never log API keys or commit cached model responses.
Preserve the pinned OpenLID revision in `config.py` unless a model upgrade is
deliberate, tested, and documented.

## Mandatory Project Documentation Maintenance

`PROJECT_DOCUMENTATION.md` is the canonical, thesis-oriented technical record for this repository. Study the relevant implementation, data, tests, and artifacts before changing it; do not treat it as a simple activity log.

Whenever a change affects the research design, data sources, collection or cleaning procedure, preprocessing, schemas, models, prompts, thresholds, validation, outputs, interfaces, commands, dependencies, limitations, or project status, update the relevant sections of `PROJECT_DOCUMENTATION.md` in the same task. Document what changed, when it changed using the strongest available evidence, where it is implemented or stored, how it works, why the decision was made, and the verified outcome. Keep the document detailed enough to support later thesis methodology, implementation, results, limitations, and reproducibility chapters.

Maintain a clear distinction between implemented, executed, validated, planned, and blocked stages. Never claim that a stage ran merely because code exists. Derive current counts and results from artifacts or commands, identify the evidence and as-of date, and record validation failures as fully as successes. Preserve data lineage from raw inputs to outputs and update the file/artifact register when paths or schemas change.

Do not put secrets, API keys, private cache contents, or unnecessary personal data in the documentation. Do not invent historical dates: use run manifests, embedded collection dates, file metadata, or other repository evidence, state which source supports the date, and label uncertain dates as estimates. Documentation-only wording or formatting changes that do not alter the project's substantive record do not require a separate history entry.
