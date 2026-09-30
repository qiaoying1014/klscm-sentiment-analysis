# Historical Streamlit deployment

Updated 30 September 2026, Asia/Kuala_Lumpur (UTC+08:00).

## Git boundary

- First commit today: `025c7478c344d145ad3db475b2390c9cd727e223`, 2026-09-28T15:50:54+08:00, Fix dashboard theme and chart rendering.
- PRE_TODAY_COMMIT: `73c71a6f5062237340a70302215bb0ee56041e27`, 2026-09-19T22:07:04+08:00, Support Python 3.14 in Streamlit Cloud dependencies.
- Historical branch: `deployment/pre-today`, created from PRE_TODAY_COMMIT and deliberately includes `025c7478` at the user's request. This documentation is the only other addition.
- Preserved main: `9d13b9a4d506d2a6ffb7bddb75d4971acc07c443`.
- Both author and committer timestamps identify the same boundary. Four commits were made today on main.

## Deployment architecture and verification

Entry point `deploy/streamlit_app.py` resolves the repository root from its own file, sets KLSCM_DASHBOARD_BUNDLE to `deploy/dashboard_data`, and executes historical `absa_dashboard.py`. Its sibling requirements pins the Cloud dependencies; root `packages.txt` installs `fonts-noto-cjk`. The dashboard supports the corresponding Linux Noto CJK font path and Windows font fallbacks. No .streamlit configuration exists in this snapshot or is required. No local research data, secrets, model downloads or export pipeline is needed.

No deployment compatibility modifications were required. Before the requested theme/chart commit was added, all 205 original tracked working files were checked against historical Git blob IDs using Git clean filters; all matched. All 34 deployment artifacts passed their manifest SHA-256 checks. Every active pinned dependency matched the existing isolated Python 3.14.7 environment: pandas 2.3.3, pyarrow 24.0.0, streamlit 1.59.2, plotly 6.9.0, numpy 2.3.5, wordcloud 1.9.6, jieba 0.42.1.

Changes between PRE_TODAY_COMMIT and main affect PROJECT_DOCUMENTATION.md, absa_dashboard.py, marathon_absa/dashboard_presentation.py, marathon_absa/participant_experience_page.py, tests/test_absa_dashboard_ui.py, tests/test_blog_dashboard_data.py, tests/test_cloud_deployment.py, tests/test_dashboard_presentation.py, and tests/test_participant_experience_dashboard_finalized.py. This branch includes only the requested theme/chart commit's changes to PROJECT_DOCUMENTATION.md, absa_dashboard.py and tests/test_absa_dashboard_ui.py. The later scope and marathon-presentation commits, including the two dashboard_presentation files, remain excluded. The deployment bundle and requirements were unchanged by all four commits.

Executed from the historical checkout with the existing sibling `.local/cloud314/Scripts/python.exe` interpreter:

- With KLSCM_DASHBOARD_BUNDLE set to the historical deployment bundle: `python -m pytest tests/test_cloud_deployment.py tests/test_absa_dashboard_ui.py -q`: **22 passed in 27.89 seconds**.
- Existing Cloud test imports and executes the entry point through Streamlit AppTest in a temporary source/bundle-only directory without research data; all 12 historical pages and caption-context word cloud pass. Tamper rejection also passes.
- Started `python -m streamlit run deploy/streamlit_app.py --server.headless=true --server.address=127.0.0.1 --server.port=18529 --browser.gatherUsageStats=false`: health endpoint returned `ok`; only this temporary server was stopped.
- `git diff --check`: passed before documentation.

These are Windows execution checks, not a new Linux Cloud deployment. Full research tests were intentionally not run: the historical documentation records missing earlier-stage local fixtures, and deployment requires no research pipeline execution. The requested theme/chart repair is included; the later scope and marathon-presentation changes are intentionally excluded.

Main and its working tree were not modified, reset, reverted, switched or pushed. No existing Streamlit app settings were changed. Frozen research artifacts were not modified; no research outputs were regenerated. The new checkout contains only Git-tracked historical files plus this record, with no copied local research workspace.

## Publish and deploy the second app

Push only the historical branch (never force or push all branches):

```powershell
git push -u origin deployment/pre-today
```

On Streamlit Community Cloud, choose **Create app** and create a NEW app with:

- Repository: `qiaoying1014/klscm-sentiment-analysis`
- Branch: `deployment/pre-today`
- Main file path: `deploy/streamlit_app.py`
- Advanced settings Python version: `3.14` (locally tested 3.14.7).
- Choose a separate available app URL, for example `klscm-sentiment-analysis-pre-today` if available.

Leave the existing app configured on `main`. Do not merge the historical branch into main or repoint the existing app. Both apps can use the same repository and entry point with different branches.

Official instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

The live second Cloud app has not been created; the owner must complete Create app. Remote main was verified at the preserved hash before publication, and no historical remote branch existed at that check. Push outcome is reported in the accompanying task response.
