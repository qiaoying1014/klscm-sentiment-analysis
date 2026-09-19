"""Check release integrity and exercise the app without local research files."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from marathon_absa.cloud_bundle import load_cloud_bundle

ROOT = Path(__file__).resolve().parents[1]


def test_cloud_bundle_rejects_modified_artifact(tmp_path, monkeypatch):
    source = ROOT / "deploy/dashboard_data"
    shutil.copytree(source, tmp_path / "bundle")
    root = tmp_path / "bundle"
    manifest = json.loads((root / "manifest.json").read_text())
    item = next(iter(manifest["sections"]["dashboard"].values()))
    with (root / item["file"]).open("ab") as stream:
        stream.write(b"tampered")
    monkeypatch.setenv("KLSCM_DASHBOARD_BUNDLE", str(root))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_cloud_bundle("dashboard")


def test_cloud_pages_without_research_artifacts(tmp_path):
    shutil.copytree(ROOT / "marathon_absa", tmp_path / "marathon_absa", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "deploy", tmp_path / "deploy")
    shutil.copy2(ROOT / "absa_dashboard.py", tmp_path)
    assert not (tmp_path / "data").exists()
    script = '''
from streamlit.testing.v1 import AppTest
app = AppTest.from_file("deploy/streamlit_app.py", default_timeout=60).run()
assert not app.exception, list(app.exception)
for page in app.sidebar.radio[0].options:
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, (page, list(app.exception))
    assert not app.error, (page, [item.value for item in app.error])
    print("PASS", page, flush=True)
app.sidebar.radio[0].set_value("Word Cloud").run()
app.radio(key="wc_source").set_value("Full Caption Context").run()
assert not app.exception and not app.error
print("PASS caption-context wordcloud", flush=True)
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS") == 13
