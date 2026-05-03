"""Phase 4 — FastAPI smoke tests (no real generation, just route shape)."""

import os
import tempfile

import pytest

from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Redirect the projects root to a tmp path so tests don't pollute outputs/.
    monkeypatch.setenv("PHASE4_PROJECTS_ROOT", str(tmp_path))
    # Re-import so PROJECTS_ROOT picks up the env var. (Our orchestrator computes
    # it at import time, so this test only exercises route shape.)
    from phase4_web.api import app

    return TestClient(app)


def test_health_endpoint_returns_status(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok"
    assert "openai_configured" in j
    assert "moviepy_available" in j
    assert "ffmpeg_available" in j


def test_list_projects_works_when_empty(client):
    r = client.get("/api/projects")
    assert r.status_code == 200
    assert "projects" in r.json()


def test_get_unknown_project_returns_404(client):
    r = client.get("/api/projects/this-does-not-exist")
    assert r.status_code == 404
