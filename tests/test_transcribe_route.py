from __future__ import annotations

import io

import pytest

from app import create_app
from app.config import AppConfig
from app.audio.transcription import TranscriptionResult
from app.web import routes


class DummySTT:
    model_name = "dummy"

    def transcribe(self, audio_path):
        assert audio_path.exists()
        return TranscriptionResult(
            text="hello world",
            language="en",
            segments=[],
            duration=1.0,
        )


@pytest.fixture()
def app_client(monkeypatch):
    routes._stt_service = None
    dummy = DummySTT()
    monkeypatch.setattr(routes, "_get_stt_service", lambda: dummy)
    config = AppConfig.from_env()
    app = create_app(config)
    app.config.update(TESTING=True)
    return app.test_client()


def test_transcribe_success(app_client):
    response = app_client.post(
        "/transcribe",
        data={"audio": (io.BytesIO(b"fake audio"), "sample.webm")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["text"] == "hello world"
    assert payload["language"] == "en"


def test_transcribe_requires_audio(app_client):
    response = app_client.post("/transcribe")
    assert response.status_code == 400
    payload = response.get_json()
    assert payload["success"] is False


def test_transcribe_handles_missing_service(monkeypatch):
    routes._stt_service = None
    monkeypatch.setattr(routes, "_get_stt_service", lambda: None)
    config = AppConfig.from_env()
    app = create_app(config)
    app.config.update(TESTING=True)
    client = app.test_client()

    response = client.post(
        "/transcribe",
        data={"audio": (io.BytesIO(b"fake audio"), "sample.webm")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 503
    payload = response.get_json()
    assert payload["success"] is False
