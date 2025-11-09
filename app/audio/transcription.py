from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.config import AppConfig

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - runtime availability
    torch = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import whisper
except ImportError:  # pragma: no cover - runtime availability
    whisper = None  # type: ignore


@dataclass(frozen=True)
class TranscriptionSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: Optional[str]
    segments: List[TranscriptionSegment]
    duration: Optional[float] = None


class SpeechToTextService:
    """Thin wrapper around OpenAI Whisper for local speech-to-text."""

    _MODEL_CACHE: Dict[Tuple[str, str], Any] = {}

    def __init__(self, model_name: str = "medium", device_preference: str = "auto") -> None:
        if whisper is None:  # pragma: no cover - handled during startup
            raise ImportError(
                "openai-whisper is not installed. Install it via `pip install openai-whisper`."
            )

        self.model_name = model_name
        self.device = self._resolve_device(device_preference)
        self.fp16 = self.device not in {"cpu", "mps"}
        logger.info(
            "whisper.init",
            extra={"model": model_name, "device": self.device, "fp16": self.fp16},
        )
        self._model = self._load_model(model_name, self.device)

    @classmethod
    def from_config(cls, config: AppConfig) -> "SpeechToTextService":
        return cls(
            model_name=config.whisper_model_name,
            device_preference=config.whisper_device,
        )

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        logger.info(
            "whisper.transcribe.start",
            extra={"path": str(path), "model": self.model_name, "device": self.device},
        )
        result = self._model.transcribe(
            str(path),
            task="transcribe",
            language=None,  # auto-detect
            fp16=self.fp16,
        )
        text = (result.get("text") or "").strip()
        language = (result.get("language") or "").strip() or None
        segments_payload: Iterable[Dict[str, Any]] = result.get("segments") or []
        segments = [
            TranscriptionSegment(
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                text=(segment.get("text") or "").strip(),
            )
            for segment in segments_payload
            if segment.get("text")
        ]
        duration = result.get("duration")
        logger.info(
            "whisper.transcribe.complete",
            extra={
                "path": str(path),
                "language": language,
                "segments": len(segments),
                "duration": duration,
            },
        )
        return TranscriptionResult(
            text=text,
            language=language,
            segments=segments,
            duration=float(duration) if duration is not None else None,
        )

    @classmethod
    def _load_model(cls, model_name: str, device: str):
        key = (model_name, device)
        if key not in cls._MODEL_CACHE:
            cls._MODEL_CACHE[key] = whisper.load_model(model_name, device=device)
        return cls._MODEL_CACHE[key]

    @staticmethod
    def _resolve_device(preference: str) -> str:
        pref = (preference or "auto").lower()
        candidates = ["auto", "cuda", "mps", "cpu"]
        if pref not in candidates:
            pref = "auto"

        if pref in {"auto", "cuda"} and SpeechToTextService._cuda_available():
            return "cuda"
        if pref in {"auto", "mps"} and SpeechToTextService._mps_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _cuda_available() -> bool:
        if torch is None:
            return False
        try:
            return bool(torch.cuda.is_available())
        except Exception:  # pragma: no cover
            return False

    @staticmethod
    def _mps_available() -> bool:
        if torch is None:
            return False
        try:
            return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        except Exception:  # pragma: no cover
            return False
