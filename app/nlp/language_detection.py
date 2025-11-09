from __future__ import annotations

from functools import lru_cache
from typing import Optional

from langdetect import DetectorFactory, detect

# Ensure deterministic output from langdetect across runs.
DetectorFactory.seed = 42


SUPPORTED_LANGS = {"en", "ar"}


@lru_cache(maxsize=512)
def _detect_language_cached(text: str) -> str:
    return detect(text)


def detect_language(text: str) -> Optional[str]:
    """
    Detect the dominant language of the supplied text using langdetect.
    Returns a two-letter ISO code when recognized, otherwise None.
    """
    candidate = text.strip()
    if len(candidate) < 20:
        return None
    try:
        language = _detect_language_cached(candidate)
    except Exception:
        return None
    if language in SUPPORTED_LANGS:
        return language
    return language
