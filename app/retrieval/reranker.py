from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - runtime availability
    torch = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:  # pragma: no cover - runtime availability
    AutoModelForSequenceClassification = None  # type: ignore
    AutoTokenizer = None  # type: ignore

from langchain_core.documents import Document

from app.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankedDocument:
    document: Document
    score: float


class CrossEncoderReranker:
    """
    Thin wrapper around Hugging Face cross-encoder rerankers. Designed to run locally using
    PyTorch; gracefully degrades if the environment lacks the required dependencies.
    """

    def __init__(
        self,
        model_name: str,
        device_preference: str = "auto",
        max_length: int = 512,
        batch_size: int = 8,
    ) -> None:
        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise ImportError(
                "transformers is not installed; reranking requires the transformers package."
            )
        if torch is None:
            raise ImportError("PyTorch is required for local reranking but is not installed.")

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = self._resolve_device(device_preference)

        logger.info(
            "reranker.init",
            extra={"model": model_name, "device": self.device, "max_length": max_length},
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def rank(self, query: str, documents: Sequence[Document]) -> List[Document]:
        return [item.document for item in self.rank_with_scores(query, documents)]

    def rank_with_scores(self, query: str, documents: Sequence[Document]) -> List[RerankedDocument]:
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores: List[RerankedDocument] = []

        with torch.no_grad():
            for start in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[start : start + self.batch_size]
                queries, passages = zip(*batch_pairs)
                inputs = self.tokenizer(
                    list(queries),
                    list(passages),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.model(**inputs)
                logits = outputs.logits.squeeze(-1)
                if logits.ndim == 0:
                    logits = logits.unsqueeze(0)
                batch_scores = logits.detach().cpu().tolist()
                if not isinstance(batch_scores, list):
                    batch_scores = [batch_scores]
                docs_slice = list(documents[start : start + len(batch_pairs)])
                for doc, score in zip(docs_slice, batch_scores):
                    scores.append(RerankedDocument(document=doc, score=float(score)))

        ordered = sorted(scores, key=lambda item: item.score, reverse=True)
        return ordered

    @staticmethod
    def _resolve_device(preference: str) -> str:
        pref = (preference or "auto").lower()
        candidates = {"auto", "cuda", "mps", "cpu"}
        if pref not in candidates:
            pref = "auto"

        if torch is None:
            return "cpu"

        if pref in {"auto", "cuda"} and torch.cuda.is_available():
            return "cuda"
        if pref in {"auto", "mps"} and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def build_reranker(config: AppConfig) -> Optional[CrossEncoderReranker]:
    if not config.reranker_model_name:
        logger.info("reranker.disabled", extra={"reason": "config_empty"})
        return None

    try:
        return CrossEncoderReranker(
            model_name=config.reranker_model_name,
            device_preference=config.reranker_device,
        )
    except Exception as exc:  # pragma: no cover - optional path
        logger.warning(
            "reranker.init_failed",
            extra={"model": config.reranker_model_name, "error": str(exc)},
        )
        return None
