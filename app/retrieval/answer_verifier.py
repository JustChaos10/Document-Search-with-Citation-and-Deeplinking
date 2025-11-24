from __future__ import annotations

import logging
from typing import List
import re

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class AnswerVerifier:
    """
    Verifies that generated answers are grounded in source documents
    and assigns confidence scores.
    """

    def __init__(self, min_confidence_threshold: float = 0.5):
        """
        Initialize answer verifier.

        Args:
            min_confidence_threshold: Minimum confidence score (0-1) for valid answers
        """
        self.min_confidence_threshold = min_confidence_threshold

    def verify_answer(
        self,
        answer: str,
        question: str,
        sources: List[Document]
    ) -> tuple[float, str]:
        """
        Verify answer against source documents and compute confidence score.

        Args:
            answer: Generated answer text
            question: Original question
            sources: Source documents used for generation

        Returns:
            Tuple of (confidence_score, verification_status)
            - confidence_score: 0.0 to 1.0
            - verification_status: "high", "medium", "low", or "unverified"
        """
        if not answer or not sources:
            return 0.0, "unverified"

        # Calculate multiple confidence signals
        factual_grounding_score = self._check_factual_grounding(answer, sources)
        citation_coverage_score = self._check_citation_coverage(answer, sources)
        specificity_score = self._check_answer_specificity(answer)
        relevance_score = self._check_question_relevance(answer, question)

        # Weighted average of scores
        confidence_score = (
            factual_grounding_score * 0.4 +
            citation_coverage_score * 0.3 +
            specificity_score * 0.2 +
            relevance_score * 0.1
        )

        # Determine status
        if confidence_score >= 0.75:
            status = "high"
        elif confidence_score >= 0.5:
            status = "medium"
        elif confidence_score >= 0.3:
            status = "low"
        else:
            status = "unverified"

        logger.info(
            "answer.verification_complete",
            extra={
                "confidence_score": round(confidence_score, 3),
                "status": status,
                "factual_grounding": round(factual_grounding_score, 3),
                "citation_coverage": round(citation_coverage_score, 3),
                "specificity": round(specificity_score, 3),
                "relevance": round(relevance_score, 3)
            }
        )

        return confidence_score, status

    def _check_factual_grounding(
        self,
        answer: str,
        sources: List[Document]
    ) -> float:
        """
        Check if answer claims are grounded in source documents.

        Returns score between 0.0 and 1.0
        """
        # Extract key phrases from answer (nouns, numbers, dates)
        answer_phrases = self._extract_key_phrases(answer)

        if not answer_phrases:
            return 0.5  # Neutral if no extractable phrases

        # Combine all source text
        source_text = " ".join(doc.page_content.lower() for doc in sources)

        # Check how many answer phrases appear in sources
        grounded_count = sum(
            1 for phrase in answer_phrases
            if phrase.lower() in source_text
        )

        score = grounded_count / len(answer_phrases) if answer_phrases else 0.0
        return min(score, 1.0)

    def _check_citation_coverage(
        self,
        answer: str,
        sources: List[Document]
    ) -> float:
        """
        Check if answer has adequate citation coverage.

        Returns score between 0.0 and 1.0
        """
        # Count sentences in answer
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        num_sentences = len(sentences)

        if num_sentences == 0:
            return 0.0

        # Expect at least one citation per 2 sentences
        expected_citations = max(1, num_sentences // 2)
        actual_citations = len(sources)

        # Score based on citation ratio
        score = min(actual_citations / expected_citations, 1.0)
        return score

    def _check_answer_specificity(self, answer: str) -> float:
        """
        Check if answer contains specific information (numbers, dates, names).

        Returns score between 0.0 and 1.0
        """
        # Look for specificity indicators
        has_numbers = bool(re.search(r'\d+', answer))
        has_percentages = bool(re.search(r'\d+%', answer))
        has_dates = bool(re.search(r'\d{4}|\d{1,2}/\d{1,2}', answer))
        has_proper_nouns = bool(re.search(r'\b[A-Z][a-z]+\b', answer))

        # Count indicators
        specificity_indicators = sum([
            has_numbers,
            has_percentages,
            has_dates,
            has_proper_nouns
        ])

        # More indicators = more specific
        score = min(specificity_indicators * 0.25, 1.0)
        return score

    def _check_question_relevance(
        self,
        answer: str,
        question: str
    ) -> float:
        """
        Check if answer is relevant to the question.

        Returns score between 0.0 and 1.0
        """
        # Extract question keywords
        question_keywords = set(
            word.lower() for word in question.split()
            if len(word) > 3
        )

        if not question_keywords:
            return 0.5  # Neutral

        # Check answer mentions question keywords
        answer_lower = answer.lower()
        mentioned_keywords = sum(
            1 for keyword in question_keywords
            if keyword in answer_lower
        )

        score = min(mentioned_keywords / len(question_keywords), 1.0)
        return score

    def _extract_key_phrases(self, text: str) -> List[str]:
        """
        Extract key phrases from text (numbers, proper nouns, etc.).

        Returns list of key phrases
        """
        phrases = []

        # Extract numbers (including percentages, decimals)
        numbers = re.findall(r'\d+\.?\d*%?', text)
        phrases.extend(numbers)

        # Extract potential proper nouns (capitalized words)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        phrases.extend(proper_nouns)

        # Extract dates
        dates = re.findall(r'\d{4}|\d{1,2}/\d{1,2}/\d{2,4}', text)
        phrases.extend(dates)

        # Extract quoted text
        quoted = re.findall(r'"([^"]+)"', text)
        phrases.extend(quoted)

        return phrases
