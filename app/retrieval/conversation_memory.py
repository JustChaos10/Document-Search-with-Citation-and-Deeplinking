from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Represents a single question-answer turn in a conversation."""
    question: str
    answer: str
    timestamp: datetime
    language: str


class ConversationMemory:
    """
    Manages conversation history for context-aware follow-up questions.
    """

    def __init__(
        self,
        max_turns: int = 3,
        max_age_minutes: int = 30
    ):
        """
        Initialize conversation memory.

        Args:
            max_turns: Maximum number of turns to keep in memory
            max_age_minutes: Maximum age of conversation turns in minutes
        """
        self.max_turns = max_turns
        self.max_age_minutes = max_age_minutes
        self.history: List[ConversationTurn] = []

    def add_turn(
        self,
        question: str,
        answer: str,
        language: str = "en"
    ) -> None:
        """
        Add a new conversation turn to memory.

        Args:
            question: User's question
            answer: System's answer
            language: Language of the conversation
        """
        turn = ConversationTurn(
            question=question,
            answer=answer,
            timestamp=datetime.now(),
            language=language
        )
        self.history.append(turn)

        # Clean up old history
        self._cleanup_history()

        logger.debug(
            "conversation.turn_added",
            extra={
                "turns_in_memory": len(self.history),
                "question_preview": question[:50]
            }
        )

    def get_context(self, current_language: str = None) -> str:
        """
        Get conversation context for the LLM prompt.

        Args:
            current_language: Current conversation language (optional filter)

        Returns:
            Formatted conversation history string
        """
        if not self.history:
            return ""

        # Filter by language if specified
        relevant_turns = self.history
        if current_language:
            relevant_turns = [
                turn for turn in self.history
                if turn.language == current_language
            ]

        if not relevant_turns:
            return ""

        # Build context string
        context_lines = ["Previous conversation:"]
        for i, turn in enumerate(relevant_turns[-self.max_turns:], 1):
            context_lines.append(f"Q{i}: {turn.question}")
            context_lines.append(f"A{i}: {turn.answer}")

        context = "\n".join(context_lines)

        logger.debug(
            "conversation.context_retrieved",
            extra={
                "turns": len(relevant_turns),
                "context_length": len(context)
            }
        )

        return context

    def has_history(self) -> bool:
        """Check if there is any conversation history."""
        self._cleanup_history()
        return len(self.history) > 0

    def clear(self) -> None:
        """Clear all conversation history."""
        self.history.clear()
        logger.debug("conversation.history_cleared")

    def _cleanup_history(self) -> None:
        """Remove old conversation turns based on max_turns and max_age."""
        if not self.history:
            return

        # Remove turns older than max_age_minutes
        cutoff_time = datetime.now() - timedelta(minutes=self.max_age_minutes)
        self.history = [
            turn for turn in self.history
            if turn.timestamp > cutoff_time
        ]

        # Keep only the most recent max_turns
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def get_last_question(self) -> str | None:
        """Get the last question asked, if any."""
        if not self.history:
            return None
        return self.history[-1].question

    def expand_query_with_context(self, current_query: str) -> str:
        """
        Expand current query with conversation context if it seems like a follow-up.

        Args:
            current_query: Current user query

        Returns:
            Expanded query with context, or original if standalone
        """
        # Detect follow-up indicators
        follow_up_indicators = [
            "what about",
            "and",
            "also",
            "how about",
            "what else",
            "more about",
            "further",
            "in addition",
            "additionally",
            # Arabic indicators
            "وماذا عن",
            "و",
            "أيضا",
            "ماذا أيضا",
            "المزيد",
            "بالإضافة"
        ]

        query_lower = current_query.lower()
        is_follow_up = any(
            indicator in query_lower
            for indicator in follow_up_indicators
        )

        # Check if query is very short (likely a follow-up)
        is_short = len(current_query.split()) < 4

        if (is_follow_up or is_short) and self.has_history():
            last_question = self.get_last_question()
            if last_question:
                expanded = f"{last_question} {current_query}"
                logger.info(
                    "conversation.query_expanded",
                    extra={
                        "original": current_query,
                        "expanded": expanded[:100]
                    }
                )
                return expanded

        return current_query
