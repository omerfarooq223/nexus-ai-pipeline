"""Text processing using spaCy with fallback sentiment lexicon from NLTK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import nltk
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer

from src.config import settings


@dataclass
class TextFeatures:
    """Tokenization and sentiment outputs for a query string."""

    tokens: List[str]
    lemmas: List[str]
    noun_phrases: List[str]
    sentiment_label: str
    sentiment_score: float


class TextProcessor:
    """spaCy parser with VADER sentiment scoring."""

    def __init__(self) -> None:
        self.nlp = self._load_spacy_pipeline()
        nltk.download("vader_lexicon", quiet=True)
        self.sentiment = SentimentIntensityAnalyzer()

    @staticmethod
    def _load_spacy_pipeline():
        """Load configured spaCy model and fallback to blank English."""
        try:
            return spacy.load(settings.spacy_model)
        except OSError:
            return spacy.blank("en")

    def process(self, text: str) -> TextFeatures:
        """Run NLP parsing and sentiment scoring."""
        doc = self.nlp(text)
        tokens = [token.text for token in doc if not token.is_space]
        lemmas = [token.lemma_ for token in doc if not token.is_space]
        noun_phrases = [chunk.text for chunk in doc.noun_chunks] if doc.has_annotation("DEP") else []

        score = self.sentiment.polarity_scores(text)["compound"]
        if score > 0.05:
            label = "positive"
        elif score < -0.05:
            label = "negative"
        else:
            label = "neutral"

        return TextFeatures(
            tokens=tokens,
            lemmas=lemmas,
            noun_phrases=noun_phrases,
            sentiment_label=label,
            sentiment_score=score,
        )
