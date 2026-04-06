"""Compare custom RNN baseline against DistilBERT transformer."""

from __future__ import annotations

import pandas as pd

from src.sentiment.rnn_baseline import RNNSentimentBaseline
from src.sentiment.transformer_model import DistilBERTSentiment


LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


def run_comparison(dataset_path: str = "data/sample_reviews.csv") -> dict:
    """Train/evaluate both models and return metric summary."""
    df = pd.read_csv(dataset_path)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].map(LABEL_MAP).tolist()

    rnn = RNNSentimentBaseline()
    rnn_metrics = rnn.train_and_evaluate(texts, labels)

    transformer = DistilBERTSentiment()
    transformer_metrics = transformer.train_and_evaluate(texts, labels)

    return {
        "rnn": {
            "accuracy": rnn_metrics.accuracy,
            "f1_macro": rnn_metrics.f1_macro,
        },
        "distilbert": {
            "accuracy": transformer_metrics.accuracy,
            "f1_macro": transformer_metrics.f1_macro,
        },
    }


if __name__ == "__main__":
    result = run_comparison()
    print(result)
