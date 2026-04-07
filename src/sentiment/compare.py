"""Compare custom RNN baseline against DistilBERT transformer."""

from __future__ import annotations

import random
import os
import numpy as np
import torch
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

from src.sentiment.rnn_baseline import RNNSentimentBaseline
from src.sentiment.transformer_model import DistilBERTSentiment

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_reviews.csv"


def run_comparison(dataset_path: str | Path = DEFAULT_DATASET_PATH) -> dict:
    """Train/evaluate both models and return metric summary."""
    df = pd.read_csv(Path(dataset_path))
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
