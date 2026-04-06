"""Custom RNN sentiment baseline using Keras."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


@dataclass
class RNNEvaluation:
    """Evaluation metrics for baseline model."""

    accuracy: float
    f1_macro: float


class RNNSentimentBaseline:
    """A lightweight LSTM classifier for comparison."""

    def __init__(self, vocab_size: int = 10000, max_len: int = 120) -> None:
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
        self.model = self._build_model()

    def _build_model(self) -> Sequential:
        model = Sequential(
            [
                Embedding(input_dim=self.vocab_size, output_dim=64),
                LSTM(64),
                Dense(32, activation="relu"),
                Dense(3, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _vectorize(self, texts: list[str]) -> np.ndarray:
        seq = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(seq, maxlen=self.max_len, padding="post", truncating="post")

    def train_and_evaluate(
        self,
        texts: list[str],
        labels: list[int],
        epochs: int = 3,
        batch_size: int = 16,
    ) -> RNNEvaluation:
        """Train and evaluate baseline using a holdout split."""
        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )
        self.tokenizer.fit_on_texts(x_train)
        train_vec = self._vectorize(x_train)
        test_vec = self._vectorize(x_test)

        self.model.fit(
            train_vec,
            np.array(y_train),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )
        pred = self.model.predict(test_vec, verbose=0)
        y_pred = pred.argmax(axis=1)

        return RNNEvaluation(
            accuracy=float(accuracy_score(y_test, y_pred)),
            f1_macro=float(f1_score(y_test, y_pred, average="macro")),
        )
