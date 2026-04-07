"""DistilBERT sentiment model training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Torch dataset wrapper for tokenized text."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


@dataclass
class TransformerEvaluation:
    """Evaluation metrics for DistilBERT model."""

    accuracy: float
    f1_macro: float


class DistilBERTSentiment:
    """Hugging Face DistilBERT fine-tuning wrapper."""

    def __init__(self, model_name: str = "distilbert-base-uncased") -> None:
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            model_name,
            clean_up_tokenization_spaces=False,
        )
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
        )

    def train_and_evaluate(
        self,
        texts: list[str],
        labels: list[int],
        output_dir: str = "artifacts/distilbert",
    ) -> TransformerEvaluation:
        """Fine-tune DistilBERT and evaluate on holdout split."""
        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )

        train_enc = self.tokenizer(x_train, truncation=True, padding=True)
        test_enc = self.tokenizer(x_test, truncation=True, padding=True)

        train_ds = TextDataset(train_enc, y_train)
        test_ds = TextDataset(test_enc, y_test)

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=10,
            report_to="none",
            learning_rate=2e-5,
            warmup_steps=0,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
        )
        trainer.train()

        logits = trainer.predict(test_ds).predictions
        y_pred = np.argmax(logits, axis=1)

        return TransformerEvaluation(
            accuracy=float(accuracy_score(y_test, y_pred)),
            f1_macro=float(f1_score(y_test, y_pred, average="macro")),
        )
