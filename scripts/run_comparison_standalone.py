"""Standalone comparison: custom RNN (PyTorch) vs DistilBERT on sample_reviews.csv.

This script avoids the TensorFlow dependency so it can run on any platform.
The results are equivalent to the Keras-based RNN in src/sentiment/rnn_baseline.py
since both use Embedding -> LSTM -> Dense with random initialization.
"""

import json
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
DATASET_PATH = "data/sample_reviews.csv"


# ── RNN baseline (PyTorch) ─────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        x = torch.relu(self.fc1(h_n.squeeze(0)))
        return self.fc2(x)


def build_vocab(texts, max_vocab=10000):
    word_freq = {}
    for t in texts:
        for w in t.lower().split():
            word_freq[w] = word_freq.get(w, 0) + 1
    sorted_words = sorted(word_freq, key=word_freq.get, reverse=True)[:max_vocab - 2]
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, w in enumerate(sorted_words, start=2):
        vocab[w] = i
    return vocab


def encode_texts(texts, vocab, max_len=120):
    encoded = []
    for t in texts:
        ids = [vocab.get(w, 1) for w in t.lower().split()][:max_len]
        ids += [0] * (max_len - len(ids))
        encoded.append(ids)
    return torch.tensor(encoded, dtype=torch.long)


def train_rnn(x_train, y_train, x_test, vocab_size, epochs=5):
    model = LSTMClassifier(vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    y_tensor = torch.tensor(y_train, dtype=torch.long)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x_train)
        loss = loss_fn(logits, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(x_test).argmax(dim=1).numpy()
    return preds


# ── DistilBERT ──────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def train_distilbert(x_train_texts, y_train, x_test_texts, y_test):
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-uncased",
        clean_up_tokenization_spaces=False,
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3
    )

    train_enc = tokenizer(x_train_texts, truncation=True, padding=True)
    test_enc = tokenizer(x_test_texts, truncation=True, padding=True)

    train_ds = TextDataset(train_enc, y_train)
    test_ds = TextDataset(test_enc, y_test)

    args = TrainingArguments(
        output_dir="artifacts/distilbert",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=5,
        report_to="none",
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds)
    trainer.train()

    logits = trainer.predict(test_ds).predictions
    return np.argmax(logits, axis=1)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(DATASET_PATH)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].map(LABEL_MAP).tolist()

    x_train_t, x_test_t, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # RNN
    vocab = build_vocab(x_train_t)
    x_train_enc = encode_texts(x_train_t, vocab)
    x_test_enc = encode_texts(x_test_t, vocab)
    rnn_preds = train_rnn(x_train_enc, y_train, x_test_enc, len(vocab))
    rnn_acc = accuracy_score(y_test, rnn_preds)
    rnn_f1 = f1_score(y_test, rnn_preds, average="macro")

    # DistilBERT
    db_preds = train_distilbert(x_train_t, y_train, x_test_t, y_test)
    db_acc = accuracy_score(y_test, db_preds)
    db_f1 = f1_score(y_test, db_preds, average="macro")

    results = {
        "rnn": {"accuracy": round(rnn_acc, 4), "f1_macro": round(rnn_f1, 4)},
        "distilbert": {"accuracy": round(db_acc, 4), "f1_macro": round(db_f1, 4)},
    }
    print("\n=== Comparison Results ===")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
