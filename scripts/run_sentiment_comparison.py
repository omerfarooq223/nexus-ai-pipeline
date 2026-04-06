"""Convenience script to run RNN vs DistilBERT comparison."""

from src.sentiment.compare import run_comparison

if __name__ == "__main__":
    metrics = run_comparison()
    print("Comparison metrics:")
    print(metrics)
