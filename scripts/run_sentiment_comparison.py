"""Convenience script to run RNN vs DistilBERT comparison."""

from __future__ import annotations

import sys
from pathlib import Path


# Ensure script works when run directly from repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.sentiment.compare import run_comparison
except Exception as exc:
    error_text = str(exc)
    if (
        isinstance(exc, ModuleNotFoundError)
        and exc.name == "tensorflow"
    ) or "tf-keras" in error_text or "Keras 3" in error_text:
        raise SystemExit(
            "Challenge 2 TensorFlow stack is not ready in this environment. "
            "Use Python 3.11 and install requirements (including tf-keras), "
            "or run `python scripts/run_comparison_standalone.py` instead."
        ) from exc
    raise

if __name__ == "__main__":
    metrics = run_comparison()
    print("Comparison metrics:")
    print(metrics)
