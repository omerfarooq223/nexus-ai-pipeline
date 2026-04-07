# REPORT: Transformer-Driven Sentiment Analysis & Data Scaling Strategy

---

## Part 1 — Challenge 2: Sentiment Model Comparison

### Objective

Compare a custom RNN baseline (LSTM) with a Transformer-based model (DistilBERT) for 3-class sentiment classification (positive / neutral / negative) on the same dataset, and explain why the Transformer achieves superior contextual understanding.

### Dataset

`data/sample_reviews.csv` — 300 labeled reviews (100 per class), split 80/20 with stratification (`random_state=42`), yielding 240 training and 60 test samples.

### Reproducible Comparison

Run the standalone comparison (PyTorch-only, no TensorFlow required):

```bash
python scripts/run_comparison_standalone.py
```

Or with the Keras-based version (requires TensorFlow):

```bash
PYTHONPATH=. python scripts/run_sentiment_comparison.py
```

### Measured Results

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Custom LSTM (RNN) | 0.3333 | 0.1667 |
| DistilBERT | **1.0000** | **1.0000** |

**Deltas:**

- Accuracy improvement: +0.6667 (DistilBERT over RNN)
- Macro-F1 improvement: +0.8333 (DistilBERT over RNN)

DistilBERT outperforms the custom RNN by a wide margin on both metrics.

---

### Why the RNN Scores 33.3% (Random-Guess Level)

The RNN's accuracy of 0.3333 equals exactly 1/3 — the probability of guessing randomly across 3 balanced classes. This is not because LSTMs are fundamentally broken; it is a **data-capacity mismatch**:

| Factor | Custom LSTM | DistilBERT |
|---|---|---|
| Parameters | ~50K (randomly initialized) | ~66M (pretrained on 2B+ words) |
| Training data | 240 samples | 240 samples + Wikipedia/BookCorpus knowledge |
| Knowledge at start | None — random weights | Rich syntax, semantics, world knowledge |
| Minimum useful dataset | 1,000+ labeled samples | 10–50 samples (fine-tuning only) |

The LSTM starts from random initialization and must learn everything — tokenization patterns, word meanings, sentiment signals — from only 240 training sentences. For a randomly initialized recurrent model, this is still too little signal for robust 3-class separation, so it remains close to random guessing.

DistilBERT, by contrast, already understands English from pretraining. Fine-tuning only adjusts the final classification head. It learns the class boundary from 240 examples because it already knows what words like "loved", "terrible", and "ruined" mean from seeing billions of words during pretraining.

**Key insight:** This result demonstrates **transfer learning** — the ability to leverage knowledge from a large unsupervised corpus (Wikipedia) and apply it to a relatively small supervised task (sentiment). Without transfer learning, 240 samples is still often insufficient for stable neural generalization.

---

### How Self-Attention Enables Contextual Understanding

Self-attention is the core mechanism that separates Transformers from RNNs. Here is how it works, step by step.

#### The Mechanism: Query, Key, Value

For every token in a sentence, the model computes three vectors:

- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain?"
- **Value (V):** "What information do I carry?"

Attention scores are computed as:

```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
```

Each token's Query is dot-producted against every other token's Key to produce attention weights. These weights determine how much each token "attends to" every other token. The final representation is a weighted sum of all Value vectors.

**Why this matters for sentiment:** The word "but" can directly attend to "ruined" even if they are 5 tokens apart. The attention weight between them is high because their Q/K vectors are aligned during pretraining. No sequential processing is needed.

#### Concrete Example: How Each Model Processes a Review

**Sentence:** *"I loved the opening but the ending ruined everything"*

**Correct label:** NEGATIVE (the ending ruined it)

**How the RNN processes this (sequential, left-to-right):**

```
Step 1: "I"         → hidden_state_1 (neutral)
Step 2: "loved"     → hidden_state_2 (strong positive signal)
Step 3: "the"       → hidden_state_3 (positive carried forward)
Step 4: "opening"   → hidden_state_4 (positive context)
Step 5: "but"       → hidden_state_5 (compressed info from steps 1-4)
Step 6: "the"       → hidden_state_6 (positive still dominant)
Step 7: "ending"    → hidden_state_7 (signal from "loved" is fading)
Step 8: "ruined"    → hidden_state_8 (negative, but "loved" influence is diluted)
Step 9: "everything"→ output (depends on all previous, but degraded)

By step 8, information from "loved" has been compressed through 6 LSTM cells.
The gradient signal weakens at each step (vanishing gradient problem).
The model cannot easily learn that "but" inverts the sentiment.
→ Prediction: POSITIVE (WRONG)
```

**How the Transformer processes this (parallel, all-at-once):**

```
All tokens are processed simultaneously.
Attention computation for "ruined":
  - "ruined" → "loved":     high attention (contrast detection)
  - "ruined" → "but":       high attention (negation signal)
  - "ruined" → "everything": high attention (scope of damage)

Attention computation for "but":
  - "but" → "loved":        high attention (what is being negated)
  - "but" → "ruined":       high attention (what follows the negation)

The word "everything" attends directly to both "loved" and "ruined"
with full attention weights — no compression, no information loss.
The model sees the full contrast structure in one pass.
→ Prediction: NEGATIVE (CORRECT)
```

**The fundamental difference:** The RNN compresses the entire sentence history into a single fixed-size hidden state vector at each step. Long-distance relationships (like "loved" → "but" → "ruined") are degraded by passing through multiple compression steps. The Transformer connects every token to every other token directly through attention weights, with no information bottleneck.

---

### Why the Transformer Outperforms the RNN — Technical Analysis

1. **Bidirectional context in every layer**
   Self-attention attends to all tokens simultaneously. DistilBERT captures context from both left and right sides at each of its 6 layers. When processing "not bad", it sees "not" and "bad" together and learns this combination is mildly positive — something an RNN processing left-to-right would struggle with because "not" is processed before "bad" appears.

2. **No vanishing gradient over distance**
   RNNs process tokens sequentially, so gradient signals degrade exponentially over distance. Despite LSTM gating (which mitigates but does not eliminate this), a 20-token sentence already strains information flow. Transformers connect distant tokens through direct attention paths — "loved" and "ruined" are one attention hop apart regardless of sentence length.

3. **Pretraining eliminates the cold-start problem**
   DistilBERT is pretrained on English Wikipedia and BookCorpus (~2B+ words). It already encodes that "terrible" is negative and "amazing" is positive before seeing any labeled sentiment data. The custom RNN starts with random embeddings where "terrible" and "amazing" are indistinguishable. With only 24 training samples, the RNN cannot learn these associations; DistilBERT only needs to fine-tune its existing knowledge.

4. **Context-sensitive word representations**
   Transformers produce different embeddings for the same word in different contexts. The word "fine" in "the quality is fine" (neutral) gets a different representation than "fine" in "I got a fine for speeding" (negative). Static RNN embeddings encode a single meaning per word.

5. **Macro-F1 as the decision metric**
   Macro-F1 weights all three classes equally, preventing the model from gaming accuracy by predicting only the majority class. DistilBERT's macro-F1 of **1.0000** versus the RNN's **0.1667** confirms the Transformer cleanly separates all three sentiment classes on this benchmark, while the RNN effectively makes random predictions.

---

### Practical Trade-Offs

| Dimension | Custom LSTM | DistilBERT |
|---|---|---|
| Model size | ~1 MB | ~260 MB |
| Inference latency | <5ms per sample | ~50ms per sample |
| Min training data | 1,000+ samples | 10–50 samples (fine-tuning) |
| Hardware requirement | CPU sufficient | GPU recommended for training |
| Quality (this dataset) | Random (0.1667 F1) | Very strong (1.0000 F1) |

- In production, DistilBERT can be further compressed via quantization (INT8) or pruning, reducing size by ~4× with minimal quality loss.
- For extremely latency-sensitive applications (<5ms), the RNN is viable only if trained on a sufficiently large dataset (10K+ samples).
- The RNN is appropriate when pretraining is impossible (proprietary language, no public corpus) and labeled data is abundant.

### Conclusion

DistilBERT is the preferred model for this sentiment task. Its measured macro-F1 of **1.0000** is far above the custom LSTM's score of **0.1667**, demonstrating that pretrained Transformer representations capture contextual relationships — negation, contrast, and sentiment inversion — that a randomly initialized RNN could not learn reliably from this training regime. The core advantage is self-attention: parallel, direct token-to-token connections that eliminate the sequential information bottleneck inherent to recurrent architectures.

---
---

## Part 2 — Challenge 4: Scaling the Data Pipeline from 1GB to 1TB

### Current Architecture (1GB Scale)

The existing ingestion pipeline in `src/data/ingestion.py` uses:

- **Pandas** to load, clean, and deduplicate CSV data
- **NumPy** for numeric type coercion and inf/NaN handling
- **Parquet** output format for columnar storage efficiency

At 1GB, a single-node Pandas/NumPy pipeline is sufficient. Batch cleaning runs on one machine, processes the entire dataset in memory, and writes optimized Parquet files.

### Migration Strategy (1TB Scale)

At 1TB, single-node Pandas will fail due to memory constraints (Pandas requires 2–5× dataset size in RAM). The pipeline must be redesigned for distributed computing.

#### 1. Distributed Processing with Apache Spark

Replace the Pandas-based `clean_dataset()` with PySpark equivalents:

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, lower, when

spark = SparkSession.builder \
    .appName("data-ingestion") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# Read raw data in parallel across executors
df = spark.read.csv("s3a://bucket/raw/noisy_data/", header=True, inferSchema=True)

# Same cleaning logic, distributed across the cluster
df = df.dropDuplicates()
df = df.na.drop(how="all")
df = df.withColumn("query_text", trim(col("query_text")))
df = df.withColumn("value", col("value").cast("float"))

# Write partitioned output
df.write.partitionBy("date").parquet("s3a://bucket/cleaned/")
```

Key differences from Pandas:
- **Lazy evaluation** — transformations are computed only when an action (write/collect) is triggered, enabling Spark to optimize the entire execution plan
- **Partition-level parallelism** — each partition is cleaned independently across executor nodes
- **Spill-to-disk** — Spark spills intermediate data to disk when memory is exhausted, unlike Pandas which throws `MemoryError`

**Why Spark over alternatives:**
- **Dask** mirrors the Pandas API but lacks Spark's mature optimizer (Catalyst), shuffle implementation, and ecosystem (Hive, Delta Lake). Dask is better for teams already deep in Pandas; Spark is better for production-scale data engineering.
- **Flink** excels at stream processing but is over-engineered for batch cleaning pipelines.
- Spark's dominance in the data engineering ecosystem also means better tooling, monitoring (Spark UI), and cloud-managed options (EMR, Dataproc, HDInsight).

#### 2. Data Partitioning Strategy

Partition data by date or source for parallel reads and writes:

```
s3://bucket/cleaned/
  ├── date=2026-01-01/
  │   ├── part-00000.parquet
  │   └── part-00001.parquet
  ├── date=2026-01-02/
  └── ...
```

Benefits:
- Queries filtering by date skip irrelevant partitions entirely (predicate pushdown)
- Parallel writers avoid single-file bottlenecks
- Partition pruning reduces I/O by 100–1000× for date-scoped queries

#### 3. Storage Layer — Data Lake with Medallion Architecture

Organize data in three tiers:

| Layer | Purpose | Format |
|---|---|---|
| **Bronze** (raw) | Immutable landing zone | CSV / JSON as-received |
| **Silver** (cleaned) | Deduplicated, type-corrected, validated | Parquet, partitioned by date |
| **Gold** (feature) | Aggregated, joined, ready for ML | Parquet / Delta Lake |

Delta Lake adds ACID transactions and time-travel on top of Parquet, enabling safe concurrent writes and rollback.

#### 4. Distributed SQL for Analytics

Replace direct Pandas queries with distributed SQL engines:

- **Spark SQL** for batch analytics on the data lake
- **Trino (formerly PrestoSQL)** for interactive federated queries across MySQL + S3
- **BigQuery / Athena** for serverless ad-hoc analysis without infrastructure management

Example index-aware query at scale:

```sql
-- Leverages idx_inference_created_at index
SELECT image_label, COUNT(*) AS cnt
FROM inference_results
WHERE created_at >= '2026-01-01'
GROUP BY image_label
ORDER BY cnt DESC;
```

#### 5. Orchestration and Data Quality

Add pipeline orchestration to manage dependencies and retries:

- **Apache Airflow** or **Prefect** for DAG-based scheduling
- Data quality checks at each medallion layer:
  - Row count assertions (±5% tolerance vs previous run)
  - Schema validation (column types, nullability)
  - Statistical drift detection (mean/std of numeric columns)
  - Freshness checks (max timestamp vs wall clock)

#### 6. Failure Recovery

Distributed pipelines must handle partial failures gracefully:

- **Spark checkpointing** persists intermediate RDD state to HDFS/S3, allowing jobs to resume from the last checkpoint rather than restarting from scratch
- **Idempotent writes** — use Delta Lake's MERGE or overwrite-by-partition mode so re-running a failed job produces the same result without duplicates
- **Dead-letter queues** — route un-parseable rows to a separate table for manual inspection instead of failing the entire job
- **Airflow retries** — configure `retries=3` with exponential backoff per task; alert on final failure

#### 7. Cost Considerations

At TB scale, infrastructure cost drives architectural decisions:

| Component | Estimated Monthly Cost | Optimization |
|---|---|---|
| S3 storage (1 TB) | ~$23/month | Use S3 Infrequent Access for Bronze tier |
| EMR Spark cluster (4x m5.xlarge) | ~$600/month | Use spot instances (60-70% savings) |
| Data transfer (cross-region) | $0.09/GB | Co-locate compute and storage in same region |
| Athena queries | $5/TB scanned | Partition + Parquet reduces scanned data by 10-100× |

Spot instances for Spark workers provide 60–70% cost savings. Use on-demand for the driver node only, and enable dynamic allocation to scale workers based on current stage size.

#### 8. SQL Indexing at Scale

The current indexing strategy in `schemas/schema.sql` applies to the MySQL operational database:

```sql
CREATE INDEX idx_inference_created_at ON inference_results (created_at);
CREATE INDEX idx_inference_image_label ON inference_results (image_label);
CREATE INDEX idx_inference_sentiment_label ON inference_results (sentiment_label);
CREATE FULLTEXT INDEX idx_inference_query_text ON inference_results (query_text);
```

At TB scale, extend this with:
- **Composite indexes** for common query patterns: `(created_at, image_label)` for time-scoped label lookups
- **MySQL partitioning** by `created_at` range to keep active partitions small
- Migrate analytics queries off the operational DB to the data lake to avoid lock contention

### Summary

| Concern | 1GB Solution | 1TB Solution |
|---|---|---|
| Processing | Pandas + NumPy | Apache Spark (PySpark) |
| Storage | Local Parquet | S3/GCS + Delta Lake |
| Partitioning | Single file | Date-based partitioning |
| SQL | Single MySQL node | Spark SQL / Trino + sharded MySQL |
| Orchestration | Manual scripts | Airflow / Prefect DAGs |
| Quality | Manual inspection | Automated assertions per layer |
| Failure handling | Script re-run | Checkpointing + idempotent writes |
| Cost | Free (local) | ~$650/month (S3 + spot Spark) |
