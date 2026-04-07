# DEPLOYMENT: Production-Grade MLOps Strategy

## 1) Containerization (Local)

A Dockerfile is included to package:

- FastAPI service (`src/app.py`)
- OpenCV + MediaPipe + TensorFlow runtime dependencies
- NLP and sentiment modules

Local steps:

```bash
docker build -t multimodal-mlops-pipeline:latest .
docker run --rm -p 8000:8000 --env-file .env multimodal-mlops-pipeline:latest
```

## 2) Cloud Migration Strategy

### Option A: AWS

- **Container registry**: Amazon ECR
- **Orchestration**: ECS Fargate or EKS
- **Auto-scaling**: Target tracking on CPU/memory and request count
- **Load balancing**: Application Load Balancer
- **Database**: Amazon RDS for MySQL
- **Object storage**: S3 for image inputs and model artifacts

### Option B: GCP

- **Container registry**: Artifact Registry
- **Orchestration**: Cloud Run or GKE
- **Auto-scaling**: Cloud Run concurrency + min/max instances
- **Database**: Cloud SQL (MySQL)
- **Storage**: GCS

### Option C: Azure

- **Container registry**: Azure Container Registry
- **Orchestration**: Azure Container Apps or AKS
- **Auto-scaling**: KEDA-based scale rules
- **Database**: Azure Database for MySQL
- **Storage**: Blob Storage

## 3) Model Versioning Strategy

Use semantic model version tags and a model registry pattern:

- Naming: `mobilenet-v1.0.0`, `distilbert-sent-v1.0.0`
- Store artifacts in versioned cloud paths (S3/GCS/Blob)
- Track metadata:
  - training dataset version/hash
  - hyperparameters
  - evaluation metrics
  - commit SHA
  - preprocessing version (tokenizer/image transforms)
  - model card owner and approval timestamp

Serve-time traceability requirements:

- Include `model_name`, `model_version`, and `feature_pipeline_version` in every inference log row.
- Don't keep v1.0 and v2.0 running simultaneously (wastes memory).
   Instead: Store both in S3, use "live pointer" config.
   Update pointer → Kubernetes reloads weights → rollback complete..
- Block promotion unless candidate metrics beat baseline thresholds defined in CI.

### Preprocessing & Model Version Compatibility
   
   Preprocessing (tokenizer, image transforms) must be tracked alongside model versions:
   - Model v2.0 trained with spaCy v3.0
   - Old requests using spaCy v2.3 tokenizer → garbage predictions
   - Solution: Deploy as (model_version, preprocess_version) pair

Promotion workflow:

1. Train candidate model in CI pipeline.
2. Evaluate against baseline thresholds.
3. Register as staging version.
4. Canary deploy to small traffic slice.
5. Promote to production if SLO and accuracy criteria pass.

## 4) Canary Deployment Playbook

Recommended rollout gates:

1. Deploy candidate model to 5% traffic for 30-60 minutes.
2. Compare candidate vs current model on:
  - p95 latency
  - 5xx rate
  - abstain/low-confidence rate
  - online proxy quality metric (or delayed labeled accuracy)
3. Auto-rollback if any guardrail is breached.
4. Increase to 25%, then 50%, then 100% only after each gate passes.

Minimum rollback conditions (example):

- Candidate p95 latency is more than 20% worse than baseline.
- Candidate error rate is more than 1.5x baseline.
- Confidence drift or class-distribution drift exceeds agreed control limits.

### Cold Start Mitigation
   
   Model loading (400MB total) slows new container startup:
   - Use pod readiness probes (initialDelaySeconds: 30)
   - Pre-warm models on startup
   - Keep minimum 2 replicas always ready

## 5) Monitoring and Observability

### Application Monitoring

- Request latency (p50/p95/p99)
- Error rates (4xx/5xx)
- Throughput (RPS)
- Container CPU/memory usage

### Model Monitoring

- Prediction distribution drift over time
- Input data drift (text length, image brightness, face-count statistics)
- Sentiment class imbalance drift
- Confidence-score drift
- Accuracy decay against delayed ground-truth labels

Operational metric definitions:

- Latency SLO: p95 and p99 per endpoint and per model version.
- Quality SLO: rolling macro-F1 on recent labeled samples.
- Stability SLO: prediction entropy and class-ratio drift vs training baseline.

Accuracy-decay handling:

1. Track rolling 7-day macro-F1 for the live model.
2. Trigger retraining investigation when drop exceeds threshold (for example 3-5 percentage points).
3. Re-validate on a holdout set before promoting a replacement model.

### Alerting

- Trigger alerts when:
  - p95 latency exceeds threshold
  - error rate spikes
  - drift metrics exceed baseline bounds
  - rolling macro-F1 drops below release floor

## 6) CI/CD Outline

1. Lint + unit tests
2. Build container image
3. Push image to registry
4. Deploy to staging
5. Run smoke tests (`/health`, `/infer`)
6. Progressive rollout to production (canary/blue-green)

## 7) Security and Reliability

- Store secrets in cloud secret managers (not in image)
- Use least-privilege IAM roles
- Enable image vulnerability scanning
- Use read replicas and backups for MySQL
- Add retry/circuit-breaker behavior around DB and model dependencies

## 8) Scaling From Prototype to Production

- Stateless API containers for horizontal scaling
- Offload heavy/batch inference to async queue workers
- Cache common model components in memory
- Separate online inference DB writes from analytics pipeline
