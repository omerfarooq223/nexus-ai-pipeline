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

## 2.1) Concrete AWS Migration Workflow

### Phase 1: Local Setup and Container Registry

**Goal:** Package app in Docker and push to AWS ECR.

1. **Build container locally and test**

```bash
docker build -t multimodal-pipeline:v1.0.0 .
docker run --rm -p 8000:8000 --env-file .env multimodal-pipeline:v1.0.0
curl http://localhost:8000/health  # Verify it works
```

2. **Create AWS ECR repository**

```bash
aws ecr create-repository --repository-name multimodal-pipeline
# Output: URI like 123456789.dkr.ecr.us-east-1.amazonaws.com/multimodal-pipeline
```

3. **Authenticate Docker to ECR and push**

```bash
aws ecr get-login-password | docker login --username AWS --password-stdin <ECR-URI>
docker tag multimodal-pipeline:v1.0.0 <ECR-URI>/multimodal-pipeline:v1.0.0
docker push <ECR-URI>/multimodal-pipeline:v1.0.0
```

### Phase 2: Database Migration (Local to RDS)

1. **Create RDS MySQL instance**

```bash
aws rds create-db-instance \
  --db-instance-identifier multimodal-db \
  --db-instance-class db.t3.micro \
  --engine mysql \
  --engine-version 8.0.35 \
  --master-username admin \
  --allocated-storage 20
# Wait 5-10 minutes for instance to be ready
```

2. **Configure security group**

```bash
aws ec2 authorize-security-group-ingress \
  --group-id <rds-sg> \
  --protocol tcp \
  --port 3306 \
  --source-security-group-id <ecs-sg>
```

3. **Initialize database schema**

```bash
mysql -h multimodal-db.xxxx.rds.amazonaws.com \
  -u admin -p < schemas/schema.sql
```

4. **Verify connection**

```bash
mysql -h multimodal-db.xxxx.rds.amazonaws.com \
  -u admin -p -e "SELECT VERSION();"
```

### Phase 3: Model and Artifact Storage (S3)

1. **Create S3 bucket for models**

```bash
aws s3 mb s3://multimodal-models-prod
aws s3api put-bucket-versioning \
  --bucket multimodal-models-prod \
  --versioning-configuration Status=Enabled
```

2. **Upload model artifacts**

```bash
# Create model versions (already in Dockerfile, but store for rollback)
aws s3 cp mobilenet-v1.0.0.tar.gz s3://multimodal-models-prod/
aws s3 cp distilbert-v1.0.0.tar.gz s3://multimodal-models-prod/
```

3. **Store inference logs in S3 for analytics**

```bash
# Separate bucket for logs (not models)
aws s3 mb s3://multimodal-inference-logs
```

### Phase 4: ECS Fargate Deployment

1. **Create ECS cluster**

```bash
aws ecs create-cluster --cluster-name multimodal-prod
```

2. **Create ECS task definition** (save as `task-definition.json`)

```json
{
  "family": "multimodal-pipeline",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [
    {
      "name": "app",
      "image": "<ECR-URI>/multimodal-pipeline:v1.0.0",
      "portMappings": [
        {
          "containerPort": 8000,
          "hostPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MYSQL_HOST",
          "value": "multimodal-db.xxxx.rds.amazonaws.com"
        },
        {
          "name": "MYSQL_PORT",
          "value": "3306"
        },
        {
          "name": "MYSQL_DATABASE",
          "value": "multimodal_ai"
        }
      ],
      "secrets": [
        {
          "name": "MYSQL_USER",
          "valueFrom": "arn:aws:secretsmanager:...MYSQL_USER"
        },
        {
          "name": "MYSQL_PASSWORD",
          "valueFrom": "arn:aws:secretsmanager:...MYSQL_PASSWORD"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/multimodal-pipeline",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

3. **Register task definition**

```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

4. **Create Application Load Balancer**

```bash
aws elbv2 create-load-balancer \
  --name multimodal-alb \
  --subnets subnet-12345 subnet-67890 \
  --security-groups sg-abcdef \
  --scheme internet-facing
```

5. **Create target group**

```bash
aws elbv2 create-target-group \
  --name multimodal-targets \
  --protocol HTTP \
  --port 8000 \
  --vpc-id vpc-12345 \
  --target-type ip
```

6. **Create ECS service**

```bash
aws ecs create-service \
  --cluster multimodal-prod \
  --service-name multimodal-service \
  --task-definition multimodal-pipeline:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345,subnet-67890],securityGroups=[sg-abcdef],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=app,containerPort=8000"
```

### Phase 5: Auto-Scaling Configuration

1. **Create auto-scaling target**

```bash
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/multimodal-prod/multimodal-service \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 2 \
  --max-capacity 10
```

2. **Create scaling policy (target tracking)**

```bash
aws application-autoscaling put-scaling-policy \
  --policy-name multimodal-scale-policy \
  --service-namespace ecs \
  --resource-id service/multimodal-prod/multimodal-service \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration "TargetValue=70.0,PredefinedMetricSpecification={PredefinedMetricType=ECSServiceAverageCPUUtilization}"
```

### Phase 6: Monitoring and Logging

1. **Create CloudWatch log group**

```bash
aws logs create-log-group --log-group-name /ecs/multimodal-pipeline
```

2. **Set up CloudWatch alarms**

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name multimodal-high-cpu \
  --alarm-description "Alert if CPU > 80%" \
  --metric-name CPUUtilization \
  --namespace AWS/ECS \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --alarm-actions arn:aws:sns:...
```

### Phase 7: CI/CD Pipeline (GitHub Actions)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to ECS

on:
  push:
    tags:
      - 'v*'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: |
          docker build -t multimodal-pipeline:${{ github.ref_name }} .

      - name: Push to ECR
        run: |
          aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URI
          docker tag multimodal-pipeline:${{ github.ref_name }} $ECR_URI/multimodal-pipeline:${{ github.ref_name }}
          docker push $ECR_URI/multimodal-pipeline:${{ github.ref_name }}

      - name: Update ECS task definition
        run: |
          aws ecs update-service \
            --cluster multimodal-prod \
            --service multimodal-service \
            --force-new-deployment
```

### Phase 8: Rollback Procedure

If model v2.0 fails in production:

1. **Identify issue from CloudWatch logs**

```bash
aws logs tail /ecs/multimodal-pipeline --follow
```

2. **Revert to v1.0**

```bash
# Update task definition to use v1.0.0 image
aws ecs update-service \
  --cluster multimodal-prod \
  --service multimodal-service \
  --force-new-deployment \
  --task-definition multimodal-pipeline:1  # Points to v1.0.0
```

3. **Verify rollback**

```bash
aws ecs describe-services \
  --cluster multimodal-prod \
  --services multimodal-service | grep taskDefinition
```

### Cost Estimation (Monthly)

| Resource | Instance Type | Quantity | Monthly Cost |
| --- | --- | --- | --- |
| Fargate | 512 CPU, 1GB RAM | 2-10 | $30-150 |
| RDS | db.t3.micro | 1 | $30 |
| S3 | Storage + requests | 1TB | $23 |
| ALB | Standard | 1 | $16 |
| Data transfer | Out to internet | Per GB | $0.09/GB |
| CloudWatch | Logs + metrics | Per GB/month | $10-50 |
| **Total** |  |  | **$110-270/month** |

Use spot instances for Fargate tasks (non-critical) to reduce cost by 60-70%.

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
- Do not keep v1.0 and v2.0 running simultaneously (wastes memory).
- Instead, store both in S3 and use a live-pointer config.
- Update pointer, reload model weights, and rollback quickly if needed.
- Block promotion unless candidate metrics beat baseline thresholds defined in CI.

### Preprocessing and Model Version Compatibility

Preprocessing (tokenizer, image transforms) must be tracked alongside model versions:

- Model v2.0 trained with spaCy v3.0
- Old requests using spaCy v2.3 tokenizer can produce unstable predictions
- Solution: Deploy as a `(model_version, preprocess_version)` pair

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

Model loading (400MB total) can slow new container startup:

- Use readiness probes (for example, initialDelaySeconds: 30)
- Pre-warm models on startup
- Keep at least 2 replicas ready

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

- Latency SLO: p95 and p99 per endpoint and per model version
- Quality SLO: rolling macro-F1 on recent labeled samples
- Stability SLO: prediction entropy and class-ratio drift vs training baseline

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
- Offload heavy and batch inference to async queue workers
- Cache common model components in memory
- Separate online inference DB writes from analytics pipeline
