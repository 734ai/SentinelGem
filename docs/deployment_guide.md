# SentinelGem Deployment Guide
# Author: Muzan Sano
# Version: 1.0

## Production Deployment Guide

This comprehensive deployment guide covers everything needed to deploy SentinelGem in production environments, from single-server installations to enterprise-scale distributed deployments.

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Infrastructure Requirements](#infrastructure-requirements)
3. [Deployment Options](#deployment-options)
4. [Security Configuration](#security-configuration)
5. [Monitoring and Observability](#monitoring-and-observability)
6. [Maintenance and Updates](#maintenance-and-updates)
7. [Troubleshooting](#troubleshooting)

---

## Deployment Overview

### Deployment Architecture

SentinelGem can be deployed in multiple configurations to meet different organizational needs:

```
Production Deployment Options:

1. Single Server Deployment
   ┌─────────────────────────────────┐
   │         Single Server           │
   │  ┌─────────────────────────────┐ │
   │  │      SentinelGem App        │ │
   │  │   • Web UI                  │ │
   │  │   • API Server              │ │
   │  │   • AI Models               │ │
   │  │   • Database                │ │
   │  └─────────────────────────────┘ │
   └─────────────────────────────────┘

2. Multi-Server Deployment
   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
   │   Web Tier  │  │   App Tier  │  │  Data Tier  │
   │             │  │             │  │             │
   │ • Load      │  │ • API       │  │ • Database  │
   │   Balancer  │  │ • AI Models │  │ • Cache     │
   │ • Web UI    │  │ • Workers   │  │ • Storage   │
   └─────────────┘  └─────────────┘  └─────────────┘

3. Containerized Deployment
   ┌─────────────────────────────────────────────────────┐
   │                 Kubernetes Cluster                  │
   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
   │  │   Web Pods  │  │   API Pods  │  │ Worker Pods │ │
   │  │             │  │             │  │             │ │
   │  │ • UI        │  │ • FastAPI   │  │ • Analysis  │ │
   │  │ • Nginx     │  │ • Auth      │  │ • Models    │ │
   │  └─────────────┘  └─────────────┘  └─────────────┘ │
   └─────────────────────────────────────────────────────┘

4. Cloud-Native Deployment
   ┌─────────────────────────────────────────────────────┐
   │                 Cloud Platform                      │
   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
   │  │  Auto-Scale │  │  Managed    │  │  AI/ML      │ │
   │  │  Instances  │  │  Database   │  │  Services   │ │
   │  │             │  │             │  │             │ │
   │  │ • EC2/VM    │  │ • RDS/SQL   │  │ • SageMaker │ │
   │  │ • ELB/ALB   │  │ • Redis     │  │ • Bedrock   │ │
   │  └─────────────┘  └─────────────┘  └─────────────┘ │
   └─────────────────────────────────────────────────────┘
```

---

## Infrastructure Requirements

### Hardware Requirements

#### Minimum Production Requirements
```yaml
single_server_minimum:
  cpu: "8 cores (Intel Xeon or AMD EPYC)"
  memory: "32GB RAM"
  storage: "500GB SSD"
  network: "1Gbps connection"
  gpu: "Optional: NVIDIA RTX 4060 or better"

multi_server_minimum:
  web_tier:
    cpu: "4 cores"
    memory: "8GB RAM"
    storage: "100GB SSD"
  
  app_tier:
    cpu: "16 cores"
    memory: "64GB RAM"
    storage: "1TB SSD"
    gpu: "NVIDIA RTX 4080 or A4000"
  
  data_tier:
    cpu: "8 cores"
    memory: "32GB RAM"
    storage: "2TB SSD (RAID 1)"
```

#### Recommended Production Requirements
```yaml
high_availability_setup:
  load_balancer:
    cpu: "4 cores"
    memory: "8GB RAM"
    storage: "100GB SSD"
    instances: 2
  
  web_servers:
    cpu: "8 cores"
    memory: "16GB RAM"
    storage: "200GB SSD"
    instances: 3
  
  api_servers:
    cpu: "16 cores"
    memory: "64GB RAM"
    storage: "500GB SSD"
    gpu: "NVIDIA A5000 or better"
    instances: 2
  
  database_cluster:
    cpu: "12 cores"
    memory: "64GB RAM"
    storage: "4TB SSD (RAID 10)"
    instances: 3
```

### Software Requirements

#### Operating System
```yaml
supported_os:
  linux:
    - "Ubuntu 20.04 LTS or 22.04 LTS"
    - "CentOS 8+ or Rocky Linux 8+"
    - "Red Hat Enterprise Linux 8+"
    - "Amazon Linux 2"
  
  containers:
    - "Docker 20.10+"
    - "Kubernetes 1.24+"
    - "OpenShift 4.10+"
```

#### Runtime Dependencies
```yaml
system_packages:
  python: "3.11+"
  nodejs: "18.x LTS"
  nginx: "1.20+"
  postgresql: "14+"
  redis: "6.2+"
  docker: "20.10+"

python_packages:
  torch: "2.0.0+"
  transformers: "4.35.0+"
  fastapi: "0.104.0+"
  streamlit: "1.28.0+"
  
gpu_support:
  cuda: "11.8 or 12.1"
  cudnn: "8.6+"
  nvidia_driver: "525+"
```

---

## Deployment Options

### Option 1: Docker Compose Deployment

This is the recommended approach for small to medium deployments.

#### Step 1: Prepare Environment
```bash
# Create deployment directory
mkdir -p /opt/sentinelgem
cd /opt/sentinelgem

# Clone repository
git clone https://github.com/734ai/SentinelGem.git
cd SentinelGem

# Create environment file
cp .env.example .env
```

#### Step 2: Configure Environment
```bash
# .env file configuration
cat > .env << EOF
# Application Configuration
ENVIRONMENT=production
SECRET_KEY=your-super-secret-key-here
DEBUG=false

# Database Configuration
DATABASE_URL=postgresql://sentinelgem:password@db:5432/sentinelgem
REDIS_URL=redis://redis:6379/0

# AI Model Configuration
GEMMA_MODEL=google/gemma-3n-9b
WHISPER_MODEL=base
MODEL_CACHE_DIR=/app/models
GPU_ENABLED=true

# Security Configuration
JWT_SECRET_KEY=another-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=60

# External Services
SMTP_HOST=smtp.yourdomain.com
SMTP_PORT=587
SMTP_USERNAME=sentinelgem@yourdomain.com
SMTP_PASSWORD=smtp-password

# Monitoring
PROMETHEUS_ENABLED=true
METRICS_PORT=9090
EOF
```

#### Step 3: Docker Compose Configuration
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - GEMMA_MODEL=${GEMMA_MODEL}
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on:
      - db
      - redis
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  web:
    build:
      context: .
      dockerfile: Dockerfile.web
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://app:8000
    depends_on:
      - app
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - app
      - web
    restart: unless-stopped

  db:
    image: postgres:14
    environment:
      - POSTGRES_DB=sentinelgem
      - POSTGRES_USER=sentinelgem
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    restart: unless-stopped

  redis:
    image: redis:alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

#### Step 4: Deploy with Docker Compose
```bash
# Download models
python scripts/download_models.py

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f
```

### Option 2: Kubernetes Deployment

For enterprise-scale deployments with high availability.

#### Step 1: Kubernetes Manifests

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: sentinelgem
---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: sentinelgem-config
  namespace: sentinelgem
data:
  ENVIRONMENT: "production"
  GEMMA_MODEL: "google/gemma-3n-9b"
  WHISPER_MODEL: "base"
  GPU_ENABLED: "true"
---
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: sentinelgem-secrets
  namespace: sentinelgem
type: Opaque
stringData:
  SECRET_KEY: "your-super-secret-key-here"
  DATABASE_URL: "postgresql://sentinelgem:password@postgres:5432/sentinelgem"
  JWT_SECRET_KEY: "another-secret-key"
---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentinelgem-api
  namespace: sentinelgem
spec:
  replicas: 2
  selector:
    matchLabels:
      app: sentinelgem-api
  template:
    metadata:
      labels:
        app: sentinelgem-api
    spec:
      containers:
      - name: api
        image: sentinelgem/api:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: sentinelgem-config
        - secretRef:
            name: sentinelgem-secrets
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: models
          mountPath: /app/models
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: sentinelgem-models
---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: sentinelgem-api-service
  namespace: sentinelgem
spec:
  selector:
    app: sentinelgem-api
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sentinelgem-ingress
  namespace: sentinelgem
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - sentinelgem.yourdomain.com
    secretName: sentinelgem-tls
  rules:
  - host: sentinelgem.yourdomain.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: sentinelgem-api-service
            port:
              number: 8000
      - path: /
        pathType: Prefix
        backend:
          service:
            name: sentinelgem-web-service
            port:
              number: 8501
---
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sentinelgem-api-hpa
  namespace: sentinelgem
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sentinelgem-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### Step 2: Deploy to Kubernetes
```bash
# Apply manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n sentinelgem
kubectl get services -n sentinelgem
kubectl get ingress -n sentinelgem

# Scale deployment
kubectl scale deployment sentinelgem-api --replicas=4 -n sentinelgem

# Update deployment
kubectl set image deployment/sentinelgem-api api=sentinelgem/api:v1.1.0 -n sentinelgem
```

### Option 3: Cloud Provider Deployment

#### AWS Deployment with EKS

```bash
# Create EKS cluster
eksctl create cluster \
  --name sentinelgem-cluster \
  --region us-west-2 \
  --nodegroup-name gpu-nodes \
  --node-type p3.2xlarge \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 5 \
  --managed

# Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name sentinelgem-cluster

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.13.0/nvidia-device-plugin.yml

# Deploy application
kubectl apply -f k8s/
```

#### Azure Deployment with AKS

```bash
# Create resource group
az group create --name sentinelgem-rg --location eastus

# Create AKS cluster with GPU support
az aks create \
  --resource-group sentinelgem-rg \
  --name sentinelgem-cluster \
  --node-count 2 \
  --enable-addons monitoring \
  --node-vm-size Standard_NC6s_v3 \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group sentinelgem-rg --name sentinelgem-cluster

# Install GPU drivers
kubectl apply -f https://raw.githubusercontent.com/Azure/aks-engine/master/examples/addons/nvidia-device-plugin/nvidia-device-plugin.yaml
```

#### Google Cloud Deployment with GKE

```bash
# Create GKE cluster with GPU support
gcloud container clusters create sentinelgem-cluster \
  --zone us-central1-a \
  --machine-type n1-standard-4 \
  --accelerator type=nvidia-tesla-k80,count=1 \
  --enable-autorepair \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 5 \
  --num-nodes 2

# Get credentials
gcloud container clusters get-credentials sentinelgem-cluster --zone us-central1-a

# Install NVIDIA drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

---

## Security Configuration

### SSL/TLS Configuration

#### Nginx SSL Configuration
```nginx
# /etc/nginx/sites-available/sentinelgem
server {
    listen 80;
    server_name sentinelgem.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name sentinelgem.yourdomain.com;

    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/sentinelgem.crt;
    ssl_certificate_key /etc/nginx/ssl/sentinelgem.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options DENY always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';" always;

    # API Proxy
    location /api {
        proxy_pass http://sentinelgem-api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Rate limiting
        limit_req zone=api burst=20 nodelay;
        
        # Timeout settings
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    # Web UI Proxy
    location / {
        proxy_pass http://sentinelgem-web:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support for Streamlit
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}

# Rate limiting configuration
http {
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=general:10m rate=1r/s;
}
```

### Authentication and Authorization

#### JWT Configuration
```python
# security/auth.py
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import os

# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def create_access_token(data: dict, expires_delta: timedelta = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
        return username
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

# Role-based access control
def require_role(required_role: str):
    """Decorator for role-based access control"""
    def role_checker(current_user: str = Depends(verify_token)):
        user_role = get_user_role(current_user)  # Implement this function
        if user_role != required_role and user_role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return role_checker
```

### Database Security

#### PostgreSQL Security Configuration
```sql
-- Create dedicated user
CREATE USER sentinelgem WITH PASSWORD 'strong-password-here';

-- Create database
CREATE DATABASE sentinelgem OWNER sentinelgem;

-- Grant minimum required permissions
GRANT CONNECT ON DATABASE sentinelgem TO sentinelgem;
GRANT USAGE ON SCHEMA public TO sentinelgem;
GRANT CREATE ON SCHEMA public TO sentinelgem;

-- Enable SSL
ALTER SYSTEM SET ssl = on;
ALTER SYSTEM SET ssl_cert_file = '/etc/ssl/certs/server.crt';
ALTER SYSTEM SET ssl_key_file = '/etc/ssl/private/server.key';

-- Configure authentication
-- In pg_hba.conf:
# hostssl sentinelgem sentinelgem 0.0.0.0/0 md5
```

#### Connection Pool Configuration
```python
# database/pool.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
import os

DATABASE_URL = os.getenv("DATABASE_URL")

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False
)
```

### Network Security

#### Firewall Configuration
```bash
# UFW configuration (Ubuntu)
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow from 10.0.0.0/8 to any port 5432  # Database access
ufw enable

# Fail2ban configuration
cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[nginx-http-auth]
enabled = true
port = http,https
logpath = /var/log/nginx/error.log

[nginx-limit-req]
enabled = true
port = http,https
logpath = /var/log/nginx/error.log
maxretry = 10
EOF
```

---

## Monitoring and Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "sentinelgem_rules.yml"

scrape_configs:
  - job_name: 'sentinelgem-api'
    static_configs:
      - targets: ['app:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'sentinelgem-web'
    static_configs:
      - targets: ['web:8501']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Custom Metrics

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

# Define metrics
analysis_counter = Counter(
    'sentinelgem_analyses_total',
    'Total number of analyses performed',
    ['analysis_type', 'threat_detected']
)

analysis_duration = Histogram(
    'sentinelgem_analysis_duration_seconds',
    'Time spent on analysis',
    ['analysis_type']
)

model_memory_usage = Gauge(
    'sentinelgem_model_memory_bytes',
    'Memory usage by AI models',
    ['model_name']
)

threat_confidence = Histogram(
    'sentinelgem_threat_confidence',
    'Distribution of threat confidence scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Metric collection decorator
def track_analysis_metrics(analysis_type: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                analysis_counter.labels(
                    analysis_type=analysis_type,
                    threat_detected=str(result.threat_detected).lower()
                ).inc()
                threat_confidence.observe(result.confidence_score)
                return result
            finally:
                analysis_duration.labels(analysis_type=analysis_type).observe(
                    time.time() - start_time
                )
        return wrapper
    return decorator
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "SentinelGem Operations Dashboard",
    "panels": [
      {
        "title": "Analysis Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(sentinelgem_analyses_total[5m])",
            "legendFormat": "Analyses per second"
          }
        ]
      },
      {
        "title": "Threat Detection Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(sentinelgem_analyses_total{threat_detected=\"true\"}[5m]) / rate(sentinelgem_analyses_total[5m]) * 100",
            "legendFormat": "Threat detection %"
          }
        ]
      },
      {
        "title": "Analysis Duration",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(sentinelgem_analysis_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(sentinelgem_analysis_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Model Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "sentinelgem_model_memory_bytes",
            "legendFormat": "{{model_name}}"
          }
        ]
      }
    ]
  }
}
```

### Application Logging

```python
# logging/config.py
import logging
import sys
from pythonjsonlogger import jsonlogger

def setup_logging():
    """Configure application logging"""
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # JSON formatter for structured logging
    json_formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(json_formatter)
    logger.addHandler(console_handler)
    
    # File handler for errors
    file_handler = logging.FileHandler('/app/logs/sentinelgem.log')
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(json_formatter)
    logger.addHandler(file_handler)
    
    return logger

# Usage in application
logger = setup_logging()

def analyze_content(content: str):
    logger.info("Starting content analysis", extra={
        "content_length": len(content),
        "analysis_id": generate_analysis_id()
    })
    
    try:
        result = perform_analysis(content)
        logger.info("Analysis completed", extra={
            "threat_detected": result.threat_detected,
            "confidence": result.confidence_score,
            "analysis_time": result.processing_time
        })
        return result
    except Exception as e:
        logger.error("Analysis failed", extra={
            "error": str(e),
            "content_preview": content[:100]
        })
        raise
```

---

## Maintenance and Updates

### Automated Backup Strategy

```bash
#!/bin/bash
# scripts/backup.sh

# Configuration
BACKUP_DIR="/opt/backups/sentinelgem"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Create backup directory
mkdir -p $BACKUP_DIR

# Database backup
docker exec sentinelgem-db pg_dump -U sentinelgem sentinelgem | gzip > $BACKUP_DIR/database_$DATE.sql.gz

# Application data backup
tar -czf $BACKUP_DIR/app_data_$DATE.tar.gz /opt/sentinelgem/models /opt/sentinelgem/logs

# Configuration backup
tar -czf $BACKUP_DIR/config_$DATE.tar.gz /opt/sentinelgem/config

# Clean old backups
find $BACKUP_DIR -name "*.gz" -mtime +$RETENTION_DAYS -delete

# Verify backups
if [ $? -eq 0 ]; then
    echo "Backup completed successfully: $DATE"
else
    echo "Backup failed: $DATE"
    exit 1
fi
```

### Update Process

```bash
#!/bin/bash
# scripts/update.sh

# Pre-update checks
echo "Starting SentinelGem update process..."

# Check system health
docker-compose -f docker-compose.prod.yml exec app python -c "
import requests
response = requests.get('http://localhost:8000/health')
assert response.status_code == 200
print('Health check passed')
"

# Create backup
./scripts/backup.sh

# Pull latest images
docker-compose -f docker-compose.prod.yml pull

# Update application
docker-compose -f docker-compose.prod.yml up -d --no-deps app web

# Run database migrations
docker-compose -f docker-compose.prod.yml exec app alembic upgrade head

# Verify deployment
sleep 30
docker-compose -f docker-compose.prod.yml exec app python -c "
import requests
response = requests.get('http://localhost:8000/health')
assert response.status_code == 200
print('Update verification passed')
"

echo "Update completed successfully"
```

### Health Checks

```python
# health/checks.py
from typing import Dict, Any
import psutil
import redis
import sqlalchemy
from datetime import datetime

class HealthChecker:
    def __init__(self):
        self.checks = {
            'database': self.check_database,
            'redis': self.check_redis,
            'models': self.check_models,
            'disk_space': self.check_disk_space,
            'memory': self.check_memory
        }
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        for check_name, check_func in self.checks.items():
            try:
                check_result = check_func()
                results['checks'][check_name] = {
                    'status': 'healthy',
                    'details': check_result
                }
            except Exception as e:
                results['checks'][check_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                results['overall_status'] = 'unhealthy'
        
        return results
    
    def check_database(self) -> Dict[str, Any]:
        """Check database connectivity"""
        engine = sqlalchemy.create_engine(DATABASE_URL)
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            return {'connection': 'ok', 'query_result': result.scalar()}
    
    def check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity"""
        r = redis.from_url(REDIS_URL)
        r.ping()
        return {'connection': 'ok', 'memory_usage': r.info()['used_memory_human']}
    
    def check_models(self) -> Dict[str, Any]:
        """Check AI model availability"""
        model_hub = get_model_hub()
        return {
            'loaded_models': list(model_hub.loaded_models.keys()),
            'model_cache_usage': len(model_hub.loaded_models)
        }
    
    def check_disk_space(self) -> Dict[str, Any]:
        """Check disk space usage"""
        disk_usage = psutil.disk_usage('/')
        return {
            'total_gb': round(disk_usage.total / (1024**3), 2),
            'used_gb': round(disk_usage.used / (1024**3), 2),
            'free_gb': round(disk_usage.free / (1024**3), 2),
            'usage_percent': round((disk_usage.used / disk_usage.total) * 100, 2)
        }
    
    def check_memory(self) -> Dict[str, Any]:
        """Check memory usage"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': round(memory.total / (1024**3), 2),
            'used_gb': round(memory.used / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'usage_percent': memory.percent
        }
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: High Memory Usage
```bash
# Symptoms
- Application becomes slow
- Out of memory errors
- High swap usage

# Diagnosis
docker stats
free -h
ps aux --sort=-%mem | head

# Solutions
# 1. Increase model cache size limits
# 2. Use smaller model variants
# 3. Enable model quantization
# 4. Add more RAM or swap

# Configuration fix
# In docker-compose.yml
services:
  app:
    deploy:
      resources:
        limits:
          memory: 16G
    environment:
      - MODEL_CACHE_SIZE=2
      - ENABLE_QUANTIZATION=true
```

#### Issue 2: Slow Analysis Performance
```bash
# Symptoms
- Analysis takes >5 seconds
- API timeouts
- Queue backlog

# Diagnosis
# Check GPU utilization
nvidia-smi

# Check CPU usage
htop

# Check analysis metrics
curl http://localhost:9090/api/v1/query?query=sentinelgem_analysis_duration_seconds

# Solutions
# 1. Enable GPU acceleration
# 2. Use smaller models for real-time analysis
# 3. Implement batch processing
# 4. Scale horizontally

# Configuration fix
services:
  app:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - BATCH_SIZE=4
```

#### Issue 3: Database Connection Issues
```bash
# Symptoms
- Connection timeouts
- "Too many connections" errors
- Slow queries

# Diagnosis
# Check database connections
docker exec sentinelgem-db psql -U sentinelgem -c "SELECT * FROM pg_stat_activity;"

# Check connection pool
curl http://localhost:8000/metrics | grep db_pool

# Solutions
# 1. Increase connection pool size
# 2. Optimize database queries
# 3. Add read replicas
# 4. Enable connection pooling

# Configuration fix
# In application config
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
DATABASE_POOL_RECYCLE=3600
```

### Performance Tuning

#### Model Optimization
```python
# performance/optimization.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def optimize_model_for_production(model_name: str):
    """Optimize model for production deployment"""
    
    # Load model with optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use half precision
        device_map="auto",          # Automatic device placement
        low_cpu_mem_usage=True,     # Reduce CPU memory usage
        use_cache=True             # Enable KV cache
    )
    
    # Apply optimizations
    if torch.cuda.is_available():
        # Compile model for faster inference (PyTorch 2.0+)
        model = torch.compile(model, mode="reduce-overhead")
        
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    return model

def batch_inference_optimization(texts: list, model, tokenizer, batch_size: int = 4):
    """Optimize inference for batch processing"""
    
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize batch with padding
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        
        # Batch inference
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode batch results
        batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(batch_results)
    
    return results
```

### Disaster Recovery

#### Recovery Procedures
```bash
#!/bin/bash
# scripts/disaster_recovery.sh

# Full system recovery procedure

echo "Starting disaster recovery process..."

# 1. Stop all services
docker-compose -f docker-compose.prod.yml down

# 2. Restore from backup
BACKUP_DATE=$1
if [ -z "$BACKUP_DATE" ]; then
    echo "Usage: $0 <backup_date>"
    echo "Available backups:"
    ls -la /opt/backups/sentinelgem/
    exit 1
fi

# 3. Restore database
echo "Restoring database..."
gunzip -c /opt/backups/sentinelgem/database_$BACKUP_DATE.sql.gz | \
    docker exec -i sentinelgem-db psql -U sentinelgem sentinelgem

# 4. Restore application data
echo "Restoring application data..."
tar -xzf /opt/backups/sentinelgem/app_data_$BACKUP_DATE.tar.gz -C /

# 5. Restore configuration
echo "Restoring configuration..."
tar -xzf /opt/backups/sentinelgem/config_$BACKUP_DATE.tar.gz -C /

# 6. Start services
echo "Starting services..."
docker-compose -f docker-compose.prod.yml up -d

# 7. Verify recovery
sleep 60
./scripts/health_check.sh

echo "Disaster recovery completed"
```

---

This deployment guide provides comprehensive instructions for deploying SentinelGem in production environments. Follow the security best practices and monitoring guidelines to ensure a robust and secure deployment.

For specific deployment scenarios or troubleshooting assistance, please refer to the developer documentation or contact the support team.
