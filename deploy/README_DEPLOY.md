# Deployment Guide

## Deploy instructions

### Install Docker
```bash
sudo snap install docker
```

### Setup Docker Daemon
```bash
# 1. Create group docker
sudo groupadd docker

# 2. Add user to docker group
sudo usermod -aG docker $USER

# 3. Logout and Login again
# Or run this command to apply in current terminal:
newgrp docker
```

---

## Deploy Full Stack

### Step 1: Prepare .env file
```bash
# Copy .env.example to deploy/.env
cp .env.example deploy/.env

# Edit deploy/.env with your configuration
nano deploy/.env
```

### Step 2: Build and run
```bash
cd deploy

# Build all services
docker compose build

# Run (background mode)
docker compose up -d
```

### Step 3: Check
```bash
# Check logs
docker compose logs -f

# Check status
docker compose ps
```

### Access
- **API Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Neo4j Browser**: [http://localhost:7474](http://localhost:7474)
- **Qdrant Dashboard**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
  - Credentials: Check your `.env` file
---

## Services

| Service | Port | Description |
|---------|------|-------------|
| Neo4j Browser | 7474 | Graph DB UI |
| Neo4j Bolt | 7687 | Graph DB Connection |
| Qdrant REST | 6333 | Vector DB REST API |
| Qdrant gRPC | 6334 | Vector DB gRPC |

---

## Deploy Each Service

### Run Neo4j only
```bash
docker compose -f docker-compose.neo4j.yml up -d
```

### Run Qdrant only
```bash
docker compose -f docker-compose.qdrant.yml up -d
```

### Build and run Backend only
```bash
cd /path/to/project

# Build
docker build -f Dockerfile -t hybridrag-backend .

# Run
docker run -d \
  --name backend \
  -p 8000:8000 \
  -e NEO4J_URI=bolt://host.docker.internal:7687 \
  -e NEO4J_USERNAME=neo4j \
  -e NEO4J_PASSWORD=your_neo4j_password \
  -e API_URL_LLM=http://host.docker.internal:5000/v1/ \
  -e API_KEY=your_api_key \
  --add-host=host.docker.internal:host-gateway \
  hybridrag-backend
```

### Build and run Frontend only
```bash
cd /path/to/project

# Build
docker build -f Dockerfile.frontend -t hybridrag-frontend .

# Run
docker run -d \
  --name frontend \
  -p 3000:80 \
  hybridrag-frontend
```

---

## Management

### Stop all services
```bash
docker compose down
```

### Stop and remove data (including Neo4j data)
```bash
docker compose down -v
```

### Check logs
```bash
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f neo4j
```

### Rebuild when change code
```bash
docker compose build --no-cache
docker compose up -d
```

---

## Environment Variables

Key variables to configure in `.env`:

```env
# LLM API
API_URL_LLM=http://host.docker.internal:5000/v1/
API_KEY=your_api_key

# Neo4j
NEO4J_PASSWORD=your_neo4j_password
```
