# GraphRAG Docker Deployment

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- OpenAI API key (or compatible endpoint)

### Setup

1. **Create environment file**:
```bash
cp .env.example .env
# Edit .env with your API credentials
```

2. **Build and run with Docker Compose**:
```bash
docker-compose up -d
```

This will start:
- Neo4j database on ports 7474 (HTTP) and 7687 (Bolt)
- GraphRAG Streamlit UI on port 8501

3. **Access the application**:
- Streamlit UI: http://localhost:8501
- Neo4j Browser: http://localhost:7474 (username: neo4j, password: password123)

### Docker Commands

**Stop services**:
```bash
docker-compose down
```

**View logs**:
```bash
docker-compose logs -f graphrag
docker-compose logs -f neo4j
```

**Rebuild after code changes**:
```bash
docker-compose up -d --build
```

**Remove all data (reset)**:
```bash
docker-compose down -v
```

## Manual Docker Build

If you prefer to build without Docker Compose:

```bash
# Build image
docker build -t graphrag:latest .

# Run Neo4j
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:5.15.0

# Run GraphRAG
docker run -d \
  --name graphrag \
  -p 8501:8501 \
  --link neo4j \
  -e API_URL=your_api_url \
  -e API_KEY=your_api_key \
  -e NEO4J_URL=bolt://neo4j:7687 \
  -e NEO4J_USERNAME=neo4j \
  -e NEO4J_PASSWORD=password123 \
  -v $(pwd)/data:/app/data \
  graphrag:latest
```
