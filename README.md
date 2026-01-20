# Hybrid Knowledge Graph RAG System

A Retrieval-Augmented Generation (RAG) system that combines **Knowledge Graph** structured data with **Vector Database** (Hybrid Semantic Search). This approach leverages the precision of graph relationships and the breadth of unstructured text to provide accurate, context-aware responses.

## Overview

HybridRAG solves the limitations of standalone vector search by integrating structured knowledge. The system extracts entities and relationships from text into a Neo4j graph while simultaneously maintaining a Hybrid Search Index (SPLADE + Qdrant Dense Vector) for document chunks.

The retrieval pipeline uses a sophisticated strategy:
- **Graph RAG**: Structured entity relationships from Neo4j with **Fuzzy Search** (typo tolerance) capabilities.
- **Vector RAG**: Uses Reciprocal Rank Fusion (RRF) to combine sparse search (SPLADE) and semantic search (Dense Vector).
- **Query Analysis**: LLM-powered entity extraction and query normalization.
- **Smart Exclusion**: Handles queries like "Besides X, what else..." by excluding specific entities.

## Key Features

### Advanced Retrieval
- **Hybrid Retrieval**: Combines graph triples and semantic chunks for comprehensive context.
- **Fuzzy Entity Search**: Uses Neo4j Fulltext Index with Lucene to find entities even with typos (e.g., "Elizabth" -> "Elizabeth").
    - Strategy: Exact Match → Fuzzy Match (Edit Distance) → Substring Match.
- **SPLADE + Dense Fusion**: Hybrid search using both neural sparse (lexical expansion) and semantic matching with Reciprocal Rank Fusion (RRF).

### Chunking
- **Two-Phase Chunking**: 
    1. **Structure-Aware**: Recognizes Vietnamese document structures (Chương, Phần, Mục) and Markdown.
    2. **Recursive Fallback**: Handles oversized chunks using OpenAI-compatible tokenization.
- **Dynamic Pattern Matching**: Prioritizes specific patterns (e.g., "1.1.1") over general ones for accurate hierarchy.

## System Workflow
![HybridRag](assets/hybrid-rag.png)

## Installation

### Prerequisites
- Neo4j Database (local or remote) with APOC installed
- Qdrant Vector Database
- OpenAI-compatible API endpoint

### Setup
1. **Using uv (Recommended)**
   ```bash
   # Install uv (Remember to restart terminal)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Create conda
   conda create -n synthetic python==3.12 -y
   conda activate synthetic

   # Install dependencies
   uv pip install -e .
   ```

2. **Using pip (conda env)**
   ```bash
   conda create -n synthetic python==3.12.8
   conda activate synthetic
   pip install -e .
   ```

3. **Configure Settings**
    ```bash
    cp .env.example .env
    # Edit .env with your Neo4j, Qdrant, and LLM credentials
    ```

## Usage

### Build and Deploy

Watch [Deploy](deploy/README_DEPLOY.md) for more details

### Start the API Server

The system is exposed via a production-ready FastAPI server:

```bash
python run.py
```

- **API Documentation**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/api/v1/health`

### Key Endpoints

- **Indexing**: `/api/v1/indexing` - Upload and process documents.
- **Querying**: `/api/v1/query` - Ask questions to the RAG system.
- **Graph Visualization**: `/api/v1/graph` - Retrieve graph data for visualization.
