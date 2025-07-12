# GraphRAG
GraphRAG implements a Retrieval-Augmented Generation (RAG) system using knowledge graphs. It extracts entities and relationships from text data (e.g., Wikipedia articles), embedding information with Sentence-Transformers, stores them in a Neo4j graph database, and provides visualization capabilities. This project is served for researchers and developers interested in NLP.

- GraphRAG includes 3 main function:
    + **Indexing**: Processing provided data and extract entities into nodes (with embeddings for retrieval) and relationships
    + **Retrieval**: Retrieving with Vector search with model "all-MiniLM-L6-v2" from the knowledge graph to get relative information about provided data and response to user.
    + **Visualization**: Visualizing these nodes and relationships into a knowledge graph in form of 2D and 3D

## Features
- **Dataloader**: Crawl text data from Wikipedia or load from pdf/json file.
- **Entity and Relationship Extraction**: Utilize a LLM Qwen2.5-72B to extract structured data from unstructured text.
- **Graph Storage**: Store extracted entities and relationships in a Neo4j database.
- **Visualization**: Query and visualize the knowledge graph in 2D and 3D.
- **Retrieval**: Using Vector search to retrieve from graph database, provide related information to LLM and response by Qwen2.5-72B 

## Installation
1. **Requirements**:
    ```sh
    conda create -n graphrag python==3.11
    poetry install
    ```

2. **Execution**:
    - **Indexing**
    Extract Entities and Relationships, then store in Neo4j:
    ```sh
    bash scripts/indexing.sh
    ```

    - **Retrieval**
    Retrieval related information from query and response by LLM: 
    ```sh
    bash scripts/retrieval.sh
    ```

    - **Visualization**
    Visualization Nodes and Relationships from Neo4j into Graph:
    ```sh
    bash scripts/visualization.sh
    ```

3. **Note**:
    - If input is JSON, then format must be like this form, else it will crawl data from wikipedia:
    ```json
    [
        {
            "page_content": "context",
            "metadata": {
                "abc": "xyz",
    
            }
        }
    ]
    ```

## Future plans
- Multiple inputs: json, pdf, wikipedia crawl. ("Done")
- Visualize Graph in form of 2D, 3D...  ("Done")
- Module Retrieval LLM: Retrieval by keyword or query of user ("Done")
- Further improvement of logic processing with input is the question of User -> use Retrieval model to retrieve related data -> Output is List schemas for LLM and query for LLM response (Done)