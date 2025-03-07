# GraphRAG
GraphRAG implements a Retrieval-Augmented Generation (RAG) system using knowledge graphs. It extracts entities and relationships from text data (e.g., Wikipedia articles), stores them in a Neo4j graph database, and provides visualization capabilities. This project is served for researchers and developers interested in NLP.

- GraphRAG includes 3 main function:
    + **indexing**: Processing provided data and extract entities into nodes and relationships
    + **visualizing**: Visualizing these nodes and relationships into a knowledge graph in form of 2D and 3D
    + **querying**: Retrieving from the knowledge graph to get relative information about provided data and response to User.

## Features
- **Dataloader**: Crawl text data from Wikipedia or load from pdf/json file.
- **Entity and Relationship Extraction**: Utilize a LLM Qwen2.5-72B to extract structured data from unstructured text.
- **Graph Storage**: Store extracted entities and relationships in a Neo4j database.
- **Visualization**: Query and visualize the knowledge graph in 2D and 3D.
- **Retrieval**: Using provided ID to retrieve from graph database and response by Qwen2.5-72B 

## Installation
1. **Requirements**:
    ```sh
    pip install -r requirements.txt
    ```

2. **Execution**:
    - **Indexing**
    ```sh
    bash scripts/indexing.py
    ```

    - **Querying**
    ```sh
    bash scripts/querying.py
    ```

    - **Visualizing**
    ```sh
    bash scripts/visualizing.py
    ```

3. **Note**:
    - Input JSON format must be like this form:
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
- Đa dạng đầu vào: json, pdf, wikipedia crawl. ("Done")
- Biểu diễn Graph đa dạng và đẹp hơn (nhiều kiểu biểu diễn: 2D, 3D...)  ("Done")
- Thêm module Retrieval LLM: đầu vào là ID để retrieval ("Done")
- sửa lại logic phần extractor query đang bị lấn cấn: giữa file pdf và index ("Done")

- Thêm đầu vào dạng relational db
- Cải tiến thêm xử lý logic với đầu vào là câu hỏi của User -> lấy ra ID từ câu hỏi để retrieval -> đưa output List graph schemas cho LLM và câu hỏi để LLM response