import argparse
from src.pipeline import Pipeline

def main():
    parser = argparse.ArgumentParser(description="Run the graphRAG pipeline.")
    parser.add_argument("--indexing", action="store_true", help="Run the indexing pipeline")
    parser.add_argument("--query_keyword", type=str, help="Query documents for indexing with keyword")
    parser.add_argument("--load_max_docs", type=int, default=10, help="Max docs to load from source")
    parser.add_argument("--querying", type=str, help="Query the graph database")
    parser.add_argument("--visualizing", action="store_true", help="Visualize the knowledge graph")
    
    args = parser.parse_args()

    pipeline = Pipeline()

    if args.indexing:
        pipeline.pipeline_indexing(query_documents=args.query_keyword, load_max_docs=args.load_max_docs)

    if args.querying:
        answer = pipeline.pipeline_querying(query=args.querying)
        print("Answer:", answer)
        
    if args.visualizing:
        pipeline.visualize_knowledge_graph(limit=100)
        
if __name__ == "__main__":
    main()