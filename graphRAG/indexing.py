import json
import argparse
from loguru import logger
from typing import Literal

from graphRAG.config.setting import neo4j_config
from graphRAG.dataloaders.loaders import DataLoader
from graphRAG.graph.extractor import GraphExtractor
from graphRAG.graph.storage import GraphStorage

class GraphIndexing:
    def __init__(self):
        self.extractor = GraphExtractor()
        self.storage = GraphStorage(
            url=neo4j_config.url,
            username=neo4j_config.username,
            password=neo4j_config.password
        )
        
    def indexing(
            self,
            format_type: Literal["wiki", "pdf", "json"],
            data_path: str,
            save_entities: bool = False,
            save_graph_dir: str = None,
            graph_data: str = None
        ) -> None:

        if graph_data:
            with open(graph_data, "r", encoding="utf-8") as f:
                combined_graph_data = json.load(f)
            logger.info(f"Loaded graph data from {graph_data}: {len(combined_graph_data['nodes'])} nodes, "
                    f"{len(combined_graph_data['relationships'])} relationships")
            
            self.storage.store(combined_graph_data, clear_old_graph=True)
            logger.info("Graph data uploaded to Neo4j successfully!")
            return
        
        loader = DataLoader(format_type=format_type)
        raw_docs = loader.load(file_path=data_path, save_to=True)
        
        all_nodes = []
        all_relationships = []
        batch_size = 5

        for i, doc in enumerate(raw_docs):
            logger.info(f"Processing document {i + 1}/{len(raw_docs)} with length {len(doc.page_content)} characters")
            graph_data = self.extractor.extract_graph(doc)
            all_nodes.extend(graph_data["nodes"])
            all_relationships.extend(graph_data["relationships"])
            logger.info(f"Extracted {len(all_nodes)} nodes and {len(all_relationships)} relationships")

            if (i + 1) % batch_size == 0 or (i + 1) == len(raw_docs):
                    combined_graph_data = {
                        "nodes": all_nodes,
                        "relationships": all_relationships
                    }
                    logger.info(f"Storing batch at document {i + 1}: {len(all_nodes)} nodes, {len(all_relationships)} relationships")
                    self.storage.store(combined_graph_data, clear_old_graph=False)


        if all_nodes or all_relationships:
            combined_graph_data = {
                "nodes": all_nodes,
                "relationships": all_relationships
            }
            self.storage.store(combined_graph_data, clear_old_graph=False)

            logger.info(f"GraphRAG extracted {len(all_nodes)} nodes and {len(all_relationships)} relationships")
            print("Final store:", combined_graph_data)

            if save_entities:
                if save_graph_dir is None:
                    save_graph_dir = "data/entities_extracted/node_relationship.json"
                    with open(save_graph_dir, "w", encoding="utf-8") as f:
                        json.dump(combined_graph_data, f, ensure_ascii=False, indent=4)
                    logger.info(f"Entities data saved to {save_graph_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Crawl data từ wikipedia, hoặc load từ pdf, json
    parser.add_argument("--format_type", type=str, default="json", 
                        help="Choose format type to load data: wikipedia(crawl) / pdf / json")
    parser.add_argument("--data_path", type=str, default="data/data_loaded/elizabeth_i.json", 
                        help="path to raw data (wikipedia, pdf, json... )")
    
    # Sau khi entities được extract thì có thể save lại dưới dạng JSON
    parser.add_argument("--save_entities", type=bool, default=True, 
                        help="Save extracted entities (nodes and relationships) to JSON file")
    parser.add_argument("--save_graph_dir", type=str, default="data/entities_extracted/node_relationship.json", 
                        help="Path to save extracted entities (nodes and relationships) to JSON file")
    
    # Nếu có graph_data thì upsert trực tiếp lên Neo4j, không cần crawl hoặc extract
    parser.add_argument("--graph_data", type=str, default="data/node_relationship.json", 
                        help="Load entities graph data from JSON file")
    args = parser.parse_args()

    graph_indexing = GraphIndexing()
    graph_indexing.indexing(
        format_type=args.format_type,
        data_path=args.data_path,
        save_entities=args.save_entities,
        save_graph_dir=args.save_graph_dir,
        graph_data=args.graph_data,
    )
