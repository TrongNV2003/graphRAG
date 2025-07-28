import os
import json
import argparse
from loguru import logger
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer

from graphRAG.services.storage import GraphStorage
from graphRAG.dataloaders.loaders import DataLoader
from graphRAG.services.extractor import GraphExtractorLLM
from graphRAG.config.setting import neo4j_config, embed_config

class GraphIndexing:
    def __init__(self):
        self.loader = DataLoader()
        self.extractor = GraphExtractorLLM()
        self.storage = GraphStorage(
            url=neo4j_config.url,
            username=neo4j_config.username,
            password=neo4j_config.password
        )
        self.embedder = SentenceTransformer(embed_config.embedder_model)
        self.output_dir = "graphRAG/data/entities_extracted"

    def _get_embeddings(self, node: Dict) -> List[float]:
        text = f"{node['id']} {node.get('role', '')}".strip()
        return self.embedder.encode(text).tolist()

    def indexing(
            self,
            data_path: Optional[str] = None,
            save_graph_path: Optional[str] = None,
        ) -> None:

        if save_graph_path and os.path.exists(save_graph_path):
            with open(save_graph_path, "r", encoding="utf-8") as f:
                combined_graph_data = json.load(f)
            logger.info(f"Loaded graph data from {save_graph_path}: {len(combined_graph_data['nodes'])} nodes, "
                    f"{len(combined_graph_data['relationships'])} relationships")
            
            for node in combined_graph_data["nodes"]:
                node["embedding"] = self._get_embeddings(node)
            
            self.storage.store(combined_graph_data, clear_old_graph=True)
            logger.info("Graph data uploaded to Neo4j successfully!")
            return
        
        
        raw_docs = self.loader.load(file_path=data_path, save_to=True)
        
        all_nodes = []
        all_relationships = []
        batch_size = 3

        for i, doc in enumerate(raw_docs):
            logger.info(f"Processing document {i + 1}/{len(raw_docs)} with length {len(doc.page_content)} characters")
            graph_data_path = self.extractor.call(doc)

            for node in graph_data_path["nodes"]:
                node["embedding"] = self._get_embeddings(node)

            all_nodes.extend(graph_data_path["nodes"])
            all_relationships.extend(graph_data_path["relationships"])
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
            
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            if save_graph_path:
                with open(save_graph_path, "w", encoding="utf-8") as f:
                    json.dump(combined_graph_data, f, ensure_ascii=False, indent=4)
                logger.info(f"Entities graph saved to {save_graph_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Crawl data từ wikipedia, hoặc load từ pdf, json
    parser.add_argument("--data_path", type=str, help="path to raw data: wikipedia, pdf, json (e.g: graphRAG/data/raw_data/sample.pdf)")

    # Sau khi entities được extracted thì có thể save lại dưới dạng JSON
    parser.add_argument("--save_graph_path", type=str, default="graphRAG/data/entities_extracted/node_relationship.json", help="Path to save extracted entities (nodes and relationships) to JSON file")
    args = parser.parse_args()

    graph_indexing = GraphIndexing()
    graph_indexing.indexing(
        data_path=args.data_path,
        save_graph_path=args.save_graph_path,
    )
