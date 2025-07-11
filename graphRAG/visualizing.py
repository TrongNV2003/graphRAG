import argparse
import numpy as np
from loguru import logger
from typing import Literal
from pyvis.network import Network
import plotly.graph_objects as go
from langchain_neo4j import Neo4jGraph

from graphRAG.config.setting import neo4j_config
from graphRAG.prompt.prompts import GET_RELATIONSHIPS_CYPHER, GET_NODE_RELATIONSHIPS_CYPHER

class GraphVisualizer:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=neo4j_config.url,
            username=neo4j_config.username,
            password=neo4j_config.password
        )

    def _query_graph(self, query: str, limit: int = 25) -> list:
        """Truy vấn Neo4j và trả về danh sách kết quả."""
        full_query = f"{query} LIMIT {limit}"
        results = self.graph.query(full_query)
        return results

    def print_relationships(self, limit: int = 10):
        """
            Tìm tất cả các cặp node từ node source có mối quan hệ có hướng trong đồ thị.
               (node n)     relationship     (node m)
            (node source) ---------------> (node target)
        """

        results = self._query_graph(query=GET_RELATIONSHIPS_CYPHER, limit=limit)
        logger.info(f"Showing {len(results)} relationships from Neo4j:")
        for record in results:
            print(f"{record['source']} -[{record['relationship']}]-> {record['target']}")

    def visualize_graph_3d(self, limit: int = 25, output_file: str = "graph.html"):
        """
            Lấy tất cả các cặp node từ node source có mối quan hệ có hướng trong đồ thị.
               (node n)     relationship     (node m)
            (node source) ---------------> (node target)
        """
    
        results = self._query_graph(query=GET_NODE_RELATIONSHIPS_CYPHER, limit=limit)

        nodes = set()
        for record in results:
            nodes.add((record["source_id"], record["source_labels"][0]))
            nodes.add((record["target_id"], record["target_labels"][0]))
        nodes = list(nodes)

        np.random.seed(42)
        x_nodes = np.random.uniform(0, 10, len(nodes))
        y_nodes = np.random.uniform(0, 10, len(nodes))
        z_nodes = np.random.uniform(0, 10, len(nodes))

        node_map = {node[0]: i for i, node in enumerate(nodes)}

        type_colors = {
            "PERSON": "#FF6B6B",
            "PLACE": "#4ECDC4",
            "TIME_PERIOD": "#FFD93D",
            "EVENT": "#95E1D3",
            "ORGANIZATION": "#F7C8E0",
            "CONCEPT": "#B9BBDF",
            "OBJECT": "#FF9F1C"
        }

        # Tạo danh sách node
        node_traces = []
        for i, (node_id, node_type) in enumerate(nodes):
            trace = go.Scatter3d(
                x=[x_nodes[i]], y=[y_nodes[i]], z=[z_nodes[i]],
                mode="markers+text",
                name=node_type,
                marker=dict(
                    size=10,
                    color=type_colors.get(node_type, "white"),
                    line=dict(width=1, color="black")
                ),
                text=[node_id],
                textposition="middle center",
                hoverinfo="text",
                hovertext=f"ID: {node_id}<br>Type: {node_type}"
            )
            node_traces.append(trace)

        # Tạo danh sách edge
        edge_x, edge_y, edge_z = [], [], []
        for record in results:
            source_idx = node_map[record["source_id"]]
            target_idx = node_map[record["target_id"]]
            edge_x.extend([x_nodes[source_idx], x_nodes[target_idx], None])
            edge_y.extend([y_nodes[source_idx], y_nodes[target_idx], None])
            edge_z.extend([z_nodes[source_idx], z_nodes[target_idx], None])

        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode="lines",
            line=dict(width=2, color="gray"),
            hoverinfo="none"
        )

        # Layout
        layout = go.Layout(
            title="3D Graph Visualization",
            showlegend=True,
            scene=dict(
                xaxis=dict(title="X", backgroundcolor="black", gridcolor="gray", showbackground=True),
                yaxis=dict(title="Y", backgroundcolor="black", gridcolor="gray", showbackground=True),
                zaxis=dict(title="Z", backgroundcolor="black", gridcolor="gray", showbackground=True),
                bgcolor="black"
            ),
            paper_bgcolor="black",
            plot_bgcolor="black",
            width=1500,
            height=800,
            margin=dict(l=0, r=0, b=0, t=40)
        )

        # Kết hợp traces và vẽ
        fig = go.Figure(data=[edge_trace] + node_traces, layout=layout)
        fig.write_html(output_file)
        logger.info(f"3D graph visualization saved to {output_file}")

    def visualize_graph_2d(self, limit: int = 25, output_file: str = "graph.html"):
        """
            Lấy tất cả các cặp node từ node source có mối quan hệ có hướng trong đồ thị.
               (node n)     relationship     (node m)
            (node source) ---------------> (node target)
        """

        results = self._query_graph(query=GET_NODE_RELATIONSHIPS_CYPHER, limit=limit)

        net = Network(
            notebook=True,
            directed=True,
            height="900px",
            width="100%",
            bgcolor="#1a1a1a",
            font_color="white",
            cdn_resources="remote"
        )
        
        net.force_atlas_2based(
            gravity=-50,
            central_gravity=0.01,
            spring_length=100,
            spring_strength=0.08,
            damping=0.4,
            overlap=0
        )

        type_colors = {
            "PERSON": "#FF6B6B",
            "PLACE": "#4ECDC4",
            "TIME_PERIOD": "#FFD93D",
            "EVENT": "#95E1D3",
            "ORGANIZATION": "#F7C8E0",
            "CONCEPT": "#B9BBDF",
            "OBJECT": "#FF9F1C"
        }

        for record in results:
            source_id = record["source_id"]
            target_id = record["target_id"]
            source_label = record["source_labels"][0]
            target_label = record["target_labels"][0]
            rel_type = record["rel_type"]
            
            net.add_node(
                source_id,
                label=source_id,
                title=f"Type: {source_label}",
                color=type_colors.get(source_label, "#FFFFFF"),
                size=15,
                shape="dot",
                font={"size": 14, "face": "arial"}
            )

            # Thêm node đích
            net.add_node(
                target_id,
                label=target_id,
                title=f"Type: {target_label}",
                color=type_colors.get(target_label, "#FFFFFF"),
                size=15,
                shape="dot",
                font={"size": 14, "face": "arial"}
            )

            # Thêm edge
            net.add_edge(
                source_id,
                target_id,
                label=rel_type,
                title=rel_type,
                color="#848484",
                width=1.5,
                font={"size": 12, "align": "middle"}
            )

        net.set_options("""
        {
            "physics": {
                "enabled": true,
                "stabilization": {
                    "enabled": true,
                    "iterations": 1000
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 200,
                "zoomView": true,
                "dragView": true
            },
            "nodes": {
                "borderWidth": 1,
                "shadow": true,
                "scaling": {
                    "min": 10,
                    "max": 30
                }
            },
            "edges": {
                "arrows": {
                    "to": {
                        "enabled": true,
                        "scaleFactor": 0.5
                    }
                },
                "smooth": {
                    "enabled": true,
                    "type": "dynamic"
                }
            },
            "configure": {
                "enabled": false
            }
        }
        """)

        net.show(output_file)
        logger.info(f"Graph visualization saved to {output_file}")

    def visualize_graph(self, limit: int = 500, visualize_type: Literal["2d", "3d"] = "2d"):
        if visualize_type == "2d":
            self.visualize_graph_2d(limit=limit, output_file=f"graph_{visualize_type}.html")
        elif visualize_type == "3d":
            self.visualize_graph_3d(limit=limit, output_file=f"graph_{visualize_type}.html")
        else:
            raise ValueError(f"Invalid visualize_type: {visualize_type}. Must be visualize in '2d' or '3d'.")

if __name__ == "__main__":
    visualizer = GraphVisualizer()

    parser = argparse.ArgumentParser(description="Visualize Neo4j graph data")
    parser.add_argument("--limit", type=int, default=500, help="Number of relationships to visualize")
    parser.add_argument("--visualize_type", type=str, default="2d", help="Type of visualization: 2d or 3d")
    args = parser.parse_args()
    
    visualizer.visualize_graph(limit=args.limit, visualize_type=args.visualize_type)