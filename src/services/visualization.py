import tempfile
import hashlib
from loguru import logger
from pyvis.network import Network
from langchain_neo4j import Neo4jGraph


def _generate_color_from_category(category: str) -> str:
    """
    Generate a consistent color for a category using hash-based color generation.
    Same category always gets the same color.
    
    Args:
        category: Entity category name
        
    Returns:
        Hex color string (e.g., "#FF5733")
    """
    hash_value = int(hashlib.md5(category.encode()).hexdigest(), 16)
    
    # Generate HSL color with:
    # - Hue: 0-360
    # - Saturation: 60-80%
    # - Lightness: 45-65%
    
    hue = hash_value % 360
    saturation = 60 + (hash_value % 20)
    lightness = 45 + (hash_value % 20)
    
    # Convert HSL to RGB
    def hsl_to_rgb(h, s, l):
        s /= 100
        l /= 100
        c = (1 - abs(2 * l - 1)) * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = l - c / 2
        
        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        r = int((r + m) * 255)
        g = int((g + m) * 255)
        b = int((b + m) * 255)
        
        return f"#{r:02X}{g:02X}{b:02X}"
    
    return hsl_to_rgb(hue, saturation, lightness)


def visualize_knowledge_graph(graph_db: Neo4jGraph, limit: int = 100):
    """Create interactive knowledge graph from Neo4j"""
    try:
        query = """
            MATCH (n:Entity)-[r]->(m:Entity)
            RETURN n.id AS source_id, n.entity_type AS source_type, n.entity_role AS source_role,
                   type(r) AS rel_type,
                   m.id AS target_id, m.entity_type AS target_type, m.entity_role AS target_role
            LIMIT $limit
        """
        
        results = graph_db.query(query, params={"limit": limit})
        
        if not results:
            return None
        
        net = Network(
            height="800px",
            width="100%",
            bgcolor="#1a1a2e",
            font_color="white",
            directed=True
        )
        
        net.force_atlas_2based(
            gravity=-100,
            central_gravity=0.01,
            spring_length=200,
            spring_strength=0.1,
            damping=0.4,
            overlap=0
        )
        
        # Cache for generated colors to ensure consistency
        color_cache = {}
        
        def get_category_color(category: str) -> str:
            """Get color for category, using cache for consistency"""
            if category not in color_cache:
                color_cache[category] = _generate_color_from_category(category)
            return color_cache[category]
        
        for record in results:
            source_id = record["source_id"]
            target_id = record["target_id"]
            source_category = record.get("source_type") or "Unknown"
            target_category = record.get("target_type") or "Unknown"
            source_role = record.get("source_role")
            target_role = record.get("target_role")
            rel_type = record["rel_type"]
            
            source_hover = f"{source_id}<br>Type: {source_category}"
            if source_role:
                source_hover += f"<br>Role: {source_role}"
            
            target_hover = f"{target_id}<br>Type: {target_category}"
            if target_role:
                target_hover += f"<br>Role: {target_role}"
            
            # Add nodes
            net.add_node(
                source_id,
                label=source_id[:25] + "..." if len(source_id) > 25 else source_id,
                title=source_hover,
                color=get_category_color(source_category),
                size=20,
                shape="dot",
                font={"size": 14, "face": "arial"}
            )
            
            net.add_node(
                target_id,
                label=target_id[:25] + "..." if len(target_id) > 25 else target_id,
                title=target_hover,
                color=get_category_color(target_category),
                size=20,
                shape="dot",
                font={"size": 14, "face": "arial"}
            )
            
            # Add edge
            net.add_edge(
                source_id,
                target_id,
                title=rel_type,
                label=rel_type,
                color="#848484",
                width=1.5,
                font={"size": 10, "align": "middle", "vadjust": -5}
            )
        
        net.set_options("""
        {
            "physics": {
                "enabled": true,
                "forceAtlas2Based": {
                    "gravitationalConstant": -100,
                    "centralGravity": 0.01,
                    "springLength": 200,
                    "springConstant": 0.1,
                    "damping": 0.4,
                    "avoidOverlap": 0.5
                },
                "solver": "forceAtlas2Based",
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
        
        html_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w', encoding='utf-8')
        net.save_graph(html_file.name)
        html_file.close()
        
        with open(html_file.name, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Inject CSS to remove margins, padding, and white background
        custom_css = """
        <style>
            * {
                margin: 0 !important;
                padding: 0 !important;
                box-sizing: border-box !important;
            }
            
            body {
                margin: 0 !important;
                padding: 0 !important;
                background-color: #1a1a2e !important;
                overflow: hidden !important;
            }
            
            html {
                margin: 0 !important;
                padding: 0 !important;
                background-color: #1a1a2e !important;
            }
            
            #mynetwork {
                margin: 0 !important;
                padding: 0 !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
                background-color: #1a1a2e !important;
                width: 100% !important;
                height: 800px !important;
            }
            
            canvas {
                border: none !important;
                outline: none !important;
                display: block !important;
            }
            
            .vis-network {
                border: none !important;
                outline: none !important;
            }
            
            /* Remove Bootstrap card styling */
            .card {
                border: none !important;
                background-color: transparent !important;
                margin: 0 !important;
                padding: 0 !important;
            }
            
            .card-body {
                border: none !important;
                padding: 0 !important;
                margin: 0 !important;
            }
            
            /* Hide any headers */
            center, h1 {
                display: none !important;
            }
        </style>
        """
        
        html_content = html_content.replace('</head>', custom_css + '</head>')
        
        with open(html_file.name, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_file.name
        
    except Exception as e:
        logger.error(f"Error creating Neo4j visualization: {e}")
        return None

if __name__ == "__main__":
    from src.config.setting import neo4j_config
    
    graph_db = Neo4jGraph(
        url=neo4j_config.url,
        username=neo4j_config.username,
        password=neo4j_config.password
    )
    html_path = visualize_knowledge_graph(graph_db, limit=50)
    if html_path:
        logger.info(f"Knowledge graph visualization saved to: {html_path}")
    else:
        logger.error("Failed to create knowledge graph visualization.")