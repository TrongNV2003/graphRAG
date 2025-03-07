python -m graphRAG.indexing \
    --format_type "wikipedia" \
    --data_path "data/data_loaded/elizabeth_i.json" \
    --save_entities True \
    --save_graph_dir "data/entities_extracted/node_relationship.json" \
    --graph_data "data/node_relationship.json" \
