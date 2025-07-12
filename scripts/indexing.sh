export PYTHONPATH=$PYTHONPATH:$PWD
mkdir -p graphRAG/data/entities_extracted


python -m graphRAG.indexing \
    --save_graph_path "graphRAG/data/entities_extracted/node_relationship.json" \
    # --data_path "data/format_data/elizabeth_i.json" \
