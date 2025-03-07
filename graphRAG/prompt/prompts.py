EXTRACT_SYSTEM_PROMPT = """You are an expert in extracting entities and relationships from text. Provide accurate and structured JSON output."""

EXTRACT_PROMPT = (
    "### Role:\n"
    "You are an expert in entity extraction.\n"
    "\n"
    "### Instruction: \n"
    "- Analyze the following text and extract entities (people, places, events...) and relationships between them.\n"
    "- Output returned in JSON format is as follows, remember output return stays in <output> tag:\n"
    "<output>\n"
    '{{"nodes": [{{"id": "entity_name", "type": "entity_type"}}],\n'
    '"relationships": [{{"source": "entity_1", "target": "entity_2", "type": "relationship_type"}}]}}\n'
    "</output>"
    "\n\n"
    "### Execute with the following input\n"
    "<input>\n"
    "{text}\n"
    "</input>\n"
)

ANSWERING_SYSTEM_PROMPT = """You are an assistant expert in describing user's input based on the provided graphs."""

ANSWERING_PROMPT = (
    "### Role:\n"
    "You are an expert in describing users' input based on provided graphs.\n"
    "\n"
    "### Instruction: \n"
    "- Make sure that you use 'List graph schemas' provided to describe the query accurately.\n"
    "- 'List graph schemas' is information about relationships and attributes that you can utilize to describe.\n"
    "- Output returned in JSON format is as follows, remember output return stays in <output> tag:\n"
    "<output>\n"
    '{{"response": "your_response", "additional_data": "your_additional_data"}}\n'
    "</output>"
    "\n\n"
    "### List graph schemas\n"
    "{schemas}"
    "\n\n"
    "### Describe query with the following 'List graph schemas' provided\n"
    "<input>\n"
    "{query}\n"
    "</input>\n"
)

# ANSWERING_PROMPT = (
#     "### Role:\n"
#     "You are an expert in answering users' questions based on provided graphs.\n"
#     "\n"
#     "### Instruction: \n"
#     "- Make sure that you use 'List graphs' provided to answer the query.\n"
#     "- Make sure to answer truthfully, if you are unsure about a question, just say I don't know, don't make up an answer.\n"
#     "- Output returned in JSON format is as follows, remember output return stays in <output> tag:\n"
#     "<output>\n"
#     '{{"context": "your_response", "additional_data": "your_additional_data"}}\n'
#     "</output>"
#     "\n\n"
#     "### List graphs\n"
#     "{cypher}"
#     "\n\n"
#     "### Answer with the following context\n"
#     "<input>\n"
#     "{query}\n"
#     "</input>\n"
# )

GET_RELATIONSHIPS_CYPHER = (
    "MATCH (n)-[r]->(m)\n"
    "RETURN n.id AS source, type(r) AS relationship, m.id AS target\n"
)

GET_NODE_RELATIONSHIPS_CYPHER = (
    "MATCH (n)-[r]->(m)\n"
    "RETURN n.id AS source_id, n.name AS source_name, type(r) AS rel_type,\n"
    "       m.id AS target_id, m.name AS target_name, labels(n) AS source_labels, labels(m) AS target_labels\n"
)