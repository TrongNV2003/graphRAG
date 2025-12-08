EXTRACT_SYSTEM_PROMPT = """You are an expert in extracting entities and relationships from text. Provide accurate and structured JSON output."""

EXTRACT_PROMPT = (
    "### Role:\n"
    "You are an expert in entity extraction.\n"
    "\n"
    "### Instruction: \n"
    "- Analyze the following text and extract entities (people, places, events...) and relationships between them.\n"
    "- For each entity, include an 'id' (the entity name), 'type' (e.g., Person, Place, Event, Organization), and 'role' (the role or function, e.g., 'Queen of England').\n"
    "- If no role is applicable, leave them as empty strings ('').\n"
    "- Output returned in JSON format is as follows, remember output return stays in <output> tag:\n"
    "<output>\n"
    '{{"nodes": [{{"id": "entity_name", "type": "entity_type", "role": "entity_role"}}],\n'
    '"relationships": [{{"source": "entity_1", "target": "entity_2", "type": "relationship_type"}}]}}\n'
    "</output>"
    "\n\n"
    "### Execute with the following input\n"
    "<input>\n"
    "{text}\n"
    "</input>\n"
)


GET_RELATIONSHIPS_CYPHER = (
    "MATCH (n)-[r]->(m)\n"
    "RETURN n.id AS source, type(r) AS relationship, m.id AS target\n"
)

GET_NODE_RELATIONSHIPS_CYPHER = (
    "MATCH (n)-[r]->(m)\n"
    "RETURN n.id AS source_id, n.name AS source_name, type(r) AS rel_type,\n"
    "       m.id AS target_id, m.name AS target_name, labels(n) AS source_labels, labels(m) AS target_labels\n"
)