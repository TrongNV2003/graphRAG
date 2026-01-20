EXTRACT_SYSTEM_PROMPT = """You are an AI assistant expert in data extraction. Your task is to extract numerical entities from text and return them as valid JSON according to the provided schema. Always strictly follow the instructions below."""

EXTRACT_PROMPT_TEMPLATE = (
    "### Role:\n"
    "You are an expert in entity extraction.\n"
    "\n"
    "### Instruction: \n"
    "- Analyze the following text and extract entities (people, places, events...) and relationships between them.\n"
    "1.  **Entities:**\n"
    "- For each entity, include an 'id' (the entity name), 'entity_type' (e.g., Person, Place, Event, Organization), and 'entity_role' (the role or function, e.g., 'Queen of England').\n"
    "- If no 'entity_role' is applicable, leave them as empty strings ('').\n"
    "2.  **Relationships:**\n"
    '    - `source` and `target`: MUST match exactly the `id` field of the extracted nodes.\n'
    '    - `relationship_type`: The type of relationship between two entities.\n'
    "\n"
    "## Example output (valid JSON):\n"
    '{"nodes": [{"id": "entity_name", "entity_type": "entity_type", "entity_role": "entity_role"}],\n'
    '"relationships": [{"source": "entity_1", "target": "entity_2", "relationship_type": "relationship_type"}]}\n'
    "\n"
    "### Execute with the following input\n"
    "<input>\n"
    "{{ text }}\n"
    "</input>\n"
)

EXTRACT_SCHEMA = {
    "name": "entities_extraction",
    "schema": {
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Name of the entity being extracted."
                        },
                        "entity_type": {
                            "type": "string",
                            "description": "Type of the entity (e.g., Person, Place, Event, Organization)."
                        },
                        "entity_role": {
                            "type": "string",
                            "description": "Role or function of the entity (e.g., 'Queen of England'). If not applicable, leave as an empty string ('')."
                        },
                    },
                    "required": ["id", "entity_type", "entity_role"],
                    "additionalProperties": False
                }
            },
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Name of the source entity, must match exactly the `id` field of the extracted nodes."
                        },
                        "target": {
                            "type": "string",
                            "description": "Name of the target entity, must match exactly the `id` field of the extracted nodes."
                        },
                        "relationship_type": {
                            "type": "string",
                            "description": "The type of relationship between two entities."
                        }
                    },
                    "required": ["source", "target", "relationship_type"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["nodes", "relationships"],
        "additionalProperties": False
    },
    "strict": True,
}
