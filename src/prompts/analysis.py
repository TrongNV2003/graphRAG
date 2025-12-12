ANALYZE_SYSTEM_PROMPT = """You are an AI assistant expert in query analysis. Your task is to extract entities from user queries and normalize them for accurate retrieval."""

ANALYZE_PROMPT_TEMPLATE = """
### Role:
You are an expert analyst for a Graph RAG system. Your goal is to break down user queries to optimize retrieval accuracy.

### Instruction:
Analyze the user query and extract the following information into a JSON object:

1. `target_entities`: A list of key entities that form the core subject of the query.
2. `excluded_entities`: A list of entities that the user explicitly wants to ignore or exclude (e.g., "Besides X", "Not Y").
3. `normalized_query`: A rewritten version of the query optimized for vector search. 
    - Remove negative constraints if they confuse embedding models.
    - Focus on the *category* or *topic* if specific entities are excluded.

### Rules for "Exclusion" logic:
- If user asks: "Besides X, what else?", X is an `excluded_entity`. 
- CRITICAL: For Graph retrieval to work, you often need a starting point. If the user excludes X, try to find the `Category` or `Group` X belongs to and put that in `target_entities`. 
    - Example: "Besides Earth, which planets...?" -> target: ["Planets"], excluded: ["Earth"].
    - Example: "Who else ruled besides Elizabeth?" -> target: ["Monarchs", "Rulers"], excluded: ["Elizabeth I"].
- If no category is explicit, leave `target_entities` empty but fill `excluded_entities`.

### Example Outputs:
Query: 'Besides Elizabeth I, are there any other cases?'
Output:
{{"target_entities": ["Cases", "Historical Figures"], "excluded_entities": ["Elizabeth I"], "normalized_query": "List of historical cases or figures excluding Elizabeth I"}}

Query: 'Compare Elizabeth I and Mary Stuart'
Output:
{{"target_entities": ["Elizabeth I", "Mary Stuart"], "excluded_entities": [], "normalized_query": "Comparison of traits and history between Elizabeth I and Mary Stuart"}}

### Analyze the following query:
<input>
{query}
</input>
"""

ANALYZE_SCHEMA = {
    "name": "query_analysis",
    "schema": {
        "type": "object",
        "properties": {
            "target_entities": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "Entity name mentioned in the query for graph retrieval"
                },
                "description": "List of entities to retrieve from knowledge graph. Empty if query asks to exclude certain entities."
            },
            "excluded_entities": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "Entity name that the user wants to exclude from results"
                },
                "description": "List of entities to exclude from retrieval based on user query"
            },
            "normalized_query": {
                "type": "string",
                "description": "Normalized query capturing the semantic intent for embedding search"
            }
        },
        "required": ["target_entities", "excluded_entities", "normalized_query"],
        "additionalProperties": False
    },
    "strict": True
}