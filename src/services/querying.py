import json
from loguru import logger
from openai import OpenAI
from typing import List, Dict
from langchain_neo4j import Neo4jGraph

from src.handler.retrieval import HybridRetrieval
from src.engines.search import HybridRetrievalEngine
from src.engines.llm import AnalysisQueryLLM, GenerationResponseLLM
from src.prompts.analysis import ANALYZE_SYSTEM_PROMPT, ANALYZE_PROMPT_TEMPLATE, ANALYZE_SCHEMA
from src.prompts.response import ANSWERING_SYSTEM_PROMPT, ANSWERING_PROMPT_TEMPLATE
from src.config.setting import embed_config

class GraphQuerying:
    def __init__(
        self,
        client: OpenAI,
        graph_db: Neo4jGraph,
        answering_prompt_template: str = ANSWERING_PROMPT_TEMPLATE,
        answering_prompt_system: str = ANSWERING_SYSTEM_PROMPT,
        analyze_prompt_template: str = ANALYZE_PROMPT_TEMPLATE,
        analyze_prompt_system: str = ANALYZE_SYSTEM_PROMPT,
        analyze_schema: dict = ANALYZE_SCHEMA,
    ):
        self.search_engine = HybridRetrievalEngine(dense_params={
            "model_name": embed_config.embedder_model,
        })
        
        self.retriever = HybridRetrieval(
            graph_db,
            search_engine=self.search_engine,
            graph_limit=10,
            chunk_top_k=2,
            chunk_threshold=0,
            auto_build=True
        )
        
        self.generator = GenerationResponseLLM(
            client=client,
            prompt_template=answering_prompt_template,
            system_prompt=answering_prompt_system,
        )
        
        self.analyzer = AnalysisQueryLLM(
            client=client,
            prompt_template=analyze_prompt_template,
            system_prompt=analyze_prompt_system,
            json_schema=analyze_schema
        )
    
    def response(self, query: str) -> str:
        """Generate a response to the query using graph retrieval and LLM generation.
        
        Args:
            query: The input query string.
            
        Returns:
            str: The generated response from the LLM.
        """
        target_entities, excluded_entities, normalized_query = self._analyze_query(query)
        
        retrieved_results = self.retriever.retrieve(
            query=normalized_query,
            target_entities=target_entities,
            excluded_entities=excluded_entities
        )
        
        graph_context = self._format_graph_context(retrieved_results["graph"])
        chunk_context = self._format_chunk_context(retrieved_results["chunk"])
        
        response = self.generator.call(
            query=query,
            graph_context=graph_context,
            chunk_context=chunk_context
        )
        
        response = json.loads(response)
        
        answer_text = response.get("answer", "No answer generated.")
        return answer_text
    
    def _analyze_query(self, query: str) -> dict:
        """Analyze the query to extract entities and relations using the LLM.
        
        Args:
            query: The input query string.
            
        Returns:
            dict: The analysis result containing extracted entities and relations.
        """
        analysis = self.analyzer.call(query=query)
        
        try:
            analysis = json.loads(analysis)
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {analysis}")
            logger.error(f"JSON decode error: {e}")
            analysis = {"target_entities": [], "excluded_entities": [], "normalized_query": query}
        
        target_entities = analysis.get("target_entities", [])  if isinstance(analysis, dict) else []
        excluded_entities = analysis.get("excluded_entities", [])  if isinstance(analysis, dict) else []
        normalized_query = analysis.get("normalized_query", query) if isinstance(analysis, dict) else query
        
        logger.info(f"Target entities: {target_entities}")
        logger.info(f"Excluded entities: {excluded_entities}")
        logger.info(f"Normalized query: {normalized_query}")
        
        return target_entities, excluded_entities, normalized_query
    
    def _format_graph_context(self, graph_results: List[Dict]) -> str:
        """Format graph triples into readable text.
        
        Args:
            graph_results: List of graph triples
            
        Returns:
            str: Formatted graph context
        """
        if not graph_results:
            return "No graph relationships found."
        
        formatted = []
        for i, triple in enumerate(graph_results, 1):
            source = triple.get("source", {})
            relationship = triple.get("relationship", "RELATED_TO")
            target = triple.get("target", {})
            
            # Format: "Elizabeth I (Person, Queen)" or "Elizabeth I (Person)" if no role
            source_id = source.get('id', 'Unknown')
            source_type = source.get('type', 'Entity')
            source_role = source.get('entity_role', '')
            source_str = f"{source_id} ({source_type}" + (f", {source_role})" if source_role else ")")
            
            target_id = target.get('id', 'Unknown')
            target_type = target.get('type', 'Entity')
            target_role = target.get('entity_role', '')
            target_str = f"{target_id} ({target_type}" + (f", {target_role})" if target_role else ")")
            
            formatted.append(f"{i}. {source_str} --[{relationship}]--> {target_str}")
        
        return "\n".join(formatted)
    
    def _format_chunk_context(self, chunk_results: List[Dict]) -> str:
        """Format chunks into numbered readable text.
        
        Args:
            chunk_results: List of chunk dictionaries
            
        Returns:
            str: Formatted chunk context
        """
        if not chunk_results:
            return "No relevant document chunks found."
        
        formatted = []
        for i, chunk in enumerate(chunk_results, 1):
            chunk_text = chunk.get("chunk_text", "")
            formatted.append(f"[Chunk {i}]\n{chunk_text}")
        
        return "\n\n".join(formatted)


if __name__ == "__main__":
    from src.config.setting import api_config, neo4j_config
    client = OpenAI(api_key=api_config.api_key, base_url=api_config.base_url)

    graph_db = Neo4jGraph(
        url=neo4j_config.url,
        username=neo4j_config.username,
        password=neo4j_config.password
    )
    queries = GraphQuerying(client=client, graph_db=graph_db)
    
    query = "Tell me about Elizabeth and her relationships."
    answer = queries.response(query=query)
    
    print(f"Query: {query}\nAnswer: {answer}")