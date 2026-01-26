from loguru import logger
from typing import List, Dict

from src.core.retrieval import HybridRetrieval
from src.engines.llm import AnalysisQueryLLM, GenerationResponseLLM


class GraphQuerying:
    def __init__(
        self,
        retriever: HybridRetrieval,
        generator: GenerationResponseLLM,
        analyzer: AnalysisQueryLLM
    ):
        self.retriever = retriever
        self.generator = generator
        self.analyzer = analyzer
    
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
        
        answer_text = response.get("answer", "No answer generated.")
        return answer_text

    def response_detailed(self, query: str, top_k: int = 5, threshold: float = 0.0, graph_limit: int = 10) -> dict:
        """Generate a response to the query using graph retrieval and LLM generation, returning detailed context.
        
        Args:
            query: The input query string.
            top_k: Top K results for chunk retrieval.
            threshold: Similarity threshold for chunk retrieval.
            graph_limit: Limit for graph retrieval.
            
        Returns:
            dict: The generated response containing answer, graph_context, and chunk_context.
        """
        target_entities, excluded_entities, normalized_query = self._analyze_query(query)
        
        retrieved_results = self.retriever.retrieve(
            query=normalized_query,
            target_entities=target_entities,
            excluded_entities=excluded_entities,
            graph_limit=graph_limit,
            chunk_top_k=top_k,
            chunk_threshold=threshold
        )
        
        graph_context = self._format_graph_context(retrieved_results["graph"])
        chunk_context = self._format_chunk_context(retrieved_results["chunk"])
        
        response = self.generator.call(
            query=query,
            graph_context=graph_context,
            chunk_context=chunk_context
        )
        
        answer_text = response.get("answer", "No answer generated.")
        
        return {
            "answer": answer_text,
            "graph_context": retrieved_results["graph"],
            "chunk_context": retrieved_results["chunk"]
        }
    
    def _analyze_query(self, query: str) -> tuple:
        """Analyze the query to extract entities and relations using the LLM.
        
        Args:
            query: The input query string.
            
        Returns:
            tuple: A tuple containing target_entities, excluded_entities, and normalized_query.
        """
        analysis = self.analyzer.call(query=query)
        
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


    def semantic_response(self, query: str, top_k: int = 5, threshold: float = 0.0) -> dict:
        """Generate answer using Semantic Search (Dense vector) only."""
        chunks = self.retriever.chunk_retrieval.semantic_search(
            query=query, 
            top_k=top_k, 
            threshold=threshold
        )
            
        # Format context
        chunk_context = self._format_chunk_context(chunks)
        graph_context = "No graph context (Semantic Search mode)."
        
        response = self.generator.call(
            query=query,
            graph_context=graph_context,
            chunk_context=chunk_context
        )
        answer_text = response.get("answer", "No answer generated.")
        
        return {
            "answer": answer_text,
            "graph_context": [],
            "chunk_context": chunks,
            "search_type": "semantic"
        }

    def hybrid_response(self, query: str, top_k: int = 5, threshold: float = 0.0) -> dict:
        """Generate answer using Hybrid Search (Qdrant Dense+Sparse) only."""
        chunks = self.retriever.chunk_retrieval.retrieve(
            query=query,
            top_k=top_k,
            threshold=threshold
        )
            
        # Format context
        chunk_context = self._format_chunk_context(chunks)
        graph_context = "No graph context (Hybrid Search mode)."
        
        # Generate Answer
        response = self.generator.call(
            query=query,
            graph_context=graph_context,
            chunk_context=chunk_context
        )
        answer_text = response.get("answer", "No answer generated.")
        
        return {
            "answer": answer_text,
            "graph_context": [],
            "chunk_context": chunks,
            "search_type": "hybrid"
        }


    def semantic_search(self, query: str, top_k: int = 5, threshold: float = 0.0) -> dict:
        """Perform semantic search (dense vector only, no sparse/keyword).
        
        Args:
            query: The search query string.
            top_k: Maximum number of results to return.
            threshold: Similarity threshold.
            
        Returns:
            dict: Search results with chunks and total count.
        """
        chunks = self.retriever.chunk_retrieval.semantic_search(
            query=query,
            top_k=top_k,
            threshold=threshold
        )
        return {
            "chunks": chunks,
            "total": len(chunks),
            "search_type": "semantic"
        }

    def hybrid_search(self, query: str, top_k: int = 5, threshold: float = 0.0) -> dict:
        """Perform hybrid search (dense + sparse vectors via Qdrant).
        
        Args:
            query: The search query string.
            top_k: Maximum number of results to return.
            threshold: Similarity threshold.
            
        Returns:
            dict: Search results with chunks and total count.
        """
        chunks = self.retriever.chunk_retrieval.retrieve(
            query=query, 
            top_k=top_k, 
            threshold=threshold
        )
        return {
            "chunks": chunks,
            "total": len(chunks),
            "search_type": "hybrid"
        }
