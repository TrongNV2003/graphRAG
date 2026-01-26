from openai import OpenAI
from functools import lru_cache
from langchain_neo4j import Neo4jGraph

from src.core.factory import LLMClientFactory
from src.core.retrieval import HybridRetrieval
from src.core.storage import GraphStorage, QdrantEmbedStorage
from src.services.index_service import GraphIndexing
from src.services.query_service import GraphQuerying
from src.processing.dataloaders import DataLoader
from src.processing.chunking import TwoPhaseDocumentChunker
from src.processing.postprocessing import EntityPostprocessor
from src.engines.qdrant import create_qdrant_store, QdrantVectorStore
from src.engines.llm import EntityExtractionLLM, GenerationResponseLLM, AnalysisQueryLLM
from src.prompts.ner_prompt import EXTRACT_SYSTEM_PROMPT, EXTRACT_PROMPT_TEMPLATE, EXTRACT_SCHEMA
from src.prompts.response_prompt import ANSWERING_SYSTEM_PROMPT, ANSWERING_PROMPT_TEMPLATE
from src.prompts.analysis_prompt import ANALYZE_SYSTEM_PROMPT, ANALYZE_PROMPT_TEMPLATE, ANALYZE_SCHEMA
from src.config.setting import neo4j_config, llm_config


@lru_cache()
def get_openai_client() -> OpenAI:
    return LLMClientFactory.create_sync_client()

@lru_cache()
def get_neo4j_graph() -> Neo4jGraph:
    return Neo4jGraph(
        url=neo4j_config.url,
        username=neo4j_config.username,
        password=neo4j_config.password,
    )

@lru_cache()
def get_data_loader() -> DataLoader:
    return DataLoader()

@lru_cache()
def get_qdrant_store() -> QdrantVectorStore:
    return create_qdrant_store()

@lru_cache()
def get_indexing_service() -> GraphIndexing:
    client = get_openai_client()
    graph_db = get_neo4j_graph()
    
    chunker = TwoPhaseDocumentChunker(
        chunk_size=2048,
        tokenize_model=llm_config.llm_model,
        verbose=False
    )
    extractor = EntityExtractionLLM(
        client=client,
        system_prompt=EXTRACT_SYSTEM_PROMPT,
        prompt_template=EXTRACT_PROMPT_TEMPLATE,
        json_schema=EXTRACT_SCHEMA,
    )
    vector_store = get_qdrant_store()
    storage = GraphStorage(graph_db)
    postprocessor = EntityPostprocessor()
    qdrant_storage = QdrantEmbedStorage(vector_store=vector_store)
    
    return GraphIndexing(
        client=client,
        graph_db=graph_db,
        chunker=chunker,
        extractor=extractor,
        postprocessor=postprocessor,
        storage=storage,
        qdrant_storage=qdrant_storage
    )

@lru_cache()
def get_querying_service() -> GraphQuerying:
    client = get_openai_client()
    graph_db = get_neo4j_graph()
    vector_store = get_qdrant_store()
    
    retriever = HybridRetrieval(
        graph_db,
        vector_store=vector_store,
        auto_build=True
    )
    generator = GenerationResponseLLM(
        client=client,
        prompt_template=ANSWERING_PROMPT_TEMPLATE,
        system_prompt=ANSWERING_SYSTEM_PROMPT,
    )
    analyzer = AnalysisQueryLLM(
        client=client,
        prompt_template=ANALYZE_PROMPT_TEMPLATE,
        system_prompt=ANALYZE_SYSTEM_PROMPT,
        json_schema=ANALYZE_SCHEMA
    )
    return GraphQuerying(
        retriever=retriever,
        generator=generator,
        analyzer=analyzer
    )
