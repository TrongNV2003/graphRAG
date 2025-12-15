import os
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
from langchain_neo4j import Neo4jGraph

from src.config.setting import api_config, neo4j_config
from src.processing.dataloaders import DataLoader
from src.services.indexing import GraphIndexing
from src.services.querying import GraphQuerying
from src.services.visualization import visualize_knowledge_graph


st.set_page_config(
    page_title="HyRAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
    }
    
    /* ChatGPT-style interface */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Style chat input */
    .stChatInput {
        border-radius: 1.5rem;
        position: sticky;
        bottom: 0;
        background: var(--background-color);
        padding: 1rem 0;
        z-index: 100;
    }
    
    /* Split view divider - Dark mode compatible */
    [data-testid="column"]:first-child {
        border-right: 1px solid rgba(250, 250, 250, 0.1);
        padding-right: 1.5rem;
    }
    
    [data-testid="column"]:last-child {
        padding-left: 1.5rem;
    }
    
    /* Chat container with proper spacing */
    .chat-wrapper {
        display: flex;
        flex-direction: column;
        height: calc(100vh - 12rem);
        position: relative;
    }
    
    /* Chat messages area - scrollable with space for input */
    .chat-messages-container {
        flex: 1;
        overflow-y: auto;
        padding-bottom: 6rem;
        margin-bottom: 1rem;
        border: none;
    }
    
    /* Graph container - Dark mode compatible */
    .graph-container {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        padding: 1rem;
        height: calc(100vh - 8rem);
        border: 1px solid rgba(250, 250, 250, 0.1);
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def _get_resources():
    client = OpenAI(api_key=api_config.api_key, base_url=api_config.base_url)
    graph_db = Neo4jGraph(
        url=neo4j_config.url,
        username=neo4j_config.username,
        password=neo4j_config.password,
    )
    dataloader = DataLoader()
    return client, graph_db, dataloader


def _get_db_stats(graph_db: Neo4jGraph) -> dict:
    entity_count = graph_db.query("MATCH (e:Entity) RETURN count(e) as count")[0]["count"]
    rel_count = graph_db.query("MATCH ()-[r]->() RETURN count(r) as count")[0]["count"]
    chunk_count = graph_db.query("MATCH (c:Chunk) RETURN count(c) as count")[0]["count"]
    return {
        "entities": int(entity_count),
        "relationships": int(rel_count),
        "chunks": int(chunk_count),
    }


@st.cache_data(ttl=300)
def _get_graph_html(_graph_db: Neo4jGraph, limit: int) -> str:
    """Cache graph HTML to avoid regenerating on every expand/collapse"""
    graph_html_path = visualize_knowledge_graph(graph_db=_graph_db, limit=limit)
    if not graph_html_path:
        return ""
    
    try:
        with open(graph_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return html_content
    finally:
        try:
            os.unlink(graph_html_path)
        except OSError:
            pass


def _render_graph(graph_db: Neo4jGraph, limit: int) -> None:
    html_content = _get_graph_html(graph_db, limit)
    if not html_content:
        st.warning("Could not generate graph visualization. Check Neo4j connection and data.")
        return
    
    components.html(html_content, height=750, scrolling=False)


def _init_state() -> None:
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "indexing_completed" not in st.session_state:
        st.session_state.indexing_completed = False
    if "show_graph" not in st.session_state:
        st.session_state.show_graph = False
    if "graph_expanded" not in st.session_state:
        st.session_state.graph_expanded = True


def main() -> None:
    _init_state()
    client, graph_db, dataloader = _get_resources()

    st.title("RAG System")

    with st.sidebar:
        st.header("Indexing")
        query_keyword = st.text_input("Query", value="Elizabeth I")
        load_max_docs = st.number_input("Load max docs", min_value=1, max_value=50, value=10, step=1)

        run_indexing = st.button("Run indexing", type="primary")
        if run_indexing:
            with st.spinner("Indexing documents and storing into Neo4j..."):
                docs = dataloader.load(query_keyword, load_max_docs=int(load_max_docs))
                indexing = GraphIndexing(
                    client=client,
                    graph_db=graph_db,
                    chunk_size=2048,
                    clear_old_graph=False,
                )

                chunks = []
                for doc in docs:
                    chunks.extend(indexing.chunking(doc.get("content", "")))
                indexing.indexing(chunks=chunks)

            st.session_state.indexing_completed = True
            st.session_state.show_graph = True
            stats = _get_db_stats(graph_db)
            st.success(
                f"Indexing finished. Total Entities: {stats['entities']}, "
                f"Total Relationships: {stats['relationships']}, Total Chunks: {stats['chunks']}"
            )
        
        st.divider()
        
        if st.button("Visualize Graph", type="secondary"):
            st.session_state.show_graph = True
            st.rerun()

    # Always split layout, but adjust column ratio based on expansion state
    if st.session_state.show_graph:
        if st.session_state.graph_expanded:
            left, right = st.columns([1, 1], gap="medium")
        else:
            left, right = st.columns([20, 1], gap="small")
    else:
        left = st.container()
        right = None

    # Chat section
    with left:
        messages_container = st.container(height=700, border=False)
        with messages_container:
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input at bottom
        user_input = st.chat_input("Message")
        if user_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            
            with st.spinner("Thinking..."):
                querying = GraphQuerying(client=client, graph_db=graph_db)
                answer = querying.response(query=user_input)

            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            st.rerun()

    # Graph panel - always present when show_graph is True
    if st.session_state.show_graph and right is not None:
        with right:
            if st.session_state.graph_expanded:
                # Expanded state - button on left side with graph
                col_button, col_graph = st.columns([0.3, 9.7])
                with col_button:
                    st.markdown("<br>" * 15, unsafe_allow_html=True)
                    if st.button("◀", key="collapse_graph", help="Collapse graph", use_container_width=True):
                        st.session_state.graph_expanded = False
                        st.rerun()
                
                with col_graph:
                    st.subheader("Knowledge Graph")
                    _render_graph(graph_db=graph_db, limit=150)
            else:
                # Collapsed state - show only expand button
                st.markdown("<br>" * 15, unsafe_allow_html=True)
                if st.button("▶", key="expand_graph", help="Expand graph", use_container_width=True):
                    st.session_state.graph_expanded = True
                    st.rerun()


if __name__ == "__main__":
    main()