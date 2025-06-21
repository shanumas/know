import streamlit as st
import time
from datetime import datetime, timedelta
import threading
import os
from hn_data_manager import HackerNewsDataManager
from vector_store import VectorStore
from rag_agent import RAGAgent
from auto_updater import AutoUpdater
from utils import format_timestamp, truncate_text

STORIES_LIMIT = 100  # Limit for initial data load
AUTO_UPDATE_INTERVAL = 30  # in minutes


# Import pandas with fallback
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    st.warning("Pandas not available - some features may be limited")

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'rag_agent' not in st.session_state:
    st.session_state.rag_agent = None
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = None
if 'auto_updater' not in st.session_state:
    st.session_state.auto_updater = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'initialization_complete' not in st.session_state:
    st.session_state.initialization_complete = False

def initialize_system():
    """Initialize the RAG system components"""
    try:
        # Initialize data manager with URL content extraction
        st.session_state.data_manager = HackerNewsDataManager(extract_url_content=True)
        
        # Initialize vector store
        st.session_state.vector_store = VectorStore()
        
        # Initialize RAG agent
        st.session_state.rag_agent = RAGAgent(st.session_state.vector_store)
        
        # Initialize auto-updater
        st.session_state.auto_updater = AutoUpdater(
            vector_store=st.session_state.vector_store,
            data_manager=st.session_state.data_manager,
            update_interval_minutes=AUTO_UPDATE_INTERVAL,#in minutes
            max_new_stories=50
        )
        
        # Load initial data if vector store is empty
        if st.session_state.vector_store.get_document_count() == 0:
            with st.spinner("Loading initial HackerNews data... This may take a few minutes."):
                stories = st.session_state.data_manager.fetch_top_stories(limit=STORIES_LIMIT)
                if stories:
                    st.session_state.vector_store.add_documents(stories)
                    st.session_state.last_update = datetime.now()
                    return True
        else:
            st.session_state.last_update = datetime.now()
            return True
            
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        return False
    
    return True

def update_knowledge_base():
    """Update the knowledge base with new HackerNews content"""
    try:
        if st.session_state.data_manager and st.session_state.vector_store:
            # Fetch recent stories
            stories = st.session_state.data_manager.fetch_top_stories(limit=50)
            if stories:
                # Filter for new stories (not already in vector store)
                new_stories = []
                existing_ids = st.session_state.vector_store.get_existing_ids()
                
                for story in stories:
                    if story['id'] not in existing_ids:
                        new_stories.append(story)
                
                if new_stories:
                    st.session_state.vector_store.add_documents(new_stories)
                    st.session_state.last_update = datetime.now()
                    return len(new_stories)
        return 0
    except Exception as e:
        st.error(f"Failed to update knowledge base: {str(e)}")
        return 0

def main():
    st.set_page_config(
        page_title="HackerNews RAG Search",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("HackerNews RAG Search")
    st.markdown("Semantic search and AI-powered responses over HackerNews content")
    
    # Initialize system if not done
    if not st.session_state.initialization_complete:
        with st.spinner("Initializing RAG system..."):
            if initialize_system():
                st.session_state.initialization_complete = True
                st.success("System initialized successfully!")
                st.rerun()
            else:
                st.error("Failed to initialize system. Please check your API keys and try again.")
                return
    
    # Sidebar for system status and controls
    with st.sidebar:
        st.header("System Status")
        
        if st.session_state.last_update:
            st.success("System Ready")
            st.write(f"Last updated: {format_timestamp(st.session_state.last_update)}")
            
            if st.session_state.vector_store:
                stories_count = st.session_state.vector_store.get_stories_count()
                doc_count = st.session_state.vector_store.get_document_count()
                st.metric("Number of stories", stories_count)
                st.metric("Chunks", doc_count)
        else:
            st.warning("System Not Ready")
        
        st.header("Knowledge Base Management")
        
        # Auto-update controls
        if st.session_state.auto_updater:
            status = st.session_state.auto_updater.get_status()
            
            col1, col2 = st.columns(2)
            with col1:
                if status['is_running']:
                    st.success("Auto-update: Running")
                    if st.button("Stop Auto-Update"):
                        st.session_state.auto_updater.stop_auto_update()
                        st.rerun()
                else:
                    st.info("Auto-update: Stopped")
                    if st.button("Start Auto-Update"):
                        st.session_state.auto_updater.start_auto_update()
                        st.rerun()
            
            with col2:
                if st.button("Check for New Posts Now"):
                    with st.spinner("Checking for new posts..."):
                        new_count = st.session_state.auto_updater.manual_update()
                        if new_count > 0:
                            st.success(f"Added {new_count} new posts!")
                            st.session_state.last_update = datetime.now()
                        else:
                            st.info("No new posts found")
                    st.rerun()
        
        # Display auto-update status
        if st.session_state.auto_updater:
            status = st.session_state.auto_updater.get_status()
            st.write(f"**Update interval:** {status['update_interval_minutes']:.0f} minutes")
            st.write(f"**Posts tracked:** {status['processed_count']:,}")
            if status['last_update']:
                st.write(f"**Last auto-update:** {format_timestamp(status['last_update'])}")
        
        st.divider()
        
        # URL content extraction toggle
        extract_urls = st.checkbox("Extract content from URLs & YouTube videos", value=True, 
                                 help="When enabled, the system will extract text content from web pages and YouTube video metadata. Note: YouTube transcript extraction may be limited due to bot detection.")
        
        if extract_urls:
            st.info("Content extraction enabled: Web pages and YouTube metadata will be extracted and included in search.")
        
        if st.button("Manual Full Update"):
            with st.spinner("Updating knowledge base..."):
                # Update data manager setting
                if st.session_state.data_manager:
                    st.session_state.data_manager.extract_url_content = extract_urls
                
                new_docs = update_knowledge_base()
                if new_docs > 0:
                    st.success(f"Added {new_docs} new documents!")
                else:
                    st.info("No new documents to add.")
        
        # Auto-update toggle
        auto_update = st.checkbox("Auto-update every hour", value=False)
        
        if auto_update and st.session_state.last_update:
            time_since_update = datetime.now() - st.session_state.last_update
            if time_since_update > timedelta(hours=1):
                st.info("Auto-update triggered...")
                new_docs = update_knowledge_base()
                if new_docs > 0:
                    st.success(f"Auto-update: Added {new_docs} new documents!")
    
    # Main search interface
    if not st.session_state.initialization_complete:
        return
    
    st.header("Search HackerNews")
    
    # Search input
    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., 'What are the latest developments in AI?', 'startup funding trends', 'remote work discussions'"
    )
    
    # Search options
    col1, col2 = st.columns([1, 1])
    with col1:
        search_type = st.selectbox(
            "Search Type:",
            ["RAG Response", "Semantic Search Only"],
            help="RAG Response provides AI-generated answers, Semantic Search shows matching documents"
        )
    
    with col2:
        num_results = st.slider("Number of results:", 3, 20, 5)
    
    if query:
        if search_type == "RAG Response":
            # RAG-powered response
            with st.spinner("Generating AI response..."):
                try:
                    response, sources = st.session_state.rag_agent.generate_response(query, top_k=num_results)
                    
                    # Display AI response
                    st.subheader("🤖 AI Response")
                    st.write(response)
                    
                    # Display sources
                    if sources:
                        st.subheader("Sources")
                        for i, source in enumerate(sources, 1):
                            with st.expander(f"Source {i}: {truncate_text(source.get('title', 'No title'), 60)}"):
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"**Title:** {source.get('title', 'No title')}")
                                    if source.get('text'):
                                        st.write(f"**Content:** {truncate_text(source['text'], 200)}")
                                    
                                    # Show extracted content info
                                    if source.get('extracted_content'):
                                        content_type = source.get('content_type', 'extracted')
                                        st.write(f"**Extracted {content_type} content:** {truncate_text(source['extracted_content'], 150)}")
                                    elif source.get('extraction_error'):
                                        st.write(f"**Note:** Content extraction failed: {source['extraction_error']}")
                                    
                                    st.write(f"**Score:** {source.get('score', 0)}")
                                    if source.get('time'):
                                        st.write(f"**Posted:** {format_timestamp(datetime.fromtimestamp(source['time']))}")
                                
                                with col2:
                                    if source.get('url'):
                                        st.link_button("View Original", source['url'])
                                    hn_link = f"https://news.ycombinator.com/item?id={source.get('id', '')}"
                                    st.link_button("HN Discussion", hn_link)
                
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
        
        else:
            # Semantic search only
            with st.spinner("Searching..."):
                try:
                    results = st.session_state.vector_store.search(query, top_k=num_results)
                    
                    st.subheader(f"Search Results ({len(results)} found)")
                    
                    for i, result in enumerate(results, 1):
                        with st.container():
                            st.markdown(f"### {i}. {result.get('title', 'No title')}")
                            
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                if result.get('text'):
                                    st.write(truncate_text(result['text'], 300))
                                
                                # Show extracted content info
                                if result.get('extracted_content'):
                                    content_type = result.get('content_type', 'extracted')
                                    st.write(f"**Extracted {content_type} content:** {truncate_text(result['extracted_content'], 200)}")
                                elif result.get('extraction_error'):
                                    st.write(f"**Note:** Content extraction failed: {result['extraction_error']}")
                                
                                st.write(f"**Similarity Score:** {result.get('similarity_score', 0):.3f}")
                            
                            with col2:
                                st.write(f"**Score:** {result.get('score', 0)}")
                                if result.get('time'):
                                    st.write(f"**Posted:** {format_timestamp(datetime.fromtimestamp(result['time']))}")
                            
                            with col3:
                                if result.get('url'):
                                    st.link_button("View Original", result['url'], key=f"orig_{i}")
                                hn_link = f"https://news.ycombinator.com/item?id={result.get('id', '')}"
                                st.link_button("HN Discussion", hn_link, key=f"hn_{i}")
                            
                            st.divider()
                
                except Exception as e:
                    st.error(f"Error performing search: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with Streamlit • Data from [HackerNews API](https://github.com/HackerNews/API) • "
        "Powered by OpenAI and Sentence Transformers"
    )

if __name__ == "__main__":
    main()
