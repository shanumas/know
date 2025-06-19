# HackerNews RAG Application

## Overview

This is a Retrieval-Augmented Generation (RAG) application built with Streamlit that provides intelligent search and Q&A capabilities over HackerNews data. The system fetches stories from the HackerNews API, stores them in a vector database for semantic search, and uses OpenAI's GPT-4o model to generate contextual responses based on retrieved content.

## System Architecture

The application follows a modular RAG architecture with clear separation of concerns:

- **Frontend**: Streamlit web interface for user interaction
- **Data Layer**: HackerNews API integration with rate limiting and caching
- **Vector Storage**: FAISS-based semantic search using sentence transformers
- **Generation**: OpenAI GPT-4o for response generation
- **Orchestration**: RAG agent that coordinates retrieval and generation

## Key Components

### 1. Data Management (`hn_data_manager.py`)
- **Purpose**: Fetches and processes HackerNews data
- **Features**: Rate limiting, retry logic, concurrent processing
- **API**: Uses HackerNews Firebase API v0
- **Rate Limiting**: 10 requests per second with exponential backoff

### 2. Vector Store (`vector_store.py`)
- **Technology**: Simple TF-IDF based similarity search
- **Model**: Custom TF-IDF implementation for text embeddings
- **Storage**: Persistent JSON + pickle metadata files
- **Search**: Cosine similarity with sparse vectors
- **Text Processing**: Combines title, content, extracted URL content, and top comments for embedding

### 3. RAG Agent (`rag_agent.py`)
- **Model**: OpenAI GPT-4o (latest as of May 2024)
- **Process**: Query → Retrieval → Context preparation → Generation
- **Context Window**: Configurable top-k document retrieval
- **Response**: Combines generated answer with source documents

### 4. Web Interface (`app.py`)
- **Framework**: Streamlit with session state management
- **Features**: Real-time chat interface, document preview, system status, URL extraction controls
- **Initialization**: Lazy loading of components and data
- **Updates**: Background data refresh capabilities with optional content extraction

### 5. Content Extractor (`content_extractor.py`)
- **Web Pages**: Uses trafilatura for robust content extraction
- **YouTube Videos**: Uses yt-dlp for transcript extraction
- **Fallback**: Simple HTML parsing when dependencies unavailable
- **Integration**: Automatically enhances HackerNews documents with extracted content

### 6. Utilities (`utils.py`)
- **Timestamp Formatting**: Human-readable time differences
- **Text Processing**: Truncation and HTML cleaning
- **Display Helpers**: Content formatting for UI

## Data Flow

1. **Data Ingestion**: HackerNews API → Rate-limited fetching → Document processing
2. **Content Enhancement**: URL extraction → Web/YouTube content → Enhanced documents
3. **Vectorization**: Text preparation → TF-IDF encoding → In-memory index
4. **Query Processing**: User query → Vector search → Top-k retrieval
5. **Response Generation**: Retrieved context → OpenAI API → Generated response
6. **UI Rendering**: Response + sources → Streamlit interface

## External Dependencies

### Core Services
- **HackerNews API**: Primary data source (hacker-news.firebaseio.com)
- **OpenAI API**: GPT-4o for text generation (requires OPENAI_API_KEY)

### Key Libraries
- **Streamlit**: Web framework and UI
- **TF-IDF**: Simple vector similarity search (fallback implementation)
- **OpenAI**: LLM API client
- **Requests**: HTTP client with session management
- **Trafilatura**: Web content extraction
- **yt-dlp**: YouTube video transcript extraction

### ML Models
- **all-MiniLM-L6-v2**: Lightweight sentence transformer (384 dimensions)
- **GPT-4o**: OpenAI's latest multimodal model for generation

## Deployment Strategy

### Replit Configuration
- **Runtime**: Python 3.11 with Nix package management
- **Port**: 5000 (configured for Streamlit)
- **Deployment**: Autoscale target for production
- **Process**: Streamlit server with custom port binding

### Environment Requirements
- **OPENAI_API_KEY**: Required environment variable
- **Storage**: Persistent file system for FAISS indices
- **Memory**: Sufficient for sentence transformer model and vector index
- **Network**: Outbound access to HackerNews and OpenAI APIs

### Scaling Considerations
- Vector store persists to disk for quick startup
- Rate limiting prevents API abuse
- Concurrent document processing for faster ingestion
- Session state management for multi-user scenarios

## Changelog

```
Changelog:
- June 19, 2025. Initial setup
- June 19, 2025. Enhanced with URL content extraction:
  - Added trafilatura for web page text extraction
  - Added yt-dlp for YouTube video transcript extraction with HTML fallback
  - Improved YouTube metadata extraction to get proper video titles and descriptions
  - Updated vector embeddings to include extracted content from URLs
  - Added user controls for content extraction settings
  - Fixed generic YouTube content extraction issue
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```