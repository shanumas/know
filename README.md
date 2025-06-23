# know
 
🚀 Installation

Copy the .env.sample file and rename it to .env.
Fill in your OpenAI API key.

Use Python >= 3.12.0.

Install dependencies:

    pip install streamlit && pip install uv
    uv sync  # Sync and install remaining dependencies

Run the app:
    To use my snapshot of chunks, just run the script.
    To fetch fresh stories, delete hn_index.json(chunks) first, then run the script.
    
    option 1: streamlit run app.py --server.port 5000
    option 2: Or use the preconfigured VSCode debug config to run directly with debugpy.

✅ Assumptions
    This is a quick prototype (no more than 2 days of effort).

    Search accuracy is prioritized over performance.

    Python and Streamlit are used for speed and simplicity.

    Start with an in-memory vector store for fast local debugging (cost-effective, easy to debug ETL pipeline locally before to remote vector database - weaviate in this case).

    Final step: move to Weaviate 

    Embedding model: all-MiniLM-L6-v2 for fast and semantically rich embeddings.

    Scraping goal: at least 95% URL success rate; 100% is not required.

    Content is extracted from YouTube stories to preserve full context.

🧠 Solution
    Semantic search works reliably.

    Failed to enable follow-up questions like chat-gpt style, which is not so hard to implement but will take some time

    If a story doesn’t appear in the top results, it’s likely due to:
        *Failed URL extraction
        *YouTube video processing issues (~1–5% failure rate)

    A sidebar lists all included stories for reference.

    Automatic post scanning is implemented but disabled by default due to memory limits.

📉 Story Handling
    Loads 100 recent posts initially.

    Manual loading works, but:

    If many new posts arrive (e.g. >100 after a few hours), the in-memory store might crash.

    High-frequency updates (~1 post every 2 minutes) require a scalable Vector Database.

📚 Technical Notes
    Chunking: 1000 characters per chunk with a 100-character sliding window.

    URL-extracted text is included inline to ensure consistent chunking strategy.

🔧 Future Improvements
    Add unit and integration tests

    Implement LangChain ConversationBufferMemory (or similar) to support follow-up Q&A like ChatGPT

    Hybrid search with filters (e.g., by story-point, date) to improve ranking

Add LangChain Tools to:
    *Let users search through comments
    *Advanced comment-centric queries using langchain tool specially for digging throught comments

