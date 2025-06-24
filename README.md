# Hacker News Rag

This is a lean Rag based semantic search solution built on Streamlit, in-memory Vectore-store,
OpenAI Chatcompletions API for LLM calls, all-MiniLM-L6-v2 for embeddings.

In-memory Vector-store will be replaced with Weaviate whenever I get time to do that.  More details
as follows.

## 🚀 Installation

1. Copy the `.env.sample` file and rename it to `.env`.
2. Fill in your OpenAI API key.
3. Use Python >= 3.12.0.
4. Install dependencies:

```bash
pip install streamlit && pip install uv
uv sync  # Sync and install remaining dependencies
```

5. Run the app: (App available on:  localhost:5000. this will start fetch 100 new stories since the app is started in the browser)

```bash
streamlit run app.py --server.port 5000
```

Or use the preconfigured VSCode debug config to run directly with debugpy.

## ✅ Assumptions

- This is a quick prototype (no more than 2 days of effort).
- Search accuracy is prioritized over performance.
- Python and Streamlit are used for speed and simplicity.
- Start with an in-memory vector store for fast local debugging  (cost-effective, easy to debug ETL pipeline remotely). and introduce weaviate vector database later
- Embedding model: `all-MiniLM-L6-v2` for fast and semantically rich embeddings.
- Scraping goal: at least 95% URL success rate; 100% is not required.
- Content is extracted from YouTube stories to preserve full context.


## 🧠 Solution

- Semantic search works reliably.
- Failed to integrate weaviate because of time constrains, but can be done with 1-3 hours of time, if it is needed by the assesment comitee
- Failed to enable follow-up questions like ChatGPT style, which is not so hard to implement but will take some time.
- If a story doesn’t appear in the top results, it’s likely due to:
    - Failed URL extraction
    - YouTube video processing issues (~1–5% failure rate)
- A sidebar lists all included stories for reference of stories that are included
- Automatic post scanning is implemented but disabled by default due to memory limits.


## 📉 Story Handling

- Loads 100 recent posts initially.
- Manual loading works, but can be done within fe minutes after starting:
    - Rapid growth in new posts (e.g., 100+) can cause the in-memory vector store to exceed memory limits and crash the app.
    - But I tried the auto-updater with 5 minutes interval, which worked fine. You may also try it by changing the following constant in app.py
![image](https://github.com/user-attachments/assets/7e940561-7faf-4bea-9cf1-bef5c5266762)



## 📚 Technical Notes

- Chunking: 1000 characters per chunk with a 100-character overlap
- URL-extracted text is included inline with "text from story" to ensure consistent chunking strategy.


## 🔧 Future Improvements

- Add unit and integration tests
- Implement LangChain `ConversationBufferMemory` (or similar) to support follow-up Q\&A like ChatGPT
- Hybrid search with filters (e.g., by story-point, date) and re-ranking
- Introduce feedback scheme from users where they can upvote or downvoe certain document
- Self-improving prompts
- For production, bigger models with more accuracy can be employed, depending upon customer needs, scope and budget

**Add LangChain Tools to:**

- Let users search through comments
- Advanced comment-centric queries using LangChain tool, especially for digging through comments

**Key Takeaway**
    -It was easy to this assignment because I build 3 chatbots I worked at 2 different companies in the past
    -But spent a lot of time on ETL, specially with youtube data extraction
    -They were hard to build and maintain because they were of python backend and react.js frontend and weaviate for vector store
    -Since this one is built with Streamlit, it was relatively easy


![Included stories shown on sidebar](https://github.com/user-attachments/assets/be81e28a-5803-46ec-a280-78daa485f984)


![Successful Search](https://github.com/user-attachments/assets/f94f398e-10c9-406b-9ce1-c2442a753dae)

![Search for a topic](https://github.com/user-attachments/assets/db43175b-efdd-46f5-a850-df0a8a032d23)

![Failed search because of failed Website extraction](https://github.com/user-attachments/assets/fab8b275-146f-43f3-a330-b9845940f804)






