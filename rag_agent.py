import os
from typing import List, Dict, Tuple
from openai import OpenAI
import logging

class RAGAgent:
    """RAG agent that combines retrieval with generation"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        
        # Initialize OpenAI client
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = "gpt-4o"
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def generate_response(self, query: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
        """Generate RAG response for a query"""
        
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.search(query, top_k=top_k)
        
        if not retrieved_docs:
            return "I couldn't find any relevant information in the HackerNews knowledge base for your query.", []
        
        # Sort by score descending
        retrieved_docs = sorted(retrieved_docs, key=lambda x: x.get("score", 0), reverse=True)
        
        # Prepare context from retrieved documents
        context = self._prepare_context(retrieved_docs)
        
        # Generate response using OpenAI
        response = self._generate_with_context(query, context)
        
        return response, retrieved_docs
    
    def _prepare_context(self, documents: List[Dict]) -> str:
        """Prepare context string from retrieved documents"""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            doc_text = f"Document {i}:\n"
            doc_text += f"Title: {doc.get('title', 'No title')}\n"
            
            if doc.get('text'):
                # Truncate very long texts
                text = doc['text']
                if len(text) > 1000:
                    text = text[:1000] + "..."
                doc_text += f"Content: {text}\n"
            
            if doc.get('url'):
                doc_text += f"URL: {doc['url']}\n"
            
            doc_text += f"Score: {doc.get('score', 0)}\n"
            doc_text += f"Similarity: {doc.get('similarity_score', 0):.3f}\n"
            
            # Add top comments if available
            if doc.get('comments'):
                comments = doc['comments'][:2]  # Limit to top 2 comments
                if comments:
                    doc_text += "Top Comments:\n"
                    for j, comment in enumerate(comments, 1):
                        if comment.get('text'):
                            comment_text = comment['text']
                            if len(comment_text) > 200:
                                comment_text = comment_text[:200] + "..."
                            doc_text += f"  Comment {j}: {comment_text}\n"
            
            context_parts.append(doc_text)
        
        return "\n" + "="*50 + "\n".join(context_parts)
    
    def _generate_with_context(self, query: str, context: str) -> str:
        """Generate response using retrieved context"""
        
        system_prompt = """You are an AI assistant that helps users find and understand information from online discussions.

Instructions:
1. Use the provided context to answer the user's question accurately.
2. Be specific — explain how the tool or concept solves the *exact* problem described in the query.
3. Synthesize information from multiple sources when relevant.
4. Do not mention anything about the origin of the context.
5. Be conversational, under 120 words.
6. Highlight interesting capabilities or benefits relevant to the query.
7. Avoid generic summaries; focus on what directly answers the user’s request.
        """
        
        user_prompt = f"""Based on the following HackerNews content, please answer this question: {query}

Context from HackerNews:
{context}

Please provide a specific, helpful answer (under 120 words) that directly addresses the user's request."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"I encountered an error while generating a response: {str(e)}"
    
    def summarize_document(self, document: Dict) -> str:
        """Generate a summary of a specific document"""
        
        context = self._prepare_context([document])
        
        system_prompt = """You are an AI assistant that creates concise summaries of HackerNews content.
        Create a brief, informative summary that captures the key points and discussions."""
        
        user_prompt = f"""Please provide a concise summary of this HackerNews content:

{context}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.5
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return f"Error generating summary: {str(e)}"
    
    def analyze_trends(self, query: str, top_k: int = 20) -> str:
        """Analyze trends in HackerNews content"""
        
        retrieved_docs = self.vector_store.search(query, top_k=top_k)
        
        if not retrieved_docs:
            return "No relevant documents found for trend analysis."
        
        context = self._prepare_context(retrieved_docs)
        
        system_prompt = """You are an AI assistant that analyzes trends and patterns in HackerNews discussions.
        Look for recurring themes, popular topics, emerging technologies, and community sentiment."""
        
        user_prompt = f"""Based on the following HackerNews content related to "{query}", please analyze trends and patterns:

{context}

Please provide insights about:
1. Common themes and topics
2. Community sentiment and opinions
3. Emerging trends or technologies mentioned
4. Popular discussions or controversies
5. Any notable patterns in the data"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1200,
                temperature=0.6
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error analyzing trends: {e}")
            return f"Error analyzing trends: {str(e)}"
