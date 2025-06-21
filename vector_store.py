import pickle
import os
from typing import List, Dict, Any, Optional
import logging
import json
import math
import re
from collections import Counter

class VectorStore:
    
    def __init__(self, model_name: str = "simple-tfidf", index_file: str = "hn_index.json", metadata_file: str = "hn_metadata.pkl"):
        self.model_name = model_name
        self.index_file = index_file
        self.metadata_file = metadata_file
        
        self.documents = []  # Store all documents with embeddings
        self.metadata = []  # Store document metadata
        self.id_to_idx = {}  # Map document IDs to index positions
        self.vocabulary = {}
        self.idf_scores = {}
        self.doc_count = 0
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load existing index if available
        self._load_index()
    
    def _normalize_embeddings(self, embeddings: List[Dict]) -> List[Dict]:
        """Normalize embeddings for cosine similarity"""
        return embeddings  # Simple TF-IDF doesn't need normalization
    
    def _prepare_text_for_embedding(self, document: Dict) -> str:
        """Prepare document text for embedding"""
        text_parts = []
        
        # Add title
        if document.get('title'):
            text_parts.append(f"Title: {document['title']}")
        
        # Add main text
        if document.get('text'):
            text_parts.append(f"Content: {document['text']}")
        
        # Add extracted content from URLs (web pages, YouTube transcripts, etc.)
        if document.get('extracted_content'):
            content_type = document.get('content_type', 'extracted')
            text_parts.append(f"Extracted {content_type} content: {document['extracted_content']}")
        
        # Add extracted title if different from main title
        if document.get('extracted_title') and document.get('extracted_title') != document.get('title'):
            text_parts.append(f"Extracted title: {document['extracted_title']}")
        
        # Add top comments
        if document.get('comments'):
            comment_texts = []
            for comment in document['comments'][:3]:  # Limit to top 3 comments
                if comment.get('text'):
                    comment_texts.append(comment['text'][:200])  # Truncate long comments
            
            if comment_texts:
                text_parts.append(f"Comments: {' '.join(comment_texts)}")
        
        return ' '.join(text_parts)
    
    def add_documents(self, documents: List[Dict]) -> None:
        """Add documents to the vector store"""
        if not documents:
            return
        
        self.logger.info(f"Adding {len(documents)} documents to vector store...")
        
        # Filter out documents that already exist
        new_documents = []
        for doc in documents:
            doc_id = doc.get('id')
            if doc_id and doc_id not in self.id_to_idx:
                new_documents.append(doc)
        
        if not new_documents:
            self.logger.info("No new documents to add")
            return
        
        # Prepare texts for embedding
        texts = [self._prepare_text_for_embedding(doc) for doc in new_documents]
        
        # Generate embeddings
        embeddings = self._simple_encode(texts)
        
        # Add to documents
        start_idx = len(self.documents)
        for i, doc in enumerate(new_documents):
            idx = start_idx + i
            doc_with_embedding = doc.copy()
            doc_with_embedding['embedding'] = embeddings[i]
            self.documents.append(doc_with_embedding)
            self.metadata.append(doc)
            if doc.get('id'):
                self.id_to_idx[doc['id']] = idx
        
        self.doc_count = len(self.documents)
        
        self.logger.info(f"Added {len(new_documents)} new documents. Total: {self.doc_count}")
        
        # Save updated index
        self._save_index()
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform semantic search"""
        if self.doc_count == 0:
            return []
        
        # Generate query embedding
        query_embedding = self._simple_encode([query])[0]
        
        # Calculate similarity with all documents
        similarities = []
        for i, doc in enumerate(self.documents):
            similarity = self._cosine_similarity(query_embedding, doc['embedding'])
            similarities.append((similarity, i))
        
        # Sort by similarity and get top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for similarity, idx in similarities[:top_k]:
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(similarity)
                results.append(result)
        
        return results
    
    def get_document_count(self) -> int:
        """Get the number of documents in the vector store"""
        return self.doc_count
    
    def get_existing_ids(self) -> set:
        """Get set of existing document IDs"""
        return set(self.id_to_idx.keys())
    
    def _save_index(self) -> None:
        """Save index and metadata to disk"""
        try:
            # Save documents with embeddings
            with open(self.index_file, 'w') as f:
                json.dump({
                    'documents': self.documents,
                    'vocabulary': self.vocabulary,
                    'idf_scores': self.idf_scores,
                    'doc_count': self.doc_count
                }, f)
            
            with open(self.metadata_file, 'wb') as f:
                pickle.dump((self.metadata, self.id_to_idx), f)
            self.logger.info("Index saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")
    
    def _load_index(self) -> None:
        """Load index and metadata from disk"""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                with open(self.index_file, 'r') as f:
                    data = json.load(f)
                    self.documents = data.get('documents', [])
                    self.vocabulary = data.get('vocabulary', {})
                    self.idf_scores = data.get('idf_scores', {})
                    self.doc_count = data.get('doc_count', 0)
                
                with open(self.metadata_file, 'rb') as f:
                    self.metadata, self.id_to_idx = pickle.load(f)
                self.logger.info(f"Loaded index with {self.doc_count} documents")
            else:
                self.logger.info("No existing index found, starting fresh")
        except Exception as e:
            self.logger.error(f"Failed to load index: {e}")
            # Reset to empty state
            self.documents = []
            self.metadata = []
            self.id_to_idx = {}
            self.doc_count = 0
    
    def clear(self) -> None:
        """Clear the vector store"""
        self.documents = []
        self.metadata = []
        self.id_to_idx = {}
        self.doc_count = 0
        
        # Remove saved files
        for file_path in [self.index_file, self.metadata_file]:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        self.logger.info("Vector store cleared")
    
    def _simple_encode(self, texts: List[str]) -> List[Dict]:
        """Simple TF-IDF based encoding"""
        # Tokenize and build vocabulary
        all_words = set()
        tokenized_texts = []
        
        for text in texts:
            # Simple tokenization
            words = re.findall(r'\b\w+\b', text.lower())
            tokenized_texts.append(words)
            all_words.update(words)
        
        # Update vocabulary
        for word in all_words:
            if word not in self.vocabulary:
                self.vocabulary[word] = len(self.vocabulary)
        
        # Calculate IDF scores
        total_docs = max(self.doc_count, len(tokenized_texts))
        for word in all_words:
            if word not in self.idf_scores:
                # Count documents containing this word in current batch + existing docs
                doc_freq = sum(1 for tokens in tokenized_texts if word in tokens)
                # Simple IDF calculation
                self.idf_scores[word] = math.log(total_docs / max(1, doc_freq))
        
        # Create embeddings
        embeddings = []
        for tokens in tokenized_texts:
            # Create TF-IDF vector as dict
            tf_counts = Counter(tokens)
            embedding = {}
            
            for word, tf in tf_counts.items():
                if word in self.vocabulary:
                    tfidf = tf * self.idf_scores.get(word, 0)
                    embedding[word] = tfidf
            
            embeddings.append(embedding)
        
        return embeddings
    
    def _cosine_similarity(self, vec1: Dict, vec2: Dict) -> float:
        """Calculate cosine similarity between two sparse vectors"""
        # Get common words
        common_words = set(vec1.keys()) & set(vec2.keys())
        
        if not common_words:
            return 0.0
        
        # Calculate dot product
        dot_product = sum(vec1[word] * vec2[word] for word in common_words)
        
        # Calculate magnitudes
        mag1 = math.sqrt(sum(val * val for val in vec1.values()))
        mag2 = math.sqrt(sum(val * val for val in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
