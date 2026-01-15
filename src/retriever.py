"""
Enhanced retriever with semantic search and optional re-ranking.
"""
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedRetriever:
    """Retriever with semantic search and optional re-ranking."""
    
    def __init__(
        self,
        vector_store_path: str,
        collection_name: str = "complaint_chunks",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        similarity_threshold: float = 0.6,
        enable_reranking: bool = False,
        reranker_model: str = None
    ):
        """
        Initialize the enhanced retriever.
        
        Args:
            vector_store_path: Path to ChromaDB vector store
            collection_name: Name of the collection
            embedding_model_name: Model for query embedding
            top_k: Number of results to retrieve
            similarity_threshold: Minimum similarity score
            enable_reranking: Whether to use re-ranker
            reranker_model: Cross-encoder model for re-ranking
        """
        self.vector_store_path = Path(vector_store_path)
        self.collection_name = collection_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.enable_reranking = enable_reranking
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize ChromaDB client
        logger.info(f"Loading vector store from: {vector_store_path}")
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.vector_store_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get collection
        self.collection = self.chroma_client.get_collection(self.collection_name)
        
        # Initialize re-ranker if enabled
        self.reranker = None
        if enable_reranking and reranker_model:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading re-ranker model: {reranker_model}")
                self.reranker = CrossEncoder(reranker_model)
            except ImportError:
                logger.warning("CrossEncoder not installed. Install with: pip install sentence-transformers")
                self.enable_reranking = False
        
        logger.info(f"Retriever initialized. Collection has {self.collection.count()} chunks.")
    
    def retrieve(
        self,
        query: str,
        filter_dict: Optional[Dict] = None,
        k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User's question
            filter_dict: Metadata filters (e.g., {"product_category": "Credit Card"})
            k: Number of results (overrides default top_k)
        
        Returns:
            List of retrieved chunks with metadata and scores
        """
        if k is None:
            k = self.top_k
        
        # 1. Embed the query
        query_embedding = self.embedding_model.encode(query, normalize_embeddings=True)
        
        # 2. Perform similarity search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(k * 3, 100) if self.enable_reranking else k,  # Get more for re-ranking
            where=filter_dict,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['documents'] or not results['documents'][0]:
            return []
        
        # 3. Format initial results
        chunks = []
        for i in range(len(results['documents'][0])):
            # Convert cosine distance to similarity score
            # ChromaDB returns distances (0 = identical, 2 = orthogonal)
            distance = results['distances'][0][i]
            similarity = 1 - (distance / 2)  # Convert to 0-1 similarity
            
            if similarity < self.similarity_threshold:
                continue
            
            chunks.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity': similarity,
                'distance': distance,
                'id': results['ids'][0][i]
            })
        
        # 4. Apply re-ranking if enabled
        if self.enable_reranking and self.reranker and len(chunks) > 1:
            chunks = self._rerank_chunks(query, chunks, k)
        elif len(chunks) > k:
            # Just take top-k by similarity
            chunks = sorted(chunks, key=lambda x: x['similarity'], reverse=True)[:k]
        
        logger.info(f"Retrieved {len(chunks)} chunks for query: '{query[:50]}...'")
        return chunks
    
    def _rerank_chunks(self, query: str, chunks: List[Dict], k: int) -> List[Dict]:
        """Re-rank chunks using cross-encoder."""
        # Prepare pairs for re-ranking
        pairs = [(query, chunk['text']) for chunk in chunks]
        
        # Get re-ranking scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Update chunks with re-ranking scores
        for i, chunk in enumerate(chunks):
            chunk['rerank_score'] = float(rerank_scores[i])
        
        # Sort by re-rank score
        chunks = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
        
        # Take top-k
        return chunks[:k]
    
    def retrieve_with_context(
        self,
        query: str,
        filter_dict: Optional[Dict] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve chunks and format them for LLM context.
        
        Returns:
            Dictionary with formatted context and source information
        """
        chunks = self.retrieve(query, filter_dict)
        
        if not chunks:
            return {
                'context': "",
                'sources': [],
                'chunk_count': 0
            }
        
        # Format context for LLM
        context_parts = []
        source_info = []
        
        for i, chunk in enumerate(chunks):
            # Build source string
            metadata = chunk['metadata']
            source_str = f"[Source {i+1}]"
            
            if include_metadata:
                source_details = []
                if metadata.get('product_category'):
                    source_details.append(f"Product: {metadata['product_category']}")
                if metadata.get('issue'):
                    source_details.append(f"Issue: {metadata['issue']}")
                if metadata.get('date_received'):
                    source_details.append(f"Date: {metadata['date_received'][:10]}")
                
                if source_details:
                    source_str += f" ({', '.join(source_details)})"
            
            # Add to context
            context_parts.append(f"{source_str}\n{chunk['text']}")
            source_info.append({
                'text': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                'metadata': metadata,
                'similarity': chunk['similarity'],
                'id': chunk['id']
            })
        
        context = "\n\n".join(context_parts)
        
        return {
            'context': context,
            'sources': source_info,
            'chunk_count': len(chunks),
            'avg_similarity': sum(c['similarity'] for c in chunks) / len(chunks)
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a query to suggest filters."""
        # Simple keyword-based filter suggestion
        query_lower = query.lower()
        suggested_filters = {}
        
        # Product category detection
        if any(word in query_lower for word in ['credit card', 'credit', 'card']):
            suggested_filters['product_category'] = 'Credit card'
        elif any(word in query_lower for word in ['loan', 'personal loan', 'borrow']):
            suggested_filters['product_category'] = 'Personal loan'
        elif any(word in query_lower for word in ['savings', 'account', 'bank account']):
            suggested_filters['product_category'] = 'Savings account'
        elif any(word in query_lower for word in ['money transfer', 'transfer', 'wire']):
            suggested_filters['product_category'] = 'Money transfer'
        
        # Time-related detection
        if any(word in query_lower for word in ['recent', 'last month', 'this year', '2024']):
            suggested_filters['time_period'] = 'recent'
        
        return {
            'query': query,
            'suggested_filters': suggested_filters,
            'query_length': len(query),
            'word_count': len(query.split())
        }