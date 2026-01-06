"""
Vector database creation and management.
"""
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import logging

# Try to import FAISS and ChromaDB
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available. Install with: pip install faiss-cpu")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("ChromaDB not available. Install with: pip install chromadb")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """Base class for vector store operations."""
    
    def __init__(
        self,
        store_type: str = 'chroma',  # 'chroma' or 'faiss'
        persist_directory: str = './vector_store',
        collection_name: str = 'complaint_chunks'
    ):
        """
        Initialize vector store.
        
        Args:
            store_type: Type of vector store ('chroma' or 'faiss')
            persist_directory: Directory to persist the vector store
            collection_name: Name of the collection (for ChromaDB)
        """
        self.store_type = store_type
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.store = None
        
        # Create persist directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
    
    def create_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        batch_size: int = 1000
    ) -> None:
        """
        Create vector store from chunks and embeddings.
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: Corresponding embeddings
            batch_size: Batch size for adding to store
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")
        
        logger.info(f"Creating {self.store_type} vector store with {len(chunks):,} chunks...")
        
        if self.store_type == 'chroma':
            self._create_chroma_store(chunks, embeddings, batch_size)
        elif self.store_type == 'faiss':
            self._create_faiss_store(chunks, embeddings)
        else:
            raise ValueError(f"Unsupported store type: {self.store_type}")
    
    def _create_chroma_store(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        batch_size: int
    ) -> None:
        """Create ChromaDB vector store."""
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB is not installed")
        
        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(
            path=str(self.persist_directory / 'chroma_db'),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Delete existing collection if it exists
        try:
            chroma_client.delete_collection(self.collection_name)
            logger.info(f"Deleted existing collection: {self.collection_name}")
        except:
            pass
        
        # Create new collection
        collection = chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )
        
        # Prepare data for batch addition
        total_chunks = len(chunks)
        
        for i in range(0, total_chunks, batch_size):
            batch_end = min(i + batch_size, total_chunks)
            batch_chunks = chunks[i:batch_end]
            batch_embeddings = embeddings[i:batch_end]
            
            # Prepare batch data
            ids = []
            metadatas = []
            documents = []
            
            for j, chunk in enumerate(batch_chunks):
                chunk_id = f"chunk_{i + j:08d}"
                ids.append(chunk_id)
                documents.append(chunk['text'])
                metadatas.append(chunk['metadata'])
            
            # Add to collection
            collection.add(
                embeddings=batch_embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            if (i // batch_size) % 10 == 0:  # Log every 10 batches
                logger.info(f"Added {batch_end:,}/{total_chunks:,} chunks")
        
        self.store = collection
        logger.info(f"ChromaDB store created with {total_chunks:,} chunks")
    
    def _create_faiss_store(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray
    ) -> None:
        """Create FAISS vector store."""
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not installed")
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        
        # Use IndexFlatIP for inner product (cosine similarity when vectors are normalized)
        index = faiss.IndexFlatIP(dimension)
        
        # Add vectors to index
        index.add(embeddings.astype('float32'))
        
        # Save index
        index_path = self.persist_directory / 'faiss_index.bin'
        faiss.write_index(index, str(index_path))
        
        # Save metadata
        metadata_path = self.persist_directory / 'metadata.pkl'
        metadata = {
            'chunks': chunks,
            'embedding_dimension': dimension,
            'total_chunks': len(chunks)
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        self.store = index
        logger.info(f"FAISS index created with {len(chunks):,} vectors")
        logger.info(f"Index saved to: {index_path}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            filter_dict: Metadata filters (only for ChromaDB)
        
        Returns:
            List of search results
        """
        if self.store is None:
            raise ValueError("Vector store not initialized")
        
        if self.store_type == 'chroma':
            return self._chroma_search(query_embedding, k, filter_dict)
        elif self.store_type == 'faiss':
            return self._faiss_search(query_embedding, k)
    
    def _chroma_search(
        self,
        query_embedding: np.ndarray,
        k: int,
        filter_dict: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        """Search in ChromaDB."""
        results = self.store.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=filter_dict
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': results['distances'][0][i],
                'id': results['ids'][0][i]
            })
        
        return formatted_results
    
    def _faiss_search(
        self,
        query_embedding: np.ndarray,
        k: int
    ) -> List[Dict[str, Any]]:
        """Search in FAISS index."""
        # Load metadata
        metadata_path = self.persist_directory / 'metadata.pkl'
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Search
        distances, indices = self.store.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k
        )
        
        # Format results
        formatted_results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(metadata['chunks']):
                chunk = metadata['chunks'][idx]
                formatted_results.append({
                    'text': chunk['text'],
                    'metadata': chunk['metadata'],
                    'score': float(distances[0][i]),
                    'id': f"chunk_{idx:08d}"
                })
        
        return formatted_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if self.store_type == 'chroma' and self.store is not None:
            count = self.store.count()
            return {
                'store_type': 'chroma',
                'chunk_count': count,
                'collection_name': self.collection_name
            }
        elif self.store_type == 'faiss':
            metadata_path = self.persist_directory / 'metadata.pkl'
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                return {
                    'store_type': 'faiss',
                    'chunk_count': metadata['total_chunks'],
                    'embedding_dimension': metadata['embedding_dimension']
                }
        
        return {'store_type': self.store_type, 'chunk_count': 0}