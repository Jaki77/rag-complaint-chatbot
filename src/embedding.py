"""
Embedding model for converting text to vectors.
"""
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper for sentence transformer embedding model."""
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        device: str = None,
        batch_size: int = 32
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the SentenceTransformer model
            device: Device to run on ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        
        # Auto-select device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Loading model: {model_name} on {device}")
        
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.device = device
            self.batch_size = batch_size
            
            # Get model info
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            logger.info(f"Max sequence length: {self.model.max_seq_length}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def encode(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode a list of texts to embeddings.
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar
        
        Returns:
            Numpy array of embeddings (n_texts x embedding_dim)
        """
        if not texts:
            return np.array([])
        
        logger.info(f"Encoding {len(texts):,} texts in batches of {self.batch_size}...")
        
        # Clean texts
        cleaned_texts = [self._clean_text(text) for text in texts]
        
        # Encode in batches
        embeddings = []
        
        if show_progress:
            pbar = tqdm(total=len(cleaned_texts), desc="Encoding texts")
        
        for i in range(0, len(cleaned_texts), self.batch_size):
            batch = cleaned_texts[i:i + self.batch_size]
            
            # Encode batch
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Normalize for cosine similarity
                show_progress_bar=False
            )
            
            embeddings.append(batch_embeddings)
            
            if show_progress:
                pbar.update(len(batch))
        
        if show_progress:
            pbar.close()
        
        # Combine all batches
        all_embeddings = np.vstack(embeddings)
        
        logger.info(f"Encoding complete. Shape: {all_embeddings.shape}")
        
        return all_embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text."""
        return self.encode([text], show_progress=False)[0]
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text before encoding.
        
        Args:
            text: Input text
        
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (keep model's max length in tokens)
        # Note: This is approximate - models have token limits
        if len(text) > 10000:  # Very conservative limit
            text = text[:10000]
        
        return text
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'device': self.device,
            'max_sequence_length': self.model.max_seq_length,
            'vocab_size': getattr(self.model, 'vocab_size', 'Unknown')
        }


class CachedEmbeddingModel(EmbeddingModel):
    """Embedding model with caching for repeated texts."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}
    
    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Encode with caching."""
        # Find unique texts
        unique_texts = []
        text_to_idx = []
        cache_hits = 0
        
        for text in texts:
            if text in self.cache:
                cache_hits += 1
            else:
                if text not in unique_texts:
                    unique_texts.append(text)
                text_to_idx.append(len(unique_texts) - 1)
        
        if cache_hits > 0:
            logger.info(f"Cache hits: {cache_hits}/{len(texts)} ({cache_hits/len(texts)*100:.1f}%)")
        
        # Encode unique texts
        if unique_texts:
            unique_embeddings = super().encode(unique_texts, show_progress)
            
            # Cache results
            for text, embedding in zip(unique_texts, unique_embeddings):
                self.cache[text] = embedding
        
        # Build final embeddings array
        embeddings = []
        for text in texts:
            embeddings.append(self.cache[text])
        
        return np.vstack(embeddings)