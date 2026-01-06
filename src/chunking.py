"""
Text chunking utilities for splitting long complaint narratives.
"""
from typing import List, Dict, Any
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd


class ComplaintChunker:
    """Custom chunker for financial complaint narratives."""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum size of chunks (in characters)
            chunk_overlap: Overlap between chunks (in characters)
            separators: List of separators to use for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if separators is None:
            separators = ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False
        )
    
    def chunk_complaint(
        self,
        complaint_id: str,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Split a single complaint into chunks.
        
        Args:
            complaint_id: Unique identifier for the complaint
            text: Complaint narrative text
            metadata: Additional metadata for the complaint
        
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []
        
        # Split the text
        chunks = self.text_splitter.split_text(text)
        
        # Create chunk documents with metadata
        chunk_documents = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'complaint_id': complaint_id,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_text_length': len(chunk_text)
            })
            
            chunk_documents.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })
        
        return chunk_documents
    
    def chunk_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'Consumer complaint narrative_cleaned',
        id_column: str = 'complaint_id'
    ) -> List[Dict[str, Any]]:
        """
        Chunk all complaints in a DataFrame.
        
        Args:
            df: DataFrame containing complaints
            text_column: Name of the column with text to chunk
            id_column: Name of the column with complaint IDs
        
        Returns:
            List of all chunk documents
        """
        all_chunks = []
        
        for _, row in df.iterrows():
            # Extract metadata
            metadata = {
                'product_category': row.get('Product_standardized', 'Unknown'),
                'product': row.get('Product', 'Unknown'),
                'issue': row.get('Issue', 'Unknown'),
                'sub_issue': row.get('Sub-issue', 'Unknown'),
                'company': row.get('Company', 'Unknown'),
                'state': row.get('State', 'Unknown'),
                'date_received': row.get('Date received', 'Unknown'),
                'original_word_count': row.get('cleaned_word_count', 0)
            }
            
            # Chunk the complaint
            chunks = self.chunk_complaint(
                complaint_id=str(row[id_column]),
                text=str(row[text_column]),
                metadata=metadata
            )
            
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def analyze_chunking_results(
        self,
        chunks: List[Dict[str, Any]],
        original_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze chunking results for reporting.
        
        Returns:
            Dictionary with chunking statistics
        """
        total_chunks = len(chunks)
        total_complaints = len(original_df)
        
        # Calculate chunks per complaint
        chunks_per_complaint = {}
        for chunk in chunks:
            complaint_id = chunk['metadata']['complaint_id']
            chunks_per_complaint[complaint_id] = chunks_per_complaint.get(complaint_id, 0) + 1
        
        avg_chunks_per_complaint = total_chunks / total_complaints
        
        # Calculate chunk length distribution
        chunk_lengths = [len(chunk['text']) for chunk in chunks]
        
        # Calculate how many complaints were not chunked
        chunked_complaints = set(chunks_per_complaint.keys())
        total_complaint_ids = set(original_df['complaint_id'].astype(str))
        not_chunked = total_complaint_ids - chunked_complaints
        
        return {
            'total_chunks': total_chunks,
            'total_complaints': total_complaints,
            'avg_chunks_per_complaint': avg_chunks_per_complaint,
            'chunk_length_stats': {
                'min': min(chunk_lengths),
                'max': max(chunk_lengths),
                'mean': sum(chunk_lengths) / len(chunk_lengths),
                'median': sorted(chunk_lengths)[len(chunk_lengths) // 2]
            },
            'chunks_per_complaint_distribution': pd.Series(list(chunks_per_complaint.values())).describe().to_dict(),
            'complaints_not_chunked': len(not_chunked),
            'chunking_efficiency': (total_chunks / len(original_df[text_column].str.cat(sep=' ').count(' '))) if total_chunks > 0 else 0
        }


def experiment_with_chunking(
    df: pd.DataFrame,
    sample_size: int = 100
) -> Dict[str, Any]:
    """
    Experiment with different chunking parameters.
    
    Args:
        df: DataFrame with complaint narratives
        sample_size: Number of complaints to use for experimentation
    
    Returns:
        Dictionary with experiment results
    """
    sample_df = df.sample(min(sample_size, len(df)), random_state=42)
    
    experiments = []
    
    # Test different chunk sizes
    chunk_sizes = [300, 500, 800, 1000]
    chunk_overlaps = [0, 50, 100]
    
    for chunk_size in chunk_sizes:
        for chunk_overlap in chunk_overlaps:
            if chunk_overlap >= chunk_size:
                continue
            
            chunker = ComplaintChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            chunks = chunker.chunk_dataframe(sample_df)
            
            # Calculate statistics
            avg_chunks = len(chunks) / len(sample_df)
            chunk_lengths = [len(chunk['text']) for chunk in chunks]
            avg_length = sum(chunk_lengths) / len(chunk_lengths)
            
            # Calculate information preservation score
            # (how much of original text is captured without duplication)
            original_text_length = sum(sample_df['cleaned_word_count'])
            chunked_text_length = sum(chunk_lengths)
            preservation_score = min(1.0, original_text_length / chunked_text_length) if chunked_text_length > 0 else 0
            
            experiments.append({
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'avg_chunks_per_complaint': avg_chunks,
                'avg_chunk_length': avg_length,
                'preservation_score': preservation_score,
                'total_chunks': len(chunks)
            })
    
    return pd.DataFrame(experiments)