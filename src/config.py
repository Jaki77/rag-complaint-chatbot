"""
Configuration management for RAG pipeline.
"""
import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
import os

@dataclass
class RetrieverConfig:
    top_k: int = 5
    similarity_threshold: float = 0.6
    enable_reranking: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"

@dataclass
class GeneratorConfig:
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    temperature: float = 0.3
    max_new_tokens: int = 512
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    device_map: str = "auto"
    load_in_4bit: bool = True  # For memory efficiency

@dataclass
class PromptConfig:
    system_prompt: str = ""
    analysis_template: str = ""
    summarization_template: str = ""

@dataclass
class RAGConfig:
    retriever: RetrieverConfig
    generator: GeneratorConfig
    prompts: PromptConfig
    vector_store_path: str = "./vector_store/full"
    collection_name: str = "complaint_chunks"
    
    @classmethod
    def from_yaml(cls, config_path: str = None):
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "rag_config.yaml"
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        rag_config = config_dict.get('rag', {})
        
        return cls(
            retriever=RetrieverConfig(**rag_config.get('retriever', {})),
            generator=GeneratorConfig(**rag_config.get('generator', {})),
            prompts=PromptConfig(**rag_config.get('prompts', {})),
            vector_store_path=rag_config.get('vector_store_path', './vector_store/full'),
            collection_name=rag_config.get('collection_name', 'complaint_chunks')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'retriever': self.retriever.__dict__,
            'generator': self.generator.__dict__,
            'prompts': self.prompts.__dict__,
            'vector_store_path': self.vector_store_path,
            'collection_name': self.collection_name
        }