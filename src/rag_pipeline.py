"""
Main RAG pipeline integrating retriever and generator.
"""
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass
import time

from .retriever import EnhancedRetriever
from .generator import LLMGenerator, GenerationResult
from .prompts import PromptManager
from .config import RAGConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Container for complete RAG response."""
    answer: str
    sources: List[Dict[str, Any]]
    retrieved_chunks: int
    generation_stats: Dict[str, Any]
    query_analysis: Dict[str, Any]
    processing_time: float

class RAGPipeline:
    """Main RAG pipeline combining retrieval and generation."""
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        retriever: Optional[EnhancedRetriever] = None,
        generator: Optional[LLMGenerator] = None,
        prompt_manager: Optional[PromptManager] = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            config: RAG configuration
            retriever: Pre-initialized retriever
            generator: Pre-initialized generator
            prompt_manager: Pre-initialized prompt manager
        """
        # Use provided config or load default
        if config is None:
            config = RAGConfig.from_yaml()
        self.config = config
        
        # Initialize components
        self.retriever = retriever or self._init_retriever()
        self.generator = generator or self._init_generator()
        self.prompt_manager = prompt_manager or PromptManager()
        
        logger.info("RAG Pipeline initialized successfully")
    
    def _init_retriever(self) -> EnhancedRetriever:
        """Initialize the retriever from config."""
        return EnhancedRetriever(
            vector_store_path=self.config.vector_store_path,
            collection_name=self.config.collection_name,
            top_k=self.config.retriever.top_k,
            similarity_threshold=self.config.retriever.similarity_threshold,
            enable_reranking=self.config.retriever.enable_reranking,
            reranker_model=self.config.retriever.reranker_model
        )
    
    def _init_generator(self) -> LLMGenerator:
        """Initialize the generator from config."""
        return LLMGenerator(
            model_name=self.config.generator.model_name,
            temperature=self.config.generator.temperature,
            max_new_tokens=self.config.generator.max_new_tokens,
            top_p=self.config.generator.top_p,
            repetition_penalty=self.config.generator.repetition_penalty,
            device_map=self.config.generator.device_map,
            load_in_4bit=self.config.generator.load_in_4bit
        )
    
    def query(
        self,
        question: str,
        filter_dict: Optional[Dict] = None,
        template_name: Optional[str] = None,
        stream: bool = False
    ) -> RAGResponse:
        """
        Process a query through the RAG pipeline.
        
        Args:
            question: User's question
            filter_dict: Metadata filters for retrieval
            template_name: Specific prompt template to use
            stream: Whether to stream generation
        
        Returns:
            Complete RAG response
        """
        start_time = time.time()
        
        # Step 1: Analyze query
        query_analysis = self.retriever.analyze_query(question)
        logger.info(f"Processing query: '{question}'")
        
        # Step 2: Retrieve relevant chunks
        retrieval_result = self.retriever.retrieve_with_context(
            query=question,
            filter_dict=filter_dict or query_analysis['suggested_filters']
        )
        
        if retrieval_result['chunk_count'] == 0:
            logger.warning(f"No relevant chunks found for query: '{question}'")
            return RAGResponse(
                answer="I couldn't find any relevant complaints to answer your question. Try rephrasing or broadening your search.",
                sources=[],
                retrieved_chunks=0,
                generation_stats={},
                query_analysis=query_analysis,
                processing_time=time.time() - start_time
            )
        
        # Step 3: Select or auto-select template
        if template_name is None:
            template_name = self.prompt_manager.auto_select_template(question)
        
        template = self.prompt_manager.get_template(template_name)
        
        # Step 4: Format prompt
        prompt = template.format(
            context=retrieval_result['context'],
            question=question
        )
        
        # Step 5: Generate answer
        generation_result = self.generator.generate_with_retry(
            prompt=prompt,
            stream=stream
        )
        
        # Step 6: Add sources to generation result
        generation_result.sources = retrieval_result['sources']
        
        # Step 7: Prepare response
        processing_time = time.time() - start_time
        
        response = RAGResponse(
            answer=generation_result.answer,
            sources=retrieval_result['sources'],
            retrieved_chunks=retrieval_result['chunk_count'],
            generation_stats={
                'model': generation_result.model_name,
                'prompt_tokens': generation_result.prompt_tokens,
                'completion_tokens': generation_result.completion_tokens,
                'total_tokens': generation_result.total_tokens,
                'generation_time': generation_result.generation_time,
                'template_used': template_name
            },
            query_analysis=query_analysis,
            processing_time=processing_time
        )
        
        logger.info(f"Query processed in {processing_time:.2f}s. Retrieved {retrieval_result['chunk_count']} chunks.")
        
        return response
    
    def batch_query(
        self,
        questions: List[str],
        filter_dicts: Optional[List[Dict]] = None
    ) -> List[RAGResponse]:
        """Process multiple queries in batch."""
        responses = []
        
        for i, question in enumerate(questions):
            filter_dict = filter_dicts[i] if filter_dicts and i < len(filter_dicts) else None
            response = self.query(question, filter_dict)
            responses.append(response)
        
        return responses
    
    def analyze_trends(
        self,
        product_category: Optional[str] = None,
        time_period: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze trends in complaints.
        
        Args:
            product_category: Filter by product
            time_period: Time period for analysis
        
        Returns:
            Trend analysis
        """
        # Build filter
        filter_dict = {}
        if product_category:
            filter_dict['product_category'] = product_category
        if time_period:
            # This would need date parsing logic
            pass
        
        # Construct trend analysis question
        question = "What are the main trends in customer complaints"
        if product_category:
            question += f" for {product_category}"
        if time_period:
            question += f" during {time_period}"
        question += "?"
        
        # Get template
        template = self.prompt_manager.get_template('trend_detection')
        
        # Retrieve relevant chunks
        retrieval_result = self.retriever.retrieve_with_context(
            query=question,
            filter_dict=filter_dict,
            include_metadata=True
        )
        
        if retrieval_result['chunk_count'] == 0:
            return {"error": "No complaints found for the specified criteria"}
        
        # Generate analysis
        prompt = template.format(
            context=retrieval_result['context'],
            question=question
        )
        
        generation_result = self.generator.generate(prompt)
        
        return {
            'analysis': generation_result.answer,
            'sources_count': retrieval_result['chunk_count'],
            'avg_similarity': retrieval_result['avg_similarity'],
            'generation_stats': {
                'tokens': generation_result.total_tokens,
                'time': generation_result.generation_time
            }
        }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline components."""
        model_info = self.generator.get_model_info()
        
        return {
            'retriever': {
                'vector_store': self.config.vector_store_path,
                'collection': self.config.collection_name,
                'top_k': self.config.retriever.top_k,
                'similarity_threshold': self.config.retriever.similarity_threshold
            },
            'generator': {
                'model': model_info['model_name'],
                'device': model_info['device'],
                'temperature': self.config.generator.temperature,
                'max_tokens': self.config.generator.max_new_tokens
            },
            'prompts': {
                'available_templates': self.prompt_manager.list_templates()
            }
        }