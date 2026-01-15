"""
LLM generator for answer generation with various model backends.
"""
import torch
from typing import List, Dict, Any, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    TextStreamer
)
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GenerationResult:
    """Container for generation results."""
    answer: str
    sources: List[Dict[str, Any]]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    generation_time: float
    model_name: str

class LLMGenerator:
    """Generator for LLM-based answer generation."""
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        temperature: float = 0.3,
        max_new_tokens: int = 512,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        device_map: str = "auto",
        load_in_4bit: bool = True
    ):
        """
        Initialize the LLM generator.
        
        Args:
            model_name: Hugging Face model identifier
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repetition
            device_map: Device placement strategy
            load_in_4bit: Use 4-bit quantization for memory efficiency
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        
        logger.info(f"Loading model: {model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization for memory efficiency
            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map=device_map,
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=device_map,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=device_map
            )
            
            logger.info(f"Model loaded successfully: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            # Fallback to smaller model
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a smaller fallback model."""
        fallback_model = "microsoft/phi-2"  # Small but capable
        logger.info(f"Loading fallback model: {fallback_model}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(fallback_model, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            fallback_model,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
        
        self.model_name = fallback_model
    
    def generate(
        self,
        prompt: str,
        stream: bool = False,
        **generation_kwargs
    ) -> GenerationResult:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            stream: Whether to stream output token by token
            **generation_kwargs: Additional generation parameters
        
        Returns:
            GenerationResult object
        """
        import time
        
        start_time = time.time()
        
        # Prepare generation parameters
        gen_params = {
            'max_new_tokens': self.max_new_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'repetition_penalty': self.repetition_penalty,
            'do_sample': True,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        
        # Update with any custom kwargs
        gen_params.update(generation_kwargs)
        
        # Tokenize to count input tokens
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt")
        prompt_tokens = input_tokens.shape[1]
        
        # Generate
        if stream:
            # Create streamer
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            
            # Generate with streaming
            outputs = self.model.generate(
                input_tokens,
                streamer=streamer,
                **gen_params
            )
        else:
            # Generate normally
            outputs = self.model.generate(
                input_tokens,
                **gen_params
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from generated text
        answer = generated_text[len(prompt):].strip()
        
        # Count tokens
        total_tokens = outputs.shape[1]
        completion_tokens = total_tokens - prompt_tokens
        
        generation_time = time.time() - start_time
        
        logger.info(f"Generated {completion_tokens} tokens in {generation_time:.2f}s")
        
        return GenerationResult(
            answer=answer,
            sources=[],  # Sources added by RAG pipeline
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            generation_time=generation_time,
            model_name=self.model_name
        )
    
    def generate_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        **kwargs
    ) -> GenerationResult:
        """Generate with retry logic for robustness."""
        for attempt in range(max_retries):
            try:
                return self.generate(prompt, **kwargs)
            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                # Wait before retry
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
        
        raise RuntimeError("All generation attempts failed")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'model_type': type(self.model).__name__,
            'vocab_size': self.tokenizer.vocab_size,
            'max_position_embeddings': getattr(self.model.config, 'max_position_embeddings', 'Unknown'),
            'device': str(self.model.device),
            'dtype': str(self.model.dtype)
        }


class HuggingFaceGenerator(LLMGenerator):
    """Generator using Hugging Face's pipeline API."""
    
    def generate(
        self,
        prompt: str,
        stream: bool = False,
        **kwargs
    ) -> GenerationResult:
        """Generate using Hugging Face pipeline."""
        import time
        
        start_time = time.time()
        
        # Prepare generation parameters
        gen_params = {
            'max_new_tokens': self.max_new_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'repetition_penalty': self.repetition_penalty,
            'do_sample': True,
        }
        gen_params.update(kwargs)
        
        # Generate
        outputs = self.generator(
            prompt,
            **gen_params
        )
        
        generated_text = outputs[0]['generated_text']
        answer = generated_text[len(prompt):].strip()
        
        # Estimate token counts
        prompt_tokens = len(self.tokenizer.encode(prompt))
        total_tokens = len(self.tokenizer.encode(generated_text))
        completion_tokens = total_tokens - prompt_tokens
        
        generation_time = time.time() - start_time
        
        return GenerationResult(
            answer=answer,
            sources=[],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            generation_time=generation_time,
            model_name=self.model_name
        )