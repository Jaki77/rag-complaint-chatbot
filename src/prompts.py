"""
Prompt templates and management for RAG pipeline.
"""
from typing import List, Dict, Any
from dataclasses import dataclass
import json

@dataclass
class PromptTemplate:
    """A prompt template with variables."""
    template: str
    variables: List[str]
    description: str = ""
    
    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing variable in prompt: {e}")

class PromptManager:
    """Manager for different prompt templates."""
    
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize all prompt templates."""
        return {
            'financial_analyst': PromptTemplate(
                template="""SYSTEM: You are a helpful financial analyst assistant for CreditTrust Financial. Your task is to answer questions about customer complaints using ONLY the provided context.

IMPORTANT INSTRUCTIONS:
1. Use ONLY the information in the provided context. Do not use external knowledge.
2. If the context doesn't contain relevant information, say: "Based on the provided complaints, I don't have enough information to answer this question."
3. Be specific and cite details from the context when possible.
4. Structure your answer clearly with bullet points if appropriate.
5. Focus on actionable insights for product managers.

CONTEXT:
{context}

QUESTION: {question}

ANSWER: """,
                variables=['context', 'question'],
                description="Main template for answering complaint analysis questions"
            ),
            
            'summarization': PromptTemplate(
                template="""SYSTEM: You are a data analyst summarizing customer complaints.

Analyze the following complaints and provide a structured summary:
1. Main issues and their frequency
2. Severity levels (how severely customers are affected)
3. Common patterns or trends
4. Potential root causes
5. Recommended actions for the product team

COMPLAINTS:
{complaints}

SUMMARY:""",
                variables=['complaints'],
                description="Template for summarizing multiple complaints"
            ),
            
            'comparative_analysis': PromptTemplate(
                template="""SYSTEM: Compare customer complaints across different products or time periods.

CONTEXT:
{context}

QUESTION: {question}

ANALYSIS REQUIREMENTS:
1. Identify similarities and differences
2. Note frequency and severity variations
3. Suggest product-specific improvements
4. Highlight any concerning trends

ANSWER:""",
                variables=['context', 'question'],
                description="Template for comparative analysis"
            ),
            
            'trend_detection': PromptTemplate(
                template="""SYSTEM: Analyze temporal trends in customer complaints.

CONTEXT:
{context}

QUESTION: {question}

ANALYSIS STRUCTURE:
1. Overall trend direction (increasing/decreasing/stable)
2. Key time periods with spikes
3. Seasonal patterns if any
4. Correlation with product changes or external events
5. Forecast for next quarter

ANSWER:""",
                variables=['context', 'question'],
                description="Template for trend analysis over time"
            ),
            
            'simple_qna': PromptTemplate(
                template="""Answer the question based on the context below.

Context: {context}

Question: {question}

Answer:""",
                variables=['context', 'question'],
                description="Simple Q&A template for straightforward questions"
            )
        }
    
    def get_template(self, template_name: str) -> PromptTemplate:
        """Get a specific template by name."""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found. Available: {list(self.templates.keys())}")
        return self.templates[template_name]
    
    def list_templates(self) -> List[Dict[str, str]]:
        """List all available templates with descriptions."""
        return [
            {'name': name, 'description': template.description}
            for name, template in self.templates.items()
        ]
    
    def auto_select_template(self, query: str) -> str:
        """Automatically select the best template based on query content."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['summarize', 'overview', 'main issues', 'key problems']):
            return 'summarization'
        elif any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference', 'similar']):
            return 'comparative_analysis'
        elif any(word in query_lower for word in ['trend', 'over time', 'last year', 'monthly', 'quarterly']):
            return 'trend_detection'
        elif len(query.split()) < 10:  # Short, simple questions
            return 'simple_qna'
        else:
            return 'financial_analyst'  # Default
    
    def create_custom_template(
        self,
        name: str,
        template: str,
        variables: List[str],
        description: str = ""
    ) -> None:
        """Create a custom prompt template."""
        self.templates[name] = PromptTemplate(
            template=template,
            variables=variables,
            description=description
        )
    
    def export_templates(self, filepath: str) -> None:
        """Export all templates to JSON file."""
        templates_dict = {}
        for name, template in self.templates.items():
            templates_dict[name] = {
                'template': template.template,
                'variables': template.variables,
                'description': template.description
            }
        
        with open(filepath, 'w') as f:
            json.dump(templates_dict, f, indent=2)