"""
Event handlers for UI interactions.
"""
import gradio as gr
from typing import Dict, Any, List, Optional, Tuple
import json
import time
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go

from ..src.rag_pipeline import RAGPipeline
from .ui_components import UIComponents
from .utils import format_timestamp, export_conversation

class EventHandlers:
    """Handlers for UI events."""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.conversation_history = []
        self.feedback_history = []
    
    def handle_query(
        self,
        query: str,
        product_filter: List[str],
        issue_filter: List[str],
        date_start: str,
        date_end: str,
        chat_history: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], str, str, str, str, go.Figure]:
        """
        Handle user query.
        
        Returns:
            Tuple of (updated_chat_history, answer_markdown, sources_html, 
                     metrics_html, metrics_html2, visualization)
        """
        if not query or not query.strip():
            return chat_history, "Please enter a question.", "", "", "", go.Figure()
        
        start_time = time.time()
        
        # Build filter dictionary
        filter_dict = {}
        
        if product_filter and "All" not in product_filter:
            filter_dict['product_category'] = {"$in": product_filter}
        
        if issue_filter and "All" not in issue_filter:
            filter_dict['issue'] = {"$in": issue_filter}
        
        # TODO: Implement date filtering
        # if date_start and date_end:
        #     filter_dict['date_received'] = {
        #         "$gte": date_start,
        #         "$lte": date_end
        #     }
        
        # Process query
        try:
            response = self.rag_pipeline.query(
                question=query,
                filter_dict=filter_dict if filter_dict else None,
                stream=False
            )
            
            processing_time = time.time() - start_time
            
            # Format outputs
            answer_markdown = UIComponents.format_answer_markdown(
                response.answer,
                response.sources[:3]  # Show top 3 sources
            )
            
            sources_html = UIComponents.format_sources_html(
                response.sources[:5]  # Show top 5 sources in details
            )
            
            metrics_html = UIComponents.create_metrics_html(
                processing_time=response.processing_time,
                chunks_retrieved=response.retrieved_chunks,
                avg_similarity=response.generation_stats.get('avg_similarity', 0.7)
            )
            
            # Duplicate for second metric display (if needed)
            metrics_html2 = metrics_html
            
            # Create visualization
            visualization = UIComponents.create_visualization(response.sources)
            
            # Update chat history
            chat_history.append((query, response.answer))
            
            # Save to conversation history
            self.conversation_history.append({
                'timestamp': format_timestamp(),
                'question': query,
                'answer': response.answer,
                'sources': response.sources[:3],  # Save limited sources
                'processing_time': processing_time,
                'filters': filter_dict
            })
            
            return (
                chat_history,
                answer_markdown,
                sources_html,
                metrics_html,
                metrics_html2,
                visualization
            )
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            return chat_history, error_msg, "", "", "", go.Figure()
    
    def handle_example_click(self, evt: gr.EventData) -> str:
        """Handle example question click."""
        return evt.value[0]
    
    def handle_clear_conversation(self) -> Tuple[List, str, str, str, str, go.Figure]:
        """Clear conversation history."""
        self.conversation_history = []
        return [], "", "", "", "", go.Figure()
    
    def handle_export(self, format: str = "json") -> Dict[str, Any]:
        """Export conversation history."""
        if not self.conversation_history:
            return {"error": "No conversation to export"}
        
        exported = export_conversation(self.conversation_history, format)
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_export_{timestamp}.{format}"
        filepath = Path("exports") / filename
        
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            if format == "json":
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
            else:
                f.write(exported)
        
        return {
            "filename": filename,
            "filepath": str(filepath),
            "content": exported[:1000] + "..." if len(exported) > 1000 else exported
        }
    
    def handle_feedback(self, is_positive: bool, query: str, answer: str) -> Dict[str, Any]:
        """Handle user feedback."""
        feedback_entry = {
            'timestamp': format_timestamp(),
            'query': query[:500] if query else "",
            'answer_preview': answer[:500] if answer else "",
            'feedback': 'positive' if is_positive else 'negative',
            'source': 'ui_feedback'
        }
        
        self.feedback_history.append(feedback_entry)
        
        # Save feedback to file
        feedback_file = Path("feedback") / "user_feedback.json"
        feedback_file.parent.mkdir(exist_ok=True)
        
        # Load existing feedback
        if feedback_file.exists():
            with open(feedback_file, 'r') as f:
                existing_feedback = json.load(f)
        else:
            existing_feedback = []
        
        # Add new feedback
        existing_feedback.append(feedback_entry)
        
        # Save back
        with open(feedback_file, 'w') as f:
            json.dump(existing_feedback, f, indent=2)
        
        return {
            "status": "success",
            "message": "Thank you for your feedback!",
            "total_feedback": len(existing_feedback)
        }
    
    def handle_streaming_query(self, query: str) -> Any:
        """Handle streaming query (for future implementation)."""
        # This would yield tokens as they're generated
        # For now, return regular response
        response = self.rag_pipeline.query(query)
        yield response.answer
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        total_queries = len(self.conversation_history)
        total_sources = sum(len(conv.get('sources', [])) for conv in self.conversation_history)
        avg_processing_time = (
            sum(conv.get('processing_time', 0) for conv in self.conversation_history) / total_queries
            if total_queries > 0 else 0
        )
        
        return {
            'total_queries': total_queries,
            'total_sources': total_sources,
            'avg_processing_time': avg_processing_time,
            'feedback_count': len(self.feedback_history)
        }