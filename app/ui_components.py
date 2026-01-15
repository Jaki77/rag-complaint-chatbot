"""
Gradio UI components for the RAG chatbot.
"""
import gradio as gr
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

class UIComponents:
    """Factory for Gradio UI components."""
    
    @staticmethod
    def create_header() -> gr.HTML:
        """Create application header."""
        html = """
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #1e40af 0%, #0ea5e9 100%); border-radius: 10px; color: white;">
            <h1 style="margin: 0; font-size: 2.5em;">🤖 CreditTrust Complaint Analysis</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.1em; opacity: 0.9;">
                AI-powered insights from customer complaints across financial products
            </p>
            <div style="margin-top: 15px; display: flex; justify-content: center; gap: 15px;">
                <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px;">Credit Cards</span>
                <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px;">Personal Loans</span>
                <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px;">Savings Accounts</span>
                <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px;">Money Transfers</span>
            </div>
        </div>
        """
        return gr.HTML(value=html)
    
    @staticmethod
    def create_input_section() -> Dict[str, gr.components.Component]:
        """Create input section with query box and filters."""
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Ask a question about customer complaints",
                    placeholder="e.g., What are the main issues with credit card fees?",
                    lines=3,
                    elem_id="query-input"
                )
            
            with gr.Column(scale=1):
                submit_btn = gr.Button(
                    "Ask AI Assistant",
                    variant="primary",
                    size="lg",
                    elem_id="submit-btn"
                )
                clear_btn = gr.Button(
                    "Clear Conversation",
                    variant="secondary",
                    size="lg"
                )
        
        # Filters in an accordion
        with gr.Accordion("Advanced Filters", open=False):
            with gr.Row():
                product_filter = gr.Dropdown(
                    label="Product Category",
                    choices=["All", "Credit Card", "Personal Loan", "Savings Account", "Money Transfer"],
                    value="All",
                    multiselect=True
                )
                
                issue_filter = gr.Dropdown(
                    label="Issue Type",
                    choices=["All", "Billing", "Service", "Application", "Fees", "Technology"],
                    value="All",
                    multiselect=True
                )
            
            with gr.Row():
                date_start = gr.Textbox(
                    label="Start Date (YYYY-MM-DD)",
                    placeholder="e.g., 2024-01-01"
                )
                date_end = gr.Textbox(
                    label="End Date (YYYY-MM-DD)",
                    placeholder="e.g., 2024-12-31"
                )
        
        return {
            'query_input': query_input,
            'submit_btn': submit_btn,
            'clear_btn': clear_btn,
            'product_filter': product_filter,
            'issue_filter': issue_filter,
            'date_start': date_start,
            'date_end': date_end
        }
    
    @staticmethod
    def create_output_section() -> Dict[str, gr.components.Component]:
        """Create output section for answers and sources."""
        with gr.Row():
            with gr.Column(scale=2):
                # Answer display
                answer_output = gr.Markdown(
                    label="AI Assistant Response",
                    value="### Your answer will appear here\n\nAsk a question to get started!",
                    elem_id="answer-output"
                )
                
                # Metrics row
                with gr.Row():
                    processing_time = gr.HTML(
                        value="<div class='metric-item'><div class='metric-value'>-</div><div class='metric-label'>Processing Time</div></div>"
                    )
                    chunks_retrieved = gr.HTML(
                        value="<div class='metric-item'><div class='metric-value'>-</div><div class='metric-label'>Sources</div></div>"
                    )
                    confidence_score = gr.HTML(
                        value="<div class='metric-item'><div class='metric-value'>-</div><div class='metric-label'>Confidence</div></div>"
                    )
                
                # Feedback buttons
                with gr.Row():
                    thumbs_up = gr.Button("👍 Helpful", size="sm", variant="secondary")
                    thumbs_down = gr.Button("👎 Not Helpful", size="sm", variant="secondary")
                    export_btn = gr.Button("💾 Export", size="sm", variant="secondary")
            
            with gr.Column(scale=1):
                # Sources display
                sources_output = gr.HTML(
                    label="Retrieved Sources",
                    value="<div style='padding: 20px; text-align: center; color: #6b7280;'>Sources will appear here after query</div>",
                    elem_id="sources-output"
                )
                
                # Visualization
                visualization_output = gr.Plot(
                    label="Analysis Visualization",
                    visible=True
                )
        
        return {
            'answer_output': answer_output,
            'processing_time': processing_time,
            'chunks_retrieved': chunks_retrieved,
            'confidence_score': confidence_score,
            'thumbs_up': thumbs_up,
            'thumbs_down': thumbs_down,
            'export_btn': export_btn,
            'sources_output': sources_output,
            'visualization_output': visualization_output
        }
    
    @staticmethod
    def create_chat_history_section() -> gr.Chatbot:
        """Create chat history display."""
        chatbot = gr.Chatbot(
            label="Conversation History",
            height=400,
            elem_id="chat-history"
        )
        return chatbot
    
    @staticmethod
    def create_example_questions() -> gr.Dataset:
        """Create example questions for quick selection."""
        examples = [
            "What are the main complaints about credit card fees?",
            "How do customers feel about money transfer services?",
            "What issues are customers having with personal loan applications?",
            "Compare complaints between savings accounts and credit cards.",
            "What are the most common billing disputes?",
            "How have complaints about interest rates changed over time?",
            "What problems do customers report with mobile banking apps?",
            "Summarize the key issues across all financial products.",
            "What specific issues are mentioned about overdraft fees?",
            "Are there geographic patterns in complaints?"
        ]
        
        return gr.Dataset(
            components=[gr.Textbox(visible=False)],
            samples=[[ex] for ex in examples],
            label="Try these example questions:",
            samples_per_page=5
        )
    
    @staticmethod
    def format_answer_markdown(answer: str, sources: List[Dict]) -> str:
        """Format answer as markdown with styling."""
        markdown_lines = ["## 🤖 AI Assistant Response\n"]
        
        # Add answer
        markdown_lines.append(answer)
        markdown_lines.append("")
        
        # Add source citation
        if sources:
            markdown_lines.append("### 📚 Sources Used")
            markdown_lines.append("")
            
            for i, source in enumerate(sources, 1):
                metadata = source.get('metadata', {})
                text = source.get('text', '')
                
                markdown_lines.append(f"**Source {i}**")
                markdown_lines.append(f"*Product*: {metadata.get('product_category', 'Unknown')}")
                
                if issue := metadata.get('issue'):
                    markdown_lines.append(f"*Issue*: {issue}")
                
                if date := metadata.get('date_received'):
                    markdown_lines.append(f"*Date*: {date[:10] if len(date) >= 10 else date}")
                
                markdown_lines.append(f"*Relevance*: {source.get('similarity', 0)*100:.0f}%")
                markdown_lines.append("")
                
                # Truncated text preview
                preview = text[:150] + "..." if len(text) > 150 else text
                markdown_lines.append(f"> {preview}")
                markdown_lines.append("")
        
        return "\n".join(markdown_lines)
    
    @staticmethod
    def format_sources_html(sources: List[Dict]) -> str:
        """Format sources as HTML cards."""
        if not sources:
            return "<div style='padding: 20px; text-align: center; color: #6b7280;'>No sources retrieved</div>"
        
        html_lines = ["<div class='sources-container'>"]
        
        for i, source in enumerate(sources, 1):
            metadata = source.get('metadata', {})
            text = source.get('text', '')
            similarity = source.get('similarity', 0)
            
            # Determine similarity color
            similarity_pct = similarity * 100
            if similarity_pct > 80:
                sim_color = "success"
            elif similarity_pct > 60:
                sim_color = "warning"
            else:
                sim_color = "danger"
            
            html_lines.append(f"""
            <div class="source-card">
                <div class="source-header">
                    <div>
                        <strong>Source {i}</strong>
                        <span class="badge badge-{sim_color}" style="margin-left: 10px;">
                            {similarity_pct:.0f}% match
                        </span>
                    </div>
                </div>
                
                <div style="margin-bottom: 10px;">
                    {UIComponents._create_badges(metadata)}
                </div>
                
                <div class="source-text">
                    {text[:200]}{'...' if len(text) > 200 else ''}
                </div>
                
                <div style="margin-top: 10px; font-size: 12px; color: #6b7280;">
                    Complaint ID: {metadata.get('complaint_id', 'N/A')}
                </div>
            </div>
            """)
        
        html_lines.append("</div>")
        return "\n".join(html_lines)
    
    @staticmethod
    def _create_badges(metadata: Dict) -> str:
        """Create badges for metadata."""
        badges = []
        
        if product := metadata.get('product_category'):
            color_map = {
                'Credit card': 'blue',
                'Personal loan': 'green',
                'Savings account': 'purple',
                'Money transfer': 'orange'
            }
            color = color_map.get(product, 'gray')
            badges.append(f'<span class="badge badge-{color}">{product}</span>')
        
        if issue := metadata.get('issue'):
            badges.append(f'<span class="badge badge-gray">{issue}</span>')
        
        if date := metadata.get('date_received'):
            if isinstance(date, str) and len(date) >= 10:
                date_str = date[:10]
                badges.append(f'<span class="badge badge-light">{date_str}</span>')
        
        if sub_issue := metadata.get('sub_issue'):
            badges.append(f'<span class="badge badge-light">{sub_issue[:20]}</span>')
        
        return " ".join(badges)
    
    @staticmethod
    def create_metrics_html(processing_time: float, chunks_retrieved: int, avg_similarity: float) -> str:
        """Create HTML for metrics display."""
        return f"""
        <div class="metrics">
            <div class="metric-item">
                <div class="metric-value">{processing_time:.1f}s</div>
                <div class="metric-label">Processing Time</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{chunks_retrieved}</div>
                <div class="metric-label">Sources</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{avg_similarity*100:.0f}%</div>
                <div class="metric-label">Avg Relevance</div>
            </div>
        </div>
        """
    
    @staticmethod
    def create_visualization(sources: List[Dict]) -> go.Figure:
        """Create visualization of sources."""
        if not sources:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="No data to visualize",
                xaxis_title="",
                yaxis_title="",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            return fig
        
        # Prepare data
        data = []
        for source in sources:
            metadata = source.get('metadata', {})
            data.append({
                'product': metadata.get('product_category', 'Unknown'),
                'similarity': source.get('similarity', 0) * 100,
                'issue': metadata.get('issue', 'Unknown')
            })
        
        df = pd.DataFrame(data)
        
        if len(df) > 0:
            # Create grouped bar chart
            product_counts = df['product'].value_counts()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=product_counts.index,
                    y=product_counts.values,
                    marker_color=['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b'][:len(product_counts)],
                    text=product_counts.values,
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Sources by Product Category",
                xaxis_title="Product",
                yaxis_title="Number of Sources",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(240, 240, 240, 0.5)',
                showlegend=False
            )
        else:
            fig = go.Figure()
            fig.update_layout(
                title="No data to visualize",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
        
        return fig