"""
Main Gradio application for the RAG chatbot.
"""
import gradio as gr
import sys
from pathlib import Path
import yaml
import logging

# Add src to path
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))

# Import our modules
from src.rag_pipeline import RAGPipeline
from app.ui_components import UIComponents
from app.handlers import EventHandlers
from app.utils import load_css

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_ui_config():
    """Load UI configuration from YAML."""
    config_path = BASE_DIR / "config" / "ui_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_gradio_app(rag_pipeline: RAGPipeline, config: dict) -> gr.Blocks:
    """Create Gradio application."""
    
    # Initialize handlers
    handlers = EventHandlers(rag_pipeline)
    
    # Custom CSS
    css = load_css()
    
    with gr.Blocks(
        title=config['ui']['title'],
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="teal"
        ),
        css=css
    ) as app:
        
        # Header
        UIComponents.create_header()
        
        gr.Markdown(config['ui']['description'])
        
        # Create layout in tabs
        with gr.Tabs():
            with gr.TabItem("💬 Chat Assistant"):
                with gr.Row():
                    # Left column: Input and chat history
                    with gr.Column(scale=1):
                        # Example questions
                        examples = UIComponents.create_example_questions()
                        
                        # Chat history
                        chatbot = UIComponents.create_chat_history_section()
                        
                        # Clear button
                        clear_btn = gr.Button(
                            "Clear Conversation",
                            variant="secondary",
                            size="lg",
                            full_width=True
                        )
                    
                    # Right column: Main interface
                    with gr.Column(scale=2):
                        # Input section
                        components = UIComponents.create_input_section()
                        query_input = components['query_input']
                        submit_btn = components['submit_btn']
                        product_filter = components['product_filter']
                        issue_filter = components['issue_filter']
                        date_start = components['date_start']
                        date_end = components['date_end']
                        
                        # Output section
                        outputs = UIComponents.create_output_section()
                        answer_output = outputs['answer_output']
                        processing_time = outputs['processing_time']
                        chunks_retrieved = outputs['chunks_retrieved']
                        confidence_score = outputs['confidence_score']
                        thumbs_up = outputs['thumbs_up']
                        thumbs_down = outputs['thumbs_down']
                        export_btn = outputs['export_btn']
                        sources_output = outputs['sources_output']
                        visualization_output = outputs['visualization_output']
            
            with gr.TabItem("📊 Analytics Dashboard"):
                with gr.Column():
                    gr.Markdown("## 📈 Complaint Analysis Dashboard")
                    
                    with gr.Row():
                        with gr.Column():
                            product_distribution = gr.Plot(
                                label="Complaints by Product Category"
                            )
                            time_trend = gr.Plot(
                                label="Complaints Over Time"
                            )
                        
                        with gr.Column():
                            issue_breakdown = gr.Plot(
                                label="Top Issues"
                            )
                            geo_map = gr.Plot(
                                label="Geographic Distribution"
                            )
                    
                    refresh_btn = gr.Button("Refresh Analytics", variant="primary")
            
            with gr.TabItem("⚙️ Settings & Info"):
                with gr.Column():
                    gr.Markdown("## 🔧 Configuration")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Pipeline Information")
                            pipeline_info = rag_pipeline.get_pipeline_info()
                            info_json = gr.JSON(
                                value=pipeline_info,
                                label="Current Configuration"
                            )
                        
                        with gr.Column():
                            gr.Markdown("### Conversation Statistics")
                            stats_display = gr.JSON(
                                label="Session Stats"
                            )
                    
                    refresh_stats_btn = gr.Button("Refresh Statistics", variant="secondary")
        
        # Example click handler
        examples.click(
            fn=handlers.handle_example_click,
            inputs=examples,
            outputs=query_input
        )
        
        # Main query handler
        submit_btn.click(
            fn=handlers.handle_query,
            inputs=[
                query_input,
                product_filter,
                issue_filter,
                date_start,
                date_end,
                chatbot
            ],
            outputs=[
                chatbot,
                answer_output,
                sources_output,
                processing_time,
                chunks_retrieved,
                visualization_output
            ]
        ).then(
            fn=lambda: gr.update(interactive=True),
            outputs=[submit_btn]
        )
        
        # Also allow Enter key in query input
        query_input.submit(
            fn=handlers.handle_query,
            inputs=[
                query_input,
                product_filter,
                issue_filter,
                date_start,
                date_end,
                chatbot
            ],
            outputs=[
                chatbot,
                answer_output,
                sources_output,
                processing_time,
                chunks_retrieved,
                visualization_output
            ]
        ).then(
            fn=lambda: gr.update(value=""),
            outputs=[query_input]
        )
        
        # Clear conversation
        clear_btn.click(
            fn=handlers.handle_clear_conversation,
            outputs=[
                chatbot,
                answer_output,
                sources_output,
                processing_time,
                chunks_retrieved,
                visualization_output
            ]
        )
        
        # Feedback handlers
        thumbs_up.click(
            fn=lambda q, a: handlers.handle_feedback(True, q, a),
            inputs=[query_input, answer_output],
            outputs=[]
        )
        
        thumbs_down.click(
            fn=lambda q, a: handlers.handle_feedback(False, q, a),
            inputs=[query_input, answer_output],
            outputs=[]
        )
        
        # Export handler
        export_btn.click(
            fn=lambda: handlers.handle_export("json"),
            outputs=[]
        ).then(
            fn=lambda: gr.Info("Conversation exported successfully!"),
            outputs=[]
        )
        
        # Refresh stats handler
        refresh_stats_btn.click(
            fn=handlers.get_conversation_stats,
            outputs=[stats_display]
        )
        
        # Analytics refresh handler (placeholder)
        refresh_btn.click(
            fn=lambda: gr.Info("Analytics refresh would be implemented here"),
            outputs=[]
        )
        
        # Initial loading message
        gr.Markdown("---")
        gr.Markdown(
            """
            **Tips for best results:**
            - Be specific with your questions (e.g., "credit card fees" vs "fees")
            - Use filters to narrow down to specific products or time periods
            - Check the source citations to verify information
            - Export conversations for sharing with your team
            """
        )
    
    return app

def main():
    """Main function to launch the Gradio app."""
    
    # Load configuration
    config = load_ui_config()
    ui_config = config['ui']
    
    logger.info(f"Starting {ui_config['title']}")
    logger.info(f"Using framework: {ui_config['framework']}")
    
    # Initialize RAG pipeline
    logger.info("Initializing RAG pipeline...")
    try:
        rag_pipeline = RAGPipeline()
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        # Create a mock pipeline for demo purposes
        class MockPipeline:
            def query(self, *args, **kwargs):
                return type('Response', (), {
                    'answer': 'Mock response: The RAG pipeline failed to load.',
                    'sources': [],
                    'retrieved_chunks': 0,
                    'processing_time': 0.1,
                    'generation_stats': {'avg_similarity': 0.0}
                })()
            def get_pipeline_info(self):
                return {'error': 'Pipeline failed to load'}
        
        rag_pipeline = MockPipeline()
        logger.warning("Using mock pipeline for demo")
    
    # Create and launch app
    if ui_config['framework'] == "gradio":
        app = create_gradio_app(rag_pipeline, config)
        
        # Launch with appropriate settings
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True,
            favicon_path=None
        )
    elif ui_config['framework'] == "streamlit":
        # For Streamlit, we'd launch differently
        logger.info("Streamlit app would be launched via streamlit run streamlit_app.py")
        print("\nTo run Streamlit app, execute:")
        print("  streamlit run streamlit_app.py")
    else:
        raise ValueError(f"Unsupported framework: {ui_config['framework']}")

if __name__ == "__main__":
    main()