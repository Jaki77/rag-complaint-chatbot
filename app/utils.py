"""
UI utilities and helpers.
"""
import json
import time
from typing import Dict, Any, List
from datetime import datetime
import markdown
from pathlib import Path

def format_timestamp(timestamp: float = None) -> str:
    """Format timestamp for display."""
    if timestamp is None:
        timestamp = time.time()
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

def format_duration(seconds: float) -> str:
    """Format duration for display."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = seconds / 60
        return f"{minutes:.1f}min"

def markdown_to_html(markdown_text: str) -> str:
    """Convert markdown to HTML for display."""
    if not markdown_text:
        return ""
    
    # Convert markdown to HTML
    html = markdown.markdown(
        markdown_text,
        extensions=['fenced_code', 'tables', 'nl2br']
    )
    
    # Add some basic styling
    styled_html = f"""
    <div class="markdown-content">
        {html}
    </div>
    """
    
    return styled_html

def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def create_source_badge(source: Dict[str, Any]) -> str:
    """Create a styled badge for a source."""
    metadata = source.get('metadata', {})
    
    badge_elements = []
    
    # Product badge
    if product := metadata.get('product_category'):
        color_map = {
            'Credit card': 'blue',
            'Personal loan': 'green',
            'Savings account': 'purple',
            'Money transfer': 'orange'
        }
        color = color_map.get(product, 'gray')
        badge_elements.append(f'<span class="badge badge-{color}">{product}</span>')
    
    # Issue badge
    if issue := metadata.get('issue'):
        badge_elements.append(f'<span class="badge badge-gray">{issue}</span>')
    
    # Date badge
    if date := metadata.get('date_received'):
        if isinstance(date, str) and len(date) >= 10:
            date_str = date[:10]
            badge_elements.append(f'<span class="badge badge-light">{date_str}</span>')
    
    # Similarity badge
    if similarity := source.get('similarity'):
        similarity_pct = similarity * 100
        color = 'success' if similarity_pct > 80 else 'warning' if similarity_pct > 60 else 'danger'
        badge_elements.append(f'<span class="badge badge-{color}">{similarity_pct:.0f}% match</span>')
    
    return ' '.join(badge_elements)

def export_conversation(conversation: List[Dict[str, Any]], format: str = "json") -> str:
    """Export conversation to specified format."""
    if format == "json":
        return json.dumps(conversation, indent=2, ensure_ascii=False)
    elif format == "txt":
        lines = ["CreditTrust Complaint Analysis Conversation", "=" * 50, ""]
        for i, msg in enumerate(conversation, 1):
            lines.append(f"Q{i}: {msg.get('question', '')}")
            lines.append(f"A{i}: {msg.get('answer', '')}")
            lines.append(f"Time: {msg.get('timestamp', '')}")
            lines.append("-" * 50)
        return "\n".join(lines)
    elif format == "markdown":
        lines = ["# Conversation Export", ""]
        for i, msg in enumerate(conversation, 1):
            lines.append(f"## Question {i}")
            lines.append(f"**Question**: {msg.get('question', '')}")
            lines.append(f"**Answer**: {msg.get('answer', '')}")
            lines.append(f"*Time*: {msg.get('timestamp', '')}")
            
            if sources := msg.get('sources', []):
                lines.append("### Sources")
                for j, source in enumerate(sources, 1):
                    lines.append(f"{j}. **{source.get('metadata', {}).get('product_category', 'Unknown')}**")
                    lines.append(f"   *{truncate_text(source.get('text', ''), 100)}*")
            lines.append("")
        return "\n".join(lines)
    
    return ""

def load_css(filepath: str = None) -> str:
    """Load CSS from file or return default."""
    if filepath and Path(filepath).exists():
        with open(filepath, 'r') as f:
            return f.read()
    
    # Default CSS
    return """
    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    
    .chat-message {
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
        border-left: 4px solid;
    }
    
    .user-message {
        background-color: #f0f9ff;
        border-left-color: #3b82f6;
    }
    
    .ai-message {
        background-color: #f0fdf4;
        border-left-color: #10b981;
    }
    
    .badge {
        display: inline-block;
        padding: 3px 8px;
        margin: 2px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
    }
    
    .badge-blue { background-color: #dbeafe; color: #1e40af; }
    .badge-green { background-color: #d1fae5; color: #065f46; }
    .badge-purple { background-color: #f3e8ff; color: #6b21a8; }
    .badge-orange { background-color: #ffedd5; color: #9a3412; }
    .badge-gray { background-color: #f3f4f6; color: #374151; }
    .badge-light { background-color: #f9fafb; color: #6b7280; }
    .badge-success { background-color: #dcfce7; color: #166534; }
    .badge-warning { background-color: #fef3c7; color: #92400e; }
    .badge-danger { background-color: #fee2e2; color: #991b1b; }
    
    .source-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    
    .source-text {
        color: #4b5563;
        font-size: 14px;
        line-height: 1.5;
    }
    
    .metrics {
        display: flex;
        gap: 15px;
        margin: 10px 0;
        padding: 10px;
        background: #f8fafc;
        border-radius: 6px;
    }
    
    .metric-item {
        text-align: center;
    }
    
    .metric-value {
        font-size: 18px;
        font-weight: bold;
        color: #1e40af;
    }
    
    .metric-label {
        font-size: 12px;
        color: #6b7280;
    }
    """