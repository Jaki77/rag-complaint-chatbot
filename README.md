# 📊 RAG-Powered Financial Complaint Analysis Chatbot

An intelligent AI system that transforms **unstructured customer complaints** into **actionable insights** using Retrieval-Augmented Generation (RAG). Built for CreditTrust Financial to analyze thousands of complaints across financial products in seconds.

## 🌟 **Key Features**

- **Semantic Search**: FAISS/ChromaDB vector database with sentence-transformers embeddings
- **LLM-Powered Insights**: Open-source language models generate evidence-backed answers
- **Multi-Product Analysis**: Compare issues across credit cards, loans, savings, and money transfers
- **User-Friendly Interface**: Gradio/Streamlit UI for non-technical teams
- **Source Attribution**: Shows retrieved complaint excerpts for verification and trust
- **Real-time Analysis**: Reduce complaint analysis time from days to minutes

## 🎯 **Business Impact**

| Before | After |
|--------|-------|
| Product managers spend hours manually reading complaints | Get synthesized answers in **seconds** |
| Reactive problem-solving | **Proactive** issue identification |
| Data analysts needed for insights | **Self-service** for support/compliance teams |
| Scattered, hard-to-read narratives | **Structured, actionable** intelligence |

## 📁 **Project Structure**
```bash
rag-complaint-chatbot/
├── 📁 data/                    # Data directory
│   ├── raw/                   # Original CFPB dataset
│   └── processed/             # Cleaned and filtered data
├── 📁 notebooks/              # Jupyter notebooks for EDA
│   ├── 01_eda_preprocessing.ipynb
│   └── 02_chunking_embedding.ipynb
├── 📁 src/                    # Source code
│   ├── preprocessing.py      # Data cleaning and filtering
│   ├── embedding.py         # Text chunking and vectorization
│   ├── retriever.py         # Semantic search implementation
│   ├── generator.py         # LLM response generation
│   ├── rag_pipeline.py      # Main RAG orchestration
│   └── utils.py             # Utility functions
├── 📁 tests/                 # Unit and integration tests
│   ├── unit/
│   └── integration/
├── 📁 vector_store/          # FAISS/ChromaDB indices
├── app.py                    # Gradio/Streamlit UI
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

## 🚀 **Quick Start**

### 1. **Prerequisites**

```bash
# Python 3.9 or higher
python --version

# Git
git --version

# Recommended: Anaconda/Miniconda or virtual environment
```

### 2. **Installation**
```bash
# Clone the repository
git clone https://github.com/Jaki77/rag-complaint-chatbot.git
cd rag-complaint-chatbot

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. **Download Data**
```bash
1. Download the CFPB complaint dataset from https://www.consumerfinance.gov/data-research/consumer-complaints/

2. Place the CSV file in data/raw/complaints.csv
```

### 4. **Run the Pipeline**
```bash
# Step 1: Preprocess the data
python src/preprocessing.py

# Step 2: Build vector store (sample)
python src/embedding.py --sample-size 10000

# Step 3: Run the RAG chatbot
python app.py
```

### **🛠 Technical Architecture**
```bash
graph TD
    A[Customer Complaints] --> B[Preprocessing]
    B --> C[Text Chunking]
    C --> D[Embedding Model]
    D --> E[Vector Database]
    F[User Query] --> G[Query Embedding]
    G --> E
    E --> H[Retrieved Context]
    H --> I[LLM Generator]
    I --> J[Answer + Sources]
    J --> K[Web Interface]
```