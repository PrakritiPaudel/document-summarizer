# Notes/Document Summarizer

A production-ready AI-powered research paper analysis system built with **LangChain**, **RAG (Retrieval-Augmented Generation)**, and **PDO Prompt Engineering**. Specifically designed for analyzing sentiment analysis research papers and other academic documents.

## Features

- **Document Processing**: Supports PDF, DOCX, TXT, and Markdown files
- **RAG Architecture**: Advanced retrieval-augmented generation for accurate summaries
- **PDO Prompts**: Purpose-Details-Output structured prompts for consistent results
- **Multiple Summary Types**: Comprehensive, key points, methodology, and findings summaries
- **Interactive Q&A**: Ask specific questions about your research papers
- **LangSmith Integration**: Monitoring, evaluation, and performance tracking
- **Production Ready**: Comprehensive error handling, logging, and configuration management

## Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- LangSmith API key (optional but recommended)

### Installation

1. **Clone the repository**:

   ```bash
   git clone <your-repo-url>
   cd study-notes-summarizer
   ```

2. **Create virtual environment**:

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:

   ```bash
   # Copy the example environment file
   copy .env.example .env

   # Edit .env file with your API keys
   ```

5. **Set up your `.env` file**:

   ```env
   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key_here

   # LangSmith Configuration (Optional)
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langsmith_api_key_here
   LANGCHAIN_PROJECT=study-notes-summarizer
   ```

6. **Run the application**:

   ```bash
   streamlit run app.py
   ```

7. **Open your browser** to `http://localhost:8501`

## Project Structure

```
study-notes-summarizer/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # Configuration management
│   ├── core/
│   │   ├── __init__.py
│   │   ├── document_processor.py # Document loading and processing
│   │   ├── vector_store.py      # Vector embeddings management
│   │   └── rag_summarizer.py    # Main RAG system
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── summary_prompts.py   # PDO prompt templates
│   └── utils/
│       ├── __init__.py
│       ├── file_manager.py      # File operations
│       └── langsmith_monitor.py # LangSmith integration
├── data/                        # Vector store persistence
├── uploads/                     # Temporary file uploads
├── tests/                       # Test files
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## Usage

### 1. Upload Documents

- Supported formats: PDF, DOCX, TXT, MD
- Maximum file size: 10MB (configurable)
- Upload your sentiment analysis research paper or any academic document

### 2. Generate Summaries

Choose from multiple summary types:

- **Comprehensive**: Complete overview including methodology, findings, and implications
- **Key Points**: Bullet-point summary of main contributions
- **Methodology**: Detailed technical approach and experimental design
- **Findings**: Results, metrics, and conclusions with quantitative data

### 3. Ask Questions

Interactive Q&A system powered by RAG:

- "What is the main methodology used in this paper?"
- "What datasets were used for evaluation?"
- "What are the key performance metrics?"
- "How does this approach compare to existing methods?"

### 4. Focus Areas

Specify particular aspects to emphasize:

- Sentiment analysis techniques
- Deep learning approaches
- Evaluation methodologies
- Dataset characteristics

## 📊 RAG Architecture

Our system implements a sophisticated RAG pipeline:

1. **Document Processing**:

   - Multi-format document loading
   - Intelligent text chunking with overlap
   - Metadata preservation

2. **Vector Store**:

   - OpenAI embeddings (text-embedding-ada-002)
   - Chroma vector database with persistence
   - Similarity search with relevance scoring

3. **Retrieval & Generation**:
   - Context-aware document retrieval
   - LLM-powered answer generation
   - Source attribution and transparency

## PDO Prompt Engineering

All prompts follow the **Purpose-Details-Output** methodology:

- **Purpose**: Clear objective definition
- **Details**: Comprehensive instructions and context
- **Output**: Structured format specification

Example PDO structure:

```
PURPOSE: Create a comprehensive summary of this research paper...

DETAILS:
- Summarize the paper's main contribution to the field
- Include research objectives, methodology, key findings...
- For sentiment analysis papers, emphasize datasets, models...

OUTPUT FORMAT:
## Research Paper Summary
**Title & Objective:** [content]
**Methodology:** [content]
...
```

## 🔍 LangSmith Integration

Monitor and evaluate your RAG system:

- **Performance Tracking**: Token usage, execution time, success rates
- **Run Monitoring**: Detailed execution traces and debugging
- **Feedback System**: Collect and analyze user feedback
- **Cost Analysis**: Track API usage and costs

To enable LangSmith:

1. Set `LANGCHAIN_TRACING_V2=true` in your `.env`
2. Add your LangSmith API key
3. View traces at [smith.langchain.com](https://smith.langchain.com)

## ⚙️ Configuration

Key configuration options in `.env`:

```env
# Document Processing
MAX_FILE_SIZE_MB=10          # Maximum upload size
CHUNK_SIZE=1000              # Text chunk size for processing
CHUNK_OVERLAP=200            # Overlap between chunks
TEMPERATURE=0.3              # LLM temperature for consistency

# Vector Store
CHROMA_PERSIST_DIRECTORY=./data/chroma_db

# Application
UPLOAD_DIRECTORY=./uploads
STREAMLIT_SERVER_PORT=8501
```

## Deployment

### Local Production

1. **Install production server**:

   ```bash
   pip install gunicorn
   ```

2. **Run with Gunicorn**:
   ```bash
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```
