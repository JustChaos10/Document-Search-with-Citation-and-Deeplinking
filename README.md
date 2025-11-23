# Document Search with Citation and Deeplinking

An AI-powered document search and question-answering system that lets you ask questions about your documents and get accurate answers with citations. Supports PDF, HTML, DOCX files and audio recordings in both English and Arabic.

## How It Works

1. **Upload Documents**: Add PDF, HTML, DOCX, or audio files to the system
2. **Automatic Processing**: Documents are extracted, chunked, and indexed using vector embeddings and keyword search
3. **Ask Questions**: Type or speak your question in English or Arabic
4. **Get Answers**: The system uses advanced RAG (Retrieval-Augmented Generation) to find relevant information and generate accurate answers with citations
5. **View Sources**: Click on citations to see the exact page and context from your documents

## Technology Stack

### Core AI/ML Components
- **AI Model**: Groq Cloud with Llama 3.1 70B Versatile (ultra-fast answer generation)
- **Embeddings**: BAAI/bge-m3 (1024-dim multilingual, state-of-the-art)
- **Reranking**: BAAI/bge-reranker-large (cross-encoder)
- **Query Expansion**: LLM-based query rewriting with Llama 3.1 8B Instant
- **Speech-to-Text**: OpenAI Whisper (for voice queries)

### Retrieval Pipeline (Advanced RAG)
- **Hybrid Search**: Vector search (FAISS/ChromaDB) + BM25 keyword search
- **Token-Based Chunking**: tiktoken for accurate LLM context limits (500 tokens/chunk)
- **Semantic Chunking**: Embedding-based similarity to find natural document boundaries
- **Parent Document Retrieval**: Stores broader context (2000 tokens) for better answers
- **Multi-Stage Retrieval**: Retrieve → Deduplicate → Language Filter → Rerank

### Infrastructure
- **Backend Framework**: Flask (Python web framework)
- **Vector Stores**: FAISS (primary) or ChromaDB (fallback)
- **Document Processing**: pypdf, BeautifulSoup, python-docx
- **Frontend**: HTML, CSS, JavaScript with PDF.js viewer

## Use Cases

- **Knowledge Base Search**: Quickly find information in large document collections
- **Policy & Compliance**: Ask questions about policies, regulations, or compliance documents
- **Research Papers**: Search through academic papers and research documents
- **Multilingual Documents**: Handle both English and Arabic documents seamlessly
- **Voice Search**: Ask questions using voice instead of typing
- **Document Analysis**: Get summaries and insights from your documents

## Step-by-Step Setup

### Prerequisites

- **Python 3.11 or higher** ([Download here](https://www.python.org/downloads/))
- **Groq Cloud API Key** ([Get one FREE here](https://console.groq.com/keys))
- **Internet connection** (for API calls and downloading models)

### Installation Steps

#### 1. Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd SEARCH

# Or extract from ZIP and navigate to the folder
```

#### 2. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

#### 4. Configure Environment Variables

Create a `.env` file in the project root with:

```env
# Required: Your Groq Cloud API key (get it FREE at https://console.groq.com/keys)
GROQ_API_KEY=your_groq_api_key_here

# Optional: Customize these if needed
# GROQ_MODEL_NAME=llama-3.1-70b-versatile  # or mixtral-8x7b-32768, llama-3.1-8b-instant
# EMBEDDING_MODEL_NAME=BAAI/bge-m3
# RERANKER_MODEL_NAME=BAAI/bge-reranker-large
```

#### 5. Add Your Documents

Place your documents (PDF, HTML, DOCX, or audio files) in the `uploads/` folder:

```
uploads/
  ├── document1.pdf
  ├── document2.html
  ├── document3.docx
  └── audio.mp3
```

#### 6. Ingest Documents

Process and index your documents:

```bash
python ingest.py
```

This will:
- Extract text from all documents
- Split them into chunks
- Generate embeddings
- Create searchable indexes

**Note**: First run may take time as it downloads AI models.

#### 7. Start the Web Application

```bash
python run.py
```

The server will start at `http://localhost:8000`

#### 8. Open in Browser

Visit `http://localhost:8000` in your web browser and start asking questions!

## Features

### Advanced RAG Capabilities 🚀
- **Token-Based Chunking**: Uses tiktoken for precise token counting, ensuring chunks fit LLM context limits
  - English: 500 tokens/chunk with 100 token overlap
  - Arabic: 600 tokens/chunk with 120 token overlap (accounting for language differences)
- **Semantic Chunking**: Analyzes embedding similarity between sentences to split at natural topic boundaries
  - Prevents splitting related content across chunks
  - Maintains semantic coherence within each chunk
- **Parent Document Retrieval**: Stores 2000-token context windows around each chunk
  - Provides broader context to the LLM for better answers
  - Reduces information loss from aggressive chunking
- **LLM-Based Query Expansion**: Uses Gemini to generate 2-3 alternative query phrasings
  - Handles synonyms, related concepts, and different perspectives
  - Works in both English and Arabic
- **Multi-Stage Retrieval Pipeline**:
  1. Retrieve 3× candidates from both vector and keyword indexes
  2. Deduplicate by chunk ID
  3. Filter by detected language
  4. Rerank using cross-encoder (BAAI/bge-reranker-large)
  5. Return top-K results

### Document Support
- **PDF**: Text extraction with layout preservation
- **HTML**: Extracts content and sections from web pages
- **DOCX**: Processes Word documents with heading detection
- **Audio**: Transcribes MP3, WAV, M4A, MP4, FLAC files using Whisper

### Search Capabilities
- **Hybrid Retrieval**: Combines vector similarity search (semantic) with BM25 keyword search
- **State-of-the-Art Embeddings**: BAAI/bge-m3 (1024 dimensions, supports 100+ languages)
- **Reranking**: Uses cross-encoder models to improve result quality
- **Multilingual**: Supports English and Arabic with automatic language detection
- **Voice Search**: Speak your questions using the microphone button

### User Interface
- **Clean Design**: Modern, responsive web interface
- **RTL Support**: Full right-to-left support for Arabic text
- **PDF Viewer**: Built-in PDF viewer with direct page links
- **Citation Cards**: Each answer includes clickable citation cards with snippets

## Advanced Configuration

### Using Different Embedding Models

```env
# Use Gemini embeddings (requires API key)
EMBEDDING_MODEL_NAME=models/text-embedding-004

# Or use a local model path
EMBEDDING_MODEL_PATH=/path/to/local/model
```

### Customizing Reranking

```env
# Use a different reranker model
RERANKER_MODEL_NAME=BAAI/bge-reranker-base

# Set device (auto, cuda, cpu)
RERANKER_DEVICE=cuda
```

### Voice Search Setup

For voice search, you may need `ffmpeg`:

```bash
# Windows (using Chocolatey)
choco install ffmpeg

# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg
```

## Troubleshooting

**Problem**: "No module named 'xxx'  
**Solution**: Make sure virtual environment is activated and run `pip install -r requirements.txt`

**Problem**: "GEMINI_API_KEY not found"  
**Solution**: Create a `.env` file with your API key

**Problem**: Documents not appearing in search  
**Solution**: Run `python ingest.py` to process documents first

**Problem**: Port 8000 already in use  
**Solution**: Change port in `run.py` or stop the other service

## Project Structure

```
SEARCH/
├── app/
│   ├── web/          # Flask web application
│   ├── retrieval/    # Search and RAG logic
│   ├── ingestion/   # Document processing
│   ├── nlp/          # Language detection
│   └── audio/        # Speech-to-text
├── uploads/          # Place documents here
├── storage/          # Indexes and data storage
├── logs/             # Application logs
├── ingest.py         # Document ingestion script
├── run.py            # Start the web server
└── requirements.txt  # Python dependencies
```

## Next Steps

1. Upload your documents to the `uploads/` folder
2. Run `python ingest.py` to index them
3. Start the server with `python run.py`
4. Open `http://localhost:8000` and ask questions!

For questions or issues, check the logs in `logs/app.log`.

## Environment Variables Example

Create a `.env` file in the project root with the following configuration:

```env
# Required: Your Groq Cloud API key (FREE at https://console.groq.com/keys)
GROQ_API_KEY=your_groq_api_key_here

# LLM Configuration - Groq Cloud Models
GROQ_MODEL_NAME=llama-3.1-70b-versatile  # Best for RAG quality
# Other options: mixtral-8x7b-32768 (larger context), llama-3.1-8b-instant (fastest)

# Embedding Model (State-of-the-art multilingual)
EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_CACHE_DIR=storage/models
EMBEDDING_MODEL_PATH=

# Vector Store
VECTOR_STORE_BACKEND=auto  # Options: auto, faiss, chroma
VECTOR_STORE_PATH=storage/vector_store

# Legacy chunking (fallback only - token-based is now default)
CHUNK_SIZE=800
CHUNK_OVERLAP=200
CHUNK_SIZE_AR=1200
CHUNK_OVERLAP_AR=300

# Hybrid Retrieval (Vector + BM25)
ENABLE_HYBRID_RETRIEVAL=true
BM25_INDEX_PATH=storage/bm25_index.json

# Reranker Configuration
RERANKER_MODEL_NAME=BAAI/bge-reranker-large
RERANKER_DEVICE=auto  # Options: auto, cuda, mps, cpu
RERANKER_CANDIDATE_LIMIT=18

# Speech-to-Text (Whisper)
WHISPER_MODEL_NAME=medium
WHISPER_DEVICE=auto

# Logging
LOG_FILE=logs/app.log
ENVIRONMENT=development
```
