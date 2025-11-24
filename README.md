# Document Search with Citation and Deeplinking

An AI-powered document search and question-answering system that lets you ask questions about your documents and get accurate answers with citations. Supports PDF, HTML, DOCX files and audio recordings in both English and Arabic.

## ✨ New Features (Latest Update)

### 🚀 Advanced RAG Improvements
- **Contextual Compression**: Intelligently extracts the most relevant portions from retrieved documents, reducing token usage by ~40%
- **Conversation Memory**: Handles follow-up questions naturally by maintaining conversation history (last 3 turns, 30 min expiry)
- **Answer Verification**: Computes confidence scores (0-1) to validate answers against source documents
- **Semantic Chunking**: Uses paragraph and sentence boundaries instead of fixed character counts for better context preservation

### 🎨 Enhanced UI Features
- **Loading Skeleton Screens**: Beautiful animated placeholders improve perceived performance
- **Result Filtering & Sorting**: Filter by language/document type and sort by relevance/type/language
- **Dark Mode**: Complete dark theme with theme toggle and localStorage persistence
- **Search Query Highlighting**: Automatically highlights search terms in result snippets

---

## How It Works

1. **Upload Documents**: Add PDF, HTML, DOCX, or audio files to the system
2. **Automatic Processing**: Documents are extracted, chunked semantically, and indexed using vector embeddings and keyword search
3. **Ask Questions**: Type or speak your question in English or Arabic
4. **Get Answers**: The system uses advanced RAG (Retrieval-Augmented Generation) with contextual compression, reranking, and answer verification
5. **View Sources**: Click on citations to see the exact page and context from your documents

## Technology Stack

### Core Framework
- **Backend**: Flask (Python web framework)
- **Frontend**: HTML, CSS, JavaScript with PDF.js viewer

### AI & Search
- **LLM**: Google Gemini (default) or Groq
- **Vector Search**: FAISS or ChromaDB (semantic similarity)
- **Keyword Search**: BM25 (traditional matching)
- **Reranking**: Cross-encoder models (BAAI/bge-reranker-large)
- **Embeddings**: Multilingual sentence transformers (paraphrase-multilingual-MiniLM-L12-v2)
- **Speech-to-Text**: OpenAI Whisper

### RAG Enhancements
- **Contextual Compression**: Query-aware document compression
- **Conversation Memory**: Multi-turn dialogue support
- **Answer Verification**: Confidence scoring and factual grounding
- **Semantic Chunking**: Intelligent document segmentation

### Document Processing
- **PDF**: pypdf with layout mode for RTL support
- **HTML**: BeautifulSoup with section detection
- **DOCX**: python-docx with heading awareness
- **Audio**: Whisper transcription (MP3, WAV, M4A, MP4, FLAC)

## Use Cases

- **Knowledge Base Search**: Quickly find information in large document collections
- **Policy & Compliance**: Ask questions about policies, regulations, or compliance documents
- **Research Papers**: Search through academic papers and research documents
- **Multilingual Documents**: Handle both English and Arabic documents seamlessly
- **Voice Search**: Ask questions using voice instead of typing
- **Document Analysis**: Get summaries and insights with confidence scores
- **Conversational Search**: Ask follow-up questions in natural dialogue

## Step-by-Step Setup

### Prerequisites

- **Python 3.11 or higher** ([Download here](https://www.python.org/downloads/))
- **AI provider API key**: leave the default `LLM_PROVIDER` pointing at Gemini and supply `GEMINI_API_KEY`, or set `LLM_PROVIDER=groq` along with `GROQ_API_KEY`.
- **Internet connection** (for API calls and downloading models)

### Installation Steps

#### 1. Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd Document-Search-with-Citation-and-Deeplinking-main

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

- Leave the defaults in place for Google Gemini and only set `GEMINI_API_KEY`, or switch to Groq by setting `LLM_PROVIDER=groq` and providing the Groq-specific values shown below.

Create a `.env` file in the project root with one of the following:

```env
# Use Gemini (default)
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_key

OR

LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_key
GROQ_MODEL_NAME=llama-3.1-70b-versatile
GROQ_API_BASE_URL=https://api.groq.com/openai/v1

# Optional overrides
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RERANKER_MODEL_NAME=BAAI/bge-reranker-large
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
- Split them into semantic chunks
- Generate embeddings
- Create searchable indexes (vector + BM25)

**Note**: First run may take time as it downloads AI models.

#### 7. Start the Web Application

```bash
python run.py
```

The server will start at `http://localhost:8000`

#### 8. Open in Browser

Visit `http://localhost:8000` in your web browser and start asking questions!

## Features

### Document Support
- **PDF**: Text extraction with layout preservation and RTL support
- **HTML**: Extracts content and sections from web pages with surrogate PDF generation
- **DOCX**: Processes Word documents with heading detection
- **Audio**: Transcribes MP3, WAV, M4A, MP4, FLAC files using Whisper

### Advanced Search Capabilities
- **Hybrid Retrieval**: Combines vector similarity search (semantic) with BM25 keyword search
- **Contextual Compression**: Extracts only the most relevant portions of documents (reduces tokens by ~40%)
- **Cross-Encoder Reranking**: Uses BAAI/bge-reranker-large to improve result quality
- **Semantic Chunking**: Respects paragraph and sentence boundaries for better context
- **Multilingual**: Supports English and Arabic with automatic language detection
- **Voice Search**: Speak your questions using the microphone button
- **Conversation Memory**: Maintains context for follow-up questions (3 turns, 30 min)

### Answer Quality
- **Answer Verification**: Computes confidence scores (high/medium/low/unverified) based on:
  - Factual grounding in sources (40%)
  - Citation coverage (30%)
  - Answer specificity (20%)
  - Question relevance (10%)
- **Citation Tracking**: Every answer includes precise chunk-level citations
- **Deeplinking**: Click citations to jump to exact pages in source documents

### User Interface
- **Modern Design**: Clean, responsive web interface with smooth animations
- **Dark Mode**: Full dark theme with toggle button and localStorage persistence
- **Loading States**: Beautiful skeleton screens during search
- **Result Filtering**: Filter by language (EN/AR) and document type (PDF/HTML/Audio)
- **Result Sorting**: Sort by relevance, document type, or language
- **Query Highlighting**: Search terms highlighted in orange in result snippets
- **RTL Support**: Full right-to-left support for Arabic text
- **PDF Viewer**: Built-in PDF.js viewer with direct page links
- **Citation Cards**: Each answer includes clickable citation cards with snippets

## Advanced Configuration

### RAG Pipeline Configuration

```env
# Enable/disable semantic chunking (default: true)
USE_SEMANTIC_CHUNKING=true

# Contextual compression settings
COMPRESSION_MAX_CHARS=600

# Conversation memory settings
CONVERSATION_MAX_TURNS=3
CONVERSATION_MAX_AGE_MINUTES=30

# Answer verification threshold
CONFIDENCE_THRESHOLD=0.5
```

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

# Set device (auto, cuda, mps, cpu)
RERANKER_DEVICE=cuda

# Set candidate limit for reranking
RERANKER_CANDIDATE_LIMIT=18
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

**Problem**: "No module named 'xxx'"
**Solution**: Make sure virtual environment is activated and run `pip install -r requirements.txt`

**Problem**: "GEMINI_API_KEY not found"
**Solution**: Create a `.env` file with your API key

**Problem**: Documents not appearing in search
**Solution**: Run `python ingest.py` to process documents first

**Problem**: Port 8000 already in use
**Solution**: Change port in `run.py` or stop the other service

**Problem**: Low confidence scores on answers
**Solution**: This indicates the answer may not be well-grounded in sources. Try rephrasing your question or adding more relevant documents.

**Problem**: Follow-up questions not working
**Solution**: Conversation memory expires after 30 minutes. Start a new conversation or rephrase as a standalone question.

## Project Structure

```
Document-Search-with-Citation-and-Deeplinking-main/
├── app/
│   ├── web/              # Flask web application
│   │   ├── routes.py     # API endpoints and UI logic
│   │   ├── templates/    # HTML templates
│   │   └── static/       # CSS, JS, assets
│   ├── retrieval/        # Search and RAG logic
│   │   ├── query_service.py          # Main query pipeline
│   │   ├── vector_store.py           # FAISS/ChromaDB integration
│   │   ├── keyword_store.py          # BM25 index
│   │   ├── reranker.py               # Cross-encoder reranking
│   │   ├── context_compressor.py     # NEW: Contextual compression
│   │   ├── conversation_memory.py    # NEW: Conversation tracking
│   │   ├── answer_verifier.py        # NEW: Confidence scoring
│   │   ├── embeddings.py             # Embedding models
│   │   └── llm_client.py             # Gemini/Groq clients
│   ├── ingestion/        # Document processing
│   │   ├── extractors.py             # PDF/HTML/DOCX/Audio extraction
│   │   ├── chunker.py                # Document chunking
│   │   ├── semantic_chunker.py       # NEW: Semantic chunking
│   │   └── pipeline.py               # Ingestion orchestration
│   ├── nlp/              # Language detection
│   └── audio/            # Speech-to-text
├── uploads/              # Place documents here
├── storage/              # Indexes and data storage
│   ├── vector_store/     # FAISS/ChromaDB indexes
│   ├── bm25_index.json   # BM25 keyword index
│   ├── models/           # Downloaded ML models
│   └── debug_payloads/   # Failed LLM responses (debugging)
├── logs/                 # Application logs
├── ingest.py             # Document ingestion script
├── run.py                # Start the web server
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## API Endpoints

- `GET /` - Main search interface
- `POST /upload` - Upload new documents
- `POST /transcribe` - Voice-to-text transcription
- `GET /viewer/<doc_path>` - Document viewer with deeplinking
- `GET /download/<doc_path>` - Download original document

## Performance Tips

1. **Use GPU acceleration**: Set `RERANKER_DEVICE=cuda` if you have a GPU
2. **Adjust chunk sizes**: Larger chunks (1200+) for better context, smaller (400-600) for precision
3. **Enable hybrid retrieval**: Set `ENABLE_HYBRID_RETRIEVAL=true` for better recall
4. **Use semantic chunking**: Enabled by default, preserves document structure
5. **Monitor confidence scores**: Low scores indicate you may need better source documents

## Next Steps

1. Upload your documents to the `uploads/` folder
2. Run `python ingest.py` to index them
3. Start the server with `python run.py`
4. Open `http://localhost:8000` and ask questions!
5. Try follow-up questions to test conversation memory
6. Check confidence scores to assess answer quality
7. Use dark mode toggle for comfortable viewing

For questions or issues, check the logs in `logs/app.log`.

## Environment Variables Example

Create a `.env` file in the project root with the following configuration:

```env
# LLM Provider (gemini or groq)
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL_NAME=gemini-2.0-flash

# Alternative: Use Groq
# LLM_PROVIDER=groq
# GROQ_API_KEY=your_groq_key
# GROQ_MODEL_NAME=llama-3.1-70b-versatile
# GROQ_API_BASE_URL=https://api.groq.com/openai/v1

# Vector Store
VECTOR_STORE_BACKEND=auto
VECTOR_STORE_PATH=storage/vector_store

# Chunking Configuration
CHUNK_SIZE=800
CHUNK_OVERLAP=200
# Arabic-specific chunking (50% larger for semantic equivalence)
CHUNK_SIZE_AR=1200
CHUNK_OVERLAP_AR=300

# Embedding Model
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_CACHE_DIR=storage/models
EMBEDDING_MODEL_PATH=

# Reranking
RERANKER_MODEL_NAME=BAAI/bge-reranker-large
RERANKER_DEVICE=auto
RERANKER_CANDIDATE_LIMIT=18

# Hybrid Retrieval
ENABLE_HYBRID_RETRIEVAL=true
BM25_INDEX_PATH=storage/bm25_index.json

# Speech-to-Text
WHISPER_MODEL_NAME=medium
WHISPER_DEVICE=auto

# Application
LOG_FILE=logs/app.log
ENVIRONMENT=development
```

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- Built with LangChain, FAISS, and modern ML models
- PDF.js for document viewing
- OpenAI Whisper for speech recognition
- Google Gemini / Groq for LLM capabilities
