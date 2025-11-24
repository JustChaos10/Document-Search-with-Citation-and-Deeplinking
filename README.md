# Document Search with Citation and Deeplinking

An AI-powered document search and question-answering system that lets you ask questions about your documents and get accurate answers with citations. Supports PDF, HTML, DOCX files and audio recordings in both English and Arabic.

## How It Works

1. **Upload Documents**: Add PDF, HTML, DOCX, or audio files to the system
2. **Automatic Processing**: Documents are extracted, chunked, and indexed using vector embeddings and keyword search
3. **Ask Questions**: Type or speak your question in English or Arabic
4. **Get Answers**: The system uses advanced RAG (Retrieval-Augmented Generation) to find relevant information and generate accurate answers with citations
5. **View Sources**: Click on citations to see the exact page and context from your documents

- ## Technology Stack

- **Backend Framework**: Flask (Python web framework)
- **AI Model**: Google Gemini (default) or Groq (when `LLM_PROVIDER=groq`)
- **Vector Search**: FAISS or ChromaDB (semantic similarity search)
- **Keyword Search**: BM25 (traditional keyword matching)
- **Reranking**: Cross-encoder models (BAAI/bge-reranker-large)
- **Embeddings**: Multilingual sentence transformers (paraphrase-multilingual-MiniLM-L12-v2)
- **Speech-to-Text**: OpenAI Whisper (for voice queries)
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
- **AI provider API key**: leave the default `LLM_PROVIDER` pointing at Gemini and supply `GEMINI_API_KEY`, or set `LLM_PROVIDER=groq` along with `GROQ_API_KEY`.
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

### Document Support
- **PDF**: Text extraction with layout preservation
- **HTML**: Extracts content and sections from web pages
- **DOCX**: Processes Word documents with heading detection
- **Audio**: Transcribes MP3, WAV, M4A, MP4, FLAC files using Whisper

### Search Capabilities
- **Hybrid Retrieval**: Combines vector similarity search (semantic) with BM25 keyword search
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
LLM_PROVIDER=groq
GROQ_API_KEY=
GROQ_MODEL_NAME=llama-3.1-70b-versatile
GROQ_API_BASE_URL=https://api.groq.com/openai/v1
# Uncomment the lines below if you want to use Google Gemini instead
# LLM_PROVIDER=gemini
# GEMINI_API_KEY=
# GEMINI_MODEL_NAME=gemini-2.5-flash
VECTOR_STORE_BACKEND=auto
VECTOR_STORE_PATH=storage/vector_store
CHUNK_SIZE=800
CHUNK_OVERLAP=200
# Arabic-specific chunking (50% larger for semantic equivalence)
CHUNK_SIZE_AR=1200
CHUNK_OVERLAP_AR=300
LOG_FILE=logs/app.log
ENVIRONMENT=development
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_CACHE_DIR=storage/models
WHISPER_MODEL_NAME=medium
WHISPER_DEVICE=auto
EMBEDDING_MODEL_PATH=
ENABLE_HYBRID_RETRIEVAL=true
BM25_INDEX_PATH=storage/bm25_index.json
RERANKER_MODEL_NAME=BAAI/bge-reranker-large
RERANKER_DEVICE=auto
RERANKER_CANDIDATE_LIMIT=18
```
