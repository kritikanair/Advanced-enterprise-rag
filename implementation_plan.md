# Advanced Multi-Source RAG for Enterprise Knowledge Base

## Project Overview

This implementation plan details the development of a sophisticated Retrieval-Augmented Generation (RAG) system designed for enterprise knowledge bases. The system ingests data from three distinct sources (PDFs, websites, and structured CSV/databases), employs multiple retrieval strategies, fuses and re-ranks results based on semantic relevance, and generates precise answers with source citations using a Large Language Model.

### Architecture Highlights

The system follows a 10-stage pipeline as shown in the architecture diagram:

1. **Streamlit UI** - User interface for data upload and queries
2. **Data Ingestion Pipeline** - Multi-format document loaders
3. **Unified Document DB** - Centralized document storage
4. **Processing Layer** - Chunking, cleaning, metadata tagging
5. **Multi-Index Layer** - Vector, sentence-window, and graph indices
6. **Multi-Retriever System** - Three parallel retrieval strategies
7. **Re-Ranking Engine** - Semantic relevance scoring
8. **Query Construction** - Context-aware query optimization
9. **LLM Integration** - Answer generation with citations
10. **Response Formatting** - Structured output with sources

### Technology Stack

- **Framework**: Python 3.9+
- **RAG Orchestration**: LlamaIndex + LangChain
- **Vector Database**: FAISS (local, CPU-based)
- **Embeddings**: HuggingFace models (sentence-transformers)
- **Graph Database**: NetworkX (lightweight, in-memory)
- **LLM**: Open-source models via HuggingFace (Llama 2, Mistral, or Zephyr)
- **Web Framework**: Streamlit
- **Deployment**: Local development environment
- **Storage**: Local file system

> [!NOTE]
> **100% Free & Open-Source Stack**: All selected technologies are completely free with no subscription costs. FAISS is CPU-based (no GPU required for indexing), NetworkX runs in-memory, and HuggingFace models can be run locally. For optimal performance with larger models, a GPU is recommended but not required.

---

## Technology Choices Confirmed ✅

> [!NOTE]
> **Open-Source LLM**: Using HuggingFace models (Mistral-7B-Instruct or Zephyr-7B-beta) for local inference. These models can run on CPU (slower) or GPU (faster). No API costs.

> [!NOTE]
> **FAISS Vector Database**: CPU-based vector similarity search. Completely free, no subscription required. Handles 10K-1M documents efficiently on standard hardware.

> [!NOTE]
> **Local Deployment**: Application runs on your local machine. No cloud infrastructure costs. Can be containerized with Docker for portability.

> [!NOTE]
> **NetworkX Graph Database**: Lightweight, in-memory graph operations. Perfect for knowledge graph construction with thousands of entities. No external database required.

> [!IMPORTANT]
> **Data Privacy & Security**: All data processing happens locally on your machine. No data is sent to external APIs (except if you choose to use external LLM APIs later). Suitable for sensitive enterprise data with proper access controls.

### Cost Verification

| Component | Cost | License | Notes |
|-----------|------|---------|-------|
| Python 3.9+ | **Free** | PSF License | Open-source |
| LlamaIndex | **Free** | MIT License | Open-source |
| LangChain | **Free** | MIT License | Open-source |
| FAISS | **Free** | MIT License | Meta/Facebook AI |
| HuggingFace Transformers | **Free** | Apache 2.0 | Open-source |
| Sentence-Transformers | **Free** | Apache 2.0 | Open-source |
| NetworkX | **Free** | BSD License | Open-source |
| Streamlit | **Free** | Apache 2.0 | Open-source (Community Edition) |
| Mistral/Zephyr/Llama 2 | **Free** | Apache 2.0/MIT | Open-source models |
| PyPDF | **Free** | BSD License | Open-source |
| BeautifulSoup4 | **Free** | MIT License | Open-source |
| Pandas | **Free** | BSD License | Open-source |
| spaCy | **Free** | MIT License | Open-source |

**Total Cost: $0** - All components are free and open-source!

---

## Proposed Changes

### Core Infrastructure

#### [NEW] [requirements.txt](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/requirements.txt)

Python dependencies including:
- `llama-index>=0.9.0` - Core RAG orchestration (Free, MIT)
- `langchain>=0.1.0` - Additional LLM utilities (Free, MIT)
- `faiss-cpu>=1.7.4` - Vector similarity search (Free, MIT)
- `sentence-transformers>=2.2.0` - Embeddings generation (Free, Apache 2.0)
- `transformers>=4.35.0` - HuggingFace model loading (Free, Apache 2.0)
- `torch>=2.0.0` - PyTorch for model inference (Free, BSD)
- `accelerate>=0.24.0` - Optimized model loading (Free, Apache 2.0)
- `streamlit>=1.28.0` - Web interface (Free, Apache 2.0)
- `beautifulsoup4>=4.12.0` - Web scraping (Free, MIT)
- `pypdf>=3.17.0` - PDF parsing (Free, BSD)
- `pandas>=2.0.0` - CSV/data handling (Free, BSD)
- `networkx>=3.1` - Graph operations (Free, BSD)
- `spacy>=3.7.0` - NER for entity extraction (Free, MIT)
- `python-dotenv>=1.0.0` - Environment management (Free, BSD)

#### [NEW] [.env.example](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/.env.example)

Environment configuration template:
```env
# Vector Database Configuration
VECTOR_STORE=faiss
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
FAISS_INDEX_PATH=./data/faiss_index

# LLM Configuration (Open-Source Models)
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
# Alternative options:
# LLM_MODEL=HuggingFaceH4/zephyr-7b-beta
# LLM_MODEL=meta-llama/Llama-2-7b-chat-hf

LLM_DEVICE=cpu  # or 'cuda' if GPU available
LLM_MAX_LENGTH=2048
LLM_TEMPERATURE=0.7

# Document Processing
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Retrieval Configuration
TOP_K_RETRIEVERS=5
RERANK_TOP_K=3

# Graph Database
GRAPH_ENABLE=True
NER_MODEL=en_core_web_sm  # spaCy model
```

#### [NEW] [config.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/config.py)

Central configuration management for all system parameters including model selection, chunk sizes, retrieval settings, and deployment options.

---

### Data Ingestion Pipeline

#### [NEW] [src/ingestion/pdf_loader.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/ingestion/pdf_loader.py)

PDF document loader using `pypdf` and LlamaIndex's `PDFReader`:
- Extract text with metadata (page numbers, titles)
- Handle multi-page documents
- Support batch processing
- Extract embedded images and tables (optional)

#### [NEW] [src/ingestion/web_loader.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/ingestion/web_loader.py)

Web content scraper:
- Accept URLs or sitemap
- Use BeautifulSoup4 for HTML parsing
- Extract main content (remove headers/footers/ads)
- Handle pagination and multi-page articles
- Respect robots.txt and rate limiting

#### [NEW] [src/ingestion/csv_loader.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/ingestion/csv_loader.py)

Structured data loader:
- Parse CSV files with pandas
- Support database connections (SQLite, PostgreSQL, MySQL)
- Convert tabular data to document format
- Preserve schema information as metadata
- Handle data type inference

#### [NEW] [src/ingestion/document_store.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/ingestion/document_store.py)

Unified document storage:
- Define `Document` schema with metadata fields
- Implement document ID generation
- Store raw documents and processed versions
- Maintain source tracking
- Support CRUD operations

---

### Document Processing

#### [NEW] [src/processing/chunker.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/processing/chunker.py)

Text chunking with multiple strategies:
- **Fixed-size chunking**: Split by character/token count with overlap
- **Sentence-aware chunking**: Preserve sentence boundaries
- **Semantic chunking**: Split based on topic shifts
- **Recursive chunking**: Hierarchical splitting for large documents
- Implement LlamaIndex's `SentenceSplitter` and custom chunkers

#### [NEW] [src/processing/cleaner.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/processing/cleaner.py)

Text cleaning and normalization:
- Remove excessive whitespace, special characters
- Normalize unicode characters
- Fix encoding issues
- Remove boilerplate content
- Language detection and filtering

#### [NEW] [src/processing/metadata_tagger.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/processing/metadata_tagger.py)

Metadata extraction and enrichment:
- Extract document titles, authors, dates
- Classify document types
- Extract keywords using TF-IDF or KeyBERT
- Generate document summaries
- Add custom tags based on content

---

### Multi-Index System

#### [NEW] [src/indexing/vector_index.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/indexing/vector_index.py)

Vector database implementation:
- Initialize FAISS index (CPU-based, no GPU required)
- Generate embeddings using HuggingFace sentence-transformers
- Build dense vector index for semantic search
- Support incremental indexing and persistence to disk
- Implement similarity search with configurable top-k
- Use IndexFlatL2 for exact search or IndexIVFFlat for faster approximate search

#### [NEW] [src/indexing/sentence_window_index.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/indexing/sentence_window_index.py)

Sentence-window retrieval index:
- Implement LlamaIndex's `SentenceWindowNodeParser`
- Store sentences with surrounding context windows
- Enable fine-grained retrieval with context expansion
- Configure window sizes (e.g., 3 sentences before/after)

#### [NEW] [src/indexing/graph_index.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/indexing/graph_index.py)

Knowledge graph construction:
- Extract entities using NER (spaCy or Transformers)
- Identify relationships between entities
- Build graph structure with NetworkX
- Support entity resolution and linking
- Enable graph traversal queries
- Store entity attributes and metadata

---

### Multi-Retriever System

#### [NEW] [src/retrieval/vector_retriever.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/retrieval/vector_retriever.py)

Vector similarity search retriever:
- Query vector index with user question embeddings
- Return top-k most similar chunks
- Include relevance scores
- Support filtering by metadata

#### [NEW] [src/retrieval/sentence_window_retriever.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/retrieval/sentence_window_retriever.py)

Sentence-window retriever:
- Retrieve precise sentences matching query
- Expand to include context windows
- Return structured results with sentence + context

#### [NEW] [src/retrieval/graph_retriever.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/retrieval/graph_retriever.py)

Graph-based retriever:
- Extract entities from query
- Traverse knowledge graph to find related entities
- Retrieve documents containing matching entities/relationships
- Support multi-hop reasoning

#### [NEW] [src/retrieval/query_constructor.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/retrieval/query_constructor.py)

Query optimization:
- Analyze user query intent
- Generate query variations
- Extract key phrases and entities
- Support query expansion

#### [NEW] [src/retrieval/fusion.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/retrieval/fusion.py)

Context fusion mechanism:
- Merge results from multiple retrievers
- Deduplicate documents
- Combine relevance scores
- Preserve source attribution

---

### Re-Ranking & LLM Integration

#### [NEW] [src/reranking/reranker.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/reranking/reranker.py)

Semantic re-ranking:
- Implement cross-encoder re-ranking (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Score query-document pairs
- Re-order results by semantic relevance
- Support configurable re-ranking models

#### [NEW] [src/llm/llm_client.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/llm/llm_client.py)

LLM integration:
- Initialize HuggingFace pipeline for text generation
- Load open-source models (Mistral, Zephyr, or Llama 2)
- Implement text generation with temperature control
- Support CPU and GPU inference
- Optimize model loading with 4-bit quantization (optional)
- Implement retry logic and error handling
- No API costs - fully local inference

#### [NEW] [src/llm/prompt_templates.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/llm/prompt_templates.py)

Prompt engineering:
- Create system prompts for RAG tasks
- Design templates for answer generation with citations
- Implement few-shot examples
- Support custom prompt modifications

#### [NEW] [src/llm/citation_extractor.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/llm/citation_extractor.py)

Source citation extraction:
- Parse LLM responses for citation markers
- Map citations to source documents
- Format citations with document metadata
- Validate citation accuracy

#### [NEW] [src/llm/response_formatter.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/llm/response_formatter.py)

Response formatting:
- Structure final output with answer + sources
- Generate citation links
- Format latency/accuracy metrics
- Create human-readable responses

---

### Streamlit Web Interface

#### [NEW] [src/app.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/app.py)

Main Streamlit application:
- Design multi-section layout (upload, query, results)
- Implement session state management
- Coordinate backend pipeline execution
- Display real-time processing status

#### [NEW] [src/ui/upload_interface.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/ui/upload_interface.py)

File upload interface:
- Support PDF file uploads
- Accept CSV file uploads
- Provide URL input field
- Display upload progress
- Validate file types and sizes

#### [NEW] [src/ui/query_interface.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/ui/query_interface.py)

Query input interface:
- Text input for questions
- Query history tracking
- Example queries for guidance
- Search button with loading state

#### [NEW] [src/ui/results_display.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/src/ui/results_display.py)

Results presentation:
- Display answer prominently
- Show source citations with clickable links
- Present latency metrics (retrieval time, LLM time, total time)
- Display accuracy/confidence scores
- Show retrieved documents in expandable sections

---

### Testing & Quality Assurance

#### [NEW] [tests/test_ingestion.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/tests/test_ingestion.py)

Unit tests for data ingestion:
- Test PDF loader with sample documents
- Test web loader with mock HTML
- Test CSV loader with sample data
- Validate document schema

#### [NEW] [tests/test_processing.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/tests/test_processing.py)

Unit tests for processing:
- Test chunking strategies
- Test text cleaning functions
- Test metadata extraction
- Validate chunk boundaries

#### [NEW] [tests/test_retrieval.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/tests/test_retrieval.py)

Unit tests for retrieval:
- Test vector search
- Test sentence-window retrieval
- Test graph retrieval
- Test fusion logic

#### [NEW] [tests/test_integration.py](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/tests/test_integration.py)

End-to-end integration tests:
- Test full pipeline from ingestion to response
- Use sample knowledge base
- Validate query accuracy
- Test multiple query types

#### [NEW] [tests/sample_data/](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/tests/sample_data/)

Test data directory:
- Sample PDF documents
- Sample CSV files
- Mock web content
- Expected query-answer pairs

---

### Local Deployment Configuration

#### [NEW] [Dockerfile](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/Dockerfile)

Container configuration for local deployment:
- Base image: `python:3.9-slim`
- Install system dependencies (build-essential for FAISS)
- Copy application code
- Install Python packages from requirements.txt
- Expose Streamlit port (8501)
- Define entrypoint for Streamlit app
- Volume mounts for data, models, and FAISS index

#### [NEW] [docker-compose.yml](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/docker-compose.yml)

Local container orchestration:
- Single Streamlit app service
- Volume mounts for:
  - `./data` - Document storage and FAISS index
  - `./models` - Cached HuggingFace models
  - `./src` - Application source code
- Port mapping: 8501:8501 for web interface
- Environment variable configuration
- Optional GPU support with nvidia-docker (if available)

#### [NEW] [scripts/setup.sh](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/scripts/setup.sh)

Local setup script:
- Create virtual environment
- Install Python dependencies
- Download spaCy NER model
- Create necessary directories (data, models, logs)
- Initialize FAISS index
- Download and cache default HuggingFace model

#### [NEW] [scripts/run_local.sh](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/scripts/run_local.sh)

Local execution script:
- Activate virtual environment
- Set environment variables from .env
- Start Streamlit app on localhost:8501
- Display access URL

---

### Documentation

#### [MODIFY] [README.md](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/README.md)

Comprehensive project documentation:
- Architecture overview with embedded diagram
- Setup instructions
- Usage examples
- Configuration guide
- API documentation
- Deployment guide
- Troubleshooting

#### [NEW] [docs/architecture.md](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/docs/architecture.md)

Detailed architecture documentation:
- System design decisions
- Component interactions
- Data flow diagrams
- Technology choices rationale

#### [NEW] [docs/api.md](file:///c:/coding%20stuffs/rag/Advanced-enterprise-rag/docs/api.md)

API reference:
- Core functions and classes
- Interface contracts
- Usage examples
- Return value specifications

---

## Verification Plan

### Automated Tests

**Unit Tests**:
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all unit tests with coverage
pytest tests/test_ingestion.py tests/test_processing.py tests/test_retrieval.py -v --cov=src --cov-report=html

# Expected: 80%+ code coverage, all tests passing
```

**Integration Tests**:
```bash
# Run end-to-end integration tests
pytest tests/test_integration.py -v

# Expected: Complete pipeline execution with sample data, accurate responses with citations
```

**Component Tests**:
```bash
# Test vector index creation and search
python -c "from src.indexing.vector_index import VectorIndex; idx = VectorIndex(); print('Vector index OK')"

# Test LLM integration (requires API key)
python -c "from src.llm.llm_client import LLMClient; client = LLMClient(); print('LLM client OK')"

# Test graph construction
python -c "from src.indexing.graph_index import GraphIndex; graph = GraphIndex(); print('Graph index OK')"
```

### Manual Verification

**Local Development Testing**:
1. Set up virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. Run the Streamlit app locally:
   ```bash
   streamlit run src/app.py
   ```

4. Test data ingestion:
   - Upload a sample PDF (e.g., research paper, technical documentation)
   - Provide a website URL (e.g., documentation site)
   - Upload a CSV file (e.g., product catalog, FAQ data)
   - Verify documents appear in the system

5. Test query functionality:
   - Ask a question that requires information from PDFs: "What is the main conclusion of the research paper?"
   - Ask a question requiring web data: "What are the installation steps?"
   - Ask a question requiring structured data: "What products are available in category X?"
   - **Expected**: Accurate answers with source citations from correct document types

6. Validate multi-retriever system:
   - Check that responses include citations from multiple sources
   - Verify latency metrics are displayed (retrieval time, LLM time, total time)
   - Confirm source documents are clickable/expandable
   - **Expected**: Answers synthesize information from 2+ sources when relevant

7. Performance validation:
   - Query response time < 10 seconds for typical queries
   - System handles 10+ concurrent users without degradation
   - Re-ranking improves answer relevance (compare with/without re-ranking)

**Local Deployment Testing**:
1. Run Docker container or local Python environment
2. Access Streamlit app at `http://localhost:8501`
3. Upload production-scale data (100+ documents)
4. Test query performance with local LLM
5. Monitor system logs and resource usage (CPU/RAM)
6. **Expected**: Stable operation, response times < 20 seconds on CPU (< 10s with GPU), accurate citations

### Browser-Based UI Testing

**Streamlit Interface Verification**:
1. Use browser subagent to navigate to `http://localhost:8501`
2. Verify UI elements render correctly:
   - Upload buttons for PDF/CSV
   - URL input field
   - Query text area
   - Results display section
3. Test file upload flow
4. Submit sample query and verify results display
5. Check citation links are functional
6. Validate metrics display (latency, accuracy scores)

### Performance Benchmarking

**Retrieval Accuracy**:
```bash
# Run accuracy benchmark with test dataset
python tests/benchmark_accuracy.py

# Expected: Retrieval recall@10 > 0.85, answer accuracy > 0.80
```

**Latency Profiling**:
```bash
# Profile query latency
python tests/benchmark_latency.py

# Expected: Vector retrieval < 1s, re-ranking < 2s, LLM generation < 5s
```

### User Acceptance Criteria

- [ ] System ingests PDFs, web pages, and CSV files successfully
- [ ] Query interface is intuitive and responsive
- [ ] Answers are accurate and cite sources correctly
- [ ] Latency metrics are displayed for transparency
- [ ] System handles edge cases (empty queries, large files, network errors)
- [ ] Documentation is clear and comprehensive
- [ ] Local deployment works via Docker or virtual environment
- [ ] Application runs on localhost without external dependencies

---

## Implementation Timeline

**Week 1-2**: Project setup, data ingestion, and processing (Phases 1-3)
**Week 3**: Multi-index system implementation (Phase 4)
**Week 4**: Multi-retriever system and fusion (Phase 5)
**Week 5**: Re-ranking and LLM integration (Phase 6)
**Week 6**: Streamlit UI development (Phase 7)
**Week 7**: Testing, optimization, and documentation (Phase 8)
**Week 8**: Deployment and final validation (Phase 9)

**Total estimated timeline**: 4-6 weeks for a working demo, 8+ weeks for production-ready system
