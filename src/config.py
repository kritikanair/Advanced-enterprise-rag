"""
Central configuration management for the Advanced Multi-Source RAG system.
Loads configuration from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"


class VectorStoreConfig(BaseModel):
    """Vector database configuration"""
    store_type: str = Field(default="faiss", description="Vector store type")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name"
    )
    index_path: Path = Field(
        default=DATA_DIR / "faiss_index",
        description="Path to FAISS index"
    )
    dimension: int = Field(default=384, description="Embedding dimension")


class LLMConfig(BaseModel):
    """LLM configuration"""
    model_name: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
        description="HuggingFace model name"
    )
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Device for inference"
    )
    max_length: int = Field(default=2048, description="Max token length")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Top-p sampling")
    load_in_4bit: bool = Field(
        default=False,
        description="Use 4-bit quantization"
    )


class DocumentProcessingConfig(BaseModel):
    """Document processing configuration"""
    chunk_size: int = Field(default=512, description="Chunk size in characters")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")
    min_chunk_size: int = Field(default=100, description="Minimum chunk size")
    max_chunk_size: int = Field(default=1024, description="Maximum chunk size")


class RetrievalConfig(BaseModel):
    """Retrieval configuration"""
    top_k_retrievers: int = Field(
        default=5,
        description="Top K results per retriever"
    )
    rerank_top_k: int = Field(
        default=3,
        description="Top K after re-ranking"
    )
    similarity_threshold: float = Field(
        default=0.5,
        description="Minimum similarity threshold"
    )
    enable_reranking: bool = Field(default=True, description="Enable re-ranking")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Re-ranker model"
    )


class GraphConfig(BaseModel):
    """Knowledge graph configuration"""
    enable: bool = Field(default=True, description="Enable graph retrieval")
    ner_model: str = Field(default="en_core_web_sm", description="spaCy NER model")
    max_nodes: int = Field(default=1000, description="Maximum graph nodes")
    max_edges: int = Field(default=5000, description="Maximum graph edges")


class SentenceWindowConfig(BaseModel):
    """Sentence window configuration"""
    enable: bool = Field(default=True, description="Enable sentence window retrieval")
    window_size: int = Field(
        default=3,
        description="Context sentences before/after"
    )


class AppConfig(BaseModel):
    """Application configuration"""
    name: str = Field(default="Advanced Multi-Source RAG")
    port: int = Field(default=8501, description="Streamlit port")
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Path = Field(default=LOGS_DIR / "app.log", description="Log file path")
    max_workers: int = Field(default=4, description="Parallel processing workers")
    cache_embeddings: bool = Field(default=True, description="Cache embeddings")
    batch_size: int = Field(default=32, description="Batch processing size")


class Config:
    """Main configuration class"""
    
    def __init__(self):
        # Load from environment variables
        self.vector_store = VectorStoreConfig(
            store_type=os.getenv("VECTOR_STORE", "faiss"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2"
            ),
            index_path=Path(os.getenv("FAISS_INDEX_PATH", str(DATA_DIR / "faiss_index")))
        )
        
        self.llm = LLMConfig(
            model_name=os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"),
            device=os.getenv("LLM_DEVICE", "cpu"),
            max_length=int(os.getenv("LLM_MAX_LENGTH", "2048")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            top_p=float(os.getenv("LLM_TOP_P", "0.9")),
            load_in_4bit=os.getenv("LLM_LOAD_IN_4BIT", "false").lower() == "true"
        )
        
        self.processing = DocumentProcessingConfig(
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            min_chunk_size=int(os.getenv("MIN_CHUNK_SIZE", "100")),
            max_chunk_size=int(os.getenv("MAX_CHUNK_SIZE", "1024"))
        )
        
        self.retrieval = RetrievalConfig(
            top_k_retrievers=int(os.getenv("TOP_K_RETRIEVERS", "5")),
            rerank_top_k=int(os.getenv("RERANK_TOP_K", "3")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.5")),
            enable_reranking=os.getenv("ENABLE_RERANKING", "true").lower() == "true",
            reranker_model=os.getenv(
                "RERANKER_MODEL",
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
        )
        
        self.graph = GraphConfig(
            enable=os.getenv("GRAPH_ENABLE", "true").lower() == "true",
            ner_model=os.getenv("NER_MODEL", "en_core_web_sm"),
            max_nodes=int(os.getenv("MAX_GRAPH_NODES", "1000")),
            max_edges=int(os.getenv("MAX_GRAPH_EDGES", "5000"))
        )
        
        self.sentence_window = SentenceWindowConfig(
            enable=os.getenv("ENABLE_SENTENCE_WINDOW", "true").lower() == "true",
            window_size=int(os.getenv("SENTENCE_WINDOW_SIZE", "3"))
        )
        
        self.app = AppConfig(
            name=os.getenv("APP_NAME", "Advanced Multi-Source RAG"),
            port=int(os.getenv("APP_PORT", "8501")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=Path(os.getenv("LOG_FILE", str(LOGS_DIR / "app.log"))),
            max_workers=int(os.getenv("MAX_WORKERS", "4")),
            cache_embeddings=os.getenv("CACHE_EMBEDDINGS", "true").lower() == "true",
            batch_size=int(os.getenv("BATCH_SIZE", "32"))
        )
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            DATA_DIR,
            MODELS_DIR,
            LOGS_DIR,
            DATA_DIR / "documents",
            DATA_DIR / "cache",
            self.vector_store.index_path.parent,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def __repr__(self):
        return f"Config(llm={self.llm.model_name}, embedding={self.vector_store.embedding_model})"


# Global configuration instance
config = Config()
