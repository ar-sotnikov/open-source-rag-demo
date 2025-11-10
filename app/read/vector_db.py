import json
import logging
import os
from typing import Any

import psycopg2
import psycopg2.errors
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from app.agents.observability import trace_span
from app.core.config import DATA_DIR, EMBEDDING_MODEL, OLLAMA_HOST

logger = logging.getLogger(__name__)


class VectorDB:
    def __init__(
        self, json_file_path: str = None, table_name: str = None, embed_dim: int = 768
    ):
        """
        Initialize Vector Database Manager

        Args:
            json_file_path: Path to JSON file with chunks
            table_name: Database table name
            embed_dim: Embedding dimensions
        """
        load_dotenv()

        self.database = os.getenv("POSTGRES_DB")
        self.user = os.getenv("POSTGRES_USER")
        self.password = os.getenv("POSTGRES_PASSWORD")
        self.host = os.getenv("POSTGRES_HOST", "postgres")
        self.port = int(os.getenv("POSTGRES_PORT", "5432"))

        if not all([self.database, self.user, self.password]):
            raise ValueError(
                f"Missing database credentials. "
                f"DB: {self.database}, User: {self.user}, "
                f"Password: {'SET' if self.password else 'NOT SET'}"
            )

        self.json_file_path = json_file_path
        self.table_name = table_name or os.getenv("POSTGRES_TABLE", "rag_vectors")
        self.embed_dim = embed_dim
        self.conn = None
        self.embed_model = None
        self.vector_store = None
        self.index = None

    def connect_to_database(self) -> None:
        """Connect to database"""
        self.conn = psycopg2.connect(
            database=self.database,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
        )
        self.conn.autocommit = True

    def ensure_database_exists(self) -> None:
        """Create database if it doesn't exist"""
        if not self.conn:
            self.connect_to_database()

        with self.conn.cursor() as c:
            try:
                c.execute(f"CREATE DATABASE {self.database}")
                logger.info(f"Database {self.database} created")
            except psycopg2.errors.DuplicateDatabase:
                logger.info(f"Database {self.database} already exists")

    def initialize_embedding_model(self) -> None:
        """Initialize embedding model"""
        self.embed_model = OllamaEmbedding(
            model_name=EMBEDDING_MODEL, base_url=OLLAMA_HOST
        )

    def initialize_vector_store(self) -> None:
        """Initialize vector store"""
        self.vector_store = PGVectorStore.from_params(
            database=self.database,
            host=self.host,
            password=self.password,
            port=self.port,
            user=self.user,
            table_name=self.table_name,
            embed_dim=self.embed_dim,
        )

    def initialize_index(self) -> None:
        """Initialize index"""
        if not self.vector_store:
            self.initialize_vector_store()

        if not self.embed_model:
            self.initialize_embedding_model()

        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        self.index = VectorStoreIndex.from_documents(
            [],
            storage_context=storage_context,
            show_progress=True,
            embed_model=self.embed_model,
        )

    def load_json_chunks(self, json_file_path: str = None) -> list[dict[str, Any]]:
        """
        Load chunks from JSON file

        Args:
            json_file_path: Path to JSON file (if not provided during initialization)

        Returns:
            List of chunks
        """
        file_path = json_file_path or self.json_file_path
        if not file_path:
            raise ValueError("JSON file path not provided")

        data = None
        # Try different paths if only filename is provided
        possible_paths = [
            file_path,
            f"clear_docs/{file_path}",
            f"{DATA_DIR}/{file_path}",
            f"/app/data/{file_path}",
            f"../data/{file_path}",
            f"data/{file_path}",
        ]

        for path in possible_paths:
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                    logger.info(f"Loaded data from: {path}")
                    break
            except FileNotFoundError:
                continue
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in file {path}: {e}")

        if data is None:
            raise FileNotFoundError(
                f"Could not find JSON file. Tried: {', '.join(possible_paths)}"
            )

        # Extract chunks from JSON
        chunks = []
        if "chunks" in data:
            for chunk in data["chunks"]:
                chunks.append(chunk)
                logger.info(f"Loaded chunk: {chunk}")
        else:
            # If different structure, use entire data as chunks
            chunks = data if isinstance(data, list) else [data]

        return chunks

    def create_nodes_from_chunks(
        self, chunks: list[dict[str, Any]], source_name: str = None
    ) -> list[TextNode]:
        """
        Create TextNode from chunks

        Args:
            chunks: List of chunks
            source_name: Source name

        Returns:
            List of TextNode
        """
        if not source_name and self.json_file_path:
            source_name = os.path.basename(self.json_file_path).replace(".json", "")
        else:
            source_name = source_name or "unknown_source"

        nodes = []
        for i, chunk in enumerate(chunks):
            # Support different chunk formats
            if isinstance(chunk, dict):
                text = chunk.get("text", chunk.get("content", str(chunk)))
                chunk_id = chunk.get("id", f"{source_name}_{i}")
            else:
                text = str(chunk)
                chunk_id = f"{source_name}_{i}"

            node = TextNode(
                text=text,
                metadata={
                    "id": chunk_id,
                    "source": source_name,
                    "source_file": self.json_file_path or source_name,
                    "document_id": chunk_id,
                    "chunk_index": i,
                },
            )
            nodes.append(node)

        return nodes

    def insert_chunks(
        self, chunks: list[dict[str, Any]], source_name: str = None
    ) -> int:
        """
        Insert chunks into database

        Args:
            chunks: List of chunks
            source_name: Source name

        Returns:
            Number of inserted documents
        """
        if not self.index:
            self.initialize_index()

        nodes = self.create_nodes_from_chunks(chunks, source_name)
        self.index.insert_nodes(nodes)

        return len(nodes)

    def insert_from_json(
        self, json_file_path: str = None, source_name: str = None
    ) -> int:
        """
        Complete process: load JSON -> create chunks -> insert to DB

        Args:
            json_file_path: Path to JSON file
            source_name: Source name

        Returns:
            Number of inserted documents
        """
        # Lazy initialization: initialize index if not exists
        if not self.index:
            self._init()

        if json_file_path:
            self.json_file_path = json_file_path

        chunks = self.load_json_chunks()
        count = self.insert_chunks(chunks, source_name)

        logger.info(f"Successfully inserted {count} chunks from {self.json_file_path}")
        return count

    def _init(self):
        self.connect_to_database()
        self.ensure_database_exists()

        self.initialize_embedding_model()
        self.initialize_vector_store()
        self.initialize_index()

    def retrieve(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search for similar documents using vector similarity

        Args:
            query: Search query
            limit: Number of results to return

        Returns:
            List of similar documents with metadata
        """
        if not self.index:
            self.initialize_index()
        retriever = self.index.as_retriever(similarity_top_k=limit)

        # Retrieve documents
        with trace_span("rag.retrieve", {"query": query[:100], "limit": limit}) as span:
            response_nodes = retriever.retrieve(query)
            if span:
                span.set_attribute("documents_found", len(response_nodes))

        return response_nodes

    def close(self) -> None:
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
