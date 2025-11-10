import asyncio
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.requests import Request

from app.agents.agent import process_with_agents
from app.agents.memory import USER_ID, add_message, get_history
from app.agents.observability import trace_span
from app.core.config import CLEAR_DOCS_DIR, OLLAMA_HOST, RAW_DOCS_DIR
from app.handlers.response import UploadChunkResponse
from app.main import app, logger
from app.read.chunker import Chunker
from app.read.vector_db import VectorDB

router = APIRouter()

chunker = Chunker()
vector_db = VectorDB()


async def get_http_client():
    return httpx.AsyncClient(timeout=30.0)


@app.get("/api/tags")
async def get_models():
    """Get available models from Ollama"""
    async with await get_http_client() as client:
        try:
            response = await client.get(f"{OLLAMA_HOST}/api/tags")
            return response.json()
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503, detail=f"Ollama connection failed: {e!s}"
            )


@app.post("/api/pull")
async def pull_model(request: Request):
    """Pull a model from Ollama"""
    body = await request.json()
    async with await get_http_client() as client:
        try:
            response = await client.post(f"{OLLAMA_HOST}/api/pull", json=body)
            return response.json()
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503, detail=f"Ollama connection failed: {e!s}"
            )


@app.post("/api/generate")
async def generate_response(request: Request):
    """Generate response from Ollama"""
    body = await request.json()
    async with await get_http_client() as client:
        try:
            response = await client.post(f"{OLLAMA_HOST}/api/generate", json=body)
            return response.json()
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503, detail=f"Ollama connection failed: {e!s}"
            )


@app.post("/api/chat")
async def chat_completion(request: Request):
    """Chat completion endpoint with RAG agents for OpenWebUI"""
    body = await request.json()
    load_dotenv()

    body["stream"] = False

    with trace_span("rag.chat", {"model": body.get("model", "unknown")}):
        try:
            # Detect request format: OpenWebUI uses "messages", direct API uses "prompt" or "query"
            if "messages" in body:
                # Extract user query from messages array (OpenWebUI format)
                messages = body.get("messages", [])
                user_query = None
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        user_query = msg.get("content")
                        break

                if not user_query:
                    raise HTTPException(
                        status_code=400, detail="No user message found in messages"
                    )
            else:
                # Direct API format (for curl/testing)
                user_query = body.get("prompt", body.get("query", ""))
                if not user_query:
                    raise HTTPException(status_code=400, detail="No query provided")

            with trace_span("rag.retrieve", {"query": user_query[:100], "limit": 3}):
                response_nodes = vector_db.retrieve(user_query, 3)

            # Format retrieved documents with metadata
            retrieved_docs = [
                {
                    "text": node.text,
                    "score": node.score if hasattr(node, "score") else 0.0,
                    "metadata": (
                        node.metadata
                        if hasattr(node, "metadata") and node.metadata
                        else {}
                    ),
                    "source": (
                        node.metadata.get("source_file", "unknown")
                        if hasattr(node, "metadata") and node.metadata
                        else "unknown"
                    ),
                    "document_id": (
                        node.metadata.get("id", node.metadata.get("document_id", None))
                        if hasattr(node, "metadata") and node.metadata
                        else None
                    ),
                }
                for node in response_nodes
            ]

            with trace_span(
                "agents.process",
                {
                    "model": body["model"],
                    "query_length": len(user_query),
                    "docs_count": len(retrieved_docs),
                },
            ):
                result = await asyncio.to_thread(
                    process_with_agents, body["model"], user_query, retrieved_docs
                )

            await add_message(USER_ID, "assistant", result.get("response", ""))
            history = await get_history(USER_ID)
            logging.info(f"conversation memory {history}")

            if "messages" in body:
                return {
                    "model": body["model"],
                    "created_at": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "message": {
                        "role": "assistant",
                        "content": result.get("response", ""),
                    },
                    "done": True,
                }
            else:
                return result

        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Chat failed: {e!s}")


@app.get("/api/version")
async def get_ollama_version():
    """Get Ollama version"""
    async with await get_http_client() as client:
        try:
            response = await client.get(f"{OLLAMA_HOST}/api/version")
            return response.json()
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503, detail=f"Ollama connection failed: {e!s}"
            )


@app.post("/api/rag")
async def rag_query(request: Request):
    """Perform RAG query using Ollama embeddings"""
    body = await request.json()
    load_dotenv()

    try:
        response_nodes = vector_db.retrieve(body["query"])
        response = [
            {
                "text": node.text,
                "score": node.score if hasattr(node, "score") else 0.0,
                "metadata": (
                    node.metadata if hasattr(node, "metadata") and node.metadata else {}
                ),
                "source": (
                    node.metadata.get("source_file", "unknown")
                    if hasattr(node, "metadata") and node.metadata
                    else "unknown"
                ),
                "document_id": (
                    node.metadata.get("id", node.metadata.get("document_id", None))
                    if hasattr(node, "metadata") and node.metadata
                    else None
                ),
            }
            for node in response_nodes
        ]
        return {"response": response, "query": body["query"], "count": len(response)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"RAG query failed: {e!s}")


@app.post("/api/upload-and-chunk", response_model=UploadChunkResponse)
async def upload_and_chunk_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(
        1000,
        ge=100,
        le=10000,
        description="Maximum chunk size in characters (default: 1000, range: 100-10000)",
    ),
):
    """
    Unified endpoint for document upload and chunking.

    Uploads a document, chunks it into smaller pieces, and automatically indexes
    the chunks into the vector database.

    Args:
        file: Document file to upload (supports .pdf, .docx, .doc, .md)
        chunk_size: Maximum size of each chunk in characters.
                   Default: 1000 characters (~170 words, ~250 tokens).
                   Range: 100-10000 characters.

    Returns:
        UploadChunkResponse with processing status and chunk count
    """
    allowed_extensions = {".pdf", ".docx", ".doc", ".txt", ".md"}
    file_extension = Path(file.filename).suffix.lower()
    chunker = Chunker()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_extension} not supported. Allowed: {allowed_extensions}",
        )

    # Ensure RAW_DOCS_DIR exists
    Path(RAW_DOCS_DIR).mkdir(parents=True, exist_ok=True)

    file_path = Path(RAW_DOCS_DIR) / file.filename

    try:
        with trace_span(
            "document.upload", {"filename": file.filename, "chunk_size": chunk_size}
        ) as span:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            file_size = file_path.stat().st_size
            if span:
                span.set_attribute("file_size", file_size)

            chunks = chunker.process_document(file.filename, chunk_size)

            if span:
                span.set_attribute("total_chunks", len(chunks))

            chunk_filename = f"{Path(file.filename).stem}_chunks.json"
            chunk_file_path = Path(CLEAR_DOCS_DIR) / chunk_filename

            indexed_count = 0
            indexing_error = None
            try:
                indexed_count = vector_db.insert_from_json(
                    json_file_path=str(chunk_file_path),
                    source_name=Path(file.filename).stem,
                )
                logger.info(f"Indexed {indexed_count} chunks from {chunk_filename}")
                if span:
                    span.set_attribute("indexed_count", indexed_count)
            except Exception as index_error:
                indexing_error = str(index_error)
                logger.error(
                    f"Indexing failed (chunks saved but not indexed): {indexing_error}"
                )
                if span:
                    span.set_attribute("indexing_error", indexing_error)

        return UploadChunkResponse(
            filename=file.filename,
            status="completed",
            total_chunks=len(chunks),
            chunk_size=chunk_size,
            message=f"Successfully processed {len(chunks)} chunks"
            + (f", indexed {indexed_count}" if indexed_count else "")
            + (
                f". Warning: indexing failed: {indexing_error}"
                if indexing_error
                else ""
            ),
        )

    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=500, detail=f"Document processing failed: {e!s}"
        )
