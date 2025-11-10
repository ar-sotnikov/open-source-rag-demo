"""
FastAPI Backend for Agentic RAG System
Foundation Stage - Ollama Proxy API for OpenWebUI integration
"""

import logging
import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.config import OLLAMA_HOST
from app.handlers.router import router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Agentic RAG Backend",
    description="Backend API for RAG system with OpenWebUI integration",
    version="0.1.0",
)
# Configure CORS for OpenWebUI frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class HealthResponse(BaseModel):
    status: str
    service: str


@app.get("/health")
async def health_check():
    """Health check endpoint for service monitoring"""
    return HealthResponse(status="healthy", service="agentic-rag-backend")


# Ollama Endpoints
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint with basic API information"""
    return {
        "message": "Agentic RAG Backend API",
        "version": "0.1.0",
        "stage": "Foundation",
        "ollama_host": OLLAMA_HOST,
        "endpoints": {"health": "/health", "ollama_proxy": "/api/*", "docs": "/docs"},
    }


if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", "8000"))

    # Run the application
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,  # Set to True for development
        log_level="info",
    )
