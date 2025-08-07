# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure

This is a RAG (Retrieval-Augmented Generation) project with multiple implementations:

- `3.mission/` - Latest RAG implementation using LangGraph with streaming support
- `langgraph-fastapi-streaming/` - Production FastAPI service with real-time streaming
- `0.mission/` - Early experiments and prototypes
- `chunking/` - Document chunking experiments  
- `loader/` - Document loading utilities

## Key Components

### RAG Service Architecture
The core RAG system uses LangGraph to orchestrate a multi-step conversation flow:

1. **Router**: Classifies queries into categories (document, general, greeting)
2. **Retrieval**: Searches relevant documents using FAISS vector store
3. **Generation**: Produces contextual responses using Claude Haiku

### Streaming Implementation
- Real-time token streaming using AsyncGenerator
- Server-Sent Events (SSE) for web clients
- Multi-session conversation history management

## Development Commands

### FastAPI Service
```bash
cd langgraph-fastapi-streaming
pip install -r requirements.txt
python main.py
```
Server runs on port 8001 with endpoints:
- `/chat/stream` - Streaming responses
- `/chat` - Non-streaming responses
- `/health` - Health check

### Jupyter Development
```bash
cd 3.mission
pip install -r requirements.txt
jupyter notebook
```

### Dependencies
Main packages across the project:
- `langchain` + `langgraph` for orchestration
- `langchain-anthropic` for Claude integration
- `langchain-upstage` for document parsing
- `faiss-cpu` for vector search
- `sentence-transformers` for embeddings
- `fastapi` + `uvicorn` for web service

## Key Implementation Details

### Document Processing Pipeline
1. PDF parsing with Upstage DocumentParseLoader
2. Recursive text splitting (1500 chars, 300 overlap)
3. Multilingual embeddings (intfloat/multilingual-e5-large-instruct)
4. FAISS vector store with similarity search (k=5)

### Session Management
- UUID-based session tracking
- In-memory conversation histories
- State preservation across requests

## API Configuration
The service uses hardcoded API keys in `langgraph-fastapi-streaming/rag_service.py:50-57`. In production, these should be environment variables.

## Testing
No specific test framework is configured. Test manually via:
- FastAPI `/health` endpoint
- Jupyter notebooks for component testing
- Direct API calls to `/chat` endpoints