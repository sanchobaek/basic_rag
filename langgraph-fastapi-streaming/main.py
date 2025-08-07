from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager
from enum import Enum
import json
import logging
import uvicorn
import os

from rag_service import RAGService

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 전역 RAG 서비스 인스턴스
rag_service: Optional[RAGService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # 시작시 실행
    global rag_service

    try:
        logger.info("RAG 서비스 초기화 중...")
        rag_service = RAGService()
        await rag_service.initialize()
        logger.info("RAG 서비스 초기화 완료!")
    except Exception as e:
        logger.error(f"초기화 중 오류 발생: {e}")
        rag_service = None

    yield

    # 종료시 실행
    logger.info("애플리케이션 종료 중...")


# FastAPI 앱 생성
app = FastAPI(
    title="LangGraph RAG Streaming API",
    description="LangGraph 기반 RAG 스트리밍 API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic 모델
# class Category(str, Enum):
#     DOCUMENT = "document"
#     GENERAL = "general"
#     GREETING = "greeting"


class ChatRequest(BaseModel):
    message: str = Field(
        ..., description="사용자 메시지", min_length=1, max_length=2000
    )
    stream: bool = Field(default=True, description="스트리밍 응답 여부")
    session_id: Optional[str] = Field(default=None, description="세션 ID")


class StreamChunk(BaseModel):
    type: str = Field(description="청크 타입")
    content: str = Field(description="청크 내용")
    category: Optional[str] = Field(default=None, description="쿼리 카테고리")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="메타데이터")
    is_final: bool = Field(default=False, description="최종 청크 여부")


class ChatResponse(BaseModel):
    response: str = Field(description="AI 응답")
    category: str = Field(description="쿼리 카테고리")
    sources: List[str] = Field(default=[], description="참조 문서들")
    processing_time: float = Field(description="처리 시간")


@app.get("/")
async def root():
    return {
        "message": "LangGraph RAG Streaming API",
        "status": "running" if rag_service else "initialization_failed",
        "endpoints": {
            "chat_stream": "/chat/stream",
            "chat": "/chat",
            "health": "/health",
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if rag_service else "unhealthy",
        "rag_initialized": rag_service is not None,
    }


@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    """스트리밍 채팅 엔드포인트"""
    if not rag_service:
        raise HTTPException(
            status_code=503, detail="RAG 서비스가 초기화되지 않았습니다"
        )

    try:

        async def generate_stream():
            try:
                # 시작 알림
                start_chunk = StreamChunk(
                    type="start",
                    content="요청 처리를 시작합니다...",
                    metadata={"timestamp": "now"},
                )
                yield f"data: {start_chunk.model_dump_json()}\n\n"

                # RAG 서비스를 통해 스트리밍 실행
                async for chunk_data in rag_service.stream_chat_response(
                    message=request.message, session_id=request.session_id
                ):
                    chunk = StreamChunk(
                        type=chunk_data.get("type", "content"),
                        content=chunk_data.get("content", ""),
                        category=chunk_data.get("category"),
                        metadata=chunk_data.get("metadata", {}),
                        is_final=chunk_data.get("is_final", False),
                    )

                    yield f"data: {chunk.model_dump_json()}\n\n"

                    if chunk.is_final:
                        break

                # 스트리밍 완료 알림
                end_chunk = StreamChunk(
                    type="end", content="응답 생성이 완료되었습니다.", is_final=True
                )
                yield f"data: {end_chunk.model_dump_json()}\n\n"

            except Exception as e:
                error_chunk = StreamChunk(
                    type="error",
                    content=f"처리 중 오류가 발생했습니다: {str(e)}",
                    is_final=True,
                )
                yield f"data: {error_chunk.model_dump_json()}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Methods": "*",
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def non_stream_chat(request: ChatRequest):
    """비스트리밍 채팅 엔드포인트"""
    if not rag_service:
        raise HTTPException(
            status_code=503, detail="RAG 서비스가 초기화되지 않았습니다"
        )

    try:
        import time

        start_time = time.time()

        result = await rag_service.generate_full_response(
            message=request.message, session_id=request.session_id
        )

        processing_time = time.time() - start_time

        return ChatResponse(
            response=result["response"],
            category=result["category"],
            sources=result.get("sources", []),
            processing_time=round(processing_time, 2),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    PORT = 8001

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True, log_level="info")
