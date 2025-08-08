from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import json
import logging
import time
import uvicorn

# RAG 서비스 함수들 import
from rag_service import (
    initialize_rag,
    run_rag_query,
    stream_rag_query,
    get_conversation_history,
    clear_conversation_history
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 초기화 상태 변수
is_initialized = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    global is_initialized
    
    # 시작시 실행
    try:
        logger.info("RAG 서비스 초기화 중...")
        success = await initialize_rag()
        if success:
            is_initialized = True
            logger.info("RAG 서비스 초기화 완료!")
        else:
            logger.error("RAG 서비스 초기화 실패!")
            is_initialized = False
    except Exception as e:
        logger.error(f"초기화 중 오류 발생: {e}")
        is_initialized = False

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

# 테스트 페이지 제공
@app.get("/test")
async def serve_test_page():
    """스트리밍 테스트 페이지"""
    return FileResponse("test_streaming.html")


# Pydantic 모델
class ChatRequest(BaseModel):
    message: str = Field(
        ..., description="사용자 메시지", min_length=1, max_length=2000
    )
    session_id: Optional[str] = Field(default=None, description="세션 ID")


class StreamChunk(BaseModel):
    type: str = Field(description="청크 타입")
    content: str = Field(description="청크 내용")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="메타데이터")


class ChatResponse(BaseModel):
    response: str = Field(description="AI 응답")
    category: str = Field(description="쿼리 카테고리")
    sources: List[str] = Field(default=[], description="참조 문서들")
    processing_time: float = Field(description="처리 시간")


@app.get("/")
async def root():
    return {
        "message": "LangGraph RAG Streaming API",
        "status": "running" if is_initialized else "initialization_failed",
        "endpoints": {
            "chat_stream": "/chat/stream",
            "chat": "/chat",
            "health": "/health",
            "history": "/chat/history",
            "clear_history": "/chat/history (DELETE)",
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if is_initialized else "unhealthy",
        "rag_initialized": is_initialized,
    }


@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    """스트리밍 채팅 엔드포인트 - SSE 최적화"""
    if not is_initialized:
        raise HTTPException(
            status_code=503, detail="RAG 서비스가 초기화되지 않았습니다"
        )

    try:
        async def generate_stream():
            try:
                # 연결 확인 이벤트
                yield f"data: {json.dumps({'type': 'connected', 'content': 'Connected'})}\n\n"
                
                async for chunk_data in stream_rag_query(
                    user_input=request.message, session_id=request.session_id
                ):
                    chunk = StreamChunk(
                        type=chunk_data.get("type", "chunk"),
                        content=chunk_data.get("content", ""),
                        metadata=chunk_data.get("metadata", {}),
                    )

                    # SSE 형식으로 데이터 전송
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    
                    if chunk.type in ["final", "error"]:
                        break

            except Exception as e:
                # 스트리밍 오류 처리
                error_chunk = StreamChunk(
                    type="error",
                    content=f"처리 중 오류가 발생했습니다: {str(e)}",
                )
                yield f"data: {error_chunk.model_dump_json()}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def non_stream_chat(request: ChatRequest):
    """비스트리밍 채팅 엔드포인트"""
    if not is_initialized:
        raise HTTPException(
            status_code=503, detail="RAG 서비스가 초기화되지 않았습니다"
        )

    try:
        start_time = time.time()
        
        response = run_rag_query(
            user_input=request.message, session_id=request.session_id
        )
        
        return ChatResponse(
            response=response,
            category="unknown",
            sources=[],
            processing_time=round(time.time() - start_time, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/history")
async def get_chat_history():
    """대화 히스토리 조회"""
    if not is_initialized:
        raise HTTPException(
            status_code=503, detail="RAG 서비스가 초기화되지 않았습니다"
        )

    try:
        history = get_conversation_history()
        formatted_history = [
            {
                "type": "human" if msg.__class__.__name__ == "HumanMessage" else "ai",
                "content": msg.content,
            }
            for msg in history
        ]
        
        return {"history": formatted_history, "total_messages": len(formatted_history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat/history")
async def clear_chat_history():
    """대화 히스토리 삭제"""
    if not is_initialized:
        raise HTTPException(
            status_code=503, detail="RAG 서비스가 초기화되지 않았습니다"
        )

    try:
        clear_conversation_history()
        return {"message": "대화 히스토리가 삭제되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    PORT = 8001

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True, log_level="info")