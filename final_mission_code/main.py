from fastapi import FastAPI, HTTPException  # FastAPI 웹 프레임워크와 HTTP 예외 처리
from fastapi.responses import StreamingResponse, FileResponse  # 스트리밍 응답과 파일 응답 처리
from fastapi.middleware.cors import CORSMiddleware  # 크로스 오리진 요청 허용을 위한 CORS 미들웨어
from pydantic import BaseModel, Field  # 데이터 검증과 스키마 정의를 위한 Pydantic
from typing import List, Optional, Dict, Any  # 타입 힌팅을 위한 타입 정의들
from contextlib import asynccontextmanager  # 비동기 컨텍스트 매니저 (앱 생명주기 관리)
import json  # JSON 데이터 처리
import logging  # 로깅 기능
import uvicorn  # ASGI 서버 (FastAPI 실행용)

# RAG 서비스의 핵심 기능들을 rag_service.py에서 가져오기
from rag_service import (
    initialize_rag,  # RAG 시스템 초기화 함수
    stream_rag_query,  # 스트리밍 RAG 쿼리 실행 함수
    get_conversation_history,  # 대화 기록 조회 함수
    clear_conversation_history  # 대화 기록 삭제 함수
)

# 애플리케이션 로깅 설정 (INFO 레벨로 설정)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RAG 시스템 초기화 완료 여부를 추적하는 전역 변수
is_initialized = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 애플리케이션의 시작과 종료를 관리하는 생명주기 함수
    앱 시작 시 RAG 시스템을 초기화하고, 종료 시 정리 작업을 수행
    """
    global is_initialized
    
    # 애플리케이션 시작 시 실행되는 초기화 코드
    try:
        logger.info("RAG 서비스 초기화 중...")
        success = await initialize_rag()  # RAG 시스템 비동기 초기화 시도
        if success:
            is_initialized = True  # 초기화 성공 시 상태 업데이트
            logger.info("RAG 서비스 초기화 완료!")
        else:
            logger.error("RAG 서비스 초기화 실패!")
            is_initialized = False
    except Exception as e:
        logger.error(f"초기화 중 오류 발생: {e}")
        is_initialized = False

    yield  # 이 지점에서 애플리케이션이 실행됨

    # 애플리케이션 종료 시 실행되는 정리 코드
    logger.info("애플리케이션 종료 중...")


# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(
    title="LangGraph RAG Streaming API",  # API 문서에 표시될 제목
    description="LangGraph 기반 RAG 스트리밍 API",  # API 설명
    version="1.0.0",  # API 버전
    lifespan=lifespan,  # 애플리케이션 생명주기 관리 함수 연결
)

# 스트리밍 기능 테스트를 위한 HTML 페이지 제공 엔드포인트
@app.get("/test")
async def serve_test_page():
    """스트리밍 테스트용 HTML 페이지를 제공하는 엔드포인트"""
    return FileResponse("test_streaming.html")  # 정적 HTML 파일 반환


# Pydantic 데이터 모델 정의 (API 요청/응답 스키마)
class ChatRequest(BaseModel):
    """채팅 요청을 위한 데이터 모델"""
    message: str = Field(
        ...,  # 필수 필드
        description="사용자 메시지", 
        min_length=1,  # 최소 1자 이상
        max_length=2000  # 최대 2000자 제한
    )
    session_id: Optional[str] = Field(
        default=None,  # 선택적 필드 (기본값: None)
        description="대화 세션을 구분하기 위한 세션 ID"
    )

class StreamChunk(BaseModel):
    """스트리밍 응답의 각 청크를 나타내는 데이터 모델"""
    type: str = Field(description="청크의 타입 (예: 'chunk', 'final', 'error')")
    content: str = Field(description="청크의 실제 내용 (텍스트)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,  # 빈 딕셔너리로 기본값 설정
        description="추가 메타데이터 정보"
    )


@app.get("/")
async def root():
    """
    API 루트 엔드포인트 - 서비스 상태와 사용 가능한 엔드포인트 정보 제공
    """
    return {
        "message": "LangGraph RAG Streaming API",
        "status": "running" if is_initialized else "initialization_failed",
        "endpoints": {
            "chat_stream": "/chat/stream",  # 스트리밍 채팅 엔드포인트
            "health": "/health",  # 헬스체크 엔드포인트
            "history": "/chat/history",  # 대화 기록 조회 엔드포인트
            "clear_history": "/chat/history (DELETE)",  # 대화 기록 삭제 엔드포인트
        },
    }

@app.get("/health")
async def health_check():
    """
    서비스 상태를 확인하는 헬스체크 엔드포인트
    로드밸런서나 모니터링 도구에서 서비스 가용성 확인 용도
    """
    return {
        "status": "healthy" if is_initialized else "unhealthy",  # 전체 서비스 상태
        "rag_initialized": is_initialized,  # RAG 시스템 초기화 여부
    }

@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    """
    실시간 스트리밍 채팅 엔드포인트 - Server-Sent Events(SSE) 사용
    AI 응답을 토큰 단위로 실시간 스트리밍하여 사용자 경험 향상
    """
    # RAG 서비스 초기화 상태 확인
    if not is_initialized:
        raise HTTPException(
            status_code=503, detail="RAG 서비스가 초기화되지 않았습니다"
        )

    try:
        async def generate_stream():
            """스트림 데이터를 생성하는 내부 비동기 제너레이터 함수"""
            try:
                # 클라이언트와의 연결 확인을 위한 초기 이벤트 전송
                yield f"data: {json.dumps({'type': 'connected', 'content': 'Connected'})}\n\n"
                
                # RAG 서비스에서 스트리밍 응답을 비동기로 받아옴
                async for chunk_data in stream_rag_query(
                    user_input=request.message, session_id=request.session_id
                ):
                    # 받은 데이터를 StreamChunk 모델로 변환
                    chunk = StreamChunk(
                        type=chunk_data.get("type", "chunk"),
                        content=chunk_data.get("content", ""),
                        metadata=chunk_data.get("metadata", {}),
                    )

                    # SSE(Server-Sent Events) 형식으로 데이터 전송
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    
                    # 최종 응답이나 에러 시 스트림 종료
                    if chunk.type in ["final", "error"]:
                        break

            except Exception as e:
                # 스트리밍 중 발생한 오류를 클라이언트에게 전송
                error_chunk = StreamChunk(
                    type="error",
                    content=f"처리 중 오류가 발생했습니다: {str(e)}",
                )
                yield f"data: {error_chunk.model_dump_json()}\n\n"

        # 스트리밍 응답 반환 (SSE 헤더와 함께)
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",  # SSE MIME 타입
            headers={
                "Cache-Control": "no-cache",  # 캐싱 방지
                "Connection": "keep-alive",  # 연결 유지
                "Access-Control-Allow-Origin": "*",  # CORS 허용
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/history")
async def get_chat_history():
    """
    저장된 대화 기록을 조회하는 엔드포인트
    사용자와 AI의 이전 대화 내용을 확인할 수 있음
    """
    # RAG 서비스 초기화 상태 확인
    if not is_initialized:
        raise HTTPException(
            status_code=503, detail="RAG 서비스가 초기화되지 않았습니다"
        )

    try:
        # RAG 서비스에서 대화 기록 가져오기
        history = get_conversation_history()
        
        # 메시지 객체를 JSON 직렬화 가능한 형태로 변환
        formatted_history = [
            {
                "type": "human" if msg.__class__.__name__ == "HumanMessage" else "ai",
                "content": msg.content,
            }
            for msg in history
        ]
        
        return {
            "history": formatted_history,  # 포맷된 대화 기록
            "total_messages": len(formatted_history)  # 총 메시지 개수
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat/history")
async def clear_chat_history():
    """
    저장된 대화 기록을 모두 삭제하는 엔드포인트
    새로운 대화 세션을 시작하거나 메모리 정리 목적으로 사용
    """
    # RAG 서비스 초기화 상태 확인
    if not is_initialized:
        raise HTTPException(
            status_code=503, detail="RAG 서비스가 초기화되지 않았습니다"
        )

    try:
        # RAG 서비스의 대화 기록 삭제 함수 호출
        clear_conversation_history()
        return {"message": "대화 히스토리가 삭제되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    """
    스크립트가 직접 실행될 때 FastAPI 서버를 시작하는 메인 블록
    """
    PORT = 8001  # 서버 포트 번호 설정

    # Uvicorn ASGI 서버로 FastAPI 앱 실행
    uvicorn.run(
        "main:app",  # 앱 모듈:변수명
        host="0.0.0.0",  # 모든 IP에서 접근 가능하도록 설정
        port=PORT,  # 포트 번호
        reload=True,  # 코드 변경 시 자동 재시작 (개발 모드)
        log_level="info"  # 로깅 레벨 설정
    )