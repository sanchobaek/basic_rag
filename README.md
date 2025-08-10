**🔥final_mission_code 폴더만 확인하시면 됩니다.**


# LangGraph FastAPI Streaming RAG

🚀 **LangGraph와 FastAPI를 활용한 실시간 스트리밍 RAG 시스템**

PostgreSQL/PGVector 벡터 데이터베이스와 Claude Sonnet을 사용하여 문서 기반 질의응답과 일반 질의응답을 지원하는 고성능 스트리밍 API입니다.

## ✨ 주요 기능

### 🔄 실시간 스트리밍
- **Server-Sent Events (SSE)** 기반 실시간 응답 스트리밍
- **LangGraph astream()** 을 활용한 토큰 레벨 스트리밍
- 연결 상태, 진행률, 메타데이터 실시간 전송

### 🧠 지능형 라우팅
- **자동 쿼리 분류**: 문서 관련 질문 vs 일반 질문
- **문서 기반 RAG**: PostgreSQL 벡터 검색 + 컨텍스트 기반 답변
- **일반 질의응답**: 웹 검색 도구와 연동된 실시간 정보 제공

### 🗄️ 고성능 벡터 데이터베이스
- **PostgreSQL + pgvector** 확장을 통한 벡터 검색
- **AWS RDS** 기반 확장 가능한 데이터베이스
- **지속성**: 한 번 데이터 인제스션 후 재시작 불필요

### 🔧 생산 준비 완료
- **FastAPI** 기반 고성능 웹 API
- **환경변수** 기반 보안 설정
- **세션 관리** 및 대화 히스토리 지원
- **CORS** 및 에러 핸들링

## 🏗️ 아키텍처

```mermaid
graph TB
    C[Client] --> F[FastAPI Server]
    F --> R[RAG Service]
    R --> G[LangGraph Router]
    
    G --> D[Document QA]
    G --> GQ[General QA]
    
    D --> V[PGVector Search]
    D --> L[Claude Sonnet]
    
    GQ --> T[Tavily Web Search]
    GQ --> L
    
    V --> P[(PostgreSQL + pgvector)]
    
    style C fill:#e1f5fe
    style F fill:#f3e5f5
    style R fill:#e8f5e8
    style G fill:#fff3e0
    style P fill:#fce4ec
```

## 📁 프로젝트 구조

```
langgraph-fastapi-streaming/
├── 📄 README.md              # 프로젝트 문서
├── 🔧 .env.example           # 환경변수 템플릿
├── 🚫 .gitignore             # Git 제외 파일
│
├── 🌐 main.py                # FastAPI 서버 & 스트리밍 엔드포인트
├── 🧠 rag_service.py         # LangGraph RAG 로직 & 스트리밍 구현
├── 📊 data_ingestion.py      # 데이터 파이프라인 (문서 → 벡터DB)
│
└── 🖥️ test_streaming.html    # 스트리밍 테스트 페이지
```

## 🚀 빠른 시작

### 1️⃣ 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd langgraph-fastapi-streaming

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일을 실제 값으로 수정하세요
```

### 2️⃣ 데이터 인제스션 (최초 1회)

```bash
# 문서를 PostgreSQL 벡터DB에 저장
python data_ingestion.py
```

### 3️⃣ 서버 실행

```bash
# FastAPI 서버 시작
python main.py
```

### 4️⃣ 테스트

- **API 문서**: http://localhost:8001/docs
- **테스트 페이지**: http://localhost:8001/test
- **헬스 체크**: http://localhost:8001/health

## 🔐 환경변수 설정

`.env` 파일에 다음 값들을 설정하세요:

```env
# Langfuse 추적 (선택사항)
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://langfuse.nxtai.cloud

# AI 모델 API 키들
UPSTAGE_API_KEY=your_upstage_api_key          # 문서 파싱
ANTHROPIC_API_KEY=your_anthropic_api_key      # Claude Sonnet
LANGCHAIN_API_KEY=your_langchain_api_key      # LangChain 추적
TAVILY_API_KEY=your_tavily_api_key            # 웹 검색

# PostgreSQL 벡터 데이터베이스
DATABASE_URL=postgresql://user:pass@host:port/db
COLLECTION_NAME=documents
```

## 📡 API 엔드포인트

### 스트리밍 채팅
```http
POST /chat/stream
Content-Type: application/json

### pdf파일은 업로드하지 않았습니다.
{
  "message": "~대학교에 대해 알려줘", 
  "session_id": "optional-session-id"
}
```

**응답 형식 (SSE)**:
```javascript
data: {"type": "connected", "content": "Connected"}
data: {"type": "start", "content": "질문을 처리하고 있습니다..."}
data: {"type": "chunk", "content": "~대학교는", "metadata": {...}}
data: {"type": "chunk", "content": " 1973년에", "metadata": {...}}
data: {"type": "final", "content": "전체 응답", "metadata": {...}}
```

### 일반 채팅
```http
POST /chat
Content-Type: application/json

{
  "message": "오늘 날씨는 어때?",
  "session_id": "optional-session-id"  
}
```

### 대화 히스토리
```http
GET /chat/history          # 히스토리 조회
DELETE /chat/history       # 히스토리 삭제
```

## 🛠️ 기술 스택

### **Backend**
- **FastAPI** - 고성능 웹 프레임워크
- **LangGraph** - 워크플로우 오케스트레이션
- **LangChain** - LLM 통합 프레임워크

### **AI/ML**
- **Claude Sonnet** (AWS Bedrock) - 메인 LLM
- **Upstage Document AI** - 문서 파싱
- **HuggingFace Embeddings** - 다국어 벡터 임베딩
- **Tavily** - 실시간 웹 검색

### **Database**
- **PostgreSQL** - 메인 데이터베이스  
- **pgvector** - 벡터 검색 확장
- **AWS RDS** - 관리형 데이터베이스

### **Monitoring**
- **Langfuse** - LLM 관찰성 및 추적

## 🔄 데이터 플로우

### 1. 문서 기반 질문
```
사용자 질문 → 쿼리 분류 → 쿼리 변환 → 벡터 검색 → 컨텍스트 생성 → Claude 응답 → 스트리밍
```

### 2. 일반 질문  
```
사용자 질문 → 쿼리 분류 → 웹 검색 (필요시) → Claude 응답 → 스트리밍
```

### 3. 스트리밍 처리
```
LangGraph.astream() → 토큰 단위 처리 → SSE 형식 변환 → 클라이언트 전송
```

## 🔧 개발 가이드

### 로컬 개발 환경
```bash
# 개발 모드로 서버 실행 (자동 재시작)
uvicorn main:app --reload --port 8001

# 데이터 재인제스션 (문서 업데이트시)
python data_ingestion.py
```

### 새 문서 추가
1. `data_ingestion.py`의 `file_path` 수정
2. `python data_ingestion.py` 실행
3. 서버 재시작 불필요 (벡터DB에서 자동 로드)

## 📋 요구사항

### 의존성
```
fastapi
uvicorn
langchain
langgraph
langchain-anthropic
langchain-aws
langchain-upstage
langchain-huggingface
langchain-postgres
langchain-tavily
sentence-transformers
psycopg2-binary
python-dotenv
langfuse
pydantic
```

### 인프라
- **Python 3.11+**
- **PostgreSQL 13+** (pgvector 확장 설치 필요)
- **AWS 계정** (Bedrock 액세스)

## 🚨 중요 사항

### 보안
- **모든 API 키는 환경변수로 관리**
- **`.env` 파일은 Git에 커밋하지 마세요**
- **프로덕션에서는 HTTPS 사용 권장**

### 성능
- **PostgreSQL 연결 풀링 고려**
- **벡터 인덱스 최적화 권장**
- **대용량 문서는 청킹 크기 조정**

### 확장성
- **여러 문서 컬렉션 지원 가능**
- **다양한 임베딩 모델로 교체 가능**
- **다른 LLM으로 쉽게 변경 가능**

## 📝 라이센스

MIT License

## 🤝 기여

@NxtCloud_김현민

