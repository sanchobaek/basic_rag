from typing import Annotated, Optional, Literal, List, Dict, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import enum
import os
from dotenv import load_dotenv

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import uvicorn

# LangChain 및 기타 필요한 라이브러리 임포트
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.base import BaseMessage

# LangGraph 관련 임포트
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 기존 RAG 코드에서 가져온 컴포넌트
from langchain_upstage import UpstageDocumentParseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

# .env 파일 로드
load_dotenv()

# API 키 설정



# 요청/응답 모델 정의
class QuestionRequest(BaseModel):
    question: str = Field(..., description="사용자 질문")


class AnswerResponse(BaseModel):
    answer: str = Field(..., description="AI 답변")
    category: Optional[str] = Field(None, description="질문 카테고리")


# Define categories for query classification
class Category(enum.Enum):
    DOCUMENT = "document"  # 문서 관련 질문
    GENERAL = "general"  # 일반적인 질문
    GREETING = "greeting"  # 인사말


# Pydantic model for structured output
class QueryClassification(BaseModel):
    """사용자 쿼리 분류 모델"""

    category: Category = Field(
        description="쿼리를 카테고리화 하세요. DOCUMENT(문서 관련 질문) / GENERAL(일반적인 질문) / GREETING(인사) 중에 하나로 구분하세요."
    )
    reasoning: str = Field(description="왜 이 카테고리를 선택했는지 설명하세요.")


# Define the state structure
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    context: List[str]
    category: Optional[str]


# 전역 변수
rag_components = None
graph = None


# Format documents function
def format_docs(docs):
    """Format documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


# Initialize document loading and processing
def initialize_rag_components(file_path: str = "./test_modified.pdf"):
    """Initialize all components for RAG"""
    print("RAG 컴포넌트 초기화 중...")

    try:
        # Document loading
        loader = UpstageDocumentParseLoader(
            file_path,
            split="page",
            output_format="markdown",
            ocr="auto",
            coordinates=True,
        )
        docs = loader.load()
        print(f"문서 로딩 완료: {len(docs)} 페이지")

        # Document chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        docs_splitter = splitter.split_documents(docs)
        print(f"청킹 완료: {len(docs_splitter)} 청크")

        # Embedding model
        device = "cpu"  # 기본값으로 CPU 사용
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
        except ImportError:
            pass

        print(f"사용 중인 디바이스: {device}")

        hf_embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large-instruct",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Vector store
        vectorstore = FAISS.from_documents(
            documents=docs_splitter,
            embedding=hf_embeddings,
        )
        print("벡터 스토어 생성 완료")

        # Retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

        # LLM
        llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
        print("초기화 완료!")

        return {
            "retriever": retriever,
            "llm": llm,
        }

    except Exception as e:
        print(f"RAG 컴포넌트 초기화 실패: {str(e)}")
        print("기본 LLM만으로 실행합니다...")
        # 기본 LLM만으로도 동작하도록 수정
        return {
            "retriever": None,
            "llm": ChatAnthropic(model="claude-3-haiku-20240307", temperature=0),
        }


# Router function for categorizing queries
def router(state: State) -> Dict[str, Any]:
    """사용자 쿼리를 카테고리로 분류하는 라우터"""
    # Get the most recent user message
    user_message = state["messages"][-1].content

    # Create the router input
    router_input = f"""
    다음 사용자 쿼리를 분석하고 카테고리를 결정하세요.
    카테고리:
    - document: 문서 내용에 관한 질문 (예: "아주대학교에 대해 알려줘", "이 문서에서 중요한 내용은?")
    - general: 일반적인 질문으로, 문서와 관련이 없음 (예: "오늘 날씨 어때?", "파이썬이란?")
    - greeting: 인사말 (예: "안녕", "반가워", "뭐해?")
    
    쿼리: {user_message}
    """

    # Get LLM
    llm = rag_components["llm"]

    # Structured output with the classification model
    structured_llm = llm.with_structured_output(QueryClassification)

    try:
        # Get classification
        classification = structured_llm.invoke(router_input)
        category = classification.category.value
    except Exception as e:
        print(f"라우터 분류 오류: {e}")
        # 기본값으로 general 설정
        category = "general"

    return {"category": category}


# Conditional routing function
def route_by_category(state: State) -> Literal["document_qa", "general_qa", "greeting"]:
    """카테고리에 기반하여 다음 노드를 결정"""
    category = state.get("category", "").lower()

    if category == "document":
        return "document_qa"
    elif category == "general":
        return "general_qa"
    elif category == "greeting":
        return "greeting"
    else:
        return "general_qa"


# Define LangGraph nodes
def retrieve_documents(state: State) -> Dict[str, Any]:
    """문서에서 관련 내용 검색"""
    # Get the most recent user message
    user_message = state["messages"][-1]

    try:
        # Retrieve documents
        retriever = rag_components["retriever"]
        if retriever is None:
            return {"context": ["문서를 찾을 수 없습니다."]}

        docs = retriever.invoke(user_message.content)

        # Format documents
        formatted_docs = format_docs(docs)

        # Return updated state
        return {"context": [formatted_docs]}

    except Exception as e:
        print(f"문서 검색 오류: {e}")
        return {"context": ["문서 검색 중 오류가 발생했습니다."]}


def document_qa(state: State) -> Dict[str, Any]:
    """문서 기반 질의응답"""
    # Format conversation history
    history = state["messages"][:-1]  # All messages except the current question

    # Create formatted history for prompt context
    formatted_history = ""
    if history:
        for msg in history:
            role = "사용자" if isinstance(msg, HumanMessage) else "AI"
            formatted_history += f"{role}: {msg.content}\n"
    else:
        formatted_history = "이전 대화 없음"

    # Get current question and context
    question = state["messages"][-1].content
    context = state["context"][0] if state["context"] else "문서 정보 없음"

    # Create prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """너는 친절한 한국어 AI 비서야. 
            제공된 문서 내용(context)과 이전 대화 내용을 참고해서 질문에 답해.
            반드시 한국어로만 대답하고, 문서에 없는 내용은 대답하지 말고 모른다고 해.
            
            참고 문서:
            {context}
            
            이전 대화:
            {chat_history}
            """,
            ),
            ("user", "{question}"),
        ]
    )

    # Invoke the LLM
    llm = rag_components["llm"]

    try:
        # Create formatted message for LLM
        formatted_message = prompt.format_messages(
            context=context, chat_history=formatted_history, question=question
        )

        # Get response from LLM
        ai_response = llm.invoke(formatted_message)
        response_content = ai_response.content

    except Exception as e:
        print(f"문서 QA 오류: {e}")
        response_content = "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다."

    # Return the assistant message
    return {"messages": [AIMessage(content=response_content)]}


def general_qa(state: State) -> Dict[str, Any]:
    """일반 질의응답"""
    # Get current question
    question = state["messages"][-1].content

    # Create prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """너는 친절한 한국어 AI 비서야. 
            사용자의 질문에 대해 간결하고 정확하게 답변해. 한국어로 대답해.
            """,
            ),
            ("user", "{question}"),
        ]
    )

    # Invoke the LLM
    llm = rag_components["llm"]

    try:
        # Create formatted message for LLM
        formatted_message = prompt.format_messages(question=question)

        # Get response from LLM
        ai_response = llm.invoke(formatted_message)
        response_content = ai_response.content

    except Exception as e:
        print(f"일반 QA 오류: {e}")
        response_content = "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다."

    # Return the assistant message
    return {"messages": [AIMessage(content=response_content)]}


def greeting(state: State) -> Dict[str, Any]:
    """인사말에 응답"""
    # Get user message
    user_message = state["messages"][-1].content

    # Create prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """너는 친절한 한국어 AI 비서야.
            사용자의 인사에 친근하고 따뜻하게 응답해. 간결하게 한국어로 대답해.
            """,
            ),
            ("user", "{greeting}"),
        ]
    )

    # Invoke the LLM
    llm = rag_components["llm"]

    try:
        # Create formatted message for LLM
        formatted_message = prompt.format_messages(greeting=user_message)

        # Get response from LLM
        ai_response = llm.invoke(formatted_message)
        response_content = ai_response.content

    except Exception as e:
        print(f"인사말 처리 오류: {e}")
        response_content = "안녕하세요! 무엇을 도와드릴까요?"

    # Return the assistant message
    return {"messages": [AIMessage(content=response_content)]}


# Run the graph with a given user input
def run_graph(user_input: str):
    """Run the graph with a user input and return the response"""
    try:
        # Create initial state with the user message
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "context": [],
            "category": None,
        }

        # Run the graph and get the final state
        result = graph.invoke(initial_state)

        # Extract the AI response and category
        response_text = "응답을 생성할 수 없습니다."
        category = result.get("category", "unknown")

        if "messages" in result and len(result["messages"]) > 1:
            # Get the assistant message (should be the last one)
            ai_msg = result["messages"][-1]
            if isinstance(ai_msg, AIMessage):
                response_text = ai_msg.content

        return response_text, category

    except Exception as e:
        print(f"그래프 실행 오류: {e}")
        return f"오류가 발생했습니다: {str(e)}", "error"


# 스트리밍 응답을 위한 함수
async def stream_graph_response(user_input: str):
    """스트리밍 응답 생성"""
    try:
        # 일반적인 응답 생성
        response_text, category = run_graph(user_input)

        # 응답을 청크 단위로 나누어 스트리밍
        words = response_text.split()

        for i, word in enumerate(words):
            chunk_data = {
                "content": word + " ",
                "category": category,
                "is_final": i == len(words) - 1,
            }
            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.05)  # 약간의 지연으로 스트리밍 효과

    except Exception as e:
        error_data = {"error": str(e), "is_final": True}
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"


# FastAPI 앱 생성 및 설정
app = FastAPI(title="RAG LangGraph API", version="1.0.0")

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# FastAPI 엔드포인트들
@app.get("/")
def read_root():
    return {"message": "RAG LangGraph API Server", "status": "running"}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """일반 질문 응답 (비스트리밍)"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="질문이 비어있습니다.")

        # 그래프 실행
        answer, category = run_graph(request.question)

        return AnswerResponse(answer=answer, category=category)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")


@app.post("/ask/stream")
async def ask_question_stream(request: QuestionRequest):
    """스트리밍 질문 응답"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="질문이 비어있습니다.")

        return StreamingResponse(
            stream_graph_response(request.question),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")


@app.get("/health")
def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "components_loaded": rag_components is not None,
        "graph_loaded": graph is not None,
    }


# 서버 시작 시 초기화
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 처리"""
    global rag_components, graph

    try:
        print("서버 시작 중... RAG 컴포넌트 초기화")

        # RAG 컴포넌트 초기화
        rag_components = initialize_rag_components()

        # Build the LangGraph
        print("LangGraph 구성 중...")
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("router", router)
        graph_builder.add_node("retrieve", retrieve_documents)
        graph_builder.add_node("document_qa", document_qa)
        graph_builder.add_node("general_qa", general_qa)
        graph_builder.add_node("greeting", greeting)

        # Add edges
        graph_builder.add_edge(START, "router")

        # Add conditional edges based on the category
        graph_builder.add_conditional_edges(
            "router",
            route_by_category,
            {
                "document_qa": "retrieve",
                "general_qa": "general_qa",
                "greeting": "greeting",
            },
        )

        # Connect retrieve to document_qa
        graph_builder.add_edge("retrieve", "document_qa")

        # Connect all output nodes to END
        graph_builder.add_edge("document_qa", END)
        graph_builder.add_edge("general_qa", END)
        graph_builder.add_edge("greeting", END)

        # Compile the graph
        graph = graph_builder.compile()
        print("LangGraph 구성 완료!")
        print("🚀 서버가 준비되었습니다!")

        yield

    except Exception as e:
        print(f"초기화 중 오류 발생: {str(e)}")
        import traceback

        traceback.print_exc()
        yield


# FastAPI 앱에 lifespan 적용
app.router.lifespan_context = lifespan

# 서버 실행을 위한 코드
if __name__ == "__main__":
    # 사용 가능한 포트 찾기
    import socket

    def find_free_port(start_port=8000):
        for port in range(start_port, start_port + 10):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("localhost", port))
                    return port
            except OSError:
                continue
        return None

    port = find_free_port()
    if port:
        print(f"🚀 서버를 포트 {port}에서 시작합니다...")
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        print("❌ 사용 가능한 포트를 찾을 수 없습니다.")
        uvicorn.run(app, host="0.0.0.0", port=8000)  # 기본 포트로 시도
