from typing import Annotated, Optional, Literal, List, Dict, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import enum
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# LangChain 핵심 라이브러리
from langchain_aws import ChatBedrockConverse
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
# from langchain.text_splitter import RecursiveCharacterTextSplitter  # 데이터 인제스천에서만 사용
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
# from langchain_upstage import UpstageDocumentParseLoader  # 데이터 인제스천에서만 사용

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# 웹 검색 도구
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

# 랭퓨즈
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

# Langfuse 설정 (환경변수에서 읽기)
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://langfuse.nxtai.cloud"),  # host만 기본값 유지
)

langfuse = get_client()
langfuse_handler = CallbackHandler()

# API 키 설정 (환경변수에서 읽기)
os.environ["UPSTAGE_API_KEY"] = os.getenv("UPSTAGE_API_KEY", "")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")

# PostgreSQL 데이터베이스 설정
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")


# Format documents function
def format_docs(docs):
    """Format documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


def initialize_rag_components():
    """Initialize RAG components (connects to existing vector store)"""
    
    # 임베딩 모델 (검색 쿼리 벡터화용)
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large-instruct",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # PostgreSQL Vector store 연결 (기존 데이터 사용)
    vectorstore = PGVector(
        embeddings=hf_embeddings,
        connection=DATABASE_URL,
        collection_name=COLLECTION_NAME,
    )

    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    # Sonnet 4 LLM - 모든 작업용
    # LLM 초기화
    sonnet_llm = ChatBedrockConverse(
        model="us.anthropic.claude-sonnet-4-20250514-v1:0",
        temperature=0,
    )

    # 웹 검색 툴과 바인딩
    tools = [tavily_web_search]
    sonnet_llm_with_tools = sonnet_llm.bind_tools(tools)

    # LLM 초기화 완료

    return {
        "retriever": retriever,
        "llm": sonnet_llm,  # 모든 작업용 Sonnet 4
        "llm_with_tools": sonnet_llm_with_tools,  # 웹 검색용 Sonnet 4
        "tools": tools,
    }


# Define categories for query classification
class Category(enum.Enum):
    DOCUMENT = "document"  # 문서 관련 질문
    GENERAL = "general"  # 일반적인 질문


# Pydantic model for structured output
class QueryClassification(BaseModel):
    """사용자 쿼리 분류 모델"""

    category: Category = Field(
        description="쿼리를 카테고리화 하세요. DOCUMENT(문서 관련 질문) / GENERAL(일반적인 질문) 중에 하나로 구분하세요."
    )
    reasoning: str = Field(description="왜 이 카테고리를 선택했는지 설명하세요.")


# Define the state structure
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    context: List[str]
    category: Optional[str]
    transformed_query: Optional[str]  # 변환된 쿼리 저장


# TavilySearch 인스턴스를 올바르게 생성
tavily_search_tool = TavilySearch(max_results=3, topic="general")


@tool
def tavily_web_search(query: str) -> str:
    """웹에서 최신 정보를 검색합니다. 뉴스, 날씨, 실시간 데이터 등에 사용하세요."""
    # 웹 검색 실행

    try:
        # 이제 tavily_search_tool이 올바른 TavilySearch 인스턴스임
        search_results = tavily_search_tool.invoke({"query": query})
        # 검색 결과 처리

        # 결과 포맷팅
        if isinstance(search_results, list):
            if len(search_results) == 0:
                return "검색 결과가 없습니다."

            formatted_results = []
            for i, result in enumerate(search_results[:3]):  # 상위 3개만
                title = result.get("title", "N/A")
                content = result.get("content", "N/A")
                url = result.get("url", "N/A")

                formatted_results.append(
                    f"""
검색 결과 {i+1}:
제목: {title}
내용: {content[:300] if content != 'N/A' else 'N/A'}...
출처: {url}
"""
                )
            return "\n".join(formatted_results)
        else:
            return f"검색 결과: {str(search_results)[:500]}"

    except Exception as e:
        error_msg = f"웹 검색 중 오류 발생: {str(e)}"
        # 검색 오류 발생
        return error_msg


# router 함수 수정 - Sonnet 4 사용
def router(state: State, rag_components: Dict) -> Dict[str, Any]:
    """사용자 쿼리를 카테고리로 분류하는 라우터 - Sonnet 4 사용"""
    # 쿼리 분류 시작

    # Get the most recent user message
    user_message = state["messages"][-1].content

    # Create the router input
    router_input = f"""
    다음 사용자 쿼리를 분석하고 카테고리를 결정하세요.
    카테고리:
    - document: 아주대에 관한 질문 (예: "아주대학교에 대해 알려줘")
    - general: 일반적인 질문으로, 문서와 관련이 없음 (예: "오늘 날씨 어때?", "파이썬이란?")
    
    쿼리: {user_message}
    """

    # Sonnet 4 LLM 사용
    llm = rag_components["llm"]

    # Structured output with the classification model
    structured_llm = llm.with_structured_output(QueryClassification)

    # Get classification
    classification = structured_llm.invoke(router_input)

    category = classification.category.value
    # 분류 완료

    return {"category": category}


# Conditional routing function
def route_by_category(state: State) -> Literal["document_qa", "general_qa"]:
    """카테고리에 기반하여 다음 노드를 결정"""
    category = state.get("category", "").lower()

    if category == "document":
        return "document_qa"
    elif category == "general":
        return "general_qa"
    else:
        # 기본값은 일반 질의응답
        return "general_qa"


def query_transform(state: State, rag_components: Dict) -> Dict[str, Any]:
    """사용자 쿼리를 문서 검색에 최적화된 형태로 변환"""
    # 쿼리 변환 시작

    # 사용자 메시지 추출
    user_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    if not user_message:
        return {"transformed_query": ""}

    # 이전 대화 히스토리 구성
    history_messages = state["messages"][:-1]
    formatted_history = ""
    for msg in history_messages:
        role = "사용자" if isinstance(msg, HumanMessage) else "AI"
        formatted_history += f"{role}: {msg.content}\n"

    # Query Transform 프롬프트
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """너는 문서 검색 쿼리 최적화 전문가야.
        사용자의 질문을 분석해서 문서에서 관련 정보를 더 잘 찾을 수 있도록 쿼리를 변환해.
        
        ## 쿼리 변환 규칙:
        1. **핵심 키워드 추출**: 불필요한 단어 제거하고 핵심 단어만 남기기
        2. **동의어 확장**: 관련 용어들 추가 (예: "대학교" → "대학교, 대학, 학교")
        3. **검색 최적화**: 문서에서 찾기 쉬운 형태로 변환
        4. **컨텍스트 반영**: 이전 대화 맥락을 고려한 쿼리 확장
        5. **구체화**: 모호한 표현을 구체적으로 변환
        
        ## 변환 예시:
        - "아주대에 대해 알려줘" → "아주대학교 대학 정보 개요 소개"
        - "입학 조건이 뭐야?" → "입학 조건 지원 자격 요건 모집"
        - "학과는 어떤게 있어?" → "학과 전공 단과대학 계열 전공분야"
        
        ## 응답 형식:
        쿼리가 어떻게 변경되었는지 알려줘
        
        이전 대화:
        {chat_history}
        """,
            ),
            ("user", "원본 쿼리: {original_query}"),
        ]
    )

    # Sonnet 4 사용
    llm = rag_components["llm"]

    formatted_message = prompt.format_messages(
        chat_history=formatted_history, original_query=user_message
    )

    ai_response = llm.invoke(formatted_message)
    
    # content가 리스트인 경우 처리
    if isinstance(ai_response.content, list):
        # 텍스트 부분만 추출
        text_content = ""
        for item in ai_response.content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_content += item.get("text", "")
            elif isinstance(item, str):
                text_content += item
        transformed_query = text_content.strip()
    else:
        transformed_query = ai_response.content.strip()

    # 쿼리 변환 완료

    return {"transformed_query": transformed_query}


# Define LangGraph nodes
def retrieve_documents(state: State, rag_components: Dict) -> Dict[str, Any]:
    """문서에서 관련 내용 검색"""
    # 문서 검색 시작

    # Get the most recent user message
    user_message = state["messages"][-1]

    # Retrieve documents
    retriever = rag_components["retriever"]
    docs = retriever.invoke(user_message.content)

    # Format documents
    formatted_docs = format_docs(docs)
    # 문서 검색 완료

    # Return updated state
    return {"context": [formatted_docs]}


def document_qa(state: State, rag_components: Dict) -> Dict[str, Any]:
    """문서 기반 질의응답 - Sonnet 4 사용"""
    # 문서 기반 응답 생성 시작
    context = state["context"][0] if state["context"] else "문서 정보 없음"

    # Get user message
    user_message = state["messages"][-1].content

    # 이전 대화들은 별도로 히스토리 구성
    history_messages = state["messages"][:-1]
    formatted_history = ""
    for msg in history_messages:
        role = "사용자" if isinstance(msg, HumanMessage) else "AI"
        formatted_history += f"{role}: {msg.content}\n"

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
            ("user", "{user_input}"),
        ]
    )

    # Sonnet 4 LLM 사용
    llm = rag_components["llm"]

    # Create formatted message for LLM
    formatted_message = prompt.format_messages(
        context=context,
        chat_history=formatted_history,
        user_input=user_message,
    )

    # Get response from LLM
    ai_response = llm.invoke(formatted_message)
    response_content = ai_response.content
    # 문서 기반 응답 완료

    # Return the assistant message
    return {"messages": [AIMessage(content=response_content)]}


def general_qa(state: State, rag_components: Dict) -> Dict[str, Any]:
    """일반 질의응답 - Sonnet 4 with Tools 사용"""
    # 일반 응답 생성 시작

    user_message = state["messages"][-1].content

    # 이전 대화 히스토리 구성
    history_messages = state["messages"][:-1]
    formatted_history = ""
    for msg in history_messages:
        role = "사용자" if isinstance(msg, HumanMessage) else "AI"
        formatted_history += f"{role}: {msg.content}\n"

    # Sonnet 4 + 툴 사용
    llm_with_tools = rag_components["llm_with_tools"]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """너는 친절한 한국어 AI 비서야. 
        사용자의 일반적인 질문에 답변해. 
        만약 최신 정보나 실시간 데이터가 필요하다면 (날씨, 뉴스, 주식 등), tavily_web_search 툴을 사용해서 웹에서 정보를 찾아줘. 
        
## ReAct 단계별 처리 방식:

**1단계 - 사고 (Thought):** 사용자의 질문을 분석하고 어떤 정보가 필요한지 생각해
- 이 질문에 답하기 위해 무엇이 필요한가?
- 내가 이미 알고 있는 정보인가, 아니면 최신 정보가 필요한가?
- 웹 검색이 필요한가?

**2단계 - 행동 (Action):** 필요하다면 적절한 도구를 사용해
- 최신 정보나 실시간 데이터가 필요하면: tavily_web_search 툴 사용
- 일반 상식으로 답변 가능하면: 바로 답변

**3단계 - 관찰 (Observation):** 도구 사용 결과를 분석해
- 검색 결과가 질문에 적합한가?
- 추가 정보가 필요한가?

**4단계 - 답변 (Answer):** 최종 답변을 제공해

## 답변 형식:
**사고:** [질문 분석 및 필요한 정보 판단]
**행동:** [취할 행동 - 웹 검색 또는 직접 답변]
**답변:** [사용자에게 제공할 최종 답변]

## 웹 검색이 필요한 경우:
- 날씨, 뉴스, 주식 가격
- 현재 날짜, 시간
- 최신 사건, 트렌드
- 실시간 데이터

## 직접 답변 가능한 경우:
- 일반 상식, 역사적 사실
- 수학 계산, 언어 번역
- 개념 설명, 정의
- 기본적인 인사말

## 일반적인 상식으로 답변 가능한 질문은 바로 답변해.
## 너의 생각은 말하지말고, 사용자 질문에 대한 답변만 대답해
        
        이전 대화:
        {chat_history}
        """,
            ),
            ("user", "{user_input}"),
        ]
    )

    formatted_message = prompt.format_messages(
        chat_history=formatted_history,
        user_input=user_message,
    )

    ai_response = llm_with_tools.invoke(formatted_message)
    # 일반 응답 완료

    return {"messages": [ai_response]}


# 전역 변수들
conversation_history = []  # 전체 대화 히스토리 저장
rag_components = None
graph = None


def main():
    """메인 실행 함수"""
    global rag_components, graph, conversation_history

    try:
        # 1. RAG 컴포넌트 초기화 (Sonnet 4만)
        # RAG 컴포넌트 초기화
        rag_components = initialize_rag_components()

        # 2. 그래프 빌드
        # LangGraph 구성

        graph_builder = StateGraph(State)

        # 노드들 추가 - 함수와 rag_components를 함께 전달하도록 래핑
        def router_node(state):
            return router(state, rag_components)

        def query_transform_node(state):
            return query_transform(state, rag_components)

        def retrieve_node(state):
            return retrieve_documents(state, rag_components)

        def document_qa_node(state):
            return document_qa(state, rag_components)

        def general_qa_node(state):
            return general_qa(state, rag_components)

        graph_builder.add_node("router", router_node)
        graph_builder.add_node("query_transform", query_transform_node)
        graph_builder.add_node("retrieve", retrieve_node)
        graph_builder.add_node("document_qa", document_qa_node)
        graph_builder.add_node("general_qa", general_qa_node)
        graph_builder.add_node("tools", ToolNode(rag_components["tools"]))

        # 기본 엣지들
        graph_builder.add_edge(START, "router")

        # 라우터에서 분기
        graph_builder.add_conditional_edges(
            "router",
            route_by_category,
            {
                "document_qa": "query_transform",  # document → query_transform → retrieve → document_qa
                "general_qa": "general_qa",  # general → general_qa
            },
        )

        # Document 경로: query_transform → retrieve → document_qa → END (Sonnet 4)
        graph_builder.add_edge("query_transform", "retrieve")
        graph_builder.add_edge("retrieve", "document_qa")
        graph_builder.add_edge("document_qa", END)

        # General 경로: general_qa → (필요시 웹 툴) → END (Sonnet 4)
        graph_builder.add_conditional_edges(
            "general_qa",
            tools_condition,
            {"tools": "tools", "__end__": END},  # 웹 툴 호출 시  # 툴 호출 없으면 종료
        )

        # 웹 툴 실행 후 다시 general_qa로 돌아가서 최종 응답
        graph_builder.add_edge("tools", "general_qa")

        # 4. 그래프 컴파일
        graph = graph_builder.compile().with_config(
            config={"callbacks": [langfuse_handler]}
        )

        # 그래프 구성 완료

        # 5. 대화 히스토리 초기화
        conversation_history = []

        return True

    except Exception as e:
        # 초기화 오류 발생
        import traceback

        traceback.print_exc()
        return False


# FastAPI에서 사용할 함수들
async def initialize_rag():
    """FastAPI용 비동기 초기화 함수"""
    return main()


def run_rag_query(user_input: str, session_id: str = None):
    """단일 쿼리 실행"""
    global conversation_history, graph

    if not graph:
        raise RuntimeError("RAG 시스템이 초기화되지 않았습니다")

    try:
        # 사용자 메시지 추가
        user_message = HumanMessage(content=user_input)
        conversation_history.append(user_message)

        # 그래프 실행
        result = graph.invoke(
            {
                "messages": conversation_history,
                "context": [],
                "category": None,
            }
        )

        # AI 응답 추출 및 히스토리에 추가
        if "messages" in result and len(result["messages"]) > len(conversation_history):
            ai_msg = result["messages"][-1]
            if isinstance(ai_msg, AIMessage):
                conversation_history.append(ai_msg)
                return ai_msg.content
            else:
                return "응답을 생성할 수 없습니다."
        else:
            return "응답을 생성할 수 없습니다."

    except Exception as e:
        # 쿼리 실행 오류
        return f"처리 중 오류가 발생했습니다: {str(e)}"


async def stream_rag_query(user_input: str, session_id: str = None):
    """스트리밍 쿼리 실행 - SSE 최적화"""
    global conversation_history, graph

    if not graph:
        raise RuntimeError("RAG 시스템이 초기화되지 않았습니다")

    try:
        # 시작 알림
        yield {
            "type": "start",
            "content": "질문을 처리하고 있습니다...",
            "metadata": {"timestamp": "now", "session_id": session_id},
        }

        # 사용자 메시지 추가
        user_message = HumanMessage(content=user_input)
        conversation_history.append(user_message)

        # 스트리밍 실행
        full_response = ""
        chunk_count = 0

        async for msg, meta in graph.astream(
            {
                "messages": conversation_history,
                "context": [],
                "category": None,
            },
            stream_mode="messages",
        ):
            if isinstance(msg, AIMessage) and msg.content:
                # content가 문자열인 경우
                if isinstance(msg.content, str):
                    if msg.content.strip():  # 빈 문자열이 아닌 경우만 처리
                        full_response += msg.content
                        chunk_count += 1

                        yield {
                            "type": "chunk",
                            "content": msg.content,
                            "metadata": {
                                "chunk_id": chunk_count,
                                "total_length": len(full_response),
                                "word_count": len(full_response.split()),
                            },
                        }

                # content가 리스트인 경우 (tool calls 등)
                elif isinstance(msg.content, list):
                    for item in msg.content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text", "")
                            if text.strip():
                                full_response += text
                                chunk_count += 1

                                yield {
                                    "type": "chunk",
                                    "content": text,
                                    "metadata": {
                                        "chunk_id": chunk_count,
                                        "total_length": len(full_response),
                                        "word_count": len(full_response.split()),
                                    },
                                }

        # 응답이 생성되지 않은 경우 처리
        if not full_response:
            full_response = "죄송합니다. 응답을 생성할 수 없습니다."
            yield {
                "type": "chunk",
                "content": full_response,
                "metadata": {"chunk_id": 1, "total_length": len(full_response)},
            }

        # 최종 응답을 히스토리에 추가
        conversation_history.append(AIMessage(content=full_response))

        # 완료 알림
        yield {
            "type": "final",
            "content": full_response,
            "metadata": {
                "conversation_length": len(conversation_history),
                "total_chunks": chunk_count,
                "total_characters": len(full_response),
                "total_words": len(full_response.split()),
                "session_id": session_id,
            },
        }

    except Exception as e:
        # 스트리밍 오류 처리
        import traceback

        traceback.print_exc()

        yield {
            "type": "error",
            "content": f"처리 중 오류가 발생했습니다: {str(e)}",
            "metadata": {"error_type": type(e).__name__},
        }


def get_conversation_history():
    """대화 히스토리 반환"""
    global conversation_history
    return conversation_history


def clear_conversation_history():
    """대화 히스토리 초기화"""
    global conversation_history
    conversation_history = []


if __name__ == "__main__":
    main()
