from typing import List, Dict, Any, Optional, AsyncGenerator, Literal
from typing_extensions import TypedDict
import asyncio
import logging
import enum
import uuid
import os
from datetime import datetime

# Jupyter 코드에서 사용한 동일한 라이브러리들
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# RAG 컴포넌트들
from langchain_upstage import UpstageDocumentParseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field

# 랭퓨즈
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

# 스트리밍
import asyncio
from langchain_core.output_parsers import StrOutputParser

# 로깅 설정
logger = logging.getLogger(__name__)

# .env 파일 로드 (필요한 경우)
load_dotenv()




# Jupyter 코드에서 가져온 모델들
class Category(enum.Enum):
    """쿼리 분류 카테고리"""

    DOCUMENT = "document"
    GENERAL = "general"
    GREETING = "greeting"


class QueryClassification(BaseModel):
    """사용자 쿼리 분류 모델 - Jupyter 코드와 동일"""

    category: Category = Field(
        description="쿼리를 카테고리화 하세요. DOCUMENT(문서 관련 질문) / GENERAL(일반적인 질문) / GREETING(인사) 중에 하나로 구분하세요."
    )
    reasoning: str = Field(description="왜 이 카테고리를 선택했는지 설명하세요.")


class State(TypedDict):
    """LangGraph 상태 - Jupyter 코드와 동일"""

    messages: List[BaseMessage]
    context: List[str]
    category: Optional[str]


class RAGService:
    """
    Jupyter 노트북의 RAG 로직을 서비스 클래스로 캡슐화한 것입니다.

    이 클래스는 원본 코드의 모든 기능을 유지하면서,
    여러 세션을 동시에 처리할 수 있도록 확장했습니다.
    """

    def __init__(self):
        """RAG 서비스 초기화"""
        self.retriever = None
        self.llm = None
        self.graph = None
        self.conversation_histories: Dict[str, List[BaseMessage]] = {}
        self.is_initialized = False

    async def initialize(self, file_path: str = "./test_modified.pdf"):
        """
        RAG 컴포넌트들을 초기화합니다.

        이 함수는 Jupyter 코드의 initialize_rag_components()와 동일한 로직입니다.
        한 번만 실행되고, 이후 모든 요청에서 재사용됩니다.
        """

        try:
            logger.info("RAG 컴포넌트 초기화 시작...")

            # 문서 로딩 - Jupyter 코드와 동일
            logger.info("문서 로딩 중...")
            loader = UpstageDocumentParseLoader(
                file_path,
                split="page",
                output_format="markdown",
                ocr="auto",
                coordinates=True,
            )
            docs = loader.load()
            logger.info(f"문서 로딩 완료: {len(docs)} 페이지")

            # 문서 청킹 - Jupyter 코드와 동일
            logger.info("문서 청킹 중...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500, chunk_overlap=300
            )
            docs_splitter = splitter.split_documents(docs)
            logger.info(f"청킹 완료: {len(docs_splitter)} 청크")

            # 임베딩 모델 - Jupyter 코드와 동일한 로직
            logger.info("임베딩 모델 로딩 중...")
            device = self._get_device()
            logger.info(f"사용 중인 디바이스: {device}")

            hf_embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-large-instruct",
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )

            # 벡터 스토어 - Jupyter 코드와 동일
            logger.info("벡터 스토어 생성 중...")
            vectorstore = FAISS.from_documents(
                documents=docs_splitter,
                embedding=hf_embeddings,
            )
            logger.info("벡터 스토어 생성 완료")

            # 리트리버 설정
            self.retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5},
            )

            # LLM 초기화 - Jupyter 코드와 동일하게 스트리밍 활성화
            logger.info("LLM 초기화 중...")
            self.llm = ChatAnthropic(
                model="claude-3-haiku-20240307",
                temperature=0,
                streaming=True,  # 🔥 스트리밍 활성화
            )

            # LangGraph 초기화
            self._build_graph()

            self.is_initialized = True
            logger.info("RAG 서비스 초기화 완료!")

        except Exception as e:
            logger.error(f"초기화 중 오류: {e}")
            raise

    def _get_device(self) -> str:
        """디바이스 선택 - Jupyter 코드와 동일한 로직"""
        device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
        except ImportError:
            pass
        return device

    def _build_graph(self):

        # StateGraph 생성
        graph_builder = StateGraph(State)

        # 노드들 추가
        graph_builder.add_node("router", self._router)
        graph_builder.add_node("retrieve", self._retrieve_documents)
        graph_builder.add_node("document_qa", self._document_qa)
        graph_builder.add_node("general_qa", self._general_qa)
        graph_builder.add_node("greeting", self._greeting)

        # 엣지 연결 
        graph_builder.add_edge(START, "router")
        graph_builder.add_conditional_edges(
            "router",
            self._route_by_category,
            {
                "document_qa": "retrieve",
                "general_qa": "general_qa",
                "greeting": "greeting",
            },
        )
        graph_builder.add_edge("retrieve", "document_qa")
        graph_builder.add_edge("document_qa", END)
        graph_builder.add_edge("general_qa", END)
        graph_builder.add_edge("greeting", END)

        # 그래프 컴파일
        graph = graph_builder.compile().with_config(
            config={"callbacks": [langfuse_handler]}
        )
        print("LangGraph 구성 완료!")


    def _format_docs(self, docs) -> str:
        """문서 포맷팅 - Jupyter 코드와 동일"""
        return "\\n\\n".join(doc.page_content for doc in docs)

    def _router(self, state: State) -> Dict[str, Any]:
        """
        쿼리 분류 라우터 - Jupyter 코드와 동일한 로직

        이 함수는 사용자의 질문을 분석해서 어떤 유형의 질문인지 판단합니다.
        """

        logger.info("쿼리 분류 중...")

        # 가장 최근 사용자 메시지 가져오기
        user_message = state["messages"][-1].content

        # 라우터 입력 생성 - Jupyter 코드와 동일한 프롬프트
        router_input = f"""
        다음 사용자 쿼리를 분석하고 카테고리를 결정하세요.
        카테고리:
        - document: 문서 내용에 관한 질문 (예: "아주대학교에 대해 알려줘", "이 문서에서 중요한 내용은?")
        - general: 일반적인 질문으로, 문서와 관련이 없음 (예: "오늘 날씨 어때?", "파이썬이란?")
        - greeting: 인사말 (예: "안녕", "반가워", "뭐해?")
        
        쿼리: {user_message}
        """

        # 구조화된 출력으로 분류
        structured_llm = self.llm.with_structured_output(QueryClassification)
        classification = structured_llm.invoke(router_input)

        category = classification.category.value
        logger.info(f"분류 결과: {category} (이유: {classification.reasoning})")

        return {"category": category}

    def _route_by_category(
        self, state: State
    ) -> Literal["document_qa", "general_qa", "greeting"]:
        """카테고리 기반 라우팅 - Jupyter 코드와 동일"""
        category = state.get("category", "").lower()

        if category == "document":
            return "document_qa"
        elif category == "general":
            return "general_qa"
        elif category == "greeting":
            return "greeting"
        else:
            return "general_qa"

    def _retrieve_documents(self, state: State) -> Dict[str, Any]:
        """문서 검색 - Jupyter 코드와 동일한 로직"""
        logger.info("문서 검색 중...")

        # 가장 최근 사용자 메시지
        user_message = state["messages"][-1]

        # 문서 검색
        docs = self.retriever.invoke(user_message.content)

        # 문서 포맷팅
        formatted_docs = self._format_docs(docs)
        logger.info(f"검색 완료: {len(docs)} 문서 찾음")

        return {"context": [formatted_docs]}

    async def _stream_llm_response(
        self, llm, formatted_message
    ) -> AsyncGenerator[str, None]:
        """
        LLM 응답을 스트리밍으로 생성하는 함수

        이 함수가 핵심입니다. Jupyter 코드의 stream_llm_response를
        AsyncGenerator로 변환해서 FastAPI에서 사용할 수 있게 했습니다.

        주의: async generator에서는 return value를 사용할 수 없습니다.
        대신 yield를 사용해서 값들을 하나씩 생성합니다.
        """

        # 체인 생성 - Jupyter 코드와 동일
        chain = llm | StrOutputParser()

        # 스트리밍 실행
        async for chunk in chain.astream(formatted_message):
            if chunk:
                # 각 청크를 yield - 이것이 HTTP 스트리밍으로 전송됩니다
                yield chunk

        # async generator에서는 return 대신 그냥 함수가 끝나면 됩니다

    async def _document_qa(self, state: State) -> Dict[str, Any]:
        """
        문서 기반 질의응답 - Jupyter 코드의 스트리밍 로직을 그대로 유지

        이 함수에서 실제 문서 기반 답변을 생성합니다.
        원본 코드의 프롬프트와 로직을 그대로 사용합니다.

        수정된 부분: 스트리밍 응답을 누적해서 최종 AIMessage를 생성합니다.
        """

        logger.info("문서 기반 응답 생성 중...")

        context = state["context"][0] if state["context"] else "문서 정보 없음"
        user_message = state["messages"][-1].content

        # 이전 대화 히스토리 구성 - Jupyter 코드와 동일
        history_messages = state["messages"][:-1]
        formatted_history = ""
        for msg in history_messages:
            role = "사용자" if isinstance(msg, HumanMessage) else "AI"
            formatted_history += f"{role}: {msg.content}\\n"

        # 프롬프트 생성 - Jupyter 코드와 동일한 프롬프트
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

        formatted_message = prompt.format_messages(
            context=context, chat_history=formatted_history, user_input=user_message
        )

        # 스트리밍 응답을 누적하여 전체 응답 생성
        full_response = ""
        async for chunk in self._stream_llm_response(self.llm, formatted_message):
            full_response += chunk

        logger.info("문서 기반 응답 생성 완료")
        return {"messages": [AIMessage(content=full_response)]}

    async def _general_qa(self, state: State) -> Dict[str, Any]:
        """일반 질의응답 - Jupyter 코드와 동일한 로직"""
        logger.info("일반 응답 생성 중...")

        user_message = state["messages"][-1].content

        # 이전 대화 히스토리 구성
        history_messages = state["messages"][:-1]
        formatted_history = ""
        for msg in history_messages:
            role = "사용자" if isinstance(msg, HumanMessage) else "AI"
            formatted_history += f"{role}: {msg.content}\\n"

        # 프롬프트 생성 - Jupyter 코드와 동일
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """너는 친절한 한국어 AI 비서야. 
        사용자의 질문에 대해 간결하고 정확하게 답변해. 한국어로 대답해.
        
        이전 대화:
        {chat_history}
        """,
                ),
                ("user", "{user_input}"),
            ]
        )

        formatted_message = prompt.format_messages(
            user_input=user_message, chat_history=formatted_history
        )

        # 스트리밍 응답을 누적해서 전체 응답 생성
        full_response = ""
        async for chunk in self._stream_llm_response(self.llm, formatted_message):
            full_response += chunk

        logger.info("일반 응답 생성 완료")
        return {"messages": [AIMessage(content=full_response)]}

    async def _greeting(self, state: State) -> Dict[str, Any]:
        """인사말 응답 - Jupyter 코드와 동일한 로직"""
        logger.info("인사 응답 생성 중...")

        user_message = state["messages"][-1].content

        # 이전 대화 히스토리 구성
        history_messages = state["messages"][:-1]
        formatted_history = ""
        for msg in history_messages:
            role = "사용자" if isinstance(msg, HumanMessage) else "AI"
            formatted_history += f"{role}: {msg.content}\\n"

        # 프롬프트 생성 - Jupyter 코드와 동일
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """너는 친절한 한국어 AI 비서야.
        사용자의 인사에 친근하고 따뜻하게 응답해. 간결하게 한국어로 대답해.

        이전 대화:
        {chat_history}
        """,
                ),
                ("user", "{user_input}"),
            ]
        )

        formatted_message = prompt.format_messages(
            user_input=user_message, chat_history=formatted_history
        )

        # 스트리밍 응답을 누적해서 전체 응답 생성
        full_response = ""
        async for chunk in self._stream_llm_response(self.llm, formatted_message):
            full_response += chunk

        logger.info("인사 응답 생성 완료")
        return {"messages": [AIMessage(content=full_response)]}

    async def stream_chat_response(
        self, message: str, session_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        진정한 실시간 스트리밍 채팅 응답을 생성합니다.
        영상에서 보여주는 것처럼 LLM의 실제 토큰 스트리밍을 그대로 전달합니다.
        """

        if not self.is_initialized:
            raise RuntimeError("RAG 서비스가 초기화되지 않았습니다")

        # 세션 ID 생성 (제공되지 않은 경우)
        if not session_id:
            session_id = str(uuid.uuid4())

        # 대화 히스토리 가져오기 또는 초기화
        if session_id not in self.conversation_histories:
            self.conversation_histories[session_id] = []

        conversation_history = self.conversation_histories[session_id]

        try:
            # 사용자 메시지 추가
            user_message = HumanMessage(content=message)
            conversation_history.append(user_message)

            # 초기 상태 구성
            initial_state = {
                "messages": conversation_history.copy(),
                "context": [],
                "category": None,
            }

            # 시작 알림
            yield {
                "type": "start",
                "content": "요청 처리를 시작합니다...",
                "metadata": {"step": "start"},
            }

            # 1단계: 카테고리 분류
            yield {
                "type": "classification",
                "content": "쿼리를 분류하고 있습니다...",
                "metadata": {"step": "router"},
            }

            router_result = self._router(initial_state)
            category = router_result["category"]
            initial_state.update(router_result)

            yield {
                "type": "classification_result",
                "content": f"쿼리 분류 완료: {category}",
                "category": category,
                "metadata": {"step": "router_complete"},
            }

            # 2단계: 문서 검색 (필요한 경우)
            if category == "document":
                yield {
                    "type": "retrieval",
                    "content": "관련 문서를 검색하고 있습니다...",
                    "metadata": {"step": "retrieve"},
                }

                retrieve_result = self._retrieve_documents(initial_state)
                initial_state.update(retrieve_result)

                yield {
                    "type": "retrieval_result",
                    "content": "문서 검색 완료",
                    "metadata": {
                        "step": "retrieve_complete",
                        "docs_found": len(initial_state["context"]),
                    },
                }

            # 3단계: 실시간 응답 생성
            yield {
                "type": "generation_start",
                "content": "응답을 생성하고 있습니다...",
                "metadata": {"step": "generation"},
            }

            # 🔥 핵심: 여기서 실제 LLM 스트리밍을 바로 연결합니다
            full_response = ""

            # 프롬프트 준비
            context = initial_state["context"][0] if initial_state["context"] else ""
            formatted_message = self._prepare_prompt(
                message, conversation_history[:-1], context, category
            )

            # 🔥 LLM의 실제 스트리밍 출력을 직접 yield
            chain = self.llm | StrOutputParser()

            async for chunk in chain.astream(formatted_message):
                if chunk:  # 빈 청크가 아닌 경우에만
                    full_response += chunk

                    # 각 토큰을 즉시 전송 - 이것이 진짜 스트리밍입니다!
                    yield {
                        "type": "generation_chunk",
                        "content": chunk,  # 실제 LLM 토큰을 그대로 전송
                        "category": category,
                        "metadata": {
                            "partial_response": full_response,
                            "tokens_so_far": len(full_response.split()),
                        },
                    }

            # AI 응답을 히스토리에 추가
            ai_message = AIMessage(content=full_response)
            conversation_history.append(ai_message)

            # 최종 응답
            yield {
                "type": "final",
                "content": full_response,
                "category": category,
                "metadata": {
                    "session_id": session_id,
                    "total_tokens": len(full_response.split()),
                    "conversation_length": len(conversation_history),
                },
                "is_final": True,
            }

        except Exception as e:
            logger.error(f"스트리밍 처리 중 오류: {e}")
            yield {
                "type": "error",
                "content": f"처리 중 오류가 발생했습니다: {str(e)}",
                "metadata": {"error_type": type(e).__name__},
                "is_final": True,
            }

    def _prepare_prompt(
        self, user_message: str, history: List[BaseMessage], context: str, category: str
    ):
        """카테고리에 따라 적절한 프롬프트를 준비합니다."""

        # 히스토리 포맷팅
        formatted_history = ""
        for msg in history:
            role = "사용자" if isinstance(msg, HumanMessage) else "AI"
            formatted_history += f"{role}: {msg.content}\\n"

        if category == "document":
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

            return prompt.format_messages(
                context=context, chat_history=formatted_history, user_input=user_message
            )

        elif category == "greeting":
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """너는 친절한 한국어 AI 비서야.
            사용자의 인사에 친근하고 따뜻하게 응답해. 간결하게 한국어로 대답해.

            이전 대화:
            {chat_history}
            """,
                    ),
                    ("user", "{user_input}"),
                ]
            )

            return prompt.format_messages(
                user_input=user_message, chat_history=formatted_history
            )

        else:  # general
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """너는 친절한 한국어 AI 비서야. 
            사용자의 질문에 대해 간결하고 정확하게 답변해. 한국어로 대답해.
            
            이전 대화:
            {chat_history}
            """,
                    ),
                    ("user", "{user_input}"),
                ]
            )

            return prompt.format_messages(
                user_input=user_message, chat_history=formatted_history
            )

    async def generate_full_response(
        self, message: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """비스트리밍 방식으로 전체 응답을 한 번에 생성합니다."""

        full_response = ""
        category = "unknown"
        sources = []

        # 스트리밍 응답을 모두 수집
        async for chunk in self.stream_chat_response(message, session_id):
            if chunk["type"] == "final":
                full_response = chunk["content"]
                category = chunk.get("category", "unknown")
                break

        return {"response": full_response, "category": category, "sources": sources}

    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """대화 히스토리 조회"""
        if session_id not in self.conversation_histories:
            return []

        history = []
        for msg in self.conversation_histories[session_id]:
            history.append(
                {
                    "type": "human" if isinstance(msg, HumanMessage) else "ai",
                    "content": msg.content,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return history

    def clear_conversation_history(self, session_id: str):
        """대화 히스토리 삭제"""
        if session_id in self.conversation_histories:
            del self.conversation_histories[session_id]
