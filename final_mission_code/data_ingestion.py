#!/usr/bin/env python3
"""
데이터 인제스천 파이프라인
PostgreSQL/PGVector에 문서를 로딩하고 벡터화하는 별도 스크립트
"""

import os
from dotenv import load_dotenv
from langchain_upstage import UpstageDocumentParseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

# .env 파일 로드
load_dotenv()

# API 키 설정 (환경변수에서 읽기)
os.environ["UPSTAGE_API_KEY"] = os.getenv("UPSTAGE_API_KEY", "")

# PostgreSQL 데이터베이스 설정 (환경변수에서 읽기)
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")


def load_and_process_documents(file_path: str = "./test_modified.pdf"):
    """문서 로딩 및 처리"""
    print("📄 문서 로딩 중...")
    
    # Document loading
    loader = UpstageDocumentParseLoader(
        file_path,
        split="page",
        output_format="markdown",
        ocr="auto",
        coordinates=True,
    )
    docs = loader.load()
    print(f"✅ 문서 로딩 완료: {len(docs)}개 페이지")

    # Document chunking
    print("🔪 문서 청킹 중...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    docs_splitter = splitter.split_documents(docs)
    print(f"✅ 청킹 완료: {len(docs_splitter)}개 청크")

    return docs_splitter


def initialize_embeddings():
    """임베딩 모델 초기화"""
    print("🤖 임베딩 모델 로딩 중...")
    
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large-instruct",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    print("✅ 임베딩 모델 로딩 완료")
    return hf_embeddings


def create_vector_store(docs_splitter, embeddings):
    """벡터 스토어 생성 및 문서 저장"""
    print("🗄️ PostgreSQL/PGVector 벡터 스토어 생성 중...")
    
    try:
        # 기존 컬렉션이 있는지 확인하고 연결
        vectorstore = PGVector(
            embeddings=embeddings,
            connection_string=DATABASE_URL,
            collection_name=COLLECTION_NAME,
        )
        
        # 기존 데이터 삭제 (선택사항)
        print("🗑️ 기존 데이터 삭제 중...")
        vectorstore.delete_collection()
        
        # 새로운 벡터 스토어 생성
        vectorstore = PGVector.from_documents(
            documents=docs_splitter,
            embedding=embeddings,
            connection_string=DATABASE_URL,
            collection_name=COLLECTION_NAME,
        )
        
        print("✅ 벡터 스토어 생성 완료")
        return vectorstore
        
    except Exception as e:
        print(f"❌ 벡터 스토어 생성 실패: {e}")
        raise


def main():
    """메인 데이터 인제스천 실행"""
    try:
        print("🚀 데이터 인제스천 파이프라인 시작")
        print(f"📍 데이터베이스: {DATABASE_URL}")
        print(f"📁 컬렉션: {COLLECTION_NAME}")
        print("-" * 50)
        
        # 1. 문서 로딩 및 처리
        docs_splitter = load_and_process_documents()
        
        # 2. 임베딩 모델 초기화
        embeddings = initialize_embeddings()
        
        # 3. 벡터 스토어 생성 및 데이터 저장
        vectorstore = create_vector_store(docs_splitter, embeddings)
        
        # 4. 검색 테스트
        print("🔍 검색 테스트 중...")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        test_results = retriever.invoke("아주대학교")
        print(f"✅ 검색 테스트 완료: {len(test_results)}개 결과")
        
        print("-" * 50)
        print("🎉 데이터 인제스천 완료!")
        print("이제 RAG 서비스에서 벡터 스토어를 사용할 수 있습니다.")
        
    except Exception as e:
        print(f"❌ 데이터 인제스천 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()