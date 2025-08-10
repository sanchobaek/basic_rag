import os
from dotenv import load_dotenv  # 환경변수 로딩을 위한 모듈
from langchain_upstage import UpstageDocumentParseLoader  # PDF 문서 파싱을 위한 Upstage AI 로더
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 문서를 작은 청크로 분할하는 텍스트 분할기
from langchain_huggingface import HuggingFaceEmbeddings  # 텍스트를 벡터로 변환하는 임베딩 모델
from langchain_postgres import PGVector  # PostgreSQL 기반 벡터 데이터베이스
import traceback #에러 추적 도구

# .env 파일에서 환경변수 로드 (API 키, DB 연결정보 등)
load_dotenv()

# Upstage AI API 키를 환경변수에서 가져와서 설정
os.environ["UPSTAGE_API_KEY"] = os.getenv("UPSTAGE_API_KEY", "")

# PostgreSQL 데이터베이스 연결 URL과 컬렉션명을 환경변수에서 가져오기
DATABASE_URL = os.getenv("DATABASE_URL")  # PostgreSQL 연결 문자열
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")  # 벡터 데이터를 저장할 테이블명


def load_and_process_documents(file_path: str = "./test_modified.pdf"):
    print("문서 로딩 중...")
      
    loader = UpstageDocumentParseLoader( 
        file_path,
        split="page",   #페이지 단위로 분할
        output_format="markdown", #markdown 형식으로 출력
        ocr="auto", #ocr 자동 적용 
        coordinates=True, #텍스트 위치 정보 포함
    )
    docs = loader.load()  # 문서 로딩 실행
    print(f"문서 로딩 완료: {len(docs)}개 페이지") # 문서의 몇 개 페이지가 로딩되어있는 프린트

    
    print("🔪 문서 청킹 중...") #로딩된 문서를 더 작은 청크로 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300) #각 청크의 최대 문자 수, 청크 간 중복되는 문자 수
    docs_splitter = splitter.split_documents(docs) #문서를 작은 청크 단위로 나눠서 docs_splitter에 저장
    print(f"✅ 청킹 완료: {len(docs_splitter)}개 청크") #청크 길이 확인 

    return docs_splitter #문서를 청크로 나눴으니 이 결과를 다음 작업(main에서 사용)에 사용하세요~


def initialize_embeddings():    # HuggingFace에서 제공하는 다국어 임베딩 모델 초기화
    print("임베딩 모델 로딩 중...")
    
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large-instruct", #한국어를 포함한 다국어를 지원하는 E5 모델 
        model_kwargs={"device": "cpu"}, #cpu에서 실행(gpu가 없기때문, 임베딩 시간이 오래걸림.)
        encode_kwargs={"normalize_embeddings": True}, #벡터 정규화로 코사인 유사도 계산에 최적
    )
    
    print("✅ 임베딩 모델 로딩 완료")
    return hf_embeddings #임베딩한 결과를 다음 작업(main에서 사용)에 사용하세요


def create_vector_store(docs_splitter, embeddings):
    print("🗄️ PostgreSQL/PGVector 벡터 스토어 생성 중...")
    
    # 기존 컬렉션 확인 및 연결
    vectorstore = PGVector(
        embeddings=embeddings,
        connection_string=DATABASE_URL,
        collection_name=COLLECTION_NAME,
    )
    
    # 기존 데이터 삭제 (재학습 시 중복 방지)
    print("🗑️ 기존 데이터 삭제 중...")
    vectorstore.delete_collection()
    
    #실제 데이터 저장 담당
    vectorstore = PGVector.from_documents(
        documents=docs_splitter,      # 청킹된 문서들을
        embedding=embeddings,         # 임베딩 모델로 벡터화해서
        connection_string=DATABASE_URL,
        collection_name=COLLECTION_NAME,
    )                                # PostgreSQL에 저장
    
    print("✅ 벡터 스토어 생성 완료")
    return vectorstore


def main():
    """
    전체 데이터 인제스천 파이프라인을 실행하는 메인 함수
    PDF 문서 → 청킹 → 임베딩 → PostgreSQL 저장까지의 전 과정을 수행
    """
    try:
        print("🚀 데이터 인제스천 파이프라인 시작")
        print(f"📍 데이터베이스: {DATABASE_URL}")
        print(f"📁 컬렉션: {COLLECTION_NAME}")
        print("-" * 50)
        
        # 1단계: PDF 문서를 로딩하고 작은 청크로 분할
        docs_splitter = load_and_process_documents()
        
        # 2단계: 텍스트를 벡터로 변환할 임베딩 모델 초기화
        embeddings = initialize_embeddings()
        
        # 3단계: PostgreSQL/PGVector 벡터 스토어 생성 및 문서 데이터 저장
        vectorstore = create_vector_store(docs_splitter, embeddings)
        
        # 4단계: 벡터 검색 기능이 정상적으로 작동하는지 테스트
        print("🔍 검색 테스트 중...")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 상위 3개 결과 반환
        # test_results = retriever.invoke("아주대학교")  # "아주대학교" 키워드로 검색 테스트
        # print(f"✅ 검색 테스트 완료: {len(test_results)}개 결과")
        
        print("-" * 50)
        print("🎉 데이터 인제스천 완료!")
        print("이제 RAG 서비스에서 벡터 스토어를 사용할 수 있습니다.")
        
    except Exception as e:
        print(f"❌ 데이터 인제스천 실패: {e}")
        
        traceback.print_exc()  # 상세한 에러 정보 출력


# 스크립트가 직접 실행될 때만 main() 함수 호출
if __name__ == "__main__":
    main()