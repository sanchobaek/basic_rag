import os
from urllib.parse import quote_plus
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경변수에서 비밀번호 읽기
password = os.getenv("PASSWORD", "")  # 기본값은 빈 문자열
encoded_password = quote_plus(password)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://postgres:{encoded_password}@sancho-mission3.cpe0a008coe5.ap-northeast-2.rds.amazonaws.com:5432/postgres?sslmode=require",
)

print(f"사용 중인 DATABASE_URL: {DATABASE_URL}")
print(f"비밀번호 부분: {encoded_password}")
