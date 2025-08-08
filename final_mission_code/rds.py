import os
from urllib.parse import quote_plus

# 현재 비밀번호 (! 없는 버전)
password = "Ahrnta1213"  # 또는 실제 비밀번호
encoded_password = quote_plus(password)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://postgres:{encoded_password}@sancho-mission3.cpe0a008coe5.ap-northeast-2.rds.amazonaws.com:5432/postgres?sslmode=require",
)

print(f"사용 중인 DATABASE_URL: {DATABASE_URL}")
print(f"비밀번호 부분: {encoded_password}")
