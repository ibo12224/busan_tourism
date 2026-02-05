# 1. 베이스 이미지 설정test
FROM python:3.10

# 2. 작업 디렉토리 생성
WORKDIR /app

# 3. 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. 의존성 파일 복사 및 설치
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 및 데이터 복사
# 이후 코드에서 pd.read_csv("data/your_file.csv")로 접근 가능
COPY app/ ./
COPY data/ ./data/

# 6. Streamlit 포트 개방
EXPOSE 8501

# 7. 실행 명령
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]