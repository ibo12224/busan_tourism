# 1. 베이스 이미지 설정
FROM python:3.10-slim

# 2. 작업 디렉토리 생성
WORKDIR /app

# 3. 필수 리눅스 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. 의존성 파일 복사 및 설치 (openpyxl 필수 포함)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 및 데이터 복사
# 로컬의 각 폴더를 컨테이너 내부의 /app 하위 폴더로 각각 복사합니다.
COPY app/ .           
COPY data/ ./data/    
COPY images/ ./images/

# 6. Streamlit 포트 개방
EXPOSE 8501

# 7. 실행 명령 (main.py가 /app 바로 아래에 있으므로)
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]