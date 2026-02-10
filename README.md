# 🌊 오버투어리즘 대응 SLA 전략: 데이터 기반 부산관광 균형성장 모델
> **Spot-Linear-Area 전략을 활용한 부산 안심관광지(73개소) 유사도 분석 및 혼잡도 예측 시스템**

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat-square&logo=github-actions&logoColor=white)

---

## 📌 1. 프로젝트 개요
* **배경**: 2024년 부산 외국인 관광객 300만 돌파. 그러나 특정 명소(감천문화마을 등)에 집중된 오버투어리즘으로 인한 특별관리구역 지정 및 방문 제한 위기 발생.
* **목적**: 관광경영학의 **SLA(Spot-Linear-Area) 전략**을 기반으로, 유명 관광지(Spot)와 유사한 대체 관광지를 선(Linear)으로 연결하여 지역 전체(Area)의 균형 성장을 도모.
* **핵심 가설**: 방문객의 '취향의 관성'을 존중하되, '효용의 대체가능성'을 시각적/문맥적 데이터로 증명하여 수요 분산을 유도함.

---

## ⚙️ 2. 시스템 아키텍처 및 프로세스



### 1️⃣ 이미지 유사도 분석 (Visual Similarity)
- **Backbone**: `ConvNeXt` (Classification Head 제거 후 Feature Extractor로 활용)
- **Refinement**: Iterative Feedback Loop를 통해 데이터셋 정합성 80% 달성.
- **Algorithm**: **3-Stage Pipeline**
  - **Step 1**: 클래스 내 군집화 (Intra-class Clustering) 및 아웃라이어 제거.
  - **Step 2**: 군집별 대표 임베딩(Medoid) 5점 추출.
  - **Step 3**: 1:1 매칭(Pairwise Cross-Matching)을 통한 최종 코사인 유사도 산출.

### 2️⃣ 텍스트 유사도 분석 (Semantic Similarity)
- **Model**: `Ko-SBERT` (Sentence-BERT)
- **Feature Engineering**: 
  - **Weighted Sequence**: 제목 텍스트 중복 연결로 Attention 집중도 향상.
  - **Dual-Track Analysis**: 명사(물리적 속성)와 형용사(정서적 경험/Vibe) 분리 분석.
- **Optimization**: **Zero-Centering** 기법을 적용하여 이방성(Anisotropy) 문제를 해결하고 관광지별 고유 잔차 특징 극대화.

### 3️⃣ 혼잡도 분석 및 예측 (Crowd Insight)
- **Data**: 공공데이터(생활인구) + 민간데이터(네이버 검색 트렌드) 결합 추정.
- **Model**: `Prophet` 기반 2025년 월별 혼잡도 예측.
- **Strategy**: `changepoint_prior_scale` 조정을 통해 급격한 노이즈를 억제하고 안정적인 선형 트렌드 도출.

---

## 🚀 3. 주요 기능 (Service Features)
- **AI 심층 리포트**: RAG(Retrieval-Augmented Generation)를 통해 분석 결과에 대한 정성적 근거를 GPT-4가 설명.
- **사용자 맞춤형 유사도**: 시각적(Image) 요소와 문맥적(Text) 요소의 가중치를 직접 조절하여 대체지 탐색.
- **Cross-Category 추천**: 카테고리가 다르더라도 고유 유사도가 높은 '숨은 명소' 발굴 가능.

---

## 🛠 4. 기술 스택 (Tech Stack)

### AI / Data Science
- **Deep Learning**: PyTorch, ConvNeXt, HuggingFace
- **NLP**: Ko-SBERT, Konlpy (Mecab/Okt)
- **Time Series**: Prophet
- **Vector DB**: FAISS (Similarity Search)

### DevOps & Deployment
- **Container**: Docker
- **CI/CD**: GitHub Actions (자동 빌드 및 배포 최적화)
- **LLM**: OpenAI GPT-4 API (RAG 구현)

---

## 👥 5. 팀 정보 (Learning Crew)
- **경제학과**: 도메인 로직(SLA 전략) 수립 및 경제적 타당성 분석.
- **컴퓨터공학과**: 이미지 딥러닝 모델링 및 서비스 아키텍처 설계.
- **기계공학과**: 데이터 크롤링 및 파이프라인 자동화.
- **통계학과**: 혼잡도 지표 산출 및 시계열 예측 모델링.

---

## ⚠️ 6. 한계점 및 향후 과제
- **데이터 스케일**: 현재 부산 안심관광지 73개소에서 부산 전역으로 확장 필요.
- **정밀도**: 실시간 유동인구 API 연동을 통한 혼잡도 모델 고도화 예정.
