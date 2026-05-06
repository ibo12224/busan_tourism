import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY = os.getenv("OPENAI_API_KEY")

FILE_CONFIG = {
    "MAIN_DATA": os.path.join(BASE_DIR, "data", "관광지_혼잡도_찐최종결과물.csv"),
    "PRED_DATA": os.path.join(BASE_DIR, "data", "AI_예측_결과.csv"),
    "IMG_RANK_DATA": os.path.join(BASE_DIR, "data", "관광지_별_유사도_순위_refined.csv"),
    "IMG_MATRIX_DATA": os.path.join(BASE_DIR, "data", "부산_관광지_유사도_최종_결과_refined.csv"),
    "REVIEW_SIM_DATA": os.path.join(BASE_DIR, "data", "유사도.csv"),
    "SENTIMENT_DATA": os.path.join(BASE_DIR, "data", "관광지_감상유사도_분석(최종, TF-IDF적용).csv"),
    "FEATURE_DATA": os.path.join(BASE_DIR, "data", "관광지별_키워드_유사도_순위.csv"),
    "KEYWORD_NOUN": os.path.join(BASE_DIR, "data", "관광지별_키워드50_추출(정제후).csv"),
    "KEYWORD_ADJ": os.path.join(BASE_DIR, "data", "부산_관광지별_형용사_추출결과.csv"),
    "CATEGORY_INFO": os.path.join(BASE_DIR, "data", "부산_관광지명.xlsx"),
}

NAME_MAPPING = {
    '광안리SUPZONE': '광안대교sup',
    '오륙도': '오륙도스카이워크',
    '다대포낙조분수': '다대포꿈의낙조분수',
    '용호만부두': '용호만유람선',
    '을숙도생태공원': '을숙도',
    '안데르센마을': '안데르센동화마을',
    '석당박물관': '동아대석당박물관',
    '부산시립박물관': '부산박물관',
    '낙동강에코센터': '낙동강하구에코센터',
}
