import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
import os
import re
import numpy as np
import textwrap
from dotenv import load_dotenv

# =============================================================================
# [설정] 파일명 매핑
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv() # .env 파일 로드
api_key = os.getenv("OPENAI_API_KEY")

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
    "CATEGORY_INFO": os.path.join(BASE_DIR, "data", "부산_관광지명.xlsx")
}

# 이름 매핑
NAME_MAPPING = {
    '광안리SUPZONE': '광안대교sup',
    '오륙도': '오륙도스카이워크',
    '다대포낙조분수': '다대포꿈의낙조분수',
    '용호만부두': '용호만유람선',
    '을숙도생태공원': '을숙도',
    '안데르센마을': '안데르센동화마을',
    '석당박물관': '동아대석당박물관',
    '부산시립박물관': '부산박물관'
}

# 1. 페이지 설정
st.set_page_config(layout="wide", page_title="SLA PROJECT", page_icon="⚫")

# -----------------------------------------------------------------------------
# 2. [디자인] CSS 스타일링
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    @import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css");
    .stApp { font-family: 'Pretendard', sans-serif !important; background-color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #000000 !important; border-right: 1px solid #222 !important; }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary { color: #FFFFFF !important; font-weight: 700 !important; }
    [data-testid="stSidebar"] button { color: #FFFFFF !important; text-align: left !important; }
    
    div.stButton > button[kind="primary"] {
        background-color: #0F172A !important; color: #FFFFFF !important;
        border-radius: 8px; font-weight: 800 !important; border: 1px solid #0F172A !important;
    }
    div.stButton > button[kind="secondary"] {
        background-color: #F8FAFC !important; color: #333 !important;
        border: 1px solid #E2E8F0 !important; border-radius: 6px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 0px; border-bottom: 2px solid #000000; margin-bottom: 30px; }
    .stTabs [data-baseweb="tab"] { height: 60px; flex: 1; background: transparent; border: none; color: #9CA3AF; font-weight: 700; font-size: 1.2rem; }
    .stTabs [aria-selected="true"] { color: #000000 !important; background: #FAFAFA !important; border-bottom: 5px solid #000000 !important; }
    
    .sim-card { background: #FFFFFF; border: 1px solid #E5E5E5; border-radius: 12px; padding: 24px; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.03); }
    .rank-card-mini { background: #F9F9F9; border: 1px solid #EEE; border-radius: 8px; padding: 15px; min-width: 150px; text-align: center; }
    .sim-rank-badge { background: #000; color: #FFF; padding: 4px 10px; border-radius: 4px; font-size: 0.8rem; font-weight: 700; margin-right: 8px; }
    .congestion-badge { font-size: 0.75rem; font-weight: 700; padding: 4px 8px; border-radius: 4px; display: inline-block; }
    .cong-good { background: #E0F2FE; color: #0284C7; border: 1px solid #0284C7; } 
    .cong-norm { background: #DCFCE7; color: #16A34A; border: 1px solid #16A34A; } 
    .cong-bad { background: #FEE2E2; color: #DC2626; border: 1px solid #DC2626; } 
    .ai-insight-box { background: #F8FAFC; border: 1px solid #E2E8F0; border-left: 6px solid #0F172A; padding: 20px; margin: 20px 0 30px 0; color: #333; line-height: 1.7; }
    .ai-header { font-weight: 800; font-size: 1rem; margin-bottom: 10px; color: #0F172A !important; display: flex; align-items: center; gap: 8px; }
    .ai-box-full-height { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 25px; height: 100%; min-height: 350px; display: flex; flex-direction: column; justify-content: center; }
    .meta-tag { display: inline-block; font-size: 0.75rem; font-weight: 600; padding: 3px 8px; margin: 0 4px 4px 0; border-radius: 4px; }
    .tag-common { background: #F3F4F6; color: #4B5563; border: 1px solid #E5E7EB; } 
    .tag-unique-source { background: #FFFFFF; color: #000; border: 1px solid #000; } 
    .tag-unique-target { background: #000; color: #FFF; border: 1px solid #000; } 
    .keyword-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 15px; padding-top: 15px; border-top: 1px dashed #E2E8F0; }
    .keyword-col-header { font-size: 0.7rem; font-weight: 800; color: #64748B; margin-bottom: 8px; text-transform: uppercase; }
    .keyword-box { background: #FAFAFA; border: 2px solid #E5E5E5; border-radius: 12px; padding: 25px; margin-bottom: 30px; }
    .keyword-header { font-size: 1rem; font-weight: 900; color: #000; margin-bottom: 15px; border-left: 5px solid #000; padding-left: 12px; }
    .block-container { padding-top: 2rem; }
    
    /* 캡션 가운데 정렬 클래스 */
    .center-caption { text-align: center; color: #666; font-size: 0.9rem; margin-top: -10px; margin-bottom: 20px; line-height: 1.4; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. [데이터 로드]
# -----------------------------------------------------------------------------
@st.cache_data
def load_data_smart(file_key):
    # 1. FILE_CONFIG에서 경로 가져오기
    file_path = FILE_CONFIG.get(file_key)
    
    if not file_path:
        st.error(f"❌ FILE_CONFIG에 '{file_key}' 설정이 없습니다.")
        return pd.DataFrame()

    # 2. CSV 파일 로드 시도
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path, encoding='utf-8-sig')
        except Exception:
            try:
                return pd.read_csv(file_path, encoding='cp949')
            except Exception as e:
                st.warning(f"⚠️ CSV 로드 실패 ({file_key}): {e}")

    # 3. CSV가 없거나 실패 시 XLSX 파일 로드 시도
    # 확장자 교체 및 경로 정제
    xlsx_path = file_path.replace('.csv', '.xlsx').strip()
    
    if os.path.exists(xlsx_path):
        try:
            # 엑셀 로드를 위해 openpyxl 엔진 사용 권장
            return pd.read_excel(xlsx_path, engine='openpyxl')
        except Exception as e:
            st.warning(f"⚠️ XLSX 로드 실패 ({file_key}): {e}")

    # 4. 모든 시도가 실패할 경우 빈 데이터프레임 반환
    # st.error(f"📂 파일을 찾을 수 없습니다: {file_path}")
    return pd.DataFrame()

@st.cache_data
def get_category_map():
    cat_df = load_data_smart(FILE_CONFIG["CATEGORY_INFO"])
    if cat_df.empty: return {}
    cat_df.columns = cat_df.columns.str.strip()
    if '관광지명' in cat_df.columns and '카테고리' in cat_df.columns:
        cat_df['관광지명'] = cat_df['관광지명'].astype(str).str.strip().replace(NAME_MAPPING)
        cat_df['카테고리'] = cat_df['카테고리'].astype(str).str.strip()
        return dict(zip(cat_df['관광지명'], cat_df['카테고리']))
    return {}

CATEGORY_MAP = get_category_map()

@st.cache_data
def get_scaled_data():
    df_vis = load_data_smart(FILE_CONFIG["IMG_MATRIX_DATA"])
    df_sen = load_data_smart(FILE_CONFIG["SENTIMENT_DATA"])
    df_fea = load_data_smart(FILE_CONFIG["FEATURE_DATA"])
    
    if not df_fea.empty:
        df_fea['기준_관광지'] = df_fea['기준_관광지'].astype(str).str.strip().replace(NAME_MAPPING)
        df_fea['비교_대상'] = df_fea['비교_대상'].astype(str).str.strip().replace(NAME_MAPPING)

    if not df_sen.empty:
        df_sen['기준_관광지'] = df_sen['기준_관광지'].astype(str).str.strip().replace(NAME_MAPPING)
        df_sen['비교_대상'] = df_sen['비교_대상'].astype(str).str.strip().replace(NAME_MAPPING)

    vis_long = pd.DataFrame()
    if not df_vis.empty:
        df_vis.columns = df_vis.columns.str.strip()
        if df_vis.columns[0] == '관광지명':
            df_vis['관광지명'] = df_vis['관광지명'].astype(str).str.strip().replace(NAME_MAPPING)
        
        vis_long = df_vis.set_index(df_vis.columns[0]).stack().reset_index()
        vis_long.columns = ['기준_관광지', '비교_대상', 'VIS_RAW']
        vis_long['기준_관광지'] = vis_long['기준_관광지'].astype(str).str.strip().replace(NAME_MAPPING)
        vis_long['비교_대상'] = vis_long['비교_대상'].astype(str).str.strip().replace(NAME_MAPPING)
        
        v_min, v_max = vis_long['VIS_RAW'].min(), vis_long['VIS_RAW'].max()
        vis_long['VIS_SCALED'] = (vis_long['VIS_RAW'] - v_min) / (v_max - v_min + 1e-9)

    if not df_sen.empty and 'SBERT_유사도(가중적용)' in df_sen.columns:
        s_min, s_max = df_sen['SBERT_유사도(가중적용)'].min(), df_sen['SBERT_유사도(가중적용)'].max()
        df_sen['SEN_SCALED'] = (df_sen['SBERT_유사도(가중적용)'] - s_min) / (s_max - s_min + 1e-9)
    
    if not df_fea.empty and '최종_유사도' in df_fea.columns:
        f_min, f_max = df_fea['최종_유사도'].min(), df_fea['최종_유사도'].max()
        df_fea['FEA_SCALED'] = (df_fea['최종_유사도'] - f_min) / (f_max - f_min + 1e-9)

    return vis_long, df_sen, df_fea

@st.cache_data
def load_all_data():
    df_main = load_data_smart(FILE_CONFIG["MAIN_DATA"])
    if not df_main.empty:
        df_main['날짜'] = pd.to_datetime(df_main['날짜'], format='mixed', errors='coerce')
        df_main = df_main.dropna(subset=['날짜'])
        if '행정동' in df_main.columns:
            df_main['행정구'] = df_main['행정동'].astype(str).apply(lambda x: x.split()[0] if len(x.split()) > 0 else "미분류")
        else: df_main['행정구'] = "전체"
        
        df_main['관광지명'] = df_main['관광지명'].astype(str).str.strip().replace(NAME_MAPPING)
        
        if '시간대' in df_main.columns:
             df_main['시간대_int'] = df_main['시간대'].astype(str).str.replace('시', '').apply(pd.to_numeric, errors='coerce')

    df_pred = load_data_smart(FILE_CONFIG["PRED_DATA"])
    if not df_pred.empty:
        df_pred['ds'] = pd.to_datetime(df_pred['ds'])
        df_pred['관광지명'] = df_pred['관광지명'].astype(str).str.strip().replace(NAME_MAPPING)

    df_noun = load_data_smart(FILE_CONFIG["KEYWORD_NOUN"])
    df_adj = load_data_smart(FILE_CONFIG["KEYWORD_ADJ"])
    
    if not df_noun.empty and '관광지명' in df_noun.columns:
        df_noun['관광지명'] = df_noun['관광지명'].astype(str).str.strip().replace(NAME_MAPPING)
    if not df_adj.empty and '관광지명' in df_adj.columns:
        df_adj['관광지명'] = df_adj['관광지명'].astype(str).str.strip().replace(NAME_MAPPING)

    return df_main, df_pred, df_noun, df_adj

main_df, forecast_df, noun_df, adj_df = load_all_data()
df_vis_scaled, df_sen_scaled, df_fea_scaled = get_scaled_data()

@st.cache_data
def get_global_top1_avg():
    df_img = load_data_smart(FILE_CONFIG["IMG_RANK_DATA"])
    if df_img.empty: return 0.5
    scores = []
    for val in df_img['1순위'].dropna():
        m = re.search(r'\(([\d.]+)\)', str(val))
        if m: scores.append(float(m.group(1)))
    return np.mean(scores) if scores else 0.5

GLOBAL_TOP1_AVG = get_global_top1_avg()

def get_spot_category(name):
    if name in CATEGORY_MAP: return CATEGORY_MAP[name]
    return '기타'

def classify_density(val):
    if val >= 1.2: return "매우혼잡"
    elif val >= 0.7: return "혼잡"
    elif val >= 0.3: return "보통"
    else: return "쾌적"

def get_smart_active_mean(df, spot_name):
    spot_df = df[df['관광지명'] == spot_name]
    if spot_df.empty: return 0, "알수없음"
    
    hourly_grp = spot_df.groupby('시간대_int')['실질_㎡당_방문객수'].mean()
    if hourly_grp.empty: return 0, "알수없음"
    
    total_mean = hourly_grp.mean()
    base_hours = list(range(9, 19)) 
    active_hours = set(base_hours)
    
    for h in hourly_grp.index:
        val = hourly_grp[h]
        if h not in base_hours and val > total_mean:
            active_hours.add(h)
        if h in base_hours and val < total_mean:
            if h in active_hours: active_hours.remove(h)
            
    valid_vals = [hourly_grp[h] for h in active_hours if h in hourly_grp.index]
    
    if not valid_vals:
        final_val = total_mean
    else:
        final_val = np.mean(valid_vals)
        
    return final_val, classify_density(final_val)

def get_active_time_stats(df, spot_name, year=None):
    if year: spot_df = df[(df['관광지명'] == spot_name) & (df['날짜'].dt.year == year)]
    else: spot_df = df[df['관광지명'] == spot_name]
    return get_smart_active_mean(spot_df, spot_name)

def get_ranking_info(df, spot_name, year):
    y_df = df[df['날짜'].dt.year == year]
    if y_df.empty: return "정보 없음"
    
    rank_list = []
    for s in y_df['관광지명'].unique():
        val, _ = get_smart_active_mean(y_df, s)
        rank_list.append({'spot': s, 'val': val})
    
    rank_df = pd.DataFrame(rank_list).sort_values(by='val', ascending=False).reset_index(drop=True)
    
    if spot_name not in rank_df['spot'].values: return "정보 없음"
    
    my_rank = rank_df[rank_df['spot'] == spot_name].index[0] + 1
    total = len(rank_df)
    percent = (my_rank / total) * 100
    
    return f"전체 {total}곳 중 {my_rank}위 (상위 {percent:.1f}%)"

def get_spot_keywords(spot_name):
    nouns, adjs = [], []
    if not noun_df.empty:
        row = noun_df[noun_df['관광지명'] == spot_name]
        if not row.empty: nouns = [k.strip() for k in str(row.iloc[0]['정제키워드']).split(',') if k.strip()][:30]
    if not adj_df.empty:
        row = adj_df[adj_df['관광지명'] == spot_name]
        if not row.empty: adjs = [k.strip() for k in str(row.iloc[0]['추출_형용사']).split(',') if k.strip()][:30]
    if not nouns and not df_sen_scaled.empty:
        row = df_sen_scaled[df_sen_scaled['기준_관광지'] == spot_name]
        if not row.empty:
            raw_k = str(row.iloc[0].get('기준지_고유_키워드', ''))
            if raw_k and raw_k != 'nan':
                all_k = [k.strip() for k in raw_k.split(',') if k.strip()]
                nouns = all_k[:len(all_k)//2]
                adjs = all_k[len(all_k)//2:]
    return nouns, adjs

# [페르소나 정의]
AI_SYSTEM_PROMPT = """
당신은 날카롭고 깊이 있는 통찰력을 가진 수석 데이터 분석가입니다.
'분석하겠습니다', '결과입니다', '안녕하세요' 같은 형식적인 서론을 일절 생략하고, 즉시 핵심 수치와 그 이면의 의미를 파고드십시오.
문장은 명료하되 내용은 심층적이어야 하며, 분량은 충분히 길고 자세하게 작성하십시오.
톤은 전문적이고 냉철한 존댓말(~입니다/합니다)을 유지하십시오.
"""

def generate_spot_info_ai(spot_name):
    if not api_key: return "API Key Missing"
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": AI_SYSTEM_PROMPT}, 
                      {"role": "user", "content": f"'{spot_name}'의 위치, 주요 특징, 역사적/문화적 배경을 심층적으로 서술하십시오."}], 
            temperature=0.3
        )
        return response.choices[0].message.content
    except: return "정보 로드 실패"

def generate_visual_rank1_analysis(spot_name, rank1_name, rank1_score, avg_score, rank1_congestion):
    if not api_key: return "API Key Missing"
    
    score_diff = rank1_score - avg_score
    score_eval = f"평균({avg_score:.2f})보다 {score_diff:+.2f}점 높음" if score_diff > 0 else "평균 이하"
    
    policy_guide = ""
    if rank1_congestion in ["혼잡", "매우혼잡"]:
        policy_guide = "해당 대체지 역시 현재 '포화 상태'입니다. 이곳으로의 유입 유도는 풍선 효과를 초래하므로 정책적으로 '부적절'합니다."
    else:
        policy_guide = "해당 대체지는 현재 '수용 여력'이 충분합니다. 이곳으로의 유입 유도는 분산 정책상 '타당'합니다."

    user_msg = f"""
    [분석 대상]: {spot_name}
    [시각적 대체지 1위]: {rank1_name}
    [데이터]: 유사도 {rank1_score:.4f} ({score_eval}), 혼잡도 '{rank1_congestion}'
    
    [지시사항]
    당신은 엄격한 데이터 분석가입니다. 인사말(안녕하세요 등)을 생략하고 바로 분석 내용을 서술하십시오. 정중한 존댓말(~입니다/합니다)을 사용하십시오.
    
    1. **대체지 기본 정보**: {rank1_name}이 어떤 곳인지 간략히 설명하십시오.
    2. **유사도 평가**: 전체 평균 대비 유사도 수준을 수치와 함께 객관적으로 서술하십시오.
    3. **수용력 진단**: 대체지의 현재 혼잡도를 근거로, 분산 수용 가능 여부를 냉정하게 판정하십시오.
    
    (참고 가이드: {policy_guide})
    """
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": AI_SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e: return str(e)

def generate_strategic_analysis(source, candidates_data, anal_type="text"):
    if not api_key: return "⚠️ API Key Missing"
    candidates_text = ""
    for c in candidates_data:
        candidates_text += f"- {c['name']} (Rank {c['rank']}): 유사도 {c['score']:.4f}, 혼잡도 [{c['congestion']}]\n"
    
    user_msg = f"""
    [분석 대상]: {source.get('name', 'Unknown')} (현재 혼잡도: {source.get('congestion', 'Unknown')})
    [분석 유형]: {anal_type} 기반 유사도 후보군
    {candidates_text}
    
    [추가 정보]
    {source}

    [요청사항]
    데이터 분석가 입장에서 진단하십시오. 인사말을 절대 하지 마십시오. 반드시 존댓말(~입니다/합니다)을 사용하십시오.
    
    1. 유사도가 높으면서 혼잡도가 '쾌적/보통'인 곳을 **'유효 대체지'**로 분류하십시오.
    2. 유사도가 높더라도 혼잡도가 '혼잡/매우혼잡'인 곳은 **'대체 불가(포화)'**로 명시하십시오.
    3. 오직 데이터에 근거하여 분산 가능성 여부만 객관적으로 서술하십시오. (추상적 전략 제안 금지)
    """
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": AI_SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e: return f"Error: {str(e)}"

def generate_weighted_insight(spot_name, top_cand, weights):
    if not api_key: return "API Key Missing"
    user_msg = f"""
    [User Preferences - Weighted Priority]
    - Visual: {weights[0]}
    - Sentiment: {weights[1]}
    - Feature: {weights[2]}
    
    [Result]
    - Source: {spot_name}
    - Recommended: {top_cand['name']}
    - Scores: V({top_cand['raw_v']:.2f}), S({top_cand['raw_s']:.2f}), F({top_cand['raw_f']:.2f})
    
    [Task]
    Explain clearly and deeply why this spot was selected based on data scores. 
    No greetings. Start immediately.
    Use polite Korean (Honorifics).
    Max 5 sentences.
    """
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": AI_SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e: return f"Error: {str(e)}"

def generate_section_analysis(section_type, spot_name, year, data_summary, congestion_stage, ranking_info="정보 없음"):
    if  not api_key: return "⚠️ API Key Missing"
    
    tone_guide = ""
    if congestion_stage in ["쾌적", "보통"]:
        tone_guide = """
        [Diagnosis]: 수용 여력 충분 (Under Capacity).
        [Implication]: 데이터상 관광객 추가 유입이 가능하며, 분산 정책의 수용지(Destination)로서 적합함.
        """
    else: # 혼잡, 매우혼잡
        tone_guide = """
        [Diagnosis]: 수용 한계 초과 (Over Capacity).
        [Implication]: 데이터상 추가 유입 시 혼잡도 임계치를 넘음. 분산 정책의 대상지(Source)로 분류되어야 함.
        """

    msg = f"""
    [Target]: {spot_name} ({year})
    [Type]: {section_type} Analysis
    [Status]: {congestion_stage}
    [Ranking]: {ranking_info}
    [Data]: {data_summary}
    
    [Instruction]
    You are a strict Data Analyst evaluating urban data.
    Do NOT use greetings (Hello, etc). Start analysis directly.
    Do NOT propose marketing strategies or vague improvements.
    Use polite Korean (Honorifics, ~입니다/합니다).
    
    {tone_guide}
    
    1. **Quantify**: Use the ranking info (Top X%) to define the spot's relative density clearly.
    2. **Analyze**: Interpret the volatility (standard deviation/peaks) and seasonality patterns in depth.
    3. **Conclude**: Diagnose the 'Capacity' status strictly based on data.
    """
    
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(model="gpt-4o-mini", 
            messages=[{"role": "system", "content": AI_SYSTEM_PROMPT}, {"role": "user", "content": msg}], 
            temperature=0.3) 
        return response.choices[0].message.content, "#0F172A"
    except Exception as e: return str(e), "#000"

def get_ranking_dict(df, spot_name, year):
    y_df = df[df['날짜'].dt.year == year]
    if y_df.empty: return None
    rank_list = [{'spot': spot, 'val': get_smart_active_mean(y_df, spot)[0]} for spot in y_df['관광지명'].unique()]
    rank_df = pd.DataFrame(rank_list).sort_values(by='val', ascending=False).reset_index(drop=True)
    try:
        rank = rank_df[rank_df['spot'] == spot_name].index[0] + 1
        return {"rank": rank, "total": len(rank_df), "top_percent": (rank/len(rank_df))*100}
    except: return None

def style_chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#111", family="Pretendard"),
        margin=dict(l=0, r=0, t=20, b=20),
        hovermode="x unified",
        xaxis=dict(showgrid=False, showline=True, linecolor="#000", linewidth=1),
        yaxis=dict(showgrid=True, gridcolor="#EEE", zeroline=False),
    )
    return fig

# -----------------------------------------------------------------------------
# 5. UI 및 세션
# -----------------------------------------------------------------------------
if 'selected_spot' not in st.session_state: st.session_state['selected_spot'] = None
if 'sel_year' not in st.session_state: st.session_state['sel_year'] = 2024
if 'sel_month' not in st.session_state: st.session_state['sel_month'] = 1
if 'sim_sub_tab' not in st.session_state: st.session_state['sim_sub_tab'] = "이미지 유사도"
if 'weighted_result' not in st.session_state: st.session_state['weighted_result'] = None
if 'cross_result' not in st.session_state: st.session_state['cross_result'] = None 
if 'analysis_results' not in st.session_state: 
    st.session_state['analysis_results'] = {'trend': {}, 'hourly': {}, 'forecast': {}, 'sim_strat': {}, 'sim_img': {}, 'spot_info': {}, 'visual_rank1': {}, 'weighted': {}}

with st.sidebar:
    st.markdown('<h3 style="color:white; margin-bottom:30px; font-weight:900; letter-spacing:1px; padding-left:10px;">SLA PROJECT</h3>', unsafe_allow_html=True)
    if not main_df.empty:
        gu_list = sorted(main_df['행정구'].unique())
        for gu in gu_list:
            with st.expander(gu, expanded=False):
                spots = sorted(main_df[main_df['행정구'] == gu]['관광지명'].unique())
                for spot in spots:
                    btn_kind = "primary" if st.session_state['selected_spot'] == spot else "secondary"
                    # [수정] 탭 초기화 로직
                    if st.button(spot, key=f"btn_{gu}_{spot}", type=btn_kind):
                        st.session_state['selected_spot'] = spot
                        st.session_state['sel_month'] = 1 
                        st.session_state['sim_sub_tab'] = "이미지 유사도" 
                        st.session_state['weighted_result'] = None 
                        st.session_state['cross_result'] = None 
                        st.rerun()
    else: st.error("Data Load Failed")

if not st.session_state['selected_spot']:
    # [수정] 메인 화면 레이아웃 (좌: 텍스트 / 우: 이미지)
    col_text, col_img = st.columns([1, 1.3], gap="large", vertical_alignment="center")
    with col_text:
        st.markdown("""
        <div style="text-align: left; margin-left: 0px;">
            <div style="font-family:'Pretendard'; font-size:100px; font-weight:900; line-height:0.85; letter-spacing:-4px; color:#000;">SLA</div>
            <div style="font-family:'Pretendard'; font-size:100px; font-weight:900; line-height:0.85; letter-spacing:-4px; color:#000;">PROJECT</div>
            <div style="font-family:'Pretendard'; font-size:18px; font-weight:400; color:#666; margin-top:30px; letter-spacing:12px; margin-left:8px;">SPOT - LINEAR - AREA</div>
            <div style="margin-top:50px; border-left:4px solid black; padding-left:20px; color:#444;">
                <b>Sustainable Location Analysis</b><br>
                Solving Overtourism through Spatial Strategy.<br>
                "Connect the Dots, Create the Flow."
            </div>
        </div>""", unsafe_allow_html=True) 
    with col_img:
        st.markdown("""
        <div style="display:flex; justify-content: flex-end; align-items:center;">
            <img src="https://images.unsplash.com/photo-1558591710-4b4a1ae0f04d?q=80&w=1000&auto=format&fit=crop" 
                 style="width:90%; max-width:550px; filter:grayscale(100%) contrast(1.2); border:1px solid #E5E5E5; box-shadow: 0 10px 25px rgba(0,0,0,0.1);">
        </div>""", unsafe_allow_html=True)

else:
    spot_name = st.session_state["selected_spot"]
    st.title(spot_name)
    st.markdown(" ") 

    if spot_name not in st.session_state['analysis_results']['spot_info']:
        st.session_state['analysis_results']['spot_info'][spot_name] = generate_spot_info_ai(spot_name)
    
    with st.expander(f"ℹ️ ABOUT {spot_name}", expanded=True):
        ic1, ic2 = st.columns([1, 2])
        with ic1:
            img_path = os.path.join(BASE_DIR,"images", f"{spot_name}.jpg")
            if os.path.exists(img_path): 
                st.image(img_path, use_container_width=True) 
            else: st.markdown("<div style='background:#F4F4F5; height:200px; display:flex; justify-content:center; align-items:center; color:#999;'>NO IMAGE</div>", unsafe_allow_html=True)
        with ic2:
            st.markdown(f"<div style='line-height:1.6; color:#333;'>{st.session_state['analysis_results']['spot_info'][spot_name]}</div>", unsafe_allow_html=True)

    # [수정] 탭 키 제거
    tab1, tab2 = st.tabs(["⚫ CROWD ANALYSIS (혼잡도)", "⚪ SIMILARITY & DISPERSION (유사도)"])

    # TAB 1: 혼잡도
    with tab1:
        spot_data = main_df[main_df['관광지명'] == spot_name].copy()
        # [수정] 연도별 필터 적용된 배지 계산
        _, current_stage = get_active_time_stats(main_df, spot_name, st.session_state['sel_year'])
        st.markdown("<h3 style='font-size:1.5rem; font-weight:900; margin-bottom:15px; margin-top:20px; text-align:center;'>⚫ YEARLY TREND ANALYSIS</h3>", unsafe_allow_html=True)
        # [수정] 캡션 가운데 정렬
        st.markdown("<div class='center-caption'>선택한 연도의 월별 평균 혼잡도 추이입니다.</div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns([1, 6])
        with c1:
            st.markdown("**YEAR**")
            for y in [2023, 2024]:
                btn_type = "primary" if st.session_state['sel_year']==y else "secondary"
                if st.button(str(y), key=f"y_{y}", type=btn_type, use_container_width=True):
                    st.session_state['sel_year'] = y
                    st.rerun()
        with c2:
            y_df = spot_data[spot_data['날짜'].dt.year == st.session_state['sel_year']]
            if not y_df.empty:
                monthly_stats = []
                for m in range(1, 13):
                    m_data = y_df[y_df['날짜'].dt.month == m]
                    if not m_data.empty:
                        val, _ = get_smart_active_mean(m_data, spot_name)
                        monthly_stats.append({'month': m, 'val': val})
                
                if monthly_stats:
                    y_chart_df = pd.DataFrame(monthly_stats)
                    fig = px.line(y_chart_df, x='month', y='val', markers=True)
                    fig.update_traces(line_color='#000000', line_width=3)
                    fig.update_xaxes(tickmode='linear', tick0=1, dtick=1)
                    st.plotly_chart(style_chart(fig), use_container_width=True)
                    
                    cache_key = f"{spot_name}_{st.session_state['sel_year']}_trend"
                    if cache_key in st.session_state['analysis_results']['trend']:
                        content, color = st.session_state['analysis_results']['trend'][cache_key]
                        st.markdown(f"""<div class="ai-insight-box" style="border-left-color:{color};"><div class="ai-header" style="color:{color};">📉 DATA INSIGHT: {current_stage}</div>{content}</div>""", unsafe_allow_html=True)
                    else:
                        if st.button("📄 AI 심층 분석 보고서 생성 (Click)", key="btn_trend", use_container_width=True, type="primary"):
                            with st.spinner("🔄 AI 심층분석 중..."):
                                ranking = get_ranking_info(main_df, spot_name, st.session_state['sel_year'])
                                summary = y_chart_df.to_string(index=False)
                                res, color = generate_section_analysis("trend", spot_name, st.session_state['sel_year'], summary, current_stage, ranking)
                                st.session_state['analysis_results']['trend'][cache_key] = (res, color)
                                st.rerun()
                else: st.info("해당 연도 데이터 없음")
            else: st.info("해당 연도 데이터 없음")

        st.markdown("---")
        st.markdown("<h3 style='font-size:1.5rem; font-weight:900; margin-bottom:15px; text-align:center;'>⚫ MONTHLY&HOURLY TREND ANALYSIS</h3>", unsafe_allow_html=True)
        # [수정] 캡션 가운데 정렬
        st.markdown("<div class='center-caption'>선택한 월의 시간대별 평균 혼잡도 추이입니다.</div>", unsafe_allow_html=True)
        
        st.markdown(f"**MONTH ({st.session_state['sel_year']})**")
        m_rows = [st.columns(6), st.columns(6)]
        for i in range(12):
            btn_type = "primary" if st.session_state['sel_month']==i+1 else "secondary"
            if m_rows[0 if i<6 else 1][i%6].button(f"{i+1}", key=f"m_{i+1}", type=btn_type, use_container_width=True):
                st.session_state['sel_month'] = i+1
                st.rerun()
        st.markdown(" ") 
        m_df = spot_data[(spot_data['날짜'].dt.year == st.session_state['sel_year']) & (spot_data['날짜'].dt.month == st.session_state['sel_month'])]
        if not m_df.empty:
            h_df = m_df.groupby(['시간대_int', '시간대'])['실질_㎡당_방문객수'].mean().reset_index().sort_values('시간대_int')
            fig_h = px.area(h_df, x='시간대', y='실질_㎡당_방문객수')
            fig_h.update_traces(line_color='#666', fillcolor='rgba(0,0,0,0.1)')
            st.plotly_chart(style_chart(fig_h), use_container_width=True)
            
            key_h = f"hourly_{spot_name}_{st.session_state['sel_month']}"
            if key_h in st.session_state['analysis_results']['hourly']:
                content, color = st.session_state['analysis_results']['hourly'][key_h]
                st.markdown(f"""<div class="ai-insight-box" style="border-left-color:{color}; padding:15px; margin-top:10px;"><div class="ai-header" style="color:{color}; font-size:0.9rem;">⏳ TIME ANALYSIS</div>{content}</div>""", unsafe_allow_html=True)
            else:
                if st.button("📄 AI 심층 분석 보고서 생성 (Click)", key="btn_hourly", use_container_width=True, type="primary"):
                    with st.spinner("🔄 AI 심층분석 중..."):
                        ranking = get_ranking_info(main_df, spot_name, st.session_state['sel_year'])
                        res, color = generate_section_analysis("hourly", spot_name, st.session_state['sel_year'], h_df.to_string(), current_stage, ranking)
                        st.session_state['analysis_results']['hourly'][key_h] = (res, color)
                        st.rerun()
        else: st.write("해당 월 데이터 없음")

        st.markdown("---")
        st.markdown("<h3 style='font-size:1.5rem; font-weight:900; margin-bottom:15px; text-align:center;'>⚫ 2025 FUTURE FORECAST</h3>", unsafe_allow_html=True)
        st.markdown("<div class='center-caption'>머신러닝 모델(Prophet)이 예측한 2025년 월별 혼잡도 추이입니다.</div>", unsafe_allow_html=True)
        
        if not forecast_df.empty:
            f_spot = forecast_df[forecast_df['관광지명'] == spot_name]
            f_25 = f_spot[(f_spot['ds'] >= '2025-01-01') & (f_spot['ds'] <= '2025-12-31')]
            if not f_25.empty:
                mean_val = f_25['yhat'].mean()
                active_pred_val = f_25[f_25['yhat'] >= mean_val]['yhat'].mean()
                pred_stage = classify_density(active_pred_val)
                fig_f = px.line(f_25, x='ds', y='yhat')
                fig_f.update_traces(line_color='#000000', line_dash='dot')
                st.plotly_chart(style_chart(fig_f), use_container_width=True)
                key_f = f"forecast_{spot_name}"
                if key_f in st.session_state['analysis_results']['forecast']:
                    content, color = st.session_state['analysis_results']['forecast'][key_f]
                    st.markdown(f"""<div class="ai-insight-box" style="border-left-color:{color}; padding:15px; margin-top:10px;"><div class="ai-header" style="color:{color}; font-size:0.9rem;">📈 PREDICTIVE ANALYTICS</div>{content}</div>""", unsafe_allow_html=True)
                else:
                    if st.button("📄 AI 심층 분석 보고서 생성 (Click)", key="btn_forecast", use_container_width=True, type="primary"):
                        with st.spinner("🔄 AI 심층분석 중..."):
                            res, color = generate_section_analysis("forecast", spot_name, 2025, f_25.head().to_string(), pred_stage)
                            st.session_state['analysis_results']['forecast'][key_f] = (res, color)
                            st.rerun()
        else: st.write("예측 데이터 없음")
# TAB 2: 유사도 분석
    with tab2:
        st.markdown(" ") 
        
        # [필수] AI 분석 결과 저장을 위한 세션 상태 초기화
        if 'analysis_results' not in st.session_state:
            st.session_state['analysis_results'] = {'sim_img': {}, 'sim_strat': {}, 'weighted': {}}
        # 하위 키 안전 장치
        if 'sim_img' not in st.session_state['analysis_results']: st.session_state['analysis_results']['sim_img'] = {}
        if 'sim_strat' not in st.session_state['analysis_results']: st.session_state['analysis_results']['sim_strat'] = {}
        if 'weighted' not in st.session_state['analysis_results']: st.session_state['analysis_results']['weighted'] = {}

        tab_list = ["이미지 유사도", "텍스트 유사도", "종합 유사도", "Cross-Category"]
        cols = st.columns(4)
        for i, t_name in enumerate(tab_list):
            btn_style = "primary" if st.session_state['sim_sub_tab'] == t_name else "secondary"
            if cols[i].button(t_name, key=f"sub_t_{i}", use_container_width=True, type=btn_style):
                st.session_state['sim_sub_tab'] = t_name
                st.rerun()
        
        st.markdown("---")
        current_sub = st.session_state['sim_sub_tab']
        # [수정] 연도별 필터 적용된 배지 계산
        _, source_cong = get_active_time_stats(main_df, spot_name, 2024)

        if current_sub == "이미지 유사도":
            st.markdown(f"<h4 style='text-align:center;'>Visual Similarity Analysis</h4>", unsafe_allow_html=True)
            st.markdown("<div class='center-caption'>딥러닝으로 분석한 시각적 유사도 순위입니다.</div>", unsafe_allow_html=True)
            
            df_img = load_data_smart(FILE_CONFIG["IMG_RANK_DATA"])
            row = df_img[df_img['대상_관광지'] == spot_name] if not df_img.empty else pd.DataFrame()
            
            # [NEW] 전체 데이터의 1순위 평균 점수 계산 (비교 분석용)
            avg_top1_score = 0.0
            if not df_img.empty and '1순위' in df_img.columns:
                try:
                    # '관광지명(0.8123)' 형태에서 점수만 추출하여 평균 계산
                    scores = []
                    for val in df_img['1순위'].astype(str):
                        match = re.search(r'\(([\d.-]+)\)', val)
                        if match: scores.append(float(match.group(1)))
                    if scores: avg_top1_score = sum(scores) / len(scores)
                except: pass

            if not row.empty:
                visual_candidates = []
                for i in range(1, 9):
                    col = f"{i}순위"
                    if col in row.columns:
                        val = str(row.iloc[0][col])
                        match = re.search(r'(.+)\(([\d.-]+)\)', val)
                        if match:
                            t_name = match.group(1).strip()
                            t_score = float(match.group(2))
                            _, t_cong = get_active_time_stats(main_df, t_name, 2024)
                            visual_candidates.append({'rank': i, 'name': t_name, 'score': t_score, 'congestion': t_cong})
                
                if visual_candidates:
                    top1 = visual_candidates[0]
                    
                    st.markdown(f"#### 🥇 PRIMARY ALTERNATIVE (Rank 1)")
                    
                    # [레이아웃] 좌측: 이미지/정보(1), 우측: AI 분석(1.2)
                    c1, c2 = st.columns([1, 1.2])
                    
                    # --- LEFT COLUMN: Image & Basic Info ---
                    with c1:
                        st.markdown(f"<div style='font-size:1.8rem; font-weight:800; line-height:1.2; margin-bottom:5px;'>{top1['name']}</div>", unsafe_allow_html=True)
                        cong_cls = "cong-bad" if top1['congestion'] in ['혼잡', '매우혼잡'] else ("cong-norm" if top1['congestion'] == '보통' else "cong-good")
                        
                        # 점수와 혼잡도 표시
                        st.markdown(f"""
                        <div style="margin-bottom:10px;">
                            <span class='congestion-badge {cong_cls}'>{top1['congestion']}</span>
                            <span style='font-family:monospace; font-weight:bold; color:#333; margin-left:5px;'>Sim: {top1['score']:.4f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        img_path = os.path.join(BASE_DIR,"images", f"{top1['name']}.jpg")
                        if os.path.exists(img_path): st.image(img_path, use_container_width=True)
                        else: st.markdown("<div style='background:#F4F4F5; height:200px; border-radius:8px;'></div>", unsafe_allow_html=True)

                    # --- RIGHT COLUMN: AI Auto Insight (자동 실행 / 데이터 중심) ---
                    with c2:
                        auto_key = f"{spot_name}_vis_auto_analysis"
                        
                        # [핵심 수정] 세션에 결과가 없으면 버튼 클릭 없이 자동 실행
                        if auto_key not in st.session_state['analysis_results']['sim_img']:
                            with st.spinner(f"📊 {top1['name']} 데이터 분석 중..."):
                                try:
                                    sim_diff = top1['score'] - avg_top1_score
                                    
                                    # [수정] 'name' 키 추가 (오류 해결) 및 데이터 중심 정보 구성
                                    data_info = {
                                        'name': spot_name,
                                        'target_name': top1['name'],
                                        'current_score': f"{top1['score']:.4f}",
                                        'average_benchmark': f"{avg_top1_score:.4f}", 
                                        'score_deviation': f"{sim_diff:+.4f}",
                                        'congestion_status': top1['congestion']
                                    }
                                    
                                    # anal_type을 '이미지_데이터분석'으로 전달
                                    res = generate_strategic_analysis(data_info, [], anal_type="이미지_데이터분석")
                                    st.session_state['analysis_results']['sim_img'][auto_key] = res
                                except Exception as e:
                                    st.session_state['analysis_results']['sim_img'][auto_key] = f"분석 오류: {str(e)}"
                        
                        # 결과 표시
                        if auto_key in st.session_state['analysis_results']['sim_img']:
                            st.markdown(f"""
                            <div class="ai-insight-box" style="height:100%; min-height:300px;">
                                <div class="ai-header">📉 DATA ANALYSIS</div>
                                {st.session_state['analysis_results']['sim_img'][auto_key]}
                            </div>""", unsafe_allow_html=True)

                    st.markdown("---")
                    
                    # 2, 3순위 (기존 코드 유지)
                    r2_cols = st.columns(2)
                    for idx, cand in enumerate(visual_candidates[1:3]): 
                        cong_cls = "cong-bad" if cand['congestion'] in ['혼잡', '매우혼잡'] else ("cong-norm" if cand['congestion'] == '보통' else "cong-good")
                        medal = "🥈" if cand['rank'] == 2 else "🥉"
                        label = "SECONDARY ALTERNATIVE" if cand['rank'] == 2 else "THIRD ALTERNATIVE"
                        
                        with r2_cols[idx]:
                            st.markdown(f"<div style='font-weight:700; margin-bottom:5px; font-size:1.8rem;'>{medal} {label} (Rank {cand['rank']})</div>", unsafe_allow_html=True)
                            i_path = os.path.join(BASE_DIR,"images", f"{cand['name']}.jpg")
                            if os.path.exists(i_path): st.image(i_path, use_container_width=True)
                            else: st.markdown("<div style='background:#EEE; height:150px; display:flex; align-items:center; justify-content:center; color:#999;'>NO IMAGE</div>", unsafe_allow_html=True)
                            st.markdown(f"<div style='font-weight:800; font-size:1.1rem;'>{cand['name']}</div>", unsafe_allow_html=True)
                            st.markdown(f"<span class='congestion-badge {cong_cls}'>{cand['congestion']}</span>", unsafe_allow_html=True)
                    
                    if len(visual_candidates) > 3:
                        st.markdown("#### OTHER CANDIDATES")
                        cols = st.columns(len(visual_candidates[3:]))
                        for idx, cand in enumerate(visual_candidates[3:]):
                            cong_cls = "cong-bad" if cand['congestion'] in ['혼잡', '매우혼잡'] else ("cong-norm" if cand['congestion'] == '보통' else "cong-good")
                            with cols[idx]:
                                st.markdown(f"""<div class="rank-card-mini"><div style="font-size:0.7rem; color:#666;">RANK {cand['rank']}</div><div style="font-weight:700; font-size:0.85rem; margin:5px 0;">{cand['name']}</div><span class="congestion-badge {cong_cls}" style="font-size:0.6rem;">{cand['congestion']}</span></div>""", unsafe_allow_html=True)
                    
                    # [NEW] 하단 종합 AI 심층 분석 버튼
                    st.markdown("---")
                    
                    total_key = f"{spot_name}_vis_total_report"
                    
                    if total_key in st.session_state['analysis_results']['sim_img']:
                        st.markdown(f"""<div class="ai-insight-box"><div class="ai-header">🧠 VISUAL DEEP DIVE</div>{st.session_state['analysis_results']['sim_img'][total_key]}</div>""", unsafe_allow_html=True)

                    if st.button("📄 AI 심층 분석 보고서 생성 (Click)", key="btn_sim_img_total", use_container_width=True, type="primary"):
                        with st.spinner("🔄 시각 데이터 종합 분석 중..."):
                            try:
                                summary_info = {
                                    'name': spot_name,
                                    'avg_score': avg_top1_score,
                                    'candidate_count': len(visual_candidates)
                                }
                                # 전체 후보군(visual_candidates)을 넘겨서 종합 분석
                                res = generate_strategic_analysis(summary_info, visual_candidates, anal_type="이미지_종합분석")
                                st.session_state['analysis_results']['sim_img'][total_key] = res
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")

                else: st.info("유사도 데이터 없음")
            else: st.warning("이미지 분석 데이터 없음")

        elif current_sub == "텍스트 유사도":
            st.markdown(f"<h4 style='text-align:center;'>Contextual Similarity Analysis</h4>", unsafe_allow_html=True)
            st.markdown("<div class='center-caption'>딥러닝으로 분석한 리뷰 유사도 순위입니다.</div>", unsafe_allow_html=True)
            
            nouns, adjs = get_spot_keywords(spot_name)
            if nouns or adjs:
                st.markdown(f"""<div class="keyword-box"><div class="keyword-header">IDENTITY OF {spot_name}</div><div style="margin-bottom:8px;"><span style="font-size:0.8rem; font-weight:700; margin-right:10px;">VIBE (감성):</span>{' '.join([f"<span class='meta-tag tag-common'>#{k}</span>" for k in adjs[:5]])}</div><div><span style="font-size:0.8rem; font-weight:700; margin-right:10px;">FEATURE (시설):</span>{' '.join([f"<span class='meta-tag tag-common'>#{k}</span>" for k in nouns[:5]])}</div></div>""", unsafe_allow_html=True)
            else: st.info("키워드 데이터가 없습니다.")
            df_rev = load_data_smart(FILE_CONFIG["REVIEW_SIM_DATA"])
            if not df_rev.empty:
                if '관광지명' in df_rev.columns: df_rev['관광지명'] = df_rev['관광지명'].ffill()
                targets = df_rev[df_rev['관광지명'] == spot_name].head(5)
                if not targets.empty:
                    text_candidates = []
                    df_emo = load_data_smart(FILE_CONFIG["SENTIMENT_DATA"])
                    df_key = load_data_smart(FILE_CONFIG["FEATURE_DATA"])
                    for idx, row in enumerate(targets.iterrows(), 1):
                        _, data = row
                        target = data['리뷰 유사 관광지']
                        score = data['리뷰유사도']
                        _, t_cong = get_active_time_stats(main_df, target, 2024)
                        common_vibe, unique_s_vibe, unique_t_vibe = [], [], []
                        common_feat, unique_s_feat, unique_t_feat = [], [], []
                        if not df_emo.empty:
                            e = df_emo[(df_emo['기준_관광지']==spot_name) & (df_emo['비교_대상']==target)]
                            if not e.empty:
                                try: common_vibe = [k.strip() for k in str(e.iloc[0]['공통_키워드']).split(',') if k.strip()][:3]
                                except: pass
                                try: unique_s_vibe = [k.strip() for k in str(e.iloc[0]['기준지_고유_키워드']).split(',') if k.strip()][:3]
                                except: pass
                                try: unique_t_vibe = [k.strip() for k in str(e.iloc[0]['비교지_고유_키워드']).split(',') if k.strip()][:3]
                                except: pass
                        if not df_key.empty:
                            k = df_key[(df_key['기준_관광지']==spot_name) & (df_key['비교_대상']==target)]
                            if not k.empty:
                                try: common_feat = [k.strip() for k in str(k.iloc[0]['엣지_공통_키워드']).split(',') if k.strip()][:3]
                                except: pass
                                try: unique_s_feat = [k.strip() for k in str(k.iloc[0]['기준지_고유']).split(',') if k.strip()][:3]
                                except: pass
                                try: unique_t_feat = [k.strip() for k in str(k.iloc[0]['비교지_고유']).split(',') if k.strip()][:3]
                                except: pass
                        text_candidates.append({'rank': idx, 'name': target, 'score': score, 'congestion': t_cong, 'vibe_com': common_vibe, 'vibe_uniq_s': unique_s_vibe, 'vibe_uniq_t': unique_t_vibe, 'feat_com': common_feat, 'feat_uniq_s': unique_s_feat, 'feat_uniq_t': unique_t_feat})
                    def make_tags(tags, cls):
                        if not tags: return "<span style='color:#ccc; font-size:0.8rem;'>-</span>"
                        return ' '.join([f"<span class='meta-tag {cls}'>#{t}</span>" for t in tags])
                    for item in text_candidates:
                        cong_cls = "cong-bad" if item['congestion'] in ['혼잡', '매우혼잡'] else ("cong-norm" if item['congestion'] == '보통' else "cong-good")
                        st.markdown(f"""<div class="sim-card"><div class="sim-header-row" style="display:flex; justify-content:space-between;"><div><span class="sim-rank-badge">RANK {item['rank']}</span><span class="congestion-badge {cong_cls}">{item['congestion']}</span></div><span class="sim-score">Sim: {item['score']:.4f}</span></div><div class="sim-title" style="margin-bottom:15px; margin-top:10px;">{item['name']}</div><div class="keyword-grid"><div><div class="keyword-col-header">COMMON (공통)</div>{make_tags(item['vibe_com'], 'tag-common')}<br>{make_tags(item['feat_com'], 'tag-common')}</div><div><div class="keyword-col-header">ONLY {spot_name}</div>{make_tags(item['vibe_uniq_s'], 'tag-unique-source')}<br>{make_tags(item['feat_uniq_s'], 'tag-unique-source')}</div><div><div class="keyword-col-header">ONLY {item['name']}</div>{make_tags(item['vibe_uniq_t'], 'tag-unique-target')}<br>{make_tags(item['feat_uniq_t'], 'tag-unique-target')}</div></div></div>""", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    # [수정] 텍스트 유사도 AI 작동 보장
                    if spot_name in st.session_state['analysis_results']['sim_strat']:
                        st.markdown(f"""<div class="ai-insight-box"><div class="ai-header">🧠 CONTEXT DATA INSIGHT</div>{st.session_state['analysis_results']['sim_strat'][spot_name]}</div>""", unsafe_allow_html=True)
                    
                    if st.button("📄 AI 심층 분석 보고서 생성 (Click)", key="btn_sim_strat", use_container_width=True, type="primary"):
                        with st.spinner("🔄 AI 심층분석 중..."):
                            try:
                                source_info = {'name': spot_name, 'congestion': source_cong}
                                res = generate_strategic_analysis(source_info, text_candidates, anal_type="리뷰(Context)")
                                st.session_state['analysis_results']['sim_strat'][spot_name] = res
                                st.rerun()
                            except Exception as e:
                                st.error(f"분석 중 오류 발생: {str(e)}")
                else: st.info("유사도 데이터 없음")
            else: st.warning("파일 없음")

        # 종합 유사도 및 Cross-Category는 기존 로직 유지 (안전장치만 추가됨)
        elif current_sub == "종합 유사도":
            st.markdown(f"<h4 style='text-align:center;'>Weighted Integrated Similarity</h4>", unsafe_allow_html=True)
            st.markdown("<div class='center-caption'>3가지 속성(시각, 감성, 특성)에 가중치를 부여하여 최적의 대체지를 도출합니다.</div>", unsafe_allow_html=True)
            
            st.info("이미지(Visual), 감성(Sentiment), 특성(Feature) 데이터를 사용자가 설정한 가중치로 결합합니다.")
            c1, c2, c3 = st.columns(3)
            w_vis = c1.number_input("📸 Visual Weight (시각)", min_value=0, max_value=100, value=50, step=10, key="w_v_num")
            w_sen = c2.number_input("💬 Sentiment Weight (감성)", min_value=0, max_value=100, value=30, step=10, key="w_s_num")
            w_fea = c3.number_input("🏟️ Feature Weight (특성)", min_value=0, max_value=100, value=20, step=10, key="w_f_num")
            st.markdown(" ")
            
            if st.button("🔍 결과 분석 및 순위 산출 (Click)", type="primary", use_container_width=True):
                with st.spinner("데이터 분석 중..."):
                    total_w = w_vis + w_sen + w_fea
                    if total_w == 0: total_w = 1 
                    if not df_vis_scaled.empty:
                        curr_v = df_vis_scaled[df_vis_scaled['기준_관광지'] == spot_name].drop_duplicates('비교_대상').set_index('비교_대상')['VIS_SCALED']
                        curr_s = pd.Series()
                        if not df_sen_scaled.empty: curr_s = df_sen_scaled[df_sen_scaled['기준_관광지'] == spot_name].drop_duplicates('비교_대상').set_index('비교_대상')['SEN_SCALED']
                        curr_f = pd.Series()
                        if not df_fea_scaled.empty: curr_f = df_fea_scaled[df_fea_scaled['기준_관광지'] == spot_name].drop_duplicates('비교_대상').set_index('비교_대상')['FEA_SCALED']
                        merged = pd.concat([curr_v, curr_s, curr_f], axis=1).fillna(0)
                        merged.columns = ['VIS_SCALED', 'SEN_SCALED', 'FEA_SCALED'] 
                        merged['FINAL_SCORE'] = ((merged['VIS_SCALED'] * w_vis) + (merged['SEN_SCALED'] * w_sen) + (merged['FEA_SCALED'] * w_fea)) / total_w
                        st.session_state['weighted_result'] = merged[merged.index != spot_name].sort_values(by='FINAL_SCORE', ascending=False).head(5)
                    else: st.warning("데이터가 부족하여 계산할 수 없습니다.")

            if st.session_state['weighted_result'] is not None:
                res_df = st.session_state['weighted_result']
                st.markdown("---")
                st.markdown(f"<h4 style='text-align:center;'>🏆 TOP 5 WEIGHTED RECOMMENDATIONS</h4>", unsafe_allow_html=True)
                for rank, (cand_name, row) in enumerate(res_df.iterrows(), 1):
                    _, c_cong = get_active_time_stats(main_df, cand_name, 2024)
                    c_cong_cls = "cong-bad" if c_cong in ['혼잡', '매우혼잡'] else ("cong-norm" if c_cong == '보통' else "cong-good")
                    st.markdown(f"""<div class="sim-card" style="padding: 20px;"><div style="display:flex; justify-content:space-between; align-items:center;"><div><span class="sim-rank-badge" style="background:#0F172A;">#{rank}</span><span style="font-size:1.2rem; font-weight:800; margin-right:10px;">{cand_name}</span><span class="congestion-badge {c_cong_cls}">{c_cong}</span></div><div style="text-align:right;"><div style="font-size:1.3rem; font-weight:900; color:#0F172A;">{row['FINAL_SCORE']:.4f}</div><div style="font-size:0.75rem; color:#666;">WEIGHTED SCORE</div></div></div><div style="margin-top:15px; background:#F8FAFC; padding:10px; border-radius:8px; display:flex; gap:15px;"><div style="flex:1; text-align:center;"><div style="font-size:0.7rem; color:#64748B;">VISUAL ({w_vis}%)</div><div style="font-weight:700;">{row['VIS_SCALED']:.2f}</div></div><div style="flex:1; text-align:center; border-left:1px solid #E2E8F0;"><div style="font-size:0.7rem; color:#64748B;">SENTIMENT ({w_sen}%)</div><div style="font-weight:700;">{row['SEN_SCALED']:.2f}</div></div><div style="flex:1; text-align:center; border-left:1px solid #E2E8F0;"><div style="font-size:0.7rem; color:#64748B;">FEATURE ({w_fea}%)</div><div style="font-weight:700;">{row['FEA_SCALED']:.2f}</div></div></div></div>""", unsafe_allow_html=True)
                top_cand = res_df.iloc[0]
                cand_info = {'name': res_df.index[0], 'raw_v': top_cand['VIS_SCALED'], 'raw_s': top_cand['SEN_SCALED'], 'raw_f': top_cand['FEA_SCALED']}
                st.markdown("---")
                if spot_name in st.session_state['analysis_results']['weighted']:
                      st.markdown(f"""<div class="ai-insight-box"><div class="ai-header">⚖️ WEIGHTED INSIGHT</div>{st.session_state['analysis_results']['weighted'][spot_name]}</div>""", unsafe_allow_html=True)
                if st.button("📄 AI 가중치 결과 분석 (Click)", key="btn_weighted_ai", type="primary", use_container_width=True):
                    with st.spinner("가중치 기반 분석 중..."):
                        res = generate_weighted_insight(spot_name, cand_info, [w_vis, w_sen, w_fea])
                        st.session_state['analysis_results']['weighted'][spot_name] = res
                        st.rerun()

        elif current_sub == "Cross-Category":
            st.markdown(f"<h4 style='text-align:center;'>Cross-Category Analysis (Genre-Breaking)</h4>", unsafe_allow_html=True)
            st.markdown("<div class='center-caption'>동일 카테고리의 평균 유사도보다 높은 점수를 가진(3가지 기준 중 3개 모두 충족), '다른 카테고리'의 관광지 리스트입니다.</div>", unsafe_allow_html=True)
            
            with st.spinner("다차원 교차 분석 중..."):
                if not df_vis_scaled.empty:
                    curr_v = df_vis_scaled[df_vis_scaled['기준_관광지'] == spot_name].drop_duplicates('비교_대상').set_index('비교_대상')['VIS_SCALED']
                    curr_s = pd.Series()
                    if not df_sen_scaled.empty: curr_s = df_sen_scaled[df_sen_scaled['기준_관광지'] == spot_name].drop_duplicates('비교_대상').set_index('비교_대상')['SEN_SCALED']
                    curr_f = pd.Series()
                    if not df_fea_scaled.empty: curr_f = df_fea_scaled[df_fea_scaled['기준_관광지'] == spot_name].drop_duplicates('비교_대상').set_index('비교_대상')['FEA_SCALED']
                    
                    merged = pd.concat([curr_v, curr_s, curr_f], axis=1).fillna(0)
                    merged.columns = ['VIS_SCALED', 'SEN_SCALED', 'FEA_SCALED']
                    merged['FINAL_SCORE'] = ((merged['VIS_SCALED'] * 50) + (merged['SEN_SCALED'] * 30) + (merged['FEA_SCALED'] * 20)) / 100
                    
                    source_cat = get_spot_category(spot_name)
                    merged['CATEGORY'] = merged.index.map(get_spot_category)
                    
                    same_cat_group = merged[merged['CATEGORY'] == source_cat]
                    
                    if not same_cat_group.empty:
                        avg_vis = same_cat_group[same_cat_group['VIS_SCALED'] > 0]['VIS_SCALED'].mean()
                        avg_sen = same_cat_group[same_cat_group['SEN_SCALED'] > 0]['SEN_SCALED'].mean()
                        avg_fea = same_cat_group[same_cat_group['FEA_SCALED'] > 0]['FEA_SCALED'].mean()
                        if np.isnan(avg_vis): avg_vis = 0
                        if np.isnan(avg_sen): avg_sen = 0
                        if np.isnan(avg_fea): avg_fea = 0
                    else:
                        avg_vis, avg_sen, avg_fea = 0, 0, 0
                    
                    def check_all_pass(row):
                        mult = 1.0 
                        return (row['VIS_SCALED'] > avg_vis * mult) and \
                               (row['SEN_SCALED'] > avg_sen * mult) and \
                               (row['FEA_SCALED'] > avg_fea * mult)

                    candidates = merged[
                        (merged.index != spot_name) & 
                        (merged['CATEGORY'] != source_cat) & 
                        (merged['CATEGORY'] != '기타')
                    ]
                    
                    filtered = candidates[candidates.apply(check_all_pass, axis=1)]
                    
                    st.session_state['cross_result'] = filtered.sort_values(by='FINAL_SCORE', ascending=False)
                    st.session_state['source_cat'] = source_cat
                    st.session_state['debug_avg'] = (avg_vis, avg_sen, avg_fea)
                else: st.warning("데이터 부족")

            if st.session_state['cross_result'] is not None:
                res_df = st.session_state['cross_result']
                src_cat = st.session_state.get('source_cat', 'UNKNOWN')
                avgs = st.session_state.get('debug_avg', (0,0,0))
                mult = 1.0
                
                st.markdown(f"""
                <div style="padding:15px; background:#f8fafc; border-radius:8px; border:1px solid #e2e8f0; margin-bottom:20px;">
                    <div style="font-weight:700; color:#0f172a; margin-bottom:5px;">📊 ANALYSIS CONTEXT</div>
                    <div style="font-size:0.85rem; color:#475569;">
                        현재 카테고리: <b>{src_cat}</b><br>
                        통과 기준: <b>3개 지표 모두 카테고리 평균 이상 (All Pass)</b><br>
                        <hr style="margin:8px 0; border-color:#e2e8f0;">
                        <b>[카테고리 평균 점수]</b><br>
                        📸 시각: {avgs[0]:.3f} / 💬 감성: {avgs[1]:.3f} / 🏟️ 특성: {avgs[2]:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                if not res_df.empty:
                    st.success(f"✅ 총 {len(res_df)}곳의 검증된 대체지가 발견되었습니다.")
                    for rank, (cand_name, row) in enumerate(res_df.iterrows(), 1):
                        _, c_cong = get_active_time_stats(main_df, cand_name, 2024)
                        c_cong_cls = "cong-bad" if c_cong in ['혼잡', '매우혼잡'] else ("cong-norm" if c_cong == '보통' else "cong-good")
                        tgt_cat = row['CATEGORY']
                        
                        v_pass = "✅ Pass" if row['VIS_SCALED'] > avgs[0]*mult else "❌"
                        s_pass = "✅ Pass" if row['SEN_SCALED'] > avgs[1]*mult else "❌"
                        f_pass = "✅ Pass" if row['FEA_SCALED'] > avgs[2]*mult else "❌"
                        
                        st.markdown(f"""
                        <div class="sim-card" style="border-left: 5px solid #0369A1; background:#F0F9FF; margin-bottom:15px;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div>
                                    <span style="font-size:0.8rem; background:#FFF; padding:2px 8px; border:1px solid #DDD; border-radius:4px; font-weight:700; color:#333; margin-right:5px;">{tgt_cat}</span>
                                    <span style="font-size:1.3rem; font-weight:800;">{cand_name}</span>
                                </div>
                                <div style="text-align:right;">
                                    <div style="font-size:1.2rem; font-weight:900; color:#0F172A;">{row['FINAL_SCORE']:.2f}</div>
                                    <span class="congestion-badge {c_cong_cls}">{c_cong}</span>
                                </div>
                            </div>
                            <div style="margin-top:10px; font-size:0.8rem; color:#555; background:#fff; padding:10px; border-radius:6px; border:1px solid #e5e7eb;">
                                <strong>📊 상세 지표 (평균 대비)</strong><br>
                                📸 시각: <b>{row['VIS_SCALED']:.3f}</b> {v_pass} <span style="color:#999;">(Avg: {avgs[0]:.3f})</span><br>
                                💬 감성: <b>{row['SEN_SCALED']:.3f}</b> {s_pass} <span style="color:#999;">(Avg: {avgs[1]:.3f})</span><br>
                                🏟️ 특성: <b>{row['FEA_SCALED']:.3f}</b> {f_pass} <span style="color:#999;">(Avg: {avgs[2]:.3f})</span>
                            </div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.warning(f"조건을 충족하는 cross-category 대체지가 없습니다.")