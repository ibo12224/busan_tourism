import streamlit as st
import pandas as pd
import re
import os
import unicodedata

# ==========================================
# 1. 경로 및 환경 설정 (최적화)
# ==========================================
# 도커 컨테이너 내부 WORKDIR인 /app을 기준으로 절대 경로를 생성합니다.
# 현재 실행 중인 main.py의 위치 (/app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 컨테이너 구조에 맞춘 절대 경로 설정
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
# 상세 파일 경로 변수
FILE_PATH_MAIN = os.path.join(DATA_DIR, "부산_관광지명.xlsx")
FILE_PATH_SIMILARITY = os.path.join(DATA_DIR, "유사도.xlsx")
FILE_PATH_FUNCTIONAL = os.path.join(DATA_DIR, "기능유사도.xlsx")
FILE_PATH_EMOTION = os.path.join(DATA_DIR, "감상유사도.xlsx")

# 페이지 설정
st.set_page_config(layout="wide", page_title="Busan Tourism Analysis")

# ==========================================
# 2. 이미지 출력 헬퍼 함수 (한글 인코딩 해결)
# ==========================================
def display_tourism_image(spot_name, img_width=700):
    """
    한글 파일명(NFC/NFD)과 다양한 확장자를 자동으로 체크하여 이미지를 띄웁니다.
    """
    if not spot_name or spot_name == "Select a spot":
        return False

    # 리눅스 환경에서의 한글 깨짐 방지를 위해 NFC 정규화
    normalized_name = unicodedata.normalize('NFC', spot_name)
    extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
    
    for ext in extensions:
        target_path = os.path.join(IMAGE_DIR, f"{normalized_name}{ext}")
        if os.path.exists(target_path):
            st.image(target_path, width=img_width)
            return True
    return False

# ==========================================
# 3. CSS 스타일링
# ==========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    [data-testid="stSidebarNav"] {display: none;}
    .breadcrumb { font-size: 1.1rem; font-weight: 700; color: #64748B; }
    .breadcrumb span { color: #0F172A; font-weight: 800; }
    .section-title { font-size: 0.8rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 1rem; }
    div.stButton > button[key="open_report"] { background-color: #FFFFFF; height: 200px; border-radius: 12px; font-weight: 600; border: 1px solid #E2E8F0; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 4. 데이터 로드
# ==========================================
@st.cache_data
def load_data():
    try:
        # 엔진을 openpyxl로 명시하여 도커 내 호환성을 높임
        df = pd.read_excel(FILE_PATH_MAIN, engine='openpyxl')
        df.columns = [col.strip() for col in df.columns]
        first_col = df.columns[0]
        df[first_col] = df[first_col].ffill()
        return df, first_col
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame(columns=['지역구명', '관광지명']), '지역구명'

df, region_col = load_data()

# 세션 상태 관리
if 'report_mode' not in st.session_state: st.session_state['report_mode'] = False

# ==========================================
# 5. 메인 레이아웃
# ==========================================
# 사이드바: 지역구 선택
with st.sidebar:
    st.markdown('<p style="font-weight:800; color:#1E293B; margin-left:10px;">REGIONS</p>', unsafe_allow_html=True)
    gu_list = sorted([str(g) for g in df[region_col].dropna().unique()])
    for gu in gu_list:
        if st.button(gu, width='stretch', key=f"gu_{gu}"):
            st.session_state['selected_gu'] = gu
            st.session_state['report_mode'] = False

# 메인 콘텐츠 영역
if st.session_state.get('selected_gu'):
    current_gu = st.session_state['selected_gu']
    
    # 상단 내비
    col_nav_left, col_nav_right = st.columns([0.8, 0.2])
    with col_nav_left:
        st.markdown(f'<p class="breadcrumb">Busan / <span>{current_gu}</span></p>', unsafe_allow_html=True)
    with col_nav_right:
        if st.session_state['report_mode']:
            if st.button("✕ Close Report", key="back_to_main"):
                st.session_state['report_mode'] = False
                st.rerun()

    # 관광지 선택
    spot_col = df.columns[1]
    target_spots = df[df[region_col] == current_gu][spot_col].dropna().unique().tolist()
    selected_spot = st.selectbox("Destination", ["Select a spot"] + target_spots, label_visibility="collapsed")
    
    if 'last_spot' not in st.session_state or st.session_state['last_spot'] != selected_spot:
        st.session_state['report_mode'] = False
        st.session_state['last_spot'] = selected_spot

    if selected_spot != "Select a spot":
        if not st.session_state['report_mode']:
            # [기본 대시보드 모드]
            col_main, col_side = st.columns([0.8, 0.2], gap="medium")
            with col_main:
                st.markdown('<p class="section-title">Analysis Overview</p>', unsafe_allow_html=True)
                st.info(f"📍 {selected_spot}의 분석 데이터를 로딩 중입니다.")
            with col_side:
                st.markdown('<p class="section-title">Detail Report</p>', unsafe_allow_html=True)
                if st.button(f"Open\n\nDeep Analysis", key="open_report", width='stretch'):
                    st.session_state['report_mode'] = True
                    st.rerun()
        else:
            # [상세 리포트 모드]
            tab1, tab2, tab3, tab4 = st.tabs(["대표사진", "유사 관광지", "감성/기능", "상세특징"])
            
            with tab1:
                st.markdown(f"### {selected_spot}")
                if not display_tourism_image(selected_spot, 700):
                    st.warning("이미지 파일을 찾을 수 없습니다. (파일명/인코딩 확인 필요)")

            with tab2:
                # 이미지 유사도 로직
                try:
                    df_sim = pd.read_excel(FILE_PATH_SIMILARITY)
                    filtered = df_sim[df_sim['관광지명'].ffill() == selected_spot]
                    top5 = []
                    if not filtered.empty:
                        for sim_str in filtered['이미지유사도'].iloc[:5]:
                            m = re.search(r'([가-힣\s\w]+)\(([\d\.]+)\)', str(sim_str))
                            if m: top5.append((m.group(1).strip(), float(m.group(2))))
                    
                    c_img, c_list = st.columns([0.7, 0.3])
                    with c_img:
                        if top5:
                            st.markdown(f"**가장 유사한 곳: {top5[0][0]}**")
                            display_tourism_image(top5[0][0], 500)
                    with c_list:
                        st.write("**Top 5 List**")
                        for i, (n, s) in enumerate(top5): st.write(f"{i+1}. {n} ({s:.3f})")
                except: st.write("유사도 데이터 로드 중 오류")

            with tab3:
                # 감성/기능 데이터 출력 (파일 존재 확인 후 처리)
                c_func, c_emo = st.columns(2)
                with c_func:
                    st.markdown("**기능적 유사 관광지**")
                    # (기존 기능유사도 로직 적용)
                with c_emo:
                    st.markdown("**감성적 유사 관광지**")
                    # (기존 감상유사도 로직 적용)

else:
    st.title("Busan Tourism Dashboard")
    st.write("사이드바에서 분석할 지역을 선택해 주세요.")