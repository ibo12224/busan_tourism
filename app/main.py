import streamlit as st
import pandas as pd
import re
import os
import unicodedata

# ==========================================
# 1. 경로 및 환경 설정 (절대 경로 유지)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(BASE_DIR, "images")

# 상세 파일 경로 변수 (이 경로들을 아래 로직에서 참조합니다)
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
    if not spot_name or spot_name == "Select a spot":
        return False
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
    div.stButton > button[key="back_to_main"] { background-color: transparent; border: 1px solid #E2E8F0; color: #64748B; padding: 2px 10px; font-size: 0.75rem; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 4. 데이터 로드 (캐싱 및 ffill 적용)
# ==========================================
@st.cache_data
def load_data(path):
    try:
        df = pd.read_excel(path, engine='openpyxl')
        df.columns = [col.strip() for col in df.columns]
        df[df.columns[0]] = df[df.columns[0]].ffill()
        return df
    except Exception as e:
        return pd.DataFrame()

df_main = load_data(FILE_PATH_MAIN)
region_col = df_main.columns[0] if not df_main.empty else "지역구명"

# 세션 상태 관리
if 'report_mode' not in st.session_state: st.session_state['report_mode'] = False

# ==========================================
# 5. 메인 레이아웃 및 로직
# ==========================================
with st.sidebar:
    st.markdown('<p style="font-weight:800; color:#1E293B; margin-left:10px;">REGIONS</p>', unsafe_allow_html=True)
    if not df_main.empty:
        gu_list = sorted([str(g) for g in df_main[region_col].unique()])
        for gu in gu_list:
            if st.button(gu, width='stretch', key=f"gu_{gu}"):
                st.session_state['selected_gu'] = gu
                st.session_state['report_mode'] = False

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
    spot_col = df_main.columns[1]
    target_spots = df_main[df_main[region_col] == current_gu][spot_col].dropna().unique().tolist()
    selected_spot = st.selectbox("Destination", ["Select a spot"] + target_spots, label_visibility="collapsed")
    
    if 'last_spot' not in st.session_state or st.session_state['last_spot'] != selected_spot:
        st.session_state['report_mode'] = False
        st.session_state['last_spot'] = selected_spot

    if selected_spot != "Select a spot":
        if not st.session_state['report_mode']:
            # [A 모드: 대시보드 레이아웃 유지]
            col_main, col_side = st.columns([0.8, 0.2], gap="medium")
            with col_main:
                st.markdown('<p class="section-title">Analysis Overview</p>', unsafe_allow_html=True)
                st.info(f"📍 {selected_spot}의 지표 분석 데이터 영역")
            with col_side:
                st.markdown('<p class="section-title">Detail Report</p>', unsafe_allow_html=True)
                if st.button(f"Open\n\nDeep Analysis", key="open_report", width='stretch'):
                    st.session_state['report_mode'] = True
                    st.rerun()
        else:
            # [B 모드: 상세 리포트 탭 레이아웃 유지]
            tab1, tab2, tab3, tab4 = st.tabs(["대표사진", "이미지유사관광지", "EMOTION", "FEATURE"])
            
            with tab1:
                st.markdown(f"### {selected_spot}")
                if not display_tourism_image(selected_spot, 700):
                    st.warning("이미지 파일을 찾을 수 없습니다.")

            with tab2:
                st.markdown(f"### {selected_spot} 이미지 유사 분석")
                df_sim = load_data(FILE_PATH_SIMILARITY)
                top5 = []
                if not df_sim.empty:
                    filtered = df_sim[df_sim['관광지명'] == selected_spot]
                    if not filtered.empty:
                        for sim_str in filtered['이미지유사도'].iloc[:5]:
                            m = re.search(r'([가-힣\s\w]+)\(([\d\.]+)\)', str(sim_str))
                            if m: top5.append((m.group(1).strip(), float(m.group(2))))

                col_img, col_list = st.columns([0.8, 0.2])
                with col_img:
                    if top5:
                        st.markdown(f"**Top 1: {top5[0][0]}** (Score: {top5[0][1]:.4f})")
                        display_tourism_image(top5[0][0], 600)
                with col_list:
                    st.markdown("**유사 리스트**")
                    for i, (n, s) in enumerate(top5[1:], 2):
                        st.write(f"{i}. {n} ({s:.4f})")

            with tab3:
                # 3단 레이아웃 (리뷰/기능/감성)
                col_rev, col_func, col_emo = st.columns(3)
                
                with col_rev:
                    st.markdown("**리뷰 유사 관광지**")
                    df_sim = load_data(FILE_PATH_SIMILARITY)
                    filtered = df_sim[df_sim['관광지명'] == selected_spot]
                    if not filtered.empty:
                        for n, s in zip(filtered['리뷰 유사 관광지'].iloc[:5], filtered['리뷰유사도'].iloc[:5]):
                            if pd.notna(n): st.write(f"{n} ({s:.4f})")

                with col_func:
                    st.markdown("**기능적 TOP 5**")
                    df_f = load_data(FILE_PATH_FUNCTIONAL)
                    filtered = df_f[df_f['기준_관광지'] == selected_spot]
                    for i in range(min(5, len(filtered))):
                        row = filtered.iloc[i]
                        st.write(f"{i+1}. {row['비교_대상']}")
                        if '엣지_공통_키워드' in row:
                            st.caption(f"ㄴ {row['엣지_공통_키워드']}")

                with col_emo:
                    st.markdown("**감성적 TOP 5**")
                    df_e = load_data(FILE_PATH_EMOTION)
                    filtered = df_e[df_e['기준_관광지'] == selected_spot]
                    for i in range(min(5, len(filtered))):
                        row = filtered.iloc[i]
                        st.write(f"{i+1}. {row['비교_대상']}")
                        if '기준지_고유_키워드' in row:
                            st.caption(f"ㄴ {row['기준지_고유_키워드']}")

            with tab4:
                st.write("상세 특징 분석 데이터 준비 중")

else:
    st.title("Busan Tourism Dashboard")
    st.write("사이드바에서 지역을 선택해 주세요.")