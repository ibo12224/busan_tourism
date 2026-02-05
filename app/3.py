import streamlit as st
import pandas as pd

# 1. 페이지 설정
st.set_page_config(layout="wide", page_title="Busan Tourism Analysis")

# 2. CSS: 버튼 미니멀라이즈 및 여백 최적화
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* 사이드바 설정 */
    [data-testid="stSidebarNav"] {display: none;}
    .st-emotion-cache-1avcm0n {padding-top: 0rem !important;}
    .st-emotion-cache-6q9sum {padding-top: 2rem !important; background-color: #F8FAFC;}

    /* 경로 및 Back 버튼 정렬 */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .breadcrumb {
        font-size: 1.1rem;
        font-weight: 700;
        color: #64748B;
        letter-spacing: -0.02em;
    }
    .breadcrumb span { color: #0F172A; font-weight: 800; }

    /* 섹션 타이틀 */
    .section-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #94A3B8;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    /* 리포트 진입 버튼 (20% 영역에 맞는 세로형 디자인) */
    div.stButton > button[key="open_report"] {
        background-color: #FFFFFF;
        color: #475569;
        border: 1px solid #E2E8F0;
        height: 200px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    div.stButton > button[key="open_report"]:hover {
        border-color: #0F172A;
        color: #0F172A;
        background-color: #F1F5F9;
    }

    /* Back 버튼 (매우 작고 깔끔하게) */
    div.stButton > button[key="back_to_main"] {
        background-color: transparent;
        border: 1px solid #E2E8F0;
        color: #64748B;
        padding: 2px 10px;
        font-size: 0.75rem;
        border-radius: 4px;
    }
    div.stButton > button[key="back_to_main"]:hover {
        border-color: #64748B;
        color: #0F172A;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. 데이터 로드
@st.cache_data
def load_data():
    file_path = "부산_관광지명.xlsx"
    try:
        df = pd.read_excel(file_path)
        df.columns = [col.strip() for col in df.columns]
        first_col = df.columns[0]
        df[first_col] = df[first_col].ffill()
        return df, first_col
    except:
        return pd.DataFrame(columns=['지역구명', '관광지명']), '지역구명'

df, region_col = load_data()

# 세션 상태 관리
if 'report_mode' not in st.session_state: st.session_state['report_mode'] = False

# --- 사이드바 ---
with st.sidebar:
    st.markdown('<p style="font-size:0.9rem; font-weight:800; color:#1E293B; margin-bottom:1.5rem; padding-left:10px; border-left:3px solid #334155;">REGIONS</p>', unsafe_allow_html=True)
    gu_list = sorted([str(g) for g in df[region_col].dropna().unique()])
    for gu in gu_list:
        if st.button(gu, width='stretch', key=f"gu_{gu}", type="primary" if st.session_state.get('selected_gu') == gu else "secondary"):
            st.session_state['selected_gu'] = gu
            st.session_state['report_mode'] = False

# --- 메인 화면 ---
if st.session_state.get('selected_gu'):
    current_gu = st.session_state['selected_gu']
    
    # 상단 내비게이션 영역
    col_nav_left, col_nav_right = st.columns([0.8, 0.2])
    with col_nav_left:
        st.markdown(f'<p class="breadcrumb">Busan Tourism Analysis / <span>{current_gu}</span></p>', unsafe_allow_html=True)
    with col_nav_right:
        # 리포트 모드일 때만 아주 작은 Back 버튼 표시
        if st.session_state['report_mode']:
            if st.button("✕ Close Report", key="back_to_main", width='stretch'):
                st.session_state['report_mode'] = False
                st.rerun()

    spot_col = df.columns[1]
    target_spots = df[df[region_col] == current_gu][spot_col].dropna().unique().tolist()
    selected_spot = st.selectbox("Select Destination", ["Select a spot"] + target_spots, label_visibility="collapsed")
    
    # 관광지 변경 시 모드 리셋
    if 'last_spot' not in st.session_state or st.session_state['last_spot'] != selected_spot:
        st.session_state['report_mode'] = False
        st.session_state['last_spot'] = selected_spot

    if selected_spot != "Select a spot":
        if not st.session_state['report_mode']:
            # [A 모드: 8:2 비율]
            col_main, col_side = st.columns([0.8, 0.2], gap="medium")
            with col_main:
                st.markdown('<p class="section-title">Congestion Analysis</p>', unsafe_allow_html=True)
                st.info(f"📊 {selected_spot}의 혼잡도 지표 시각화 영역 (80% 너비)")
                # 혼잡도 관련 대형 차트가 들어올 자리

            with col_side:
                st.markdown('<p class="section-title">Report</p>', unsafe_allow_html=True)
                if st.button(f"Analysis\n\nReport", key="open_report", width='stretch'):
                    st.session_state['report_mode'] = True
                    st.rerun()
        else:
            # [B 모드: Full 리포트]
            st.markdown('<p class="section-title" style="margin-top:1rem;">Deep Similarity Report</p>', unsafe_allow_html=True)
            tab1, tab2, tab3, tab4 = st.tabs(["OVERALL", "VISUAL", "EMOTION", "FEATURE"])
            with tab1:
                st.write(f"**{selected_spot}** 유사도 분석 결과")
else:
    st.title("Main Dashboard")
    st.write("Select a region in the sidebar.")