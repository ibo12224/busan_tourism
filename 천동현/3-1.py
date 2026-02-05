import streamlit as st
import pandas as pd
import re
import os

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
# ---------------------------------

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
            # [B 모드: Full 리포트] 이 부분에 스케치 레이아웃을 넣습니다.
            st.markdown('<p class="section-title" style="margin-top:1rem;">Deep Similarity Report</p>', unsafe_allow_html=True)
            # 탭 순서와 내용 변경
            tab1, tab2, tab3, tab4 = st.tabs(["대표사진", "이미지유사관광지", "EMOTION", "FEATURE"])
            
            # -----------------------------------------------------------
            # OVERALL 탭: 선택된 관광지 자체의 이미지만 표시 (사용자 요청)
            # -----------------------------------------------------------
            with tab1:
                st.markdown(f'### **{selected_spot}**')
                image_filename = f"{selected_spot}.jpg"
                image_path = os.path.join("images", image_filename)

                if os.path.exists(image_path):
                    st.image(image_path, width=700)
                else:
                    st.warning(f"'{selected_spot}' 이미지를 찾을 수 없습니다: {image_path}")

            # -----------------------------------------------------------
            # VISUAL 탭: 이미지 유사도 분석 결과 표시 (사용자 요청)
            # -----------------------------------------------------------
            with tab2:
                st.markdown(f'### {selected_spot} 이미지 유사 관광지 분석')
                
                # B 모드 안에서 유사도 데이터 파일 로드
                file_path_similarity = "유사도.xlsx" 
                top5_list = []
                try:
                    df_sim_local = pd.read_excel(file_path_similarity)
                    df_sim_local.columns = [col.strip() for col in df_sim_local.columns]

                    filtered_df = df_sim_local[df_sim_local['관광지명'].ffill() == selected_spot]

                    if not filtered_df.empty:
                        for sim_str in filtered_df['이미지유사도'].iloc[:5]:
                            match = re.search(r'([가-힣\s\w]+)\(([\d\.]+)\)', str(sim_str))
                            if match:
                                name = match.group(1).strip()
                                score = float(match.group(2))
                                top5_list.append((name, score))

                except FileNotFoundError:
                    st.error(f"'{file_path_similarity}' 파일을 찾을 수 없습니다.")
                except Exception as e:
                    st.error(f"데이터 처리 중 오류 발생: {e}")

                # 1. 메인 TOP1 및 사이드 리스트 (8:2 비율) 레이아웃
                col_main_img, col_side_list = st.columns([0.8, 0.2], gap="medium")

                with col_main_img:
                    if top5_list:
                        top1_name = top5_list[0][0]
                        top1_score = top5_list[0][1]
                        image_filename = f"{top1_name}.jpg" 
                        image_path = os.path.join("images", image_filename)
                        
                        st.markdown(f"#### **Top 1.{top1_name}** (Score: {top1_score:.4f})")
                        
                        if os.path.exists(image_path):
                            st.image(image_path, width=600)
                        else:
                            st.image("https://via.placeholder.com", width=600) 
                            st.warning(f"이미지 파일을 찾을 수 없습니다: {image_path}")
                    else:
                         st.markdown(f"#### {selected_spot} (데이터 없음)")
                
                with col_side_list:
                    st.markdown('**이미지 유사 관광지**') 
                    st.markdown('Top 2~5') # 줄바꿈 적용
                    if len(top5_list) >= 2:
                        for i, (name, score) in enumerate(top5_list[1:5], start=2):
                            st.write(f"{i}. {name} ({score:.4f})")
                    else:
                        st.write("추가 유사도 데이터가 없습니다.")

                st.markdown('<hr/>', unsafe_allow_html=True)
            # -----------------------------------------------------------
            # EMOTION 탭: 이미지에서 요청한 3단 레이아웃을 표시합니다.
            # -----------------------------------------------------------
            with tab3:
                st.markdown(f"### **{selected_spot}** 감성/기능 분석 결과")
                
                # 파일 경로 정의
                file_path_similarity = "유사도.xlsx" 
                file_path_functional = "기능유사도.xlsx"
                file_path_functional2 = "감상유사도.xlsx" # 감성 파일 경로
                
                # 3개의 컬럼 레이아웃으로 분할
                col_review, col_function, col_emotion_list = st.columns(3, gap="medium") 
                
                # -----------------------------------------------
                with col_review:
                    st.markdown('**유사 관광지 리뷰 TOP 5**')
                    try:
                        df_review_local = pd.read_excel(file_path_similarity)
                        df_review_local.columns = [col.strip() for col in df_review_local.columns]
                        filtered_review_df = df_review_local[df_review_local['관광지명'].ffill() == selected_spot]

                        if not filtered_review_df.empty:
                            for sim_name, sim_score in zip(filtered_review_df['리뷰 유사 관광지'].iloc[:5], filtered_review_df['리뷰유사도'].iloc[:5]):
                                if pd.notna(sim_name) and pd.notna(sim_score):
                                    st.write(f"{sim_name} ({sim_score:.4f})")
                    except:
                         st.write("리뷰 유사도 데이터가 없습니다.")
                # -----------------------------------------------
                
                # -----------------------------------------------
                with col_function:
                    st.markdown('**기능적 TOP 5**')
                    try:
                        df_func_local = pd.read_excel(file_path_functional)
                        df_func_local.columns = [col.strip() for col in df_func_local.columns]
                        filtered_func_df = df_func_local[df_func_local['기준_관광지'].ffill() == selected_spot] 

                        if not filtered_func_df.empty:
                            for i in range(min(5, len(filtered_func_df))):
                                target_spot = filtered_func_df['비교_대상'].iloc[i]
                                similarity_score = filtered_func_df['최종_유사도'].iloc[i]
                                keywords_str = filtered_func_df['엣지_공통_키워드'].iloc[i]

                                if pd.notna(target_spot) and pd.notna(similarity_score):
                                    st.write(f"{i+1}. **{target_spot}**({similarity_score:.4f})")
                                
                                if pd.notna(keywords_str):
                                    keywords = [k.strip() for k in str(keywords_str).split(',')][:3]
                                    keywords_display = ', '.join(keywords)
                                    st.markdown(f"<span style='font-size:0.75rem;'>ㄴ {keywords_display}</span>", unsafe_allow_html=True)
                    except:
                        st.write("기능적 키워드 데이터가 없습니다.")
                # -----------------------------------------------
                
                # -----------------------------------------------
                with col_emotion_list:
                    st.markdown('**감성적 TOP 5**')
                    try:                       
                        df_emotion_local = pd.read_excel(file_path_functional2)
                        df_emotion_local.columns = [col.strip() for col in df_emotion_local.columns]
                        filtered_emotion_df = df_emotion_local[df_emotion_local['기준_관광지'].ffill() == selected_spot] 

                        if not filtered_emotion_df.empty:
                            for i in range(min(5, len(filtered_emotion_df))):
                                target_spot = filtered_emotion_df['비교_대상'].iloc[i]
                                similarity_score = filtered_emotion_df['SBERT_유사도(가중적용)'].iloc[i]
                                keywords_str = filtered_emotion_df['기준지_고유_키워드'].iloc[i]
                                
                                if pd.notna(target_spot) and pd.notna(similarity_score):
                                    st.write(f"{i+1}. **{target_spot}**({similarity_score:.4f})")
                                
                                if pd.notna(keywords_str):
                                    keywords = [k.strip() for k in str(keywords_str).split(',')][:3]
                                    keywords_display = ', '.join(keywords)
                                    st.markdown(f"<span style='font-size:0.75rem;'>ㄴ {keywords_display}</span>", unsafe_allow_html=True)
                    except:
                        st.write(f"감성적 데이터 처리 중 오류 발생: {file_path_functional2} 파일을 찾을 수 없습니다.")
                # -----------------------------------------------

                # -----------------------------------------------
            # -----------------------------------------------------------
            # FEATURE 탭: 기존 더미 데이터 유지
            # -----------------------------------------------------------
            with tab4:
                st.write(f"**{selected_spot}** 특징 분석 결과")

else:
    st.title("Main Dashboard")
    st.write("Select a region in the sidebar.")
