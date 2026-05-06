import os
import re
import numpy as np

import pandas as pd
import plotly.express as px
import streamlit as st

import config
from ai_service import (
    generate_spot_info_ai,
    generate_section_analysis,
    generate_strategic_analysis,
    generate_weighted_insight,
)
from analysis import (
    classify_density,
    get_active_time_stats,
    get_ranking_info,
    get_smart_active_mean,
    get_spot_category,
    get_spot_keywords,
    style_chart,
)
from data_loader import (
    get_global_top1_avg,
    get_scaled_data,
    load_all_data,
    load_data_smart,
)


def local_css(file_name):
    css_path = os.path.join(config.BASE_DIR, file_name)
    try:
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"{css_path} 파일을 찾을 수 없습니다.")


def initialize_session_state():
    if 'selected_spot' not in st.session_state:
        st.session_state['selected_spot'] = None
    if 'sel_year' not in st.session_state:
        st.session_state['sel_year'] = 2024
    if 'sel_month' not in st.session_state:
        st.session_state['sel_month'] = 1
    if 'sim_sub_tab' not in st.session_state:
        st.session_state['sim_sub_tab'] = "이미지 유사도"
    if 'weighted_result' not in st.session_state:
        st.session_state['weighted_result'] = None
    if 'cross_result' not in st.session_state:
        st.session_state['cross_result'] = None
    if 'analysis_results' not in st.session_state:
        st.session_state['analysis_results'] = {
            'trend': {},
            'hourly': {},
            'forecast': {},
            'sim_strat': {},
            'sim_img': {},
            'spot_info': {},
            'visual_rank1': {},
            'weighted': {},
        }


def run_app():
    st.set_page_config(layout="wide", page_title="SLA PROJECT", page_icon="⚫")
    local_css("style.css")

    main_df, forecast_df, noun_df, adj_df = load_all_data()
    df_vis_scaled, df_sen_scaled, df_fea_scaled = get_scaled_data()
    GLOBAL_TOP1_AVG = get_global_top1_avg()

    initialize_session_state()

    with st.sidebar:
        st.markdown(
            '<h3 style="color:white; margin-bottom:30px; font-weight:850; letter-spacing:1px; padding-left:10px;">SLA PROJECT</h3>',
            unsafe_allow_html=True,
        )
        if not main_df.empty:
            gu_list = sorted(main_df['행정구'].unique())
            for gu in gu_list:
                with st.expander(gu, expanded=False):
                    spots = sorted(main_df[main_df['행정구'] == gu]['관광지명'].unique())
                    for spot in spots:
                        btn_kind = "primary" if st.session_state['selected_spot'] == spot else "secondary"
                        if st.button(spot, key=f"btn_{gu}_{spot}", type=btn_kind):
                            st.session_state['selected_spot'] = spot
                            st.session_state['sel_month'] = 1
                            st.session_state['sim_sub_tab'] = "이미지 유사도"
                            st.session_state['weighted_result'] = None
                            st.session_state['cross_result'] = None
                            st.rerun()
        else:
            st.error("Data Load Failed")

    if not st.session_state['selected_spot']:
        col_text, col_img = st.columns([1, 1.3], gap="large", vertical_alignment="center")
        with col_text:
            st.markdown(
                """
                <div style="text-align: left; margin-left: 0px;">
                    <div style="font-family:'Pretendard'; font-size:90px; font-weight:900; line-height:0.85; letter-spacing:-4px; color:#000;">SLA</div>
                    <div style="font-family:'Pretendard'; font-size:90px; font-weight:900; line-height:0.85; letter-spacing:-4px; color:#000;">PROJECT</div>
                    <div style="font-family:'Pretendard'; font-size:18px; font-weight:400; color:#666; margin-top:30px; letter-spacing:12px; margin-left:8px;">SPOT - LINEAR - AREA</div>
                    <div style="margin-top:50px; border-left:4px solid black; padding-left:20px; color:#444;">
                        <b>Sustainable Location Analysis</b><br>
                        Solving Overtourism through Spatial Strategy.<br>
                        "Connect the Dots, Create the Flow."
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_img:
            st.markdown(
                """
                <div style="display:flex; justify-content: flex-end; align-items:center;">
                    <img src="https://images.unsplash.com/photo-1558591710-4b4a1ae0f04d?q=80&w=1000&auto=format&fit=crop"
                         style="width:90%; max-width:550px; filter:grayscale(100%) contrast(1.2); border:1px solid #E5E5E5; box-shadow: 0 10px 25px rgba(0,0,0,0.1);">
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:
        spot_name = st.session_state['selected_spot']
        st.title(spot_name)
        st.markdown(" ")

        if spot_name not in st.session_state['analysis_results']['spot_info']:
            st.session_state['analysis_results']['spot_info'][spot_name] = generate_spot_info_ai(spot_name)

        with st.expander(f"ℹ️ ABOUT {spot_name}", expanded=True):
            ic1, ic2 = st.columns([1, 2])
            with ic1:
                img_path = os.path.join(config.BASE_DIR, "images", f"{spot_name}.jpg")
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                else:
                    st.markdown(
                        "<div style='background:#F4F4F5; height:200px; display:flex; justify-content:center; align-items:center; color:#999;'>NO IMAGE</div>",
                        unsafe_allow_html=True,
                    )
            with ic2:
                st.markdown(
                    f"<div style='line-height:1.6; color:#333;'>{st.session_state['analysis_results']['spot_info'][spot_name]}</div>",
                    unsafe_allow_html=True,
                )

        tab1, tab2 = st.tabs(["⚫ CROWD ANALYSIS (혼잡도)", "⚪ SIMILARITY & DISPERSION (유사도)"])

        with tab1:
            spot_data = main_df[main_df['관광지명'] == spot_name].copy()
            _, current_stage = get_active_time_stats(main_df, spot_name, st.session_state['sel_year'])
            st.markdown(
                "<h3 style='font-size:1.5rem; font-weight:900; margin-bottom:15px; margin-top:20px; text-align:center;'>⚫ YEARLY TREND ANALYSIS</h3>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='center-caption'>선택한 연도의 월별 평균 혼잡도 추이입니다.</div>",
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns([1, 6])
            with c1:
                st.markdown("**YEAR**")
                for y in [2023, 2024]:
                    btn_type = "primary" if st.session_state['sel_year'] == y else "secondary"
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
                            st.markdown(
                                f"""<div class="ai-insight-box" style="border-left-color:{color};"><div class="ai-header" style="color:{color};">📉 DATA INSIGHT: {current_stage}</div>{content}</div>""",
                                unsafe_allow_html=True,
                            )
                        else:
                            if st.button("📄 AI 심층 분석 보고서 생성 (Click)", key="btn_trend", use_container_width=True, type="primary"):
                                with st.spinner("🔄 AI 심층분석 중..."):
                                    ranking = get_ranking_info(main_df, spot_name, st.session_state['sel_year'])
                                    summary = y_chart_df.to_string(index=False)
                                    res, color, *_ = generate_section_analysis("trend", spot_name, st.session_state['sel_year'], summary, current_stage, ranking)
                                    st.session_state['analysis_results']['trend'][cache_key] = (res, color)
                                    st.rerun()
                    else:
                        st.info("해당 연도 데이터 없음")
                else:
                    st.info("해당 연도 데이터 없음")

            st.markdown("---")
            st.markdown(
                "<h3 style='font-size:1.5rem; font-weight:900; margin-bottom:15px; text-align:center;'>⚫ MONTHLY&HOURLY TREND ANALYSIS</h3>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='center-caption'>선택한 월의 시간대별 평균 혼잡도 추이입니다.</div>",
                unsafe_allow_html=True,
            )

            st.markdown(f"**MONTH ({st.session_state['sel_year']})**")
            m_rows = [st.columns(6), st.columns(6)]
            for i in range(12):
                btn_type = "primary" if st.session_state['sel_month'] == i + 1 else "secondary"
                if m_rows[0 if i < 6 else 1][i % 6].button(f"{i+1}", key=f"m_{i+1}", type=btn_type, use_container_width=True):
                    st.session_state['sel_month'] = i + 1
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
                    st.markdown(
                        f"""<div class="ai-insight-box" style="border-left-color:{color}; padding:15px; margin-top:10px;"><div class="ai-header" style="color:{color}; font-size:0.9rem;">⏳ TIME ANALYSIS</div>{content}</div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    if st.button("📄 AI 심층 분석 보고서 생성 (Click)", key="btn_hourly", use_container_width=True, type="primary"):
                        with st.spinner("🔄 AI 심층분석 중..."):
                            ranking = get_ranking_info(main_df, spot_name, st.session_state['sel_year'])
                            res, color, *_ = generate_section_analysis("hourly", spot_name, st.session_state['sel_year'], h_df.to_string(), current_stage, ranking)
                            st.session_state['analysis_results']['hourly'][key_h] = (res, color)
                            st.rerun()
            else:
                st.write("해당 월 데이터 없음")

            st.markdown("---")
            st.markdown(
                "<h3 style='font-size:1.5rem; font-weight:900; margin-bottom:15px; text-align:center;'>⚫ 2025 FUTURE FORECAST</h3>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='center-caption'>머신러닝 모델(Prophet)이 예측한 2025년 월별 혼잡도 추이입니다.</div>",
                unsafe_allow_html=True,
            )

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
                        st.markdown(
                            f"""<div class="ai-insight-box" style="border-left-color:{color}; padding:15px; margin-top:10px;"><div class="ai-header" style="color:{color}; font-size:0.9rem;">📈 PREDICTIVE ANALYTICS</div>{content}</div>""",
                            unsafe_allow_html=True,
                        )
                    else:
                        if st.button("📄 AI 심층 분석 보고서 생성 (Click)", key="btn_forecast", use_container_width=True, type="primary"):
                            with st.spinner("🔄 AI 심층분석 중..."):
                                res, color, *_ = generate_section_analysis("forecast", spot_name, 2025, f_25.head().to_string(), pred_stage)
                                st.session_state['analysis_results']['forecast'][key_f] = (res, color)
                                st.rerun()
                else:
                    st.write("예측 데이터 없음")
            else:
                st.write("예측 데이터 없음")

            st.markdown(
                """
                <div class='notice-box'>
                    <span class='notice-title'>ℹ️ 안내</span>
                    혼잡도 계산은 상당부분 추정에 근거하고 있어 현실데이터와 차이가 있을 수 있습니다. <br>
                    보조지표로만 사용할 것을 권장하며 특히 <b>23년과 24년의 혼잡도를 직접 비교하는 것</b>은 권장하지 않습니다.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with tab2:
            st.markdown(" ")

            if 'analysis_results' not in st.session_state:
                st.session_state['analysis_results'] = {'sim_img': {}, 'sim_strat': {}, 'weighted': {}}
            if 'sim_img' not in st.session_state['analysis_results']:
                st.session_state['analysis_results']['sim_img'] = {}
            if 'sim_strat' not in st.session_state['analysis_results']:
                st.session_state['analysis_results']['sim_strat'] = {}
            if 'weighted' not in st.session_state['analysis_results']:
                st.session_state['analysis_results']['weighted'] = {}

            tab_list = ["이미지 유사도", "텍스트 유사도", "종합 유사도", "Cross-Category"]
            cols = st.columns(4)
            for i, t_name in enumerate(tab_list):
                btn_style = "primary" if st.session_state['sim_sub_tab'] == t_name else "secondary"
                if cols[i].button(t_name, key=f"sub_t_{i}", use_container_width=True, type=btn_style):
                    st.session_state['sim_sub_tab'] = t_name
                    st.rerun()

            st.markdown("---")
            current_sub = st.session_state['sim_sub_tab']
            _, source_cong = get_active_time_stats(main_df, spot_name, 2024)

            if current_sub == "이미지 유사도":
                st.markdown(f"<h4 style='text-align:center;'>Visual Similarity Analysis</h4>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='center-caption'>딥러닝으로 분석한 시각적 유사도 순위입니다.</div>",
                    unsafe_allow_html=True,
                )

                df_img = load_data_smart(config.FILE_CONFIG["IMG_RANK_DATA"])
                row = df_img[df_img['대상_관광지'] == spot_name] if not df_img.empty else pd.DataFrame()
                avg_top1_score = 0.0
                if not df_img.empty and '1순위' in df_img.columns:
                    try:
                        scores = []
                        for val in df_img['1순위'].astype(str):
                            match = re.search(r'\(([^\d.-]+)\)', val)
                            if match:
                                scores.append(float(match.group(1)))
                        if scores:
                            avg_top1_score = sum(scores) / len(scores)
                    except Exception:
                        pass

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
                        c1, c2 = st.columns([1, 1.2])
                        with c1:
                            st.markdown(
                                f"<div style='font-size:1.8rem; font-weight:800; line-height:1.2; margin-bottom:5px;'>{top1['name']}</div>",
                                unsafe_allow_html=True,
                            )
                            cong_cls = (
                                'cong-bad' if top1['congestion'] in ['혼잡', '매우혼잡'] else
                                ('cong-norm' if top1['congestion'] == '보통' else 'cong-good')
                            )
                            st.markdown(
                                f"""
                                <div style="margin-bottom:10px;">
                                    <span class='congestion-badge {cong_cls}'>{top1['congestion']}</span>
                                    <span style='font-family:monospace; font-weight:bold; color:#333; margin-left:5px;'>Sim: {top1['score']:.4f}</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            img_path = os.path.join(config.BASE_DIR, "images", f"{top1['name']}.jpg")
                            if os.path.exists(img_path):
                                st.image(img_path, use_container_width=True)
                            else:
                                st.markdown(
                                    "<div style='background:#F4F4F5; height:200px; border-radius:8px;'></div>",
                                    unsafe_allow_html=True,
                                )
                        with c2:
                            auto_key = f"{spot_name}_vis_auto_analysis"
                            if auto_key not in st.session_state['analysis_results']['sim_img']:
                                with st.spinner(f"📊 {top1['name']} 데이터 분석 중..."):
                                    sim_diff = top1['score'] - avg_top1_score
                                    data_info = {
                                        'name': top1['name'],
                                        'target_name': top1['name'],
                                        'current_score': f"{top1['score']:.4f}",
                                        'average_benchmark': f"{avg_top1_score:.4f}",
                                        'score_deviation': f"{sim_diff:+.4f}",
                                        'congestion_status': top1['congestion'],
                                    }
                                    try:
                                        res = generate_strategic_analysis(data_info, [], anal_type="이미지_데이터분석")
                                        st.session_state['analysis_results']['sim_img'][auto_key] = res
                                    except Exception as e:
                                        st.session_state['analysis_results']['sim_img'][auto_key] = f"분석 오류: {str(e)}"

                            if auto_key in st.session_state['analysis_results']['sim_img']:
                                st.markdown(
                                    f"""<div class="ai-insight-box" style="height:100%; min-height:300px;"><div class="ai-header">📉 DATA ANALYSIS</div>{st.session_state['analysis_results']['sim_img'][auto_key]}</div>""",
                                    unsafe_allow_html=True,
                                )

                        st.markdown("---")
                        r2_cols = st.columns(2)
                        for idx, cand in enumerate(visual_candidates[1:3]):
                            cong_cls = (
                                'cong-bad' if cand['congestion'] in ['혼잡', '매우혼잡'] else
                                ('cong-norm' if cand['congestion'] == '보통' else 'cong-good')
                            )
                            medal = "🥈" if cand['rank'] == 2 else "🥉"
                            label = "SECONDARY ALTERNATIVE" if cand['rank'] == 2 else "THIRD ALTERNATIVE"
                            with r2_cols[idx]:
                                st.markdown(
                                    f"<div style='font-weight:700; margin-bottom:5px; font-size:1.8rem;'>{medal} {label} (Rank {cand['rank']})</div>",
                                    unsafe_allow_html=True,
                                )
                                i_path = os.path.join(config.BASE_DIR, "images", f"{cand['name']}.jpg")
                                if os.path.exists(i_path):
                                    st.image(i_path, use_container_width=True)
                                else:
                                    st.markdown(
                                        "<div style='background:#EEE; height:150px; display:flex; align-items:center; justify-content:center; color:#999;'>NO IMAGE</div>",
                                        unsafe_allow_html=True,
                                    )
                                st.markdown(
                                    f"<div style='font-weight:800; font-size:1.1rem;'>{cand['name']}</div>",
                                    unsafe_allow_html=True,
                                )
                                st.markdown(
                                    f"<span class='congestion-badge {cong_cls}'>{cand['congestion']}</span>",
                                    unsafe_allow_html=True,
                                )

                        if len(visual_candidates) > 3:
                            st.markdown("#### OTHER CANDIDATES")
                            cols = st.columns(len(visual_candidates[3:]))
                            for idx, cand in enumerate(visual_candidates[3:]):
                                cong_cls = (
                                    'cong-bad' if cand['congestion'] in ['혼잡', '매우혼잡'] else
                                    ('cong-norm' if cand['congestion'] == '보통' else 'cong-good')
                                )
                                with cols[idx]:
                                    st.markdown(
                                        f"""<div class="rank-card-mini"><div style="font-size:0.7rem; color:#666;">RANK {cand['rank']}</div><div style="font-weight:700; font-size:0.85rem; margin:5px 0;">{cand['name']}</div><span class="congestion-badge {cong_cls}" style="font-size:0.6rem;">{cand['congestion']}</span></div>""",
                                        unsafe_allow_html=True,
                                    )

                        st.markdown("---")
                        total_key = f"{spot_name}_vis_total_report"
                        if total_key in st.session_state['analysis_results']['sim_img']:
                            st.markdown(
                                f"""<div class="ai-insight-box"><div class="ai-header">🧠 VISUAL DEEP DIVE</div>{st.session_state['analysis_results']['sim_img'][total_key]}</div>""",
                                unsafe_allow_html=True,
                            )

                        if st.button("📄 AI 심층 분석 보고서 생성 (Click)", key="btn_sim_img_total", use_container_width=True, type="primary"):
                            with st.spinner("🔄 시각 데이터 종합 분석 중..."):
                                try:
                                    summary_info = {
                                        'name': spot_name,
                                        'avg_score': avg_top1_score,
                                        'candidate_count': len(visual_candidates),
                                    }
                                    res = generate_strategic_analysis(summary_info, visual_candidates, anal_type="이미지_종합분석")
                                    st.session_state['analysis_results']['sim_img'][total_key] = res
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")

                    else:
                        st.info("유사도 데이터 없음")
                else:
                    st.warning("이미지 분석 데이터 없음")

            elif current_sub == "텍스트 유사도":
                st.markdown(f"<h4 style='text-align:center;'>Contextual Similarity Analysis</h4>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='center-caption'>딥러닝으로 분석한 리뷰 유사도 순위입니다.</div>",
                    unsafe_allow_html=True,
                )

                nouns, adjs = get_spot_keywords(spot_name, noun_df, adj_df, df_sen_scaled)
                if nouns or adjs:
                    st.markdown(
                        f"""<div class="keyword-box"><div class="keyword-header">IDENTITY OF {spot_name}</div><div style="margin-bottom:8px;"><span style="font-size:0.8rem; font-weight:700; margin-right:10px;">VIBE (감성):</span>{' '.join([f"<span class='meta-tag tag-common'>#{k}</span>" for k in adjs[:5]])}</div><div><span style="font-size:0.8rem; font-weight:700; margin-right:10px;">FEATURE (시설):</span>{' '.join([f"<span class='meta-tag tag-common'>#{k}</span>" for k in nouns[:5]])}</div></div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("키워드 데이터가 없습니다.")

                df_rev = load_data_smart(config.FILE_CONFIG["REVIEW_SIM_DATA"])
                if not df_rev.empty:
                    if '관광지명' in df_rev.columns:
                        df_rev['관광지명'] = df_rev['관광지명'].ffill()

                    targets = df_rev[df_rev['관광지명'] == spot_name].head(5)
                    if not targets.empty:
                        text_candidates = []
                        df_emo = load_data_smart(config.FILE_CONFIG["SENTIMENT_DATA"])
                        df_key = load_data_smart(config.FILE_CONFIG["FEATURE_DATA"])
                        for idx, row in enumerate(targets.iterrows(), 1):
                            _, data = row
                            target = data['리뷰 유사 관광지']
                            score = data['리뷰유사도']
                            _, t_cong = get_active_time_stats(main_df, target, 2024)
                            common_vibe, unique_s_vibe, unique_t_vibe = [], [], []
                            common_feat, unique_s_feat, unique_t_feat = [], [], []
                            if not df_emo.empty:
                                e = df_emo[(df_emo['기준_관광지'] == spot_name) & (df_emo['비교_대상'] == target)]
                                if not e.empty:
                                    try:
                                        common_vibe = [k.strip() for k in str(e.iloc[0]['공통_키워드']).split(',') if k.strip()][:3]
                                    except Exception:
                                        pass
                                    try:
                                        unique_s_vibe = [k.strip() for k in str(e.iloc[0]['기준지_고유_키워드']).split(',') if k.strip()][:3]
                                    except Exception:
                                        pass
                                    try:
                                        unique_t_vibe = [k.strip() for k in str(e.iloc[0]['비교지_고유_키워드']).split(',') if k.strip()][:3]
                                    except Exception:
                                        pass
                            if not df_key.empty:
                                k = df_key[(df_key['기준_관광지'] == spot_name) & (df_key['비교_대상'] == target)]
                                if not k.empty:
                                    try:
                                        common_feat = [x.strip() for x in str(k.iloc[0]['엣지_공통_키워드']).split(',') if x.strip()][:3]
                                    except Exception:
                                        pass
                                    try:
                                        unique_s_feat = [x.strip() for x in str(k.iloc[0]['기준지_고유']).split(',') if x.strip()][:3]
                                    except Exception:
                                        pass
                                    try:
                                        unique_t_feat = [x.strip() for x in str(k.iloc[0]['비교지_고유']).split(',') if x.strip()][:3]
                                    except Exception:
                                        pass
                            text_candidates.append({
                                'rank': idx,
                                'name': target,
                                'score': score,
                                'congestion': t_cong,
                                'vibe_com': common_vibe,
                                'vibe_uniq_s': unique_s_vibe,
                                'vibe_uniq_t': unique_t_vibe,
                                'feat_com': common_feat,
                                'feat_uniq_s': unique_s_feat,
                                'feat_uniq_t': unique_t_feat,
                            })

                        def make_tags(tags, cls):
                            if not tags:
                                return "<span style='color:#ccc; font-size:0.8rem;'>-</span>"
                            return ' '.join([f"<span class='meta-tag {cls}'>#{t}</span>" for t in tags])

                        for item in text_candidates:
                            cong_cls = (
                                'cong-bad' if item['congestion'] in ['혼잡', '매우혼잡'] else
                                ('cong-norm' if item['congestion'] == '보통' else 'cong-good')
                            )
                            st.markdown(
                                f"""
                                <div class="sim-card"><div class="sim-header-row" style="display:flex; justify-content:space-between;"><div><span class="sim-rank-badge">RANK {item['rank']}</span><span class="congestion-badge {cong_cls}">{item['congestion']}</span></div><span class="sim-score">Sim: {item['score']:.4f}</span></div><div class="sim-title" style="margin-bottom:15px; margin-top:10px;">{item['name']}</div><div class="keyword-grid"><div><div class="keyword-col-header">COMMON (공통)</div>{make_tags(item['vibe_com'], 'tag-common')}<br>{make_tags(item['feat_com'], 'tag-common')}</div><div><div class="keyword-col-header">ONLY {spot_name}</div>{make_tags(item['vibe_uniq_s'], 'tag-unique-source')}<br>{make_tags(item['feat_uniq_s'], 'tag-unique-source')}</div><div><div class="keyword-col-header">ONLY {item['name']}</div>{make_tags(item['vibe_uniq_t'], 'tag-unique-target')}<br>{make_tags(item['feat_uniq_t'], 'tag-unique-target')}</div></div></div>
                                """,
                                unsafe_allow_html=True,
                            )

                        st.markdown("---")
                        if spot_name in st.session_state['analysis_results']['sim_strat']:
                            st.markdown(
                                f"""<div class="ai-insight-box"><div class="ai-header">🧠 CONTEXT DATA INSIGHT</div>{st.session_state['analysis_results']['sim_strat'][spot_name]}</div>""",
                                unsafe_allow_html=True,
                            )
                        if st.button("📄 AI 심층 분석 보고서 생성 (Click)", key="btn_sim_strat", use_container_width=True, type="primary"):
                            with st.spinner("🔄 AI 심층분석 중..."):
                                try:
                                    source_info = {'name': spot_name, 'congestion': source_cong}
                                    res = generate_strategic_analysis(source_info, text_candidates, anal_type="리뷰(Context)")
                                    st.session_state['analysis_results']['sim_strat'][spot_name] = res
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"분석 중 오류 발생: {str(e)}")
                    else:
                        st.info("유사도 데이터 없음")
                else:
                    st.warning("파일 없음")

            elif current_sub == "종합 유사도":
                st.markdown(f"<h4 style='text-align:center;'>Weighted Integrated Similarity</h4>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='center-caption'>3가지 속성(시각, 감성, 특성)에 가중치를 부여하여 최적의 대체지를 도출합니다.</div>",
                    unsafe_allow_html=True,
                )
                st.info("이미지(Visual), 감성(Sentiment), 특성(Feature) 데이터를 사용자가 설정한 가중치로 결합합니다.")
                c1, c2, c3 = st.columns(3)
                w_vis = c1.number_input("📸 Visual Weight (시각)", min_value=0, max_value=100, value=50, step=10, key="w_v_num")
                w_sen = c2.number_input("💬 Sentiment Weight (감성)", min_value=0, max_value=100, value=30, step=10, key="w_s_num")
                w_fea = c3.number_input("🏟️ Feature Weight (특성)", min_value=0, max_value=100, value=20, step=10, key="w_f_num")
                st.markdown(" ")

                if st.button("🔍 결과 분석 및 순위 산출 (Click)", type="primary", use_container_width=True):
                    with st.spinner("데이터 분석 중..."):
                        total_w = w_vis + w_sen + w_fea
                        if total_w == 0:
                            total_w = 1
                        if not df_vis_scaled.empty:
                            curr_v = df_vis_scaled[df_vis_scaled['기준_관광지'] == spot_name].drop_duplicates('비교_대상').set_index('비교_대상')['VIS_SCALED']
                            curr_s = pd.Series(dtype=float)
                            if not df_sen_scaled.empty:
                                curr_s = df_sen_scaled[df_sen_scaled['기준_관광지'] == spot_name].drop_duplicates('비교_대상').set_index('비교_대상')['SEN_SCALED']
                            curr_f = pd.Series(dtype=float)
                            if not df_fea_scaled.empty:
                                curr_f = df_fea_scaled[df_fea_scaled['기준_관광지'] == spot_name].drop_duplicates('비교_대상').set_index('비교_대상')['FEA_SCALED']
                            merged = pd.concat([curr_v, curr_s, curr_f], axis=1).fillna(0)
                            merged.columns = ['VIS_SCALED', 'SEN_SCALED', 'FEA_SCALED']
                            merged['FINAL_SCORE'] = ((merged['VIS_SCALED'] * w_vis) + (merged['SEN_SCALED'] * w_sen) + (merged['FEA_SCALED'] * w_fea)) / total_w
                            st.session_state['weighted_result'] = merged[merged.index != spot_name].sort_values(by='FINAL_SCORE', ascending=False).head(5)
                        else:
                            st.warning("데이터가 부족하여 계산할 수 없습니다.")

                if st.session_state['weighted_result'] is not None:
                    res_df = st.session_state['weighted_result']
                    st.markdown("---")
                    st.markdown(f"<h4 style='text-align:center;'>🏆 TOP 5 WEIGHTED RECOMMENDATIONS</h4>", unsafe_allow_html=True)
                    for rank, (cand_name, row) in enumerate(res_df.iterrows(), 1):
                        _, c_cong = get_active_time_stats(main_df, cand_name, 2024)
                        c_cong_cls = (
                            'cong-bad' if c_cong in ['혼잡', '매우혼잡'] else
                            ('cong-norm' if c_cong == '보통' else 'cong-good')
                        )
                        st.markdown(
                            f"""<div class="sim-card" style="padding: 20px;"><div style="display:flex; justify-content:space-between; align-items:center;"><div><span class="sim-rank-badge" style="background:#0F172A;">#{rank}</span><span style="font-size:1.2rem; font-weight:800; margin-right:10px;">{cand_name}</span><span class="congestion-badge {c_cong_cls}">{c_cong}</span></div><div style="text-align:right;"><div style="font-size:1.3rem; font-weight:900; color:#0F172A;">{row['FINAL_SCORE']:.4f}</div><div style="font-size:0.75rem; color:#666;">WEIGHTED SCORE</div></div></div><div style="margin-top:15px; background:#F8FAFC; padding:10px; border-radius:8px; display:flex; gap:15px;"><div style="flex:1; text-align:center;"><div style="font-size:0.7rem; color:#64748B;">VISUAL ({w_vis}%)</div><div style="font-weight:700;">{row['VIS_SCALED']:.2f}</div></div><div style="flex:1; text-align:center; border-left:1px solid #E2E8F0;"><div style="font-size:0.7rem; color:#64748B;">SENTIMENT ({w_sen}%)</div><div style="font-weight:700;">{row['SEN_SCALED']:.2f}</div></div><div style="flex:1; text-align:center; border-left:1px solid #E2E8F0;"><div style="font-size:0.7rem; color:#64748B;">FEATURE ({w_fea}%)</div><div style="font-weight:700;">{row['FEA_SCALED']:.2f}</div></div></div></div>""",
                            unsafe_allow_html=True,
                        )
                    top_cand = res_df.iloc[0]
                    cand_info = {
                        'name': res_df.index[0],
                        'raw_v': top_cand['VIS_SCALED'],
                        'raw_s': top_cand['SEN_SCALED'],
                        'raw_f': top_cand['FEA_SCALED'],
                    }
                    st.markdown("---")
                    if spot_name in st.session_state['analysis_results']['weighted']:
                        st.markdown(
                            f"""<div class="ai-insight-box"><div class="ai-header">⚖️ WEIGHTED INSIGHT</div>{st.session_state['analysis_results']['weighted'][spot_name]}</div>""",
                            unsafe_allow_html=True,
                        )
                    if st.button("📄 AI 가중치 결과 분석 (Click)", key="btn_weighted_ai", type="primary", use_container_width=True):
                        with st.spinner("가중치 기반 분석 중..."):
                            res = generate_weighted_insight(spot_name, cand_info, [w_vis, w_sen, w_fea])
                            st.session_state['analysis_results']['weighted'][spot_name] = res
                            st.rerun()

            elif current_sub == "Cross-Category":
                st.markdown(f"<h4 style='text-align:center;'>Cross-Category Analysis (Genre-Breaking)</h4>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='center-caption'>동일 카테고리의 평균 유사도보다 높은 점수를 가진(3가지 기준 중 3개 모두 충족), '다른 카테고리'의 관광지 리스트입니다.</div>",
                    unsafe_allow_html=True,
                )
                with st.spinner("다차원 교차 분석 중..."):
                    if not df_vis_scaled.empty:
                        curr_v = df_vis_scaled[df_vis_scaled['기준_관광지'] == spot_name].drop_duplicates('비교_대상').set_index('비교_대상')['VIS_SCALED']
                        curr_s = pd.Series(dtype=float)
                        if not df_sen_scaled.empty:
                            curr_s = df_sen_scaled[df_sen_scaled['기준_관광지'] == spot_name].drop_duplicates('비교_대상').set_index('비교_대상')['SEN_SCALED']
                        curr_f = pd.Series(dtype=float)
                        if not df_fea_scaled.empty:
                            curr_f = df_fea_scaled[df_fea_scaled['기준_관광지'] == spot_name].drop_duplicates('비교_대상').set_index('비교_대상')['FEA_SCALED']

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
                            avg_vis = 0 if np.isnan(avg_vis) else avg_vis
                            avg_sen = 0 if np.isnan(avg_sen) else avg_sen
                            avg_fea = 0 if np.isnan(avg_fea) else avg_fea
                        else:
                            avg_vis, avg_sen, avg_fea = 0, 0, 0

                        def check_all_pass(row):
                            return (
                                (row['VIS_SCALED'] > avg_vis) and
                                (row['SEN_SCALED'] > avg_sen) and
                                (row['FEA_SCALED'] > avg_fea)
                            )

                        candidates = merged[(merged.index != spot_name) & (merged['CATEGORY'] != source_cat) & (merged['CATEGORY'] != '기타')]
                        filtered = candidates[candidates.apply(check_all_pass, axis=1)]
                        st.session_state['cross_result'] = filtered.sort_values(by='FINAL_SCORE', ascending=False)
                        st.session_state['source_cat'] = source_cat
                        st.session_state['debug_avg'] = (avg_vis, avg_sen, avg_fea)
                    else:
                        st.warning("데이터 부족")

                if st.session_state['cross_result'] is not None:
                    res_df = st.session_state['cross_result']
                    src_cat = st.session_state.get('source_cat', 'UNKNOWN')
                    avgs = st.session_state.get('debug_avg', (0, 0, 0))
                    mult = 1.0
                    st.markdown(
                        f"""
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
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")
                    if not res_df.empty:
                        st.success(f"✅ 총 {len(res_df)}곳의 검증된 대체지가 발견되었습니다.")
                        for rank, (cand_name, row) in enumerate(res_df.iterrows(), 1):
                            _, c_cong = get_active_time_stats(main_df, cand_name, 2024)
                            c_cong_cls = (
                                'cong-bad' if c_cong in ['혼잡', '매우혼잡'] else
                                ('cong-norm' if c_cong == '보통' else 'cong-good')
                            )
                            tgt_cat = row['CATEGORY']
                            v_pass = "✅ Pass" if row['VIS_SCALED'] > avgs[0] * mult else "❌"
                            s_pass = "✅ Pass" if row['SEN_SCALED'] > avgs[1] * mult else "❌"
                            f_pass = "✅ Pass" if row['FEA_SCALED'] > avgs[2] * mult else "❌"
                            st.markdown(
                                f"""
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
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    else:
                        st.warning("조건을 충족하는 cross-category 대체지가 없습니다.")


if __name__ == '__main__':
    run_app()
