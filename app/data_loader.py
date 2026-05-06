import os
import pandas as pd
import streamlit as st
from config import FILE_CONFIG, NAME_MAPPING

@st.cache_data
def load_data_smart(file_path):
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path, encoding='utf-8-sig')
        except Exception:
            try:
                return pd.read_csv(file_path, encoding='cp949')
            except Exception:
                pass

    xlsx_path = file_path.replace('.csv', '.xlsx').strip()
    if os.path.exists(xlsx_path):
        try:
            return pd.read_excel(xlsx_path)
        except Exception:
            pass

    return pd.DataFrame()

@st.cache_data
def get_category_map():
    cat_df = load_data_smart(FILE_CONFIG["CATEGORY_INFO"])
    if cat_df.empty:
        return {}

    cat_df.columns = cat_df.columns.str.strip()
    if '관광지명' in cat_df.columns and '카테고리' in cat_df.columns:
        cat_df['관광지명'] = cat_df['관광지명'].astype(str).str.strip().replace(NAME_MAPPING)
        cat_df['카테고리'] = cat_df['카테고리'].astype(str).str.strip()
        return dict(zip(cat_df['관광지명'], cat_df['카테고리']))

    return {}

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
        else:
            df_main['행정구'] = "전체"

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

@st.cache_data
def get_global_top1_avg():
    df_img = load_data_smart(FILE_CONFIG["IMG_RANK_DATA"])
    if df_img.empty:
        return 0.5

    scores = []
    for val in df_img['1순위'].dropna():
        m = re.search(r'\(([\d.]+)\)', str(val))
        if m:
            scores.append(float(m.group(1)))

    return sum(scores) / len(scores) if scores else 0.5
