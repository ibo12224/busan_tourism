import re
import numpy as np
import pandas as pd
from config import NAME_MAPPING
from data_loader import get_category_map

CATEGORY_MAP = get_category_map()


def get_spot_category(name):
    return CATEGORY_MAP.get(name, '기타')


def classify_density(val):
    if val >= 1.2:
        return "매우혼잡"
    elif val >= 0.7:
        return "혼잡"
    elif val >= 0.3:
        return "보통"
    return "쾌적"


def get_smart_active_mean(df, spot_name):
    spot_df = df[df['관광지명'] == spot_name]
    if spot_df.empty:
        return 0, "알수없음"

    hourly_grp = spot_df.groupby('시간대_int')['실질_㎡당_방문객수'].mean()
    if hourly_grp.empty:
        return 0, "알수없음"

    total_mean = hourly_grp.mean()
    base_hours = list(range(9, 19))
    active_hours = set(base_hours)

    for h in hourly_grp.index:
        val = hourly_grp[h]
        if h not in base_hours and val > total_mean:
            active_hours.add(h)
        if h in base_hours and val < total_mean and h in active_hours:
            active_hours.remove(h)

    valid_vals = [hourly_grp[h] for h in active_hours if h in hourly_grp.index]
    final_val = np.mean(valid_vals) if valid_vals else total_mean

    return final_val, classify_density(final_val)


def get_active_time_stats(df, spot_name, year=None):
    if year:
        spot_df = df[(df['관광지명'] == spot_name) & (df['날짜'].dt.year == year)]
    else:
        spot_df = df[df['관광지명'] == spot_name]
    return get_smart_active_mean(spot_df, spot_name)


def get_ranking_info(df, spot_name, year):
    y_df = df[df['날짜'].dt.year == year]
    if y_df.empty:
        return "정보 없음"

    rank_list = []
    for s in y_df['관광지명'].unique():
        val, _ = get_smart_active_mean(y_df, s)
        rank_list.append({'spot': s, 'val': val})

    rank_df = pd.DataFrame(rank_list).sort_values(by='val', ascending=False).reset_index(drop=True)
    if spot_name not in rank_df['spot'].values:
        return "정보 없음"

    my_rank = rank_df[rank_df['spot'] == spot_name].index[0] + 1
    total = len(rank_df)
    percent = (my_rank / total) * 100
    return f"전체 {total}곳 중 {my_rank}위 (상위 {percent:.1f}%)"


def get_spot_keywords(spot_name, noun_df, adj_df, df_sen_scaled):
    nouns, adjs = [], []
    if not noun_df.empty:
        row = noun_df[noun_df['관광지명'] == spot_name]
        if not row.empty:
            nouns = [k.strip() for k in str(row.iloc[0]['정제키워드']).split(',') if k.strip()][:30]

    if not adj_df.empty:
        row = adj_df[adj_df['관광지명'] == spot_name]
        if not row.empty:
            adjs = [k.strip() for k in str(row.iloc[0]['추출_형용사']).split(',') if k.strip()][:30]

    if not nouns and not df_sen_scaled.empty:
        row = df_sen_scaled[df_sen_scaled['기준_관광지'] == spot_name]
        if not row.empty:
            raw_k = str(row.iloc[0].get('기준지_고유_키워드', ''))
            if raw_k and raw_k != 'nan':
                all_k = [k.strip() for k in raw_k.split(',') if k.strip()]
                nouns = all_k[:len(all_k)//2]
                adjs = all_k[len(all_k)//2:]

    return nouns, adjs


def get_ranking_dict(df, spot_name, year):
    y_df = df[df['날짜'].dt.year == year]
    if y_df.empty:
        return None

    rank_list = [{'spot': spot, 'val': get_smart_active_mean(y_df, spot)[0]} for spot in y_df['관광지명'].unique()]
    rank_df = pd.DataFrame(rank_list).sort_values(by='val', ascending=False).reset_index(drop=True)

    try:
        rank = rank_df[rank_df['spot'] == spot_name].index[0] + 1
        return {"rank": rank, "total": len(rank_df), "top_percent": (rank / len(rank_df)) * 100}
    except Exception:
        return None


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
