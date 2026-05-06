from openai import OpenAI
from config import API_KEY

AI_SYSTEM_PROMPT = """
당신은 날카롭고 깊이 있는 통찰력을 가진 수석 데이터 분석가입니다.
'분석하겠습니다', '결과입니다', '안녕하세요' 같은 형식적인 서론을 일절 생략하고, 즉시 핵심 수치와 그 이면의 의미를 파고드십시오.
문장은 명료하되 내용은 심층적이어야 하며, 분량은 충분히 길고 자세하게 작성하십시오.
톤은 전문적이고 냉철한 존댓말(~입니다/합니다)을 유지하십시오.
"""


def generate_spot_info_ai(spot_name):
    if not API_KEY:
        return "API Key Missing"

    try:
        client = OpenAI(api_key=API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": AI_SYSTEM_PROMPT},
                {"role": "user", "content": f"'{spot_name}'의 위치, 주요 특징, 역사적/문화적 배경을 심층적으로 서술하십시오."},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception:
        return "정보 로드 실패"


def generate_visual_rank1_analysis(spot_name, rank1_name, rank1_score, avg_score, rank1_congestion):
    if not API_KEY:
        return "API Key Missing"

    score_diff = rank1_score - avg_score
    score_eval = f"평균({avg_score:.2f})보다 {score_diff:+.2f}점 높음" if score_diff > 0 else "평균 이하"

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
        client = OpenAI(api_key=API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": AI_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)


def generate_strategic_analysis(source, candidates_data, anal_type="text"):
    if not API_KEY:
        return "⚠️ API Key Missing"

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
        client = OpenAI(api_key=API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": AI_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def generate_weighted_insight(spot_name, top_cand, weights):
    if not API_KEY:
        return "API Key Missing"

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
        client = OpenAI(api_key=API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": AI_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def generate_section_analysis(section_type, spot_name, year, data_summary, congestion_stage, ranking_info="정보 없음"):
    if not API_KEY:
        return "⚠️ API Key Missing"

    if congestion_stage in ["쾌적", "보통"]:
        tone_guide = """
        [Diagnosis]: 수용 여력 충분 (Under Capacity).
        [Implication]: 데이터상 관광객 추가 유입이 가능하며, 분산 정책의 수용지(Destination)로서 적합함.
        """
    else:
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
        client = OpenAI(api_key=API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": AI_SYSTEM_PROMPT},
                {"role": "user", "content": msg},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content, "#0F172A"
    except Exception as e:
        return str(e), "#000"
