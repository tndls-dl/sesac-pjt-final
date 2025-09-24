import streamlit as st
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from typing import List, Dict, Any

# --- 1. 기본 설정 및 API/데이터 로딩 ---

st.set_page_config(
    page_title="INGREVIA | AI 화장품 성분 분석",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="auto"
)

# .env 또는 Streamlit Secrets에서 API 키 불러오기
load_dotenv()
if "OPENAI_API_KEY" in os.environ:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
else:
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    except Exception:
        st.error("OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인하거나 Streamlit Secrets를 설정해주세요.")
        st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# 업그레이드된 브랜드 색상 정의 (그린~연두~아이보리 톤온톤)
COLORS = {
    'primary': '#1a4d2e', 'secondary': '#2d5f3f', 'accent': '#4a7c59',
    'light': '#7fb069', 'soft_green': '#a7c957', 'mint': '#c5d8a4',
    'ivory': '#f8f9f5', 'cream': '#fefffe', 'sage': '#e8f1e4',
    'border': '#d4dfcf', 'text': '#1a3d25', 'text_soft': '#2d5f3f',
    'highlight': '#f0f8ec', 'card_shadow': 'rgba(26, 77, 46, 0.08)',
    'button_primary': '#2d5f3f', 'button_hover': '#1a4d2e',
    'button_secondary': '#4a7c59', 'success': '#7fb069', 'warning': '#c5a572'
}

# 데이터 로딩 함수
@st.cache_data
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        df['효능'] = df['효능'].fillna('')
        df['가격'] = pd.to_numeric(df['가격'], errors='coerce').fillna(0)
        df['유해성_점수'] = pd.to_numeric(df['유해성_점수'], errors='coerce').fillna(0)
        category_synonyms = {"로션/에멀젼": "로션/에멀전", "선크림": "선크림/로션"}
        df['카테고리'] = df['카테고리'].replace(category_synonyms)
        return df
    except FileNotFoundError:
        st.error(f"⚠️ `{filepath}` 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요!")
        return None

# --- 2. AI 기능 및 웹 서치 ---
def analyze_ingredients_with_search(product_name: str, skin_type: str, skin_concerns: List[str], ingredients_list: str) -> Dict[str, Any]:
    """웹 검색을 통해 특정 제품 성분을 사용자 맞춤형으로 심층 분석합니다."""
    try:
        search_query = f"{product_name} 전성분 분석 {skin_type} {', '.join(skin_concerns)} 피부 효능 주의점"
        search_prompt = f"'{search_query}' 라는 검색어에 대한 가상 웹 검색 결과 요약 3개를 생성해줘. 결과는 JSON 형식으로 제품의 효능과 성분 정보를 담아야 해."
        search_response = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": search_prompt}], temperature=0.1
        )
        search_results = search_response.choices[0].message.content

        analysis_prompt = f"""
        당신은 화장품 성분 분석 전문가입니다. 아래 정보를 바탕으로 제품 성분을 분석해주세요.
        [사용자 정보]
        - 피부 타입: {skin_type}
        - 피부 고민: {', '.join(skin_concerns)}
        [제품 정보]
        - 제품명: {product_name}
        - 전성분: {ingredients_list}
        [웹 검색 결과 요약]
        {search_results}
        [분석 요청]
        1. 위 정보를 종합하여, 제품의 전성분 목록에서 **핵심 효능 성분**을 3~5개 찾아주세요.
        2. 사용자가 **특별히 유의해야 할 성분**이 있다면 1~3개 찾아주세요. (없으면 빈 리스트)
        3. 이 제품을 사용자에게 추천하는 **핵심적인 이유**를 1~2 문장으로 요약해주세요.
        [출력 형식] 반드시 아래 JSON 형식으로만 응답해주세요:
        {{
          "beneficial_ingredients": ["성분1", "성분2", ...],
          "caution_ingredients": ["성분A", "성분B", ...],
          "reason": "추천 이유 요약..."
        }}
        """
        response = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": analysis_prompt}],
            response_format={"type": "json_object"}, temperature=0.2
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.warning(f"성분 상세 분석 중 오류 발생: {e}")
        return {
            "beneficial_ingredients": ["분석 중 오류"], "caution_ingredients": [],
            "reason": "AI 상세 분석 중 오류가 발생했습니다."
        }


# --- 3. UI 스타일링 및 렌더링 (업그레이드) ---
def load_css():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');
        
        /* 기본 배경 및 폰트 설정 */
        .stApp {{ 
            background: linear-gradient(135deg, {COLORS['ivory']} 0%, {COLORS['sage']} 100%); 
            font-family: 'Noto Sans KR', sans-serif;
        }}
        
        .main .block-container {{ 
            padding: 1.5rem 2rem 3rem 2rem; 
            max-width: 950px; 
            margin: 0 auto; 
        }}
        
        /* 헤더 컨테이너 - 더 고급스럽게 */
        .header-container {{ 
            background: linear-gradient(135deg, {COLORS['cream']} 0%, {COLORS['ivory']} 100%);
            border-radius: 24px; 
            padding: 2.5rem; 
            box-shadow: 0 8px 32px {COLORS['card_shadow']}, 0 2px 8px rgba(26, 77, 46, 0.04);
            margin-bottom: 2.5rem; 
            text-align: center; 
            border: 1px solid {COLORS['border']};
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
        }}
        
        .header-container::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['light']} 50%, {COLORS['primary']} 100%);
        }}
        
        .logo-section {{ 
            font-size: 4.5rem; 
            margin-bottom: 1rem; 
            filter: drop-shadow(0 2px 4px rgba(26, 77, 46, 0.1));
        }}
        
        .brand-name {{ 
            font-size: 2.8rem; 
            font-weight: 700; 
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['accent']} 50%, {COLORS['light']} 100%);
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
            background-clip: text;
            margin: 0; 
            letter-spacing: 4px; 
            text-shadow: 0 2px 8px rgba(26, 77, 46, 0.1);
        }}
        
        .brand-tagline {{ 
            color: {COLORS['text_soft']}; 
            font-size: 0.95rem; 
            font-weight: 400; 
            letter-spacing: 2px; 
            margin-top: 0.8rem; 
            opacity: 0.8;
        }}
        
        /* 채팅 메시지 스타일 */
        .stChatMessage {{ 
            background: {COLORS['cream']}; 
            border-radius: 20px; 
            padding: 1.5rem; 
            margin: 1.2rem 0; 
            box-shadow: 0 4px 16px {COLORS['card_shadow']}, 0 1px 4px rgba(26, 77, 46, 0.04);
            border-left: 4px solid {COLORS['accent']}; 
            border: 1px solid {COLORS['border']};
        }}
        
        /* 제품 카드 - 더 세련되게 */
        .product-card {{ 
            background: linear-gradient(135deg, {COLORS['cream']} 0%, {COLORS['ivory']} 100%);
            border-radius: 24px; 
            padding: 2.5rem; 
            box-shadow: 0 8px 24px {COLORS['card_shadow']}, 0 2px 8px rgba(26, 77, 46, 0.04);
            border: 1px solid {COLORS['border']};
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            position: relative;
            overflow: hidden;
        }}
        
        .product-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 12px 32px {COLORS['card_shadow']}, 0 4px 12px rgba(26, 77, 46, 0.08);
        }}
        
        .product-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['light']} 50%, {COLORS['accent']} 100%);
        }}
        
        .product-title a {{ 
            color: {COLORS['primary']}; 
            font-size: 1.5rem; 
            font-weight: 700; 
            text-decoration: none; 
            transition: color 0.2s ease;
        }}
        
        .product-title a:hover {{
            color: {COLORS['accent']};
        }}
        
        .product-brand {{ 
            color: {COLORS['text_soft']}; 
            margin-bottom: 1.2rem; 
            font-weight: 500;
        }}
        
        /* 성분 배지 - 더 고급스럽게 */
        .ingredients-section {{ margin-top: 1.5rem; }}
        .ingredients-title {{ 
            color: {COLORS['primary']}; 
            font-weight: 600; 
            margin-bottom: 1rem; 
            font-size: 1.05rem;
        }}
        
        .ingredient-badge {{ 
            display: inline-block; 
            padding: 0.6rem 1rem; 
            border-radius: 25px; 
            font-size: 0.9rem; 
            margin: 0.4rem; 
            font-weight: 500; 
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        
        .ingredient-badge:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        
        .beneficial-badge {{ 
            background: linear-gradient(135deg, {COLORS['sage']} 0%, {COLORS['mint']} 100%); 
            color: {COLORS['primary']}; 
            border: 1px solid {COLORS['light']};
            box-shadow: 0 2px 8px rgba(127, 176, 105, 0.2);
        }}
        
        .caution-badge {{ 
            background: linear-gradient(135deg, #fff4e6 0%, #ffeaa7 100%); 
            color: #d35400; 
            border: 1px solid #fdcb6e;
            box-shadow: 0 2px 8px rgba(211, 84, 0, 0.15);
        }}
        
        /* 선택 표시 영역 */
        .selection-display {{ 
            background: linear-gradient(135deg, {COLORS['highlight']} 0%, {COLORS['sage']} 100%);
            border-radius: 18px; 
            padding: 1.5rem; 
            margin-top: 1.2rem; 
            border: 1px solid {COLORS['border']};
            box-shadow: 0 4px 16px {COLORS['card_shadow']};
        }}
        
        /* 버튼 스타일 - 브랜드 컬러로 통일 */
        .stButton > button {{
            background: linear-gradient(135deg, {COLORS['button_primary']} 0%, {COLORS['button_secondary']} 100%);
            color: white;
            border: none;
            border-radius: 15px;
            padding: 0.8rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(45, 95, 63, 0.2);
            text-transform: none;
            font-family: 'Noto Sans KR', sans-serif;
        }}
        
        .stButton > button:hover {{
            background: linear-gradient(135deg, {COLORS['button_hover']} 0%, {COLORS['button_primary']} 100%);
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(45, 95, 63, 0.3);
        }}
        
        .stButton > button:active {{
            transform: translateY(0);
            box-shadow: 0 2px 8px rgba(45, 95, 63, 0.2);
        }}
        
        /* 선택 단계 버튼들을 고급스러운 크림 톤으로 오버라이드 */
        .stButton > button:not([kind="primary"]):not(.main-action-btn) {{
            background: linear-gradient(135deg, #F7ECD8 0%, #F4E9D1 100%) !important;
            color: #2d4a35 !important;
            border: 1px solid #ebe0c8 !important;
            box-shadow: 0 2px 8px rgba(247, 236, 216, 0.3), 0 1px 3px rgba(45, 74, 53, 0.08) !important;
            font-weight: 500 !important;
        }}
        
        .stButton > button:not([kind="primary"]):not(.main-action-btn):hover {{
            background: linear-gradient(135deg, #F2E7CC 0%, #EFE4C5 100%) !important;
            color: #1a3d25 !important;
            border-color: #e6dbc3 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(242, 231, 204, 0.4), 0 2px 6px rgba(26, 61, 37, 0.12) !important;
        }}
        
        .stButton > button:not([kind="primary"]):not(.main-action-btn):active {{
            background: linear-gradient(135deg, #EDE2C0 0%, #EAE1BA 100%) !important;
            color: #1a3d25 !important;
            transform: translateY(0px) !important;
            box-shadow: 0 1px 4px rgba(237, 226, 192, 0.35), inset 0 1px 2px rgba(26, 61, 37, 0.1) !important;
        }}
        
        /* Primary 버튼 스타일 */
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {COLORS['success']} 0%, {COLORS['light']} 100%);
            color: white;
            font-weight: 700;
            font-size: 1.05rem;
            box-shadow: 0 6px 20px rgba(127, 176, 105, 0.3);
        }}
        
        .stButton > button[kind="primary"]:hover {{
            background: linear-gradient(135deg, {COLORS['light']} 0%, {COLORS['success']} 100%);
            box-shadow: 0 8px 24px rgba(127, 176, 105, 0.4);
        }}
        
        /* 사이드바 스타일 개선 */
        .css-1d391kg {{
            background: linear-gradient(180deg, {COLORS['cream']} 0%, {COLORS['sage']} 100%);
        }}
        
        .css-1v0mbdj.ebxwdo61 {{
            border-radius: 15px;
            border: 1px solid {COLORS['border']};
            background: {COLORS['cream']};
            box-shadow: 0 2px 8px {COLORS['card_shadow']};
        }}
        
        /* 구분선 스타일 */
        hr {{
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent 0%, {COLORS['border']} 50%, transparent 100%);
            margin: 1.5rem 0;
        }}
        
        /* 스피너 및 로딩 애니메이션 */
        .stSpinner > div {{
            border-top-color: {COLORS['accent']};
        }}
        
        /* 성공/경고 메시지 스타일 */
        .stSuccess {{
            background: linear-gradient(135deg, {COLORS['sage']} 0%, {COLORS['mint']} 100%);
            border-left: 4px solid {COLORS['success']};
            border-radius: 12px;
        }}
        
        .stWarning {{
            background: linear-gradient(135deg, #fff4e6 0%, #ffeaa7 100%);
            border-left: 4px solid {COLORS['warning']};
            border-radius: 12px;
        }}
        
        .stInfo {{
            background: linear-gradient(135deg, {COLORS['highlight']} 0%, {COLORS['sage']} 100%);
            border-left: 4px solid {COLORS['accent']};
            border-radius: 12px;
        }}
    </style>
    """, unsafe_allow_html=True)

def render_product_card(rec: Dict[str, Any], rank: int):
    rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "🏅")

    beneficial_html = "".join([f'<span class="ingredient-badge beneficial-badge">{ing}</span>' for ing in rec.get("beneficial_ingredients", [])])
    if not beneficial_html: beneficial_html = "<p>핵심 효능 성분을 분석 중입니다.</p>"

    caution_ingredients = rec.get("caution_ingredients", [])
    if caution_ingredients:
        caution_html = "".join([f'<span class="ingredient-badge caution-badge">{ing}</span>' for ing in caution_ingredients])
    else:
        caution_html = "<p>✅ 분석 결과, 특별히 유의할 성분은 발견되지 않았어요.</p>"

    with st.container(border=False):
        st.markdown(f"""
        <div class="product-card">
            <h2 style="margin-bottom: 0.8rem; color: {COLORS['primary']}; font-weight: 600;">{rank_emoji} TOP {rank}</h2>
            <h3 class="product-title"><a href="{rec.get('링크', '#')}" target="_blank">{rec.get('제품명', '이름 없음')}</a></h3>
            <p class="product-brand"><strong>🏢 브랜드:</strong> {rec.get('브랜드명', '브랜드 없음')}</p>
            <p style="color: {COLORS['text_soft']}; font-weight: 500;"><strong>💰 가격:</strong> {int(rec.get('가격', 0)):,}원 | <strong>📏 용량:</strong> {rec.get('용량', '정보 없음')}</p>
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.info(f"**💬 추천 포인트:** {rec.get('reason', '추천 이유 분석 중...')}")
            
            st.markdown(f"""
            <div class="ingredients-section" style="margin-top:0.8rem; border:none; background:transparent; padding:0.8rem;">
                <p class="ingredients-title">✅ 이 제품의 핵심 효능 성분</p>
                <div>{beneficial_html}</div>
            </div>
            <div class="ingredients-section" style="margin-top:1.2rem; border:none; background:transparent; padding:0.8rem;">
                <p class="ingredients-title">⚠️ 내 피부에 유의할 성분</p>
                <div>{caution_html}</div>
            </div>
            """, unsafe_allow_html=True)


# --- 4. 메인 앱 로직 ---

def handle_chat_input(prompt: str):
    st.session_state.messages.append({"role": "user", "content": prompt})

    if any(keyword in prompt for keyword in ['다시 시작', '처음부터', '리셋', '초기화']):
        st.session_state.clear()
        st.rerun()

    with st.chat_message("assistant", avatar="🌿"):
        with st.spinner("AI가 사용자의 말을 이해하고 있어요..."):
            parser_prompt = f"""
            주어진 사용자 대화에서 피부 타입, 피부 고민, 제품 종류 키워드를 추출하여 JSON으로 반환해줘.
            사용자 정보가 이미 있는 상태에서 '토너도 추천해줘' 같은 후속 질문이 들어오면, 변경된 '제품 종류'만 추출해야 해.
            - skin_type: '민감성', '지성', '건성', '아토피성', '복합성', '중성' 중 하나.
            - skin_concerns: '보습', '진정', '미백', '주름/탄력', '모공/피지', '트러블', '각질' 중 여러 개 가능.
            - product_category: '스킨/토너', '로션/에멀전', '에센스/앰플/세럼', '크림', '선크림/로션' 등.
            - 정보가 없으면 null.
            사용자 입력: "{prompt}"
            """
            response = client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": parser_prompt}],
                response_format={"type": "json_object"}
            )
            parsed_info = json.loads(response.choices[0].message.content)

            profile = st.session_state.user_profile
            updated_keys = []
            
            if parsed_info.get("skin_type") and profile["skin_type"] != parsed_info["skin_type"]:
                profile["skin_type"] = parsed_info["skin_type"]; updated_keys.append("피부 타입")
            if parsed_info.get("skin_concerns") and profile["skin_concerns"] != parsed_info["skin_concerns"]:
                profile["skin_concerns"] = parsed_info["skin_concerns"]; updated_keys.append("피부 고민")
            if parsed_info.get("product_category") and profile["product_category"] != parsed_info["product_category"]:
                profile["product_category"] = parsed_info["product_category"]; updated_keys.append("제품 종류")

            if updated_keys:
                st.session_state.analysis_done = False
            
            if not all(profile.values()):
                next_q = "피부 타입" if not profile["skin_type"] else "피부 고민" if not profile["skin_concerns"] else "제품 종류"
                response_text = f"네, **{', '.join(updated_keys)}** 정보가 확인되었어요. 계속해서 **{next_q}**을(를) 알려주시겠어요?" if updated_keys else f"죄송하지만 잘 이해하지 못했어요. **{next_q}** 정보를 알려주시겠어요?"
            else:
                response_text = "모든 정보가 확인되었어요! 바로 분석을 시작할게요. 🔬" if updated_keys else "어떤 것을 더 도와드릴까요? 예를 들어 '토너도 추천해줘' 와 같이 말씀해보세요."
            
            st.session_state.messages.append({"role": "assistant", "content": response_text})

def main():
    load_css()
    df = load_data('product_data.csv')
    if df is None: st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 🥰 AI 화장품 성분 분석가 **INGREVIA** 입니다. 아래 버튼을 누르거나 채팅으로 피부 정보를 알려주시면, 웹 분석을 통해 딱 맞는 화장품을 찾아드릴게요. 먼저 **피부 타입**을 선택해주세요."}]
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {"skin_type": None, "skin_concerns": None, "product_category": None}
    if "selected_concerns" not in st.session_state:
        st.session_state.selected_concerns = []

    st.markdown("""<div class="header-container"><div class="logo-section">🌿</div><h1 class="brand-name">INGREVIA</h1><p class="brand-tagline">AI COSMETIC INGREDIENTS ANALYZER</p></div>""", unsafe_allow_html=True)
    
    if prompt := st.chat_input("피부 타입, 고민, 제품 종류 등을 알려주세요."):
        handle_chat_input(prompt); st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🌿" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"], unsafe_allow_html=True)

    profile = st.session_state.user_profile

    if not all(profile.values()):
        if not profile["skin_type"]:
            skin_types = ['민감성', '지성', '건성', '복합성', '아토피성', '중성']; cols = st.columns(3)
            for i, skin_type in enumerate(skin_types):
                if cols[i % 3].button(f"🫧 {skin_type}", key=f"skin_{skin_type}", use_container_width=True):
                    profile["skin_type"] = skin_type
                    st.session_state.messages.extend([{"role": "user", "content": f"제 피부 타입은 **{skin_type}**이에요."}, {"role": "assistant", "content": f"네, {skin_type}이시군요! 어떤 **피부 고민**이 있으신가요? (여러 개 선택 가능)"}])
                    st.rerun()
        elif not profile["skin_concerns"]:
            concerns_options = ['보습', '진정', '미백', '주름/탄력', '모공/피지', '트러블', '각질']; emoji_map = {'보습': '💦', '진정': '🍃', '미백': '✨', '주름/탄력': '🌟', '모공/피지': '🔍', '트러블': '🩹', '각질': '🧽'}
            cols = st.columns(4)
            for i, concern in enumerate(concerns_options):
                is_selected = concern in st.session_state.selected_concerns
                if cols[i % 4].button(f"{'✅' if is_selected else emoji_map[concern]} {concern}", key=f"concern_{concern}", use_container_width=True):
                    if is_selected: st.session_state.selected_concerns.remove(concern)
                    else: st.session_state.selected_concerns.append(concern)
                    st.rerun()
            if st.session_state.selected_concerns:
                st.markdown(f"""<div class="selection-display"><strong style="color: {COLORS['primary']};">선택된 고민:</strong> {', '.join(st.session_state.selected_concerns)}</div>""", unsafe_allow_html=True)
                if st.button("✅ 선택 완료", type="primary", use_container_width=True):
                    profile["skin_concerns"] = st.session_state.selected_concerns
                    st.session_state.messages.extend([{"role": "user", "content": f"피부 고민은 **{', '.join(profile['skin_concerns'])}** 입니다."}, {"role": "assistant", "content": "마지막으로, 어떤 **제품 종류**를 찾아드릴까요?"}])
                    st.rerun()
        elif not profile["product_category"]:
            categories = [('스킨/토너', '💧'), ('로션/에멀전', '🧴'), ('에센스/앰플/세럼', '✨'), ('크림', '🍶'), ('선크림/로션', '☀️'), ('클렌징 폼', '🫧')]
            cols = st.columns(3)
            for i, (cat, emoji) in enumerate(categories):
                if cols[i % 3].button(f"{emoji} {cat}", key=f"cat_{cat}", use_container_width=True):
                    profile["product_category"] = cat
                    st.session_state.messages.extend([{"role": "user", "content": f"**{cat}** 제품을 찾고 있어요."}, {"role": "assistant", "content": "모든 정보가 확인되었어요. 지금부터 AI가 사용자님께 꼭 맞는 제품을 찾아 분석을 시작할게요! 잠시만 기다려주세요. 🔬"}])
                    st.rerun()
    else:
        if not st.session_state.get("analysis_done", False):
            with st.chat_message("assistant", avatar="🌿"):
                with st.spinner(f"🔬 **{profile['skin_type']}**, **{', '.join(profile['skin_concerns'])}** 고민을 위한 **{profile['product_category']}** 제품을 찾고 있습니다..."):
                    filtered_df = df[df['카테고리'].str.contains(profile["product_category"].split('/')[0], na=False)].copy()
                    concern_filter = filtered_df['효능'].apply(lambda x: any(c in str(x) for c in profile["skin_concerns"]))
                    filtered_df = filtered_df[concern_filter]
                    if filtered_df.empty:
                        st.warning("😔 조건에 맞는 제품을 찾지 못했습니다. 다른 조건으로 다시 시도해주세요.")
                    else:
                        candidates = filtered_df.nsmallest(3, '유해성_점수').to_dict('records')
                        recommendations = [cand | analyze_ingredients_with_search(cand['제품명'], profile['skin_type'], profile['skin_concerns'], cand['전성분']) for cand in candidates]
                        st.success(f"✨ **분석 완료!** 사용자님께 최적화된 **TOP 3 {profile['product_category']}** 제품을 추천합니다.")
                        for i, rec in enumerate(recommendations, 1):
                            render_product_card(rec, i)
            st.session_state.analysis_done = True
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1,1.5,1])
        if col2.button("🔄 처음부터 다시 시작하기", type="primary", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    # --- 사이드바 ---
    with st.sidebar:
        st.markdown(f"### 🌿 INGREVIA")
        st.markdown("---")
        if st.button("🔄 새 추천 시작하기", use_container_width=True, type="primary"):
            st.session_state.clear()
            st.rerun()
        st.markdown("---")
        st.markdown("### 💡 이렇게 대화해보세요")
        st.markdown("""
        - "나는 민감성 피부야"
        - "보습이랑 진정이 필요해"
        - "수분크림 찾아줘"
        - "민감성 피부에 보습 잘되는 크림"
        - "토너도 추천해줘"
        - "다시 시작"
        """)
        st.markdown("---")
        st.markdown("### 📊 현재 분석 조건")
        if profile.get('skin_type'): st.info(f"**피부 타입:** {profile['skin_type']}")
        else: st.info("피부 타입 선택 전")
        if profile.get('skin_concerns'): st.info(f"**피부 고민:** {', '.join(profile['skin_concerns'])}")
        else: st.info("피부 고민 선택 전")
        if profile.get('product_category'): st.info(f"**제품 종류:** {profile['product_category']}")
        else: st.info("제품 종류 선택 전")

if __name__ == "__main__":
    main()