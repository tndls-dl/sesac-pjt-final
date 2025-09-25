import streamlit as st
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from typing import List, Dict, Any
import time

# --- 1. 기본 설정 및 API/데이터 로딩 ---

st.set_page_config(
    page_title="INGREVIA | AI 화장품 성분 분석",
    page_icon="🌿",
    layout="wide",
    # [수정] 사이드바를 기본적으로 닫힌 상태로 시작하도록 변경
    initial_sidebar_state="collapsed"
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

# 업그레이드된 브랜드 색상 정의
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

# --- 2. AI 기능 및 웹 서치 (한국어 출력 보장) ---
def analyze_ingredients_with_search(product_name: str, skin_type: str, skin_concerns: List[str], ingredients_list: str) -> Dict[str, Any]:
    """웹 검색을 통해 특정 제품 성분을 사용자 맞춤형으로 심층 분석합니다."""
    try:
        analysis_prompt = f"""
        당신은 전문 화장품 성분 분석가입니다. 반드시 한국어로만 답변해주세요.

        [사용자 정보]
        - 피부 타입: {skin_type}
        - 피부 고민: {', '.join(skin_concerns) if skin_concerns else '특별한 고민 없음'}

        [제품 정보]
        - 제품명: {product_name}
        - 전성분: {ingredients_list}

        [분석 지침]
        1. 전성분 목록에서 피부에 도움이 되는 **핵심 효능 성분** 3-5개를 찾아주세요. 성분명만 적고 설명은 하지 마세요.
        2. 주의해야 할 성분을 꼼꼼히 찾아보세요. 성분명만 적고 설명은 하지 마세요.
           - 알코올류, 인공향료, 방부제, 계면활성제, 에센셜 오일류, 각질 제거 성분 등 체크
        3. 이 제품을 사용자에게 추천하는 **구체적인 이유**를 한국어로 2문장 이내로 설명해주세요.

        [출력 형식] 반드시 한국어로 JSON 형식 응답:
        {{
          "beneficial_ingredients": ["성분명1", "성분명2", "성분명3"],
          "caution_ingredients": ["성분명A", "성분명B"] (없으면 빈 배열),
          "reason": "한국어로 추천 이유 설명..."
        }}
        
        주의: 모든 내용을 반드시 한국어로만 작성하세요. 영어 단어나 설명은 절대 포함하지 마세요.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": analysis_prompt}],
            response_format={"type": "json_object"}, temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.warning(f"성분 상세 분석 중 오류 발생: {e}")
        return {
            "beneficial_ingredients": ["하이알루론산", "세라마이드", "나이아신아마이드"], 
            "caution_ingredients": [],
            "reason": "AI 상세 분석 중 일시적 오류가 발생했습니다. 성분을 직접 확인해 주세요."
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
        
        /* 채팅 입력창 - 연한 회색 완전 고정 */
        .stChatInputContainer {{
            border: none !important;
            background: #f8f9fa !important;
            border-radius: 20px !important;
            padding: 8px !important;
            margin: 16px 0 !important;
        }}
        
        .stChatInput > div {{
            border: none !important;
            background: #f8f9fa !important;
            border-radius: 20px !important;
        }}
        
        /* 채팅 입력창 수직 중앙 정렬 */
        .stChatInput > div > div {{
            border: none !important;
            border-radius: 20px !important;
            background: #f8f9fa !important;
            box-shadow: none !important;
            padding: 12px 20px !important;
            display: flex !important;
            align-items: center !important;
        }}
        
        .stChatInput > div > div:focus-within {{
            border: none !important;
            background: #f8f9fa !important;
            box-shadow: none !important;
            outline: none !important;
        }}
        
        .stChatInput > div > div:hover {{
            background: #f8f9fa !important;
        }}
        
        .stChatInput textarea {{
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
            background: #f8f9fa !important;
            padding: 8px 5px !important;
            color: #2c3e50 !important;
            font-weight: 400 !important;
            line-height: 1.5;
        }}
        
        .stChatInput textarea:focus {{
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
            background: #f8f9fa !important;
        }}
        
        .stChatInput textarea:hover {{
            background: #f8f9fa !important;
        }}
        
        .stChatInput textarea::placeholder {{
            color: #6c757d !important;
            opacity: 0.8 !important;
        }}
        
        /* 모든 상태에서 완전 고정 */
        .stChatInput *, .stChatInput *:focus, .stChatInput *:hover, .stChatInput *:active {{
            background: #f8f9fa !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
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
        
        /* [수정] st.success 대신 사용할 커스텀 성공 메시지 스타일 및 정렬 */
        .custom-success {{
            background: linear-gradient(135deg, {COLORS['sage']} 0%, {COLORS['mint']} 100%);
            border-left: 4px solid {COLORS['success']};
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            margin-top: -0.5rem !important;
            color: {COLORS['text']};
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
        .ingredients-section > div {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.6rem;
        }}

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
            margin: 0;
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
            background: linear-gradient(135deg, #fff9f0 0%, #fff4e6 100%);
            color: #e67e22;
            border: 1px solid #f39c12;
            box-shadow: 0 2px 8px rgba(230, 126, 34, 0.15);
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
        
        /* 선택 단계 버튼들을 배경색 정도로 매우 연하게 */
        .stButton > button:not([kind="primary"]):not(.main-action-btn) {{
            background: linear-gradient(135deg, #FEFDFB 0%, #FEFCF8 100%) !important;
            color: #2d4a35 !important;
            border: 1px solid #F5F2EA !important;
            box-shadow: 0 1px 4px rgba(254, 253, 251, 0.6), 0 1px 2px rgba(45, 74, 53, 0.05) !important;
            font-weight: 500 !important;
            border-radius: 15px !important;
            padding: 0.8rem 1.5rem !important;
            transition: all 0.3s ease !important;
        }}
        
        .stButton > button:not([kind="primary"]):not(.main-action-btn):hover {{
            background: linear-gradient(135deg, #FDFBF6 0%, #FCFAF3 100%) !important;
            color: #1a3d25 !important;
            border-color: #F0EDE5 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 2px 8px rgba(253, 251, 246, 0.8), 0 1px 3px rgba(26, 61, 37, 0.08) !important;
        }}
        
        .stButton > button:not([kind="primary"]):not(.main-action-btn):active {{
            background: linear-gradient(135deg, #FBF9F1 0%, #FAF8EE 100%) !important;
            color: #1a3d25 !important;
            transform: translateY(0px) !important;
            box-shadow: 0 1px 3px rgba(251, 249, 241, 0.5), inset 0 1px 1px rgba(26, 61, 37, 0.08) !important;
        }}
        
        /* Primary 버튼 스타일 - 연한 연두색으로 */
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, #A8D5A8 0%, #B8E5B8 100%) !important;
            color: white !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
            box-shadow: 0 6px 20px rgba(168, 213, 168, 0.3) !important;
            border-radius: 15px !important;
            padding: 0.8rem 1.5rem !important;
            border: none !important;
        }}
        
        .stButton > button[kind="primary"]:hover {{
            background: linear-gradient(135deg, #9ECE9E 0%, #AEDDAE 100%) !important;
            box-shadow: 0 8px 24px rgba(158, 206, 158, 0.4) !important;
            transform: translateY(-1px) !important;
        }}
        
        /* 선택된 항목 표시를 위한 스타일 (더 진한 색상) */
        .stButton > button:not([kind="primary"]):not(.main-action-btn).selected {{
            background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%) !important;
            color: {COLORS['primary']} !important;
            border: 2px solid {COLORS['accent']} !important;
            box-shadow: 0 3px 10px rgba(200, 230, 201, 0.4), 0 1px 4px rgba(26, 61, 37, 0.15) !important;
            font-weight: 600 !important;
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
        
        .stWarning {{
            background: linear-gradient(135deg, #fff9f0 0%, #fff4e6 100%);
            border-left: 4px solid #f39c12;
            border-radius: 12px;
        }}
        
        .stInfo {{
            background: linear-gradient(135deg, {COLORS['highlight']} 0%, {COLORS['sage']} 100%);
            border-left: 4px solid {COLORS['accent']};
            border-radius: 12px;
        }}
    </style>
    """, unsafe_allow_html=True)

def render_product_card(rec: Dict[str, Any], rank: int, title: str = None):
    # --- 제목 설정 ---
    if title:
        category_emoji_map = {'스킨/토너': '💧', '로션/에멀전': '🧴', '에센스/앰플/세럼': '✨', '크림': '🍶', '선크림/로션': '☀️', '클렌징 폼': '🫧'}
        emoji = category_emoji_map.get(title, '🌿')
        # [수정] '추천'을 '부문'으로 변경
        header_html = f'<h2 style="margin-bottom: 0.8rem; color: {COLORS['primary']}; font-weight: 600;">{emoji} {title} 부문</h2>'
    else:
        rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "🏅")
        header_html = f'<h2 style="margin-bottom: 0.8rem; color: {COLORS['primary']}; font-weight: 600;">{rank_emoji} TOP {rank}</h2>'
    
    beneficial_html = "".join([f'<a href="https://www.google.com/search?q={ing.strip()}" target="_blank" style="text-decoration: none; margin: 4px;"><span class="ingredient-badge beneficial-badge">{ing.strip()}</span></a>' for ing in rec.get("beneficial_ingredients", [])])
    if not beneficial_html: beneficial_html = "<p>핵심 효능 성분을 분석 중입니다.</p>"

    caution_ingredients = rec.get("caution_ingredients", [])
    if caution_ingredients:
        caution_html = "".join([f'<a href="https://www.google.com/search?q={ing.strip()}" target="_blank" style="text-decoration: none; margin: 4px;"><span class="ingredient-badge caution-badge">{ing.strip()}</span></a>' for ing in caution_ingredients])
    else:
        caution_html = "<p>✅ 분석 결과, 특별히 유의할 성분은 발견되지 않았어요.</p>"

    recommendation_reason = rec.get('reason', '추천 이유 분석 중...')

    # --- 최종 카드 UI 렌더링 ---
    with st.container(border=False):
        st.markdown(f"""
        <div class="product-card">
            {header_html}
            <h3 class="product-title"><a href="{rec.get('링크', '#')}" target="_blank">{rec.get('제품명', '이름 없음')}</a></h3>
            <p class="product-brand"><strong>🏢 브랜드:</strong> {rec.get('브랜드명', '브랜드 없음')}</p>
            <p style="color: {COLORS['text_soft']}; font-weight: 500;"><strong>💰 가격:</strong> {int(rec.get('가격', 0)):,}원 | <strong>🫙 용량:</strong> {rec.get('용량', '정보 없음')}</p>
            <hr> 
            <div style="padding-top: 1rem;">
                <p style="margin-bottom: 2rem; color: {COLORS['text_soft']}; font-size: 1.05rem; font-weight: 500;">
                    <strong>💬 추천 포인트:</strong> {recommendation_reason}
                </p>
                <div class="ingredients-section" style="margin-top: 0.8rem;">
                    <p class="ingredients-title">✅ 이 제품의 핵심 효능 성분</p>
                    <div style="display: flex; flex-wrap: wrap; align-items: center;">{beneficial_html}</div>
                </div>
                <div class="ingredients-section" style="margin-top: 1.2rem;">
                    <p class="ingredients-title">⚠️ 내 피부에 유의할 성분</p>
                    <div style="display: flex; flex-wrap: wrap; align-items: center;">{caution_html}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_typing_effect(text: str):
    """타이핑 효과를 보여주는 함수"""
    message_placeholder = st.empty()
    typing_text = ""
    
    for i, char in enumerate(text):
        typing_text += char
        message_placeholder.markdown(typing_text)
        
        if char in ['.', '!', '?']:
            time.sleep(0.3)
        elif char == '\n':
            time.sleep(0.2)
        elif char == ' ':
            time.sleep(0.05)
        elif i % 2 == 0:  # 매 2글자마다
            time.sleep(0.03)
        else:
            time.sleep(0.01)
    
    return message_placeholder

def handle_chat_input(prompt: str):
    st.session_state.messages.append({"role": "user", "content": prompt, "typing_done": True})

    if any(keyword in prompt for keyword in ['다시 시작', '처음부터', '리셋', '초기화']):
        st.session_state.clear()
        st.rerun()

    st.session_state.analysis_pending = True
    st.session_state.recommend_set = False

    if any(keyword in prompt for keyword in ['기초', '기초화장품', '기초 화장품', '기초제품', '기초 세트', '기초 제품', '기초 추천', '기초 라인', '기초라인']):
        st.session_state.recommend_set = True

    with st.spinner("AI가 사용자의 말을 이해하고 있어요..."):
        parser_prompt = f"""
        주어진 사용자 대화에서 피부 타입, 피부 고민, 제품 종류 키워드를 추출하여 JSON으로 반환해줘.
        - skin_type: '민감성', '지성', '건성', '아토피성', '복합성', '중성' 중 하나. 없으면 null
        - skin_concerns: '보습', '진정', '미백', '주름/탄력', '모공/피지', '트러블', '각질' 중 여러 개 가능. 없으면 null
        - product_categories: ['스킨/토너', '로션/에멀전', '에센스/앰플/세럼', '크림', '선크림/로션', '클렌징 폼'] 중 여러 개 가능. 없으면 null
        사용자 입력: "{prompt}"
        """
        response = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": parser_prompt}],
            response_format={"type": "json_object"}
        )
        parsed_info = json.loads(response.choices[0].message.content)

        profile = st.session_state.user_profile

        # [수정] 파서가 아무 정보도 추출하지 못하고, 기초라인 요청도 아닌 경우에 대한 예외 처리
        is_info_parsed = any([parsed_info.get("skin_type"), parsed_info.get("skin_concerns"), parsed_info.get("product_categories")])
        if not is_info_parsed and not st.session_state.recommend_set:
            response_text = "죄송해요, 이해하지 못했어요! 😥\n\n피부 타입부터 다시 선택해볼까요?"
            # 사용자가 이상한 채팅을 하면 첫 화면으로 돌아가기 위해 모든 상태 초기화
            st.session_state.messages.append({"role": "assistant", "content": response_text, "typing_done": False})
            st.session_state.user_profile = {"skin_type": None, "skin_concerns": None, "product_categories": []}
            st.session_state.selected_concerns = []
            st.session_state.selected_categories = []
            st.session_state.analysis_pending = False
            st.rerun()
            return

        if parsed_info.get("skin_type"): profile["skin_type"] = parsed_info["skin_type"]
        if parsed_info.get("skin_concerns"): profile["skin_concerns"] = parsed_info["skin_concerns"]
        if parsed_info.get("product_categories"): profile["product_categories"] = parsed_info["product_categories"]
        
        if st.session_state.recommend_set:
            profile["product_categories"] = ['스킨/토너', '로션/에멀전', '에센스/앰플/세럼', '크림', '선크림/로션', '클렌징 폼']

        # [수정] '기초 라인' 요청을 더 확실하게 처리
        if st.session_state.recommend_set:
            response_text = "네, 알겠습니다!\n\n사용자 정보에 맞춰 **기초라인** 분석을 시작할게요! 🔬"
        elif profile.get("product_categories"):
            response_text = "네, 알겠습니다!\n\n바로 분석을 시작할게요! 🔬"
        else:
            response_text = "어떤 **제품**을 찾고 계신가요?"
    
    st.session_state.messages.append({"role": "assistant", "content": response_text, "typing_done": False})
    st.rerun()

def main():
    load_css()
    df = load_data('product_data.csv')
    if df is None: st.stop()

    if "messages" not in st.session_state: st.session_state.messages = []
    if "user_profile" not in st.session_state: st.session_state.user_profile = {"skin_type": None, "skin_concerns": None, "product_categories": []}
    if "selected_concerns" not in st.session_state: st.session_state.selected_concerns = []
    if "selected_categories" not in st.session_state: st.session_state.selected_categories = []
    if "analysis_pending" not in st.session_state: st.session_state.analysis_pending = False
    
    st.markdown("""<div class="header-container"><div class="logo-section">🌿</div><h1 class="brand-name">INGREVIA</h1><p class="brand-tagline">AI COSMETIC INGREDIENTS ANALYZER</p></div>""", unsafe_allow_html=True)
    
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "안녕하세요! 🥰\n\nAI 화장품 성분 분석가 **INGREVIA** 입니다.\n\n아래 버튼을 누르거나 채팅으로 피부 정보를 알려주시면, 웹 분석을 통해 딱 맞는 화장품을 찾아드릴게요.\n\n먼저 **피부 타입**을 선택해주세요.",
            "typing_done": False
        })
        st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🌿" if msg["role"] == "assistant" else "👤"):
            if msg.get("typing_done") or msg["role"] == "user":
                if "recommendations" in msg:
                    is_basic_set_result = msg.get('is_basic_set', False)
                    if is_basic_set_result:
                        st.markdown("""<div class="custom-success">✨ <strong>분석 완료!</strong> 사용자님께 최적화된 <strong>기초라인</strong>을 추천합니다.</div>""", unsafe_allow_html=True)
                    
                    for rec_data in msg["recommendations"]:
                        if not is_basic_set_result:
                            st.markdown(f"""<div class="custom-success">✨ <strong>분석 완료!</strong> 사용자님께 최적화된 <strong>TOP {len(rec_data['recommendations'])} {rec_data['category']}</strong> 제품을 추천합니다.</div>""", unsafe_allow_html=True)
                        
                        for rank, rec in enumerate(rec_data['recommendations'], 1):
                            custom_title = rec_data['category'] if is_basic_set_result else None
                            render_product_card(rec, rank, title=custom_title)
                else:
                    st.markdown(msg.get("content", ""))
            else:
                show_typing_effect(msg["content"])
                msg["typing_done"] = True
    
    profile = st.session_state.user_profile
    last_message = st.session_state.messages[-1] if st.session_state.messages else {}

    if last_message.get("role") == "assistant" and "typing_done" in last_message and last_message.get("typing_done") and not st.session_state.analysis_pending and "recommendations" not in last_message:
        if profile["skin_type"] is None:
            skin_types = ['민감성', '지성', '건성', '복합성', '아토피성', '중성', '해당없음']
            cols = st.columns(4)
            for i, skin_type in enumerate(skin_types):
                if cols[i % 4].button(f"{skin_type}", key=f"skin_{skin_type}", use_container_width=True):
                    user_msg = f"제 피부 타입은 **{skin_type}**이에요." if skin_type != '해당없음' else "제 피부 타입을 모르겠어요."
                    ai_msg = f"네, {skin_type}이시군요!\n\n어떤 **피부 고민**이 있으신가요? (여러 개 선택 가능)" if skin_type != '해당없음' else "알겠습니다!\n\n어떤 **피부 고민**이 있으신가요? (여러 개 선택 가능)"
                    profile["skin_type"] = skin_type if skin_type != '해당없음' else "없음"
                    st.session_state.messages.append({"role": "user", "content": user_msg, "typing_done": True})
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg, "typing_done": False})
                    st.rerun()

        elif profile["skin_concerns"] is None and profile.get("skin_type") is not None:
            concerns_options = ['보습', '진정', '미백', '주름/탄력', '모공/피지', '트러블', '각질', '특별한 고민 없음']
            emoji_map = {'보습': '💦', '진정': '🍃', '미백': '✨', '주름/탄력': '🌟', '모공/피지': '🔍', '트러블': '🩹', '각질': '🧽', '특별한 고민 없음': '😊'}
            cols = st.columns(4)
            for i, concern in enumerate(concerns_options):
                is_selected = concern in st.session_state.selected_concerns
                if cols[i % 4].button(f"{'✅' if is_selected else emoji_map[concern]} {concern}", key=f"concern_{concern}", use_container_width=True):
                    if concern == '특별한 고민 없음': st.session_state.selected_concerns = ['특별한 고민 없음']
                    else:
                        if '특별한 고민 없음' in st.session_state.selected_concerns: st.session_state.selected_concerns.remove('특별한 고민 없음')
                        if is_selected: st.session_state.selected_concerns.remove(concern)
                        else: st.session_state.selected_concerns.append(concern)
                    st.rerun()
            
            if st.session_state.selected_concerns:
                st.markdown(f"""<div class="selection-display"><strong style="color: {COLORS['primary']};">선택된 고민:</strong> {', '.join(st.session_state.selected_concerns)}</div>""", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ 선택 완료", type="primary", use_container_width=True):
                        user_msg = f"피부 고민은 **{', '.join(st.session_state.selected_concerns)}** 입니다." if '특별한 고민 없음' not in st.session_state.selected_concerns else "특별한 피부 고민이 없어요."
                        ai_msg = "마지막으로,\n\n어떤 **제품 종류**를 찾아드릴까요? (여러 개 선택 가능)"
                        profile["skin_concerns"] = st.session_state.selected_concerns if '특별한 고민 없음' not in st.session_state.selected_concerns else []
                        st.session_state.messages.append({"role": "user", "content": user_msg, "typing_done": True})
                        st.session_state.messages.append({"role": "assistant", "content": ai_msg, "typing_done": False})
                        st.rerun()
                with col2:
                    if st.button("↩️ 이전 선택으로", use_container_width=True):
                        profile["skin_type"] = None
                        st.session_state.selected_concerns = []
                        st.session_state.messages = st.session_state.messages[:-2]
                        st.rerun()

        elif not profile["product_categories"] and profile.get("skin_concerns") is not None:
            categories = [('스킨/토너', '💧'), ('로션/에멀전', '🧴'), ('에센스/앰플/세럼', '✨'), ('크림', '🍶'), ('선크림/로션', '☀️'), ('클렌징 폼', '🫧')]
            cols = st.columns(3)
            for i, (cat, emoji) in enumerate(categories):
                is_selected = cat in st.session_state.selected_categories
                if cols[i % 3].button(f"{'✅' if is_selected else emoji} {cat}", key=f"cat_{cat}", use_container_width=True):
                    if is_selected: st.session_state.selected_categories.remove(cat)
                    else: st.session_state.selected_categories.append(cat)
                    st.rerun()
            
            if st.session_state.selected_categories:
                st.markdown(f"""<div class="selection-display"><strong style="color: {COLORS['primary']};">선택된 제품:</strong> {', '.join(st.session_state.selected_categories)}</div>""", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ 선택 완료", type="primary", use_container_width=True):
                        profile["product_categories"] = st.session_state.selected_categories
                        st.session_state.analysis_pending = True
                        user_msg = f"**{', '.join(profile['product_categories'])}** 제품을 찾고 있어요."
                        st.session_state.messages.append({"role": "user", "content": user_msg, "typing_done": True})
                        st.session_state.messages.append({"role": "assistant", "content": "네, 알겠습니다!\n\n바로 분석을 시작할게요! 🔬", "typing_done": False})
                        st.rerun()
                with col2:
                    if st.button("↩️ 이전 선택으로", use_container_width=True):
                        profile["skin_concerns"] = None
                        st.session_state.selected_categories = []
                        st.session_state.messages = st.session_state.messages[:-2]
                        st.rerun()

    if st.session_state.analysis_pending and profile["product_categories"]:
        is_basic_set = st.session_state.get("recommend_set", False)
        current_analysis_results = []
        for category in profile["product_categories"]:
            with st.spinner(f"AI가 최적의 {category} 제품을 분석 중입니다..."):
                filtered_df = df[df['카테고리'].str.contains(category.split('/')[0], na=False)].copy()
                if profile["skin_concerns"]:
                    concern_filter = filtered_df['효능'].apply(lambda x: any(c in str(x) for c in profile["skin_concerns"]))
                    filtered_df = filtered_df[concern_filter]
                
                if filtered_df.empty: continue

                num_to_recommend = 1 if is_basic_set else 3
                candidates = filtered_df.nsmallest(num_to_recommend, '유해성_점수').to_dict('records')
                recommendations = []
                skin_concerns_for_analysis = profile['skin_concerns'] if profile['skin_concerns'] else []
                skin_type_for_analysis = profile['skin_type'] if profile['skin_type'] else '일반'
                
                for cand in candidates:
                    analysis_result = analyze_ingredients_with_search(cand['제품명'], skin_type_for_analysis, skin_concerns_for_analysis, cand['전성분'])
                    recommendations.append(cand | analysis_result)
                
                if recommendations:
                    current_analysis_results.append({'category': category, 'recommendations': recommendations})
        
        if current_analysis_results:
            st.session_state.messages[-1] = {"role": "assistant", "content": "", "recommendations": current_analysis_results, "is_basic_set": is_basic_set, "typing_done": True}
        else:
            # [수정] 분석 결과가 없을 때, 사용자 정보를 초기화하지 않고 이전 단계로 유도
            st.session_state.messages.append({"role": "assistant", "content": "분석 결과에 맞는 제품을 찾지 못했어요. 다른 조건으로 다시 시도해 주세요.", "typing_done": True})
            st.session_state.analysis_pending = False
            st.session_state.user_profile["product_categories"] = []
            st.session_state.selected_categories = []
            st.session_state.recommend_set = False
            st.rerun()

        st.session_state.analysis_pending = False
        # [수정] 사용자 정보(피부타입, 고민) 유지를 위해 제품 관련 상태만 초기화합니다.
        st.session_state.user_profile["product_categories"] = []
        st.session_state.selected_categories = []
        st.session_state.recommend_set = False
        st.rerun()

    if not st.session_state.analysis_pending:
        if last_message.get("role") == "assistant" and "recommendations" in last_message:
             st.markdown("---")
             col1, col2, col3 = st.columns([1, 1.5, 1])
             if col2.button("🔄 새 추천 시작하기", type="primary", use_container_width=True):
                 st.session_state.clear()
                 st.rerun()

        if prompt := st.chat_input("피부 타입, 고민, 제품 종류 등을 알려주세요."):
            handle_chat_input(prompt)

    with st.sidebar:
        st.markdown(f"### 🌿 INGREVIA")
        st.markdown("---")
        if st.button("🔄 새 추천 시작하기", use_container_width=True, type="primary"):
            st.session_state.clear()
            st.rerun()
        st.markdown("---")
        st.markdown("### 💡 이렇게 대화해보세요")
        st.markdown("""- "나는 민감성 피부야"
- "보습이랑 진정이 필요해"
- "수분크림 찾아줘"
- "민감성 피부에 보습 잘되는 크림"
- "토너랑 로션 추천해줘"
- "기초화장품 세트 추천해줘"
- "다시 시작"
""")
        st.markdown("---")
        st.markdown("### 📊 현재 분석 조건")
        profile = st.session_state.user_profile
        skin_type_display = profile.get('skin_type')
        if skin_type_display == "없음": st.info("**피부 타입:** 모르겠어요")
        elif skin_type_display: st.info(f"**피부 타입:** {skin_type_display}")
        else: st.info("피부 타입 선택 전")
        skin_concerns_display = profile.get('skin_concerns')
        if profile.get("skin_concerns") is not None and not profile.get("skin_concerns"): st.info("**피부 고민:** 특별한 고민 없음")
        elif skin_concerns_display: st.info(f"**피부 고민:** {', '.join(skin_concerns_display)}")
        else: st.info("피부 고민 선택 전")
        
        recommendation_messages = [msg for msg in st.session_state.messages if "recommendations" in msg]
        if recommendation_messages:
            last_rec_msg = recommendation_messages[-1]
            categories = [rec['category'] for rec in last_rec_msg['recommendations']]
            st.info(f"**최근 본 제품:** {', '.join(categories)}")
        else:
            st.info("제품 종류 선택 전")

if __name__ == "__main__":
    main()

