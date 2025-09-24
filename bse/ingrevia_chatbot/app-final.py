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

# 브랜드 색상 정의
COLORS = {
    'primary': '#2d5f3f', 'secondary': '#3a7a50', 'accent': '#4a9060',
    'light': '#6fa570', 'bg_main': '#f8faf8', 'bg_card': '#ffffff',
    'border': '#e0e8e0', 'text': '#2c3e2c', 'highlight': '#e8f5e9',
    'emoji_bg': '#d4e8d4'
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


# --- 3. UI 스타일링 및 렌더링 ---
def load_css():
    st.markdown(f"""
    <style>
        .stApp {{ background: linear-gradient(135deg, {COLORS['bg_main']} 0%, #f0f7f0 100%); }}
        .main .block-container {{ padding: 1rem 2rem 2rem 2rem; max-width: 900px; margin: 0 auto; }}
        .header-container {{ background: {COLORS['bg_card']}; border-radius: 20px; padding: 2rem; box-shadow: 0 4px 20px rgba(45, 95, 63, 0.1); margin-bottom: 2rem; text-align: center; border: 2px solid {COLORS['border']}; }}
        .logo-section {{ font-size: 4rem; margin-bottom: 1rem; }}
        .brand-name {{ font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['accent']} 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; letter-spacing: 3px; }}
        .brand-tagline {{ color: {COLORS['secondary']}; font-size: 0.9rem; font-weight: 300; letter-spacing: 2px; margin-top: 0.5rem; }}
        .stChatMessage {{ background: {COLORS['bg_card']}; border-radius: 15px; padding: 1.2rem; margin: 1rem 0; box-shadow: 0 2px 10px rgba(45, 95, 63, 0.08); border-left: 4px solid {COLORS['accent']}; }}
        .product-card {{ background: {COLORS['bg_card']}; border-radius: 20px; padding: 2rem; box-shadow: 0 4px 15px rgba(45, 95, 63, 0.1); border: 2px solid {COLORS['border']}; }}
        .product-title a {{ color: {COLORS['primary']}; font-size: 1.4rem; font-weight: 700; text-decoration: none; }}
        .product-brand {{ color: {COLORS['secondary']}; margin-bottom: 1rem; }}
        .ingredients-section {{ margin-top: 1rem; }}
        .ingredients-title {{ color: {COLORS['primary']}; font-weight: 600; margin-bottom: 0.8rem; }}
        .ingredient-badge {{ display: inline-block; padding: 0.4rem 0.8rem; border-radius: 20px; font-size: 0.85rem; margin: 0.3rem; font-weight: 500; }}
        .beneficial-badge {{ background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); color: {COLORS['primary']}; border: 1px solid #a5d6a7; }}
        .caution-badge {{ background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); color: #e65100; border: 1px solid #ffcc80; }}
        .selection-display {{ background: {COLORS['highlight']}; border-radius: 15px; padding: 1rem; margin-top: 1rem; border: 2px solid {COLORS['border']}; }}
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
            <h2 style="margin-bottom: 0.5rem;">{rank_emoji} TOP {rank}</h2>
            <h3 class="product-title"><a href="{rec.get('링크', '#')}" target="_blank">{rec.get('제품명', '이름 없음')}</a></h3>
            <p class="product-brand"><strong>🏢 브랜드:</strong> {rec.get('브랜드명', '브랜드 없음')}</p>
            <p><strong>💰 가격:</strong> {int(rec.get('가격', 0)):,}원 | <strong>📏 용량:</strong> {rec.get('용량', '정보 없음')}</p>
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.info(f"**💬 추천 포인트:** {rec.get('reason', '추천 이유 분석 중...')}")
            
            st.markdown(f"""
            <div class="ingredients-section" style="margin-top:0.5rem; border:none; background:transparent; padding:0.5rem;">
                <p class="ingredients-title">✅ 이 제품의 핵심 효능 성분</p>
                <div>{beneficial_html}</div>
            </div>
            <div class="ingredients-section" style="margin-top:1rem; border:none; background:transparent; padding:0.5rem;">
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
        # UI 개선: 불필요한 줄바꿈(\n\n) 제거하여 공백 축소
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
            # UI 개선: 버튼 앞 구분선(---) 제거하여 공백 축소
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
                st.markdown(f"""<div class="selection-display"><strong>선택된 고민:</strong> {', '.join(st.session_state.selected_concerns)}</div>""", unsafe_allow_html=True)
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
        st.markdown("### 🌿 INGREVIA")
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