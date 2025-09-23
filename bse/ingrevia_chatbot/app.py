import streamlit as st
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import time
from typing import List, Dict, Any

# --- 1. 기본 설정 및 데이터 로딩 ---
st.set_page_config(
    page_title="INGREVIA | AI 화장품 성분 분석", 
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# .env 파일에서 API 키 불러오기
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 브랜드 색상 정의 (톤온톤 그린 계열)
COLORS = {
    'primary': '#2d5f3f',      # 진한 그린
    'secondary': '#3a7a50',    # 중간 그린
    'accent': '#4a9060',       # 밝은 그린
    'light': '#6fa570',        # 연한 그린
    'bg_main': '#f8faf8',      # 메인 배경
    'bg_card': '#ffffff',      # 카드 배경
    'border': '#e0e8e0',       # 테두리
    'text': '#2c3e2c',         # 텍스트
    'highlight': '#e8f5e9',    # 하이라이트
    'emoji_bg': '#d4e8d4'      # 이모지 배경
}

# 데이터 로딩 함수 (캐시 사용)
@st.cache_data
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        df['효능'] = df['효능'].fillna('')
        df['가격'] = pd.to_numeric(df['가격'], errors='coerce').fillna(0)
        df['유해성_점수'] = pd.to_numeric(df['유해성_점수'], errors='coerce').fillna(0)
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"데이터 로딩 중 오류 발생: {str(e)}")
        return None

# 개선된 AI 추천 함수
@st.cache_data
def get_ai_recommendations(skin_type: str, skin_concerns: List[str], 
                          product_category: str, filtered_df_json: str) -> List[Dict[str, Any]]:
    """AI를 사용하여 개인화된 제품 추천을 생성합니다."""
    filtered_df = pd.read_json(filtered_df_json)
    
    if filtered_df.empty:
        return []
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # 상위 30개 제품 선택
        candidate_df = filtered_df.nsmallest(30, '유해성_점수')
        
        # 제품 정보를 구조화된 형태로 준비
        products_info = []
        for _, row in candidate_df.iterrows():
            product = {
                'name': row['제품명'],
                'brand': row['브랜드명'],
                'ingredients': row['전성분'],
                'harmful_score': row['유해성_점수'],
                'efficacy': row['효능'],
                'price': row['가격'],
                'volume': row.get('용량', ''),
                'link': row.get('링크', '#')
            }
            products_info.append(product)
        
        # 피부 타입별 피해야 할 성분
        avoid_ingredients = {
            '민감성': ['알코올', '향료', '파라벤', '페녹시에탄올', 'PEG', '설페이트'],
            '지성': ['미네랄오일', '코코넛오일', '올리브오일', '라놀린', '바셀린'],
            '건성': ['알코올', '설페이트', '레티놀(고농도)', '벤조일퍼옥사이드'],
            '아토피성': ['향료', '알코올', '파라벤', '페녹시에탄올', '프로필렌글리콜', '염료'],
            '복합성': ['미네랄오일', '알코올(과도한)', '실리콘(과도한)'],
            '중성': [],  # 특별히 피해야 할 성분 없음
        }
        
        # 고민별 유익한 성분
        beneficial_ingredients = {
            '보습': ['히알루론산', '세라마이드', '글리세린', '스쿠알란', '판테놀', '베타인', '콜라겐'],
            '진정': ['센텔라아시아티카', '알로에베라', '판테놀', '마데카소사이드', '아줄렌', '카모마일', '티트리'],
            '미백': ['나이아신아마이드', '비타민C', '알부틴', '트라넥사믹애씨드', '알파알부틴', '감초추출물'],
            '주름/탄력': ['레티놀', '펩타이드', '아데노신', '콜라겐', '엘라스틴', '비타민E', 'EGF'],
            '모공/피지': ['나이아신아마이드', 'BHA', 'AHA', '위치하젤', '티트리', '녹차추출물', '징크옥사이드'],
            '트러블': ['살리실산', '티트리', '센텔라', '프로폴리스', '아연', '칼라민'],
            '각질': ['AHA', 'BHA', 'PHA', '요소', '젖산', '글리콜산']
        }
        
        # 개선된 프롬프트
        prompt = f"""
        당신은 화장품 성분 분석 전문가입니다. 화해 웹사이트의 성분 정보를 우선적으로 참고하여 분석합니다.

        [사용자 정보]
        - 피부 타입: {skin_type}
        - 피부 고민: {', '.join(skin_concerns)}
        - 찾는 제품: {product_category}

        [피부 타입별 피해야 할 성분]
        {avoid_ingredients.get(skin_type, [])}

        [고민별 유익한 성분]
        {[beneficial_ingredients.get(concern, []) for concern in skin_concerns if concern != '해당 없음']}

        [분석 대상 제품]
        {json.dumps(products_info, ensure_ascii=False, indent=2)}

        [분석 지침]
        1. 각 제품의 전성분을 분석하여 사용자 피부 타입과 고민에 유익한 성분 개수를 정확히 계산
        2. 사용자 피부 타입에 해로운 성분이 있는지 확인
        3. 유익한 성분 개수가 많은 순서로 정렬
        4. 유익한 성분 개수가 같다면 유해성_점수가 낮은 순서로 정렬
        5. 화해 사이트 기준으로 성분 평가

        [출력 형식]
        정확히 3개 제품을 선정하여 JSON 형식으로 반환:
        {{
            "recommendations": [
                {{
                    "name": "제품명",
                    "brand": "브랜드명",
                    "price": 가격(숫자),
                    "volume": "용량",
                    "link": "링크",
                    "beneficial_count": 유익한 성분 개수,
                    "harmful_score": 유해성점수,
                    "beneficial_ingredients": ["유익한 성분1", "성분2"],
                    "caution_ingredients": ["주의해야 할 성분1", "성분2"],
                    "reason": "이 제품이 사용자에게 적합한 구체적 이유"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 화해 사이트의 성분 정보를 기반으로 정확한 분석을 제공하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=1500
        )
        
        result_json = json.loads(response.choices[0].message.content)
        return result_json.get("recommendations", [])
        
    except Exception as e:
        st.error(f"AI 추천 중 오류 발생: {str(e)}")
        # 폴백: 유해성 점수 기준 상위 3개 반환
        top_products = filtered_df.nsmallest(3, '유해성_점수')
        fallback = []
        for _, row in top_products.iterrows():
            fallback.append({
                "name": row['제품명'],
                "brand": row['브랜드명'],
                "price": row['가격'],
                "volume": row.get('용량', ''),
                "link": row.get('링크', '#'),
                "harmful_score": row['유해성_점수'],
                "reason": "유해성 점수가 낮은 안전한 제품입니다."
            })
        return fallback

# --- 2. 고급 UI 스타일링 ---
def load_css():
    st.markdown(f"""
    <style>
        /* 전체 앱 스타일 */
        .stApp {{
            background: linear-gradient(135deg, {COLORS['bg_main']} 0%, #f0f7f0 100%);
        }}
        
        /* 메인 컨테이너 */
        .main .block-container {{
            padding: 2rem;
            max-width: 900px;
            margin: 0 auto;
        }}
        
        /* 헤더 스타일 */
        .header-container {{
            background: white;
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 4px 20px rgba(45, 95, 63, 0.1);
            margin-bottom: 2rem;
            text-align: center;
            border: 2px solid {COLORS['border']};
        }}
        
        .logo-section {{
            font-size: 4rem;
            margin-bottom: 1rem;
            filter: hue-rotate(-10deg) brightness(0.9);
        }}
        
        .brand-name {{
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['accent']} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            letter-spacing: 3px;
        }}
        
        .brand-tagline {{
            color: {COLORS['secondary']};
            font-size: 0.9rem;
            font-weight: 300;
            letter-spacing: 2px;
            margin-top: 0.5rem;
        }}
        
        /* 채팅 메시지 스타일 */
        .stChatMessage {{
            background: white;
            border-radius: 15px;
            padding: 1.2rem;
            margin: 1rem 0;
            box-shadow: 0 2px 10px rgba(45, 95, 63, 0.08);
            border-left: 4px solid {COLORS['accent']};
        }}
        
        /* 채팅 아이콘 스타일 */
        .stChatMessage .stMarkdown {{
            color: {COLORS['text']};
        }}
        
        /* 버튼 스타일 */
        .stButton > button {{
            background: linear-gradient(135deg, white 0%, {COLORS['highlight']} 100%);
            color: {COLORS['primary']};
            border: 2px solid {COLORS['border']};
            border-radius: 30px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 2px 5px rgba(45, 95, 63, 0.1);
        }}
        
        .stButton > button:hover {{
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(45, 95, 63, 0.2);
            border-color: {COLORS['primary']};
        }}
        
        /* Primary 버튼 */
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['accent']} 100%);
            color: white;
            border: none;
        }}
        
        /* 진행 상태 표시 */
        .progress-step {{
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
            text-align: center;
            font-weight: 500;
        }}
        
        /* 제품 카드 스타일 */
        .product-card {{
            background: white;
            border-radius: 20px;
            padding: 2rem;
            margin: 1.5rem 0;
            box-shadow: 0 4px 15px rgba(45, 95, 63, 0.1);
            transition: all 0.3s ease;
            border: 2px solid {COLORS['border']};
            position: relative;
        }}
        
        .product-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(45, 95, 63, 0.15);
            border-color: {COLORS['accent']};
        }}
        
        .product-rank {{
            position: absolute;
            top: 1.5rem;
            right: 1.5rem;
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['accent']} 100%);
            color: white;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1.3rem;
            box-shadow: 0 3px 10px rgba(45, 95, 63, 0.3);
        }}
        
        .product-title {{
            color: {COLORS['primary']};
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
            text-decoration: none;
            display: inline-block;
            transition: color 0.3s ease;
        }}
        
        .product-title:hover {{
            color: {COLORS['accent']};
            text-decoration: underline;
        }}
        
        .product-brand {{
            color: {COLORS['secondary']};
            font-size: 1rem;
            font-weight: 500;
            margin-bottom: 1.2rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .product-info-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin: 1.5rem 0;
        }}
        
        .info-item {{
            background: linear-gradient(135deg, {COLORS['highlight']} 0%, {COLORS['emoji_bg']} 100%);
            padding: 0.8rem 1rem;
            border-radius: 12px;
            font-size: 0.95rem;
            border: 1px solid {COLORS['border']};
        }}
        
        .info-label {{
            color: {COLORS['secondary']};
            font-weight: 600;
            display: inline-block;
            margin-right: 0.5rem;
        }}
        
        .ingredients-section {{
            margin-top: 1.5rem;
            padding: 1rem;
            background: {COLORS['bg_main']};
            border-radius: 12px;
            border: 1px solid {COLORS['border']};
        }}
        
        .ingredients-title {{
            color: {COLORS['primary']};
            font-weight: 600;
            margin-bottom: 0.8rem;
            font-size: 1rem;
        }}
        
        .ingredient-badge {{
            display: inline-block;
            padding: 0.4rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            margin: 0.3rem;
            font-weight: 500;
        }}
        
        .beneficial-badge {{
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            color: {COLORS['primary']};
            border: 1px solid #a5d6a7;
        }}
        
        .caution-badge {{
            background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
            color: #e65100;
            border: 1px solid #ffcc80;
        }}
        
        .matching-info {{
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
            color: white;
            padding: 0.8rem 1.2rem;
            border-radius: 12px;
            margin-top: 1rem;
            font-size: 0.95rem;
            box-shadow: 0 3px 10px rgba(45, 95, 63, 0.2);
        }}
        
        /* 선택된 항목 표시 */
        .selection-display {{
            background: linear-gradient(135deg, {COLORS['highlight']} 0%, white 100%);
            border-radius: 15px;
            padding: 1.2rem;
            margin: 1rem 0;
            border: 2px solid {COLORS['border']};
            box-shadow: 0 2px 8px rgba(45, 95, 63, 0.1);
        }}
        
        /* 스피너 스타일 */
        .stSpinner > div {{
            border-color: {COLORS['primary']} !important;
        }}
        
        /* 이모지 스타일링 */
        .emoji-icon {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 30px;
            height: 30px;
            background: {COLORS['emoji_bg']};
            border-radius: 50%;
            margin-right: 0.5rem;
            font-size: 1.2rem;
        }}
    </style>
    """, unsafe_allow_html=True)

# --- 3. 세션 상태 관리 ---
def initialize_session():
    """세션 상태를 초기화하고 첫 질문을 표시"""
    st.session_state.clear()
    st.session_state.step = "start"
    st.session_state.user_info = {}
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": """
            <br>            
            안녕하세요! 🥰 AI 화장품 성분 분석 전문가 <strong>INGREVIA</strong> 입니다

            당신의 피부 타입과 고민 분석 및 웹서치를 기반으로
            맞춤형 화장품을 추천해드리겠습니다 💉
            
            **먼저, 피부 타입을 선택해주세요:**
            """
        }
    ]
    st.session_state.selected_concerns = []
    st.session_state.analysis_complete = False

# --- 4. 메인 앱 ---
def main():
    # CSS 로드
    load_css()
    
    # 데이터 로드
    df = load_data('product_data.csv')
    if df is None:
        st.error("⚠️ `product_data.csv` 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요!")
        st.stop()
    
    # 세션 초기화
    if "step" not in st.session_state:
        initialize_session()
    
    # 헤더 섹션
    st.markdown("""
        <div class="header-container">
            <div class="logo-section">🌿</div>
            <h1 class="brand-name">INGREVIA</h1>
            <p class="brand-tagline">AI COSMETIC INGREDIENTS ANALYZER</p>
        </div>
    """, unsafe_allow_html=True)
    
    # 진행 상태 표시
    if st.session_state.step != "start":
        progress_steps = ["🌱 피부 타입", "💧 피부 고민", "🧴 제품 선택", "🔬 AI 분석"]
        current_step = {"ask_concerns": 1, "ask_category": 2, "show_results": 3, "done": 3}.get(st.session_state.step, 0)
        
        cols = st.columns(len(progress_steps))
        for i, (col, step_name) in enumerate(zip(cols, progress_steps)):
            with col:
                if i < current_step:
                    st.success(f"✓ {step_name}")
                elif i == current_step:
                    st.info(f"→ {step_name}")
                else:
                    st.text(f"○ {step_name}")
    
    # 대화 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🌿" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"], unsafe_allow_html=True)
    
    # 단계별 UI 렌더링
    if st.session_state.step == "start":
        skin_types = ['민감성', '지성', '건성', '아토피성', '복합성', '중성', '해당 없음']
        cols = st.columns(4)
        for i, skin_type in enumerate(skin_types):
            with cols[i % 4]:
                if st.button(f"🫧 {skin_type}", key=f"skin_{skin_type}", use_container_width=True):
                    st.session_state.user_info["skin_type"] = skin_type
                    st.session_state.messages.append({"role": "user", "content": f"<span class='emoji-icon'>🫧</span> **{skin_type}** 피부입니다"})
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "💧 피부 고민을 모두 선택해주세요. (복수 선택 가능)"
                    })
                    st.session_state.step = "ask_concerns"
                    st.rerun()
    
    elif st.session_state.step == "ask_concerns":
        concerns_options = ['보습', '진정', '미백', '주름/탄력', '모공/피지', '트러블', '각질', '해당 없음']
        
        # 선택된 고민 표시
        if st.session_state.selected_concerns:
            st.markdown(f"""
                <div class="selection-display">
                    <strong>💧 선택된 고민:</strong> {', '.join([f'<span class="beneficial-badge">{c}</span>' for c in st.session_state.selected_concerns])}
                </div>
            """, unsafe_allow_html=True)
        
        # 고민 선택 버튼
        cols = st.columns(4)
        for i, concern in enumerate(concerns_options):
            with cols[i % 4]:
                emoji_map = {
                    '보습': '💦', '진정': '🍃', '미백': '✨', '주름/탄력': '🌟',
                    '모공/피지': '🔍', '트러블': '🩹', '각질': '🧽', '해당 없음': '⭕'
                }
                emoji = emoji_map.get(concern, '○')
                
                button_label = f"{emoji} {concern}"
                if concern in st.session_state.selected_concerns:
                    button_label = f"✅ {concern}"
                
                if st.button(button_label, key=f"concern_{concern}", use_container_width=True):
                    if concern == '해당 없음':
                        st.session_state.selected_concerns = ['해당 없음']
                    elif '해당 없음' in st.session_state.selected_concerns:
                        st.session_state.selected_concerns = [concern]
                    elif concern in st.session_state.selected_concerns:
                        st.session_state.selected_concerns.remove(concern)
                    else:
                        st.session_state.selected_concerns.append(concern)
                    st.rerun()
        
        # 선택 완료 버튼
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✅ 선택 완료", type="primary", use_container_width=True, 
                        disabled=not st.session_state.selected_concerns):
                st.session_state.user_info["skin_concerns"] = st.session_state.selected_concerns
                st.session_state.messages.append({
                    "role": "user", 
                    "content": f"<span class='emoji-icon'>💧</span> 피부 고민: **{', '.join(st.session_state.selected_concerns)}**"
                })
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "🧴 마지막으로, 어떤 종류의 제품을 찾으시나요?"
                })
                st.session_state.step = "ask_category"
                st.rerun()
    
    elif st.session_state.step == "ask_category":
        categories = [
            ('스킨/토너', '💧'), ('로션/에멀전', '🧴'), ('에센스/앰플/세럼', '✨'), ('크림', '🍶'), 
            ('밤/멀티밤', '🌟'), ('클렌징 폼', '🫧'), ('시트마스크', '🎭'), ('선크림/로션', '☀️')
        ]
        
        cols = st.columns(4)
        for i, (category, emoji) in enumerate(categories):
            with cols[i % 4]:
                if st.button(f"{emoji} {category}", key=f"cat_{category}", use_container_width=True):
                    st.session_state.user_info["product_category"] = category
                    st.session_state.messages.append({"role": "user", "content": f"<span class='emoji-icon'>{emoji}</span> **{category}** 제품을 찾고 있습니다"})
                    st.session_state.step = "show_results"
                    st.rerun()
    
    elif st.session_state.step == "show_results":
        with st.chat_message("assistant", avatar="🌿"):
            with st.spinner("🔬 AI가 성분을 분석하고 최적의 제품을 찾고 있습니다..."):
                # 제품 필터링
                filtered_df = df[df['카테고리'] == st.session_state.user_info["product_category"]].copy()
                
                # 효능 필터링
                if '해당 없음' not in st.session_state.user_info["skin_concerns"]:
                    concern_filter = filtered_df['효능'].apply(
                        lambda x: any(concern in str(x) for concern in st.session_state.user_info["skin_concerns"])
                    )
                    filtered_df = filtered_df[concern_filter]
                
                if filtered_df.empty:
                    st.warning("😔 조건에 맞는 제품을 찾지 못했습니다. 다른 조건으로 다시 시도해주세요.")
                    if st.button("🔄 처음부터 다시 시작", type="primary"):
                        initialize_session()
                        st.rerun()
                else:
                    # AI 추천 받기
                    recommendations = get_ai_recommendations(
                        st.session_state.user_info["skin_type"],
                        st.session_state.user_info["skin_concerns"],
                        st.session_state.user_info["product_category"],
                        filtered_df.to_json()
                    )
                    
                    if recommendations:
                        st.success(f"""
                        ✨ **분석 완료!** 
                        
                        {st.session_state.user_info['skin_type']} 피부와 {', '.join(st.session_state.user_info['skin_concerns'])} 고민에 
                        최적화된 TOP 3 제품을 추천합니다.
                        """)
                        
                        # 제품 카드 표시 (심플 버전)
                        for i, rec in enumerate(recommendations[:3], 1):
                            # 순위별 이모지
                            rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}[i]
                            
                            # 제품 카드 컨테이너
                            with st.container():
                                st.markdown(f"""
                                <style>
                                    .product-card-{i} {{
                                        background: white;
                                        border-radius: 20px;
                                        padding: 2rem;
                                        margin: 1.5rem 0;
                                        box-shadow: 0 4px 15px rgba(45, 95, 63, 0.1);
                                        border: 2px solid #e0e8e0;
                                    }}
                                    .product-link {{
                                        color: #2d5f3f;
                                        text-decoration: none;
                                        font-size: 1.5rem;
                                        font-weight: 700;
                                    }}
                                    .product-link:hover {{
                                        color: #4a9060;
                                        text-decoration: underline;
                                    }}
                                </style>
                                <div class="product-card-{i}">
                                    <h2>{rank_emoji} TOP {i}</h2>
                                    <h3><a href="{rec.get('link', '#')}" target="_blank" class="product-link">{rec.get('name', '')}</a></h3>
                                    <p><strong>🏢 브랜드:</strong> {rec.get('brand', '')}</p>
                                    <p><strong>💰 가격:</strong> {int(rec.get('price', 0)):,}원</p>
                                    <p><strong>📏 용량:</strong> {rec.get('volume', '')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.error("추천 생성 중 문제가 발생했습니다. 다시 시도해주세요.")
        
        st.session_state.step = "done"
    
    elif st.session_state.step == "done":
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 새로운 제품 추천받기", type="primary", use_container_width=True):
                initialize_session()
                st.rerun()
            
            # 추가 기능 버튼들
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("📊 상세 분석 보기", use_container_width=True):
                    st.info("상세 성분 분석 기능은 준비 중입니다.")
            with col_b:
                if st.button("💾 추천 저장하기", use_container_width=True):
                    st.info("추천 저장 기능은 준비 중입니다.")
    
    # 채팅 입력 처리 (모든 단계에서 활성화)
    if st.session_state.step != "start":
        user_input = st.chat_input("💬 메시지를 입력하세요... (예: 다시 시작, 처음부터, 민감성 피부)")
        
        if user_input:
            # 사용자 입력 표시
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # 입력 처리
            user_input_lower = user_input.lower().strip()
            
            # 다시 시작 명령
            if any(keyword in user_input_lower for keyword in ['다시 시작', '처음부터', '초기화', '리셋', 'reset', 'restart']):
                initialize_session()
                st.rerun()
            
            # 피부 타입 감지
            elif st.session_state.step in ["start", "ask_concerns"]:
                skin_types_map = {
                    '민감': '민감성', '민감성': '민감성',
                    '지성': '지성', '기름': '지성', '번들': '지성',
                    '건성': '건성', '건조': '건성',
                    '아토피': '아토피성', '아토피성': '아토피성',
                    '복합': '복합성', '복합성': '복합성',
                    '중성': '중성', '보통': '중성'
                }
                
                for keyword, skin_type in skin_types_map.items():
                    if keyword in user_input_lower:
                        st.session_state.user_info["skin_type"] = skin_type
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"🌿 {skin_type} 피부로 선택하셨네요! 이제 피부 고민을 선택해주세요."
                        })
                        st.session_state.step = "ask_concerns"
                        st.rerun()
                        break
            
            # 피부 고민 감지
            elif st.session_state.step == "ask_concerns":
                concerns_map = {
                    '보습': '보습', '건조': '보습', '수분': '보습',
                    '진정': '진정', '민감': '진정', '빨갛': '진정',
                    '미백': '미백', '화이트': '미백', '브라이트': '미백', '기미': '미백',
                    '주름': '주름/탄력', '탄력': '주름/탄력', '노화': '주름/탄력',
                    '모공': '모공/피지', '피지': '모공/피지', '블랙헤드': '모공/피지',
                    '트러블': '트러블', '여드름': '트러블', '뾰루지': '트러블',
                    '각질': '각질', '죽은세포': '각질'
                }
                
                detected_concerns = []
                for keyword, concern in concerns_map.items():
                    if keyword in user_input_lower:
                        if concern not in detected_concerns:
                            detected_concerns.append(concern)
                
                if detected_concerns:
                    st.session_state.selected_concerns = detected_concerns
                    st.session_state.user_info["skin_concerns"] = detected_concerns
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"💧 {', '.join(detected_concerns)} 고민을 선택하셨습니다! 이제 찾으시는 제품 종류를 선택해주세요."
                    })
                    st.session_state.step = "ask_category"
                    st.rerun()
                
                # "완료", "다음" 등의 명령 처리
                elif any(keyword in user_input_lower for keyword in ['완료', '다음', '선택', 'ok']):
                    if st.session_state.selected_concerns:
                        st.session_state.user_info["skin_concerns"] = st.session_state.selected_concerns
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "🧴 마지막으로, 어떤 종류의 제품을 찾으시나요?"
                        })
                        st.session_state.step = "ask_category"
                        st.rerun()
            
            # 제품 카테고리 감지
            elif st.session_state.step == "ask_category":
                category_map = {
                    '토너': '스킨/토너', '스킨': '스킨/토너', '스킨토너': '스킨/토너',
                    '로션': '로션/에멀전', '에멀전': '로션/에멀전', '에멀젼': '로션/에멀전',
                    '에센스': '에센스/앰플/세럼', '앰플': '에센스/앰플/세럼', '세럼': '에센스/앰플/세럼',
                    '크림': '크림', '수분크림': '크림', '영양크림': '크림',
                    '밤': '밤/멀티밤', '멀티밤': '밤/멀티밤',
                    '클렌징': '클렌징 폼', '폼': '클렌징 폼', '세안': '클렌징 폼',
                    '마스크': '시트마스크', '시트': '시트마스크', '팩': '시트마스크',
                    '선크림': '선크림/로션', '자외선': '선크림/로션', 'spf': '선크림/로션'
                }
                
                for keyword, category in category_map.items():
                    if keyword in user_input_lower:
                        st.session_state.user_info["product_category"] = category
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"🧴 {category} 제품을 선택하셨습니다! 지금 AI가 분석을 시작합니다..."
                        })
                        st.session_state.step = "show_results"
                        st.rerun()
                        break
            
            # 알 수 없는 명령
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "죄송합니다, 이해하지 못했어요. 버튼을 클릭하시거나 '다시 시작'이라고 입력해주세요."
                })
                st.rerun()
    
    # 우측 상단 다시 시작 버튼 (항상 표시)
    with st.sidebar:
        st.markdown("### 🌿 INGREVIA")
        st.markdown("---")
        
        if st.button("🔄 처음부터 다시 시작", use_container_width=True, type="primary"):
            initialize_session()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 💡 사용 팁")
        st.markdown("""
        **채팅으로도 대화할 수 있어요!**
        
        예시:
        - "민감성 피부예요"
        - "보습이랑 진정이 필요해요"
        - "토너 추천해주세요"
        - "다시 시작할게요"
        
        **빠른 명령어:**
        - 다시 시작 / 처음부터
        - 완료 / 다음
        """)
        
        st.markdown("---")
        st.markdown("### 📊 현재 상태")
        if 'user_info' in st.session_state:
            if 'skin_type' in st.session_state.user_info:
                st.info(f"피부 타입: {st.session_state.user_info['skin_type']}")
            if 'skin_concerns' in st.session_state.user_info:
                st.info(f"피부 고민: {', '.join(st.session_state.user_info['skin_concerns'])}")
            if 'product_category' in st.session_state.user_info:
                st.info(f"제품: {st.session_state.user_info['product_category']}")
    
    # 푸터
    st.markdown("---")
    st.markdown(f"""
        <div style="text-align: center; color: {COLORS['secondary']}; padding: 2rem 0;">
            <p style="font-size: 0.9rem;">
                🌿 <strong>INGREVIA</strong> - AI Cosmetic Ingredients Analyzer<br/>
                <span style="font-size: 0.8rem;">Powered by GPT-4 & Streamlit</span>
            </p>
        </div>
    """, unsafe_allow_html=True)

# --- 5. 앱 실행 ---
if __name__ == "__main__":
    main()