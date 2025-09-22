import streamlit as st
import pandas as pd
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time
import json

# --- ⚙️ 환경 설정 ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    st.error("❌ .env 파일에서 OPENAI_API_KEY를 찾을 수 없습니다.")
    st.stop()

# OpenAI 클라이언트 설정
client = OpenAI(api_key=api_key)

# LangChain 설정
@st.cache_resource
def init_langchain():
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7, openai_api_key=api_key)
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    
    template = """당신은 친절하고 전문적인 화장품 추천 전문가 'INGREVIA'입니다.
    사용자의 피부 타입과 고민을 바탕으로 최적의 화장품을 추천합니다.
    이모지를 적절히 사용하여 친근하고 읽기 쉽게 대답합니다.
    
    {history}
    
    사용자: {input}
    INGREVIA:"""
    
    prompt = PromptTemplate(input_variables=["history", "input"], template=template)
    conversation = ConversationChain(llm=llm, memory=memory, prompt=prompt, verbose=False)
    return conversation

# --- 🎨 페이지 설정 ---
st.set_page_config(
    page_title="INGREVIA 챗봇",
    page_icon="🌿",
    layout="wide"
)

# --- 🎨 CSS 스타일링 ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .chat-container {
        background: white;
        border-radius: 20px;
        padding: 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .chat-header {
        background: linear-gradient(135deg, #2d5a27 0%, #5a9754 100%);
        color: white;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 0;
    }
    
    .chat-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 300;
        letter-spacing: 2px;
    }
    
    .chat-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .message-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background: #fafafa;
    }
    
    .chat-message {
        display: flex;
        margin-bottom: 1rem;
        animation: slideIn 0.3s ease-out;
    }
    
    .bot-message {
        justify-content: flex-start;
    }
    
    .user-message {
        justify-content: flex-end;
    }
    
    .message-content {
        max-width: 70%;
        padding: 1rem 1.5rem;
        border-radius: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .bot-message .message-content {
        background: linear-gradient(135deg, #e8f5e8 0%, #d4f1d4 100%);
        border-bottom-left-radius: 5px;
    }
    
    .user-message .message-content {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-bottom-right-radius: 5px;
    }
    
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin: 0 0.5rem;
        flex-shrink: 0;
    }
    
    .bot-avatar {
        background: #5a9754;
        color: white;
    }
    
    .user-avatar {
        background: #2196f3;
        color: white;
    }
    
    .product-card {
        background: white;
        border: 2px solid #5a9754;
        border-radius: 15px;
        padding: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .product-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .product-rank {
        font-size: 2rem;
        font-weight: bold;
        color: #5a9754;
        margin-bottom: 0.5rem;
    }
    
    .product-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2d5a27;
        margin-bottom: 0.8rem;
    }
    
    .product-info {
        font-size: 0.9rem;
        color: #555;
        line-height: 1.6;
        margin: 0.3rem 0;
    }
    
    .ingredient-tag {
        display: inline-block;
        background: #e8f5e8;
        color: #2d5a27;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.85rem;
        margin: 0.2rem;
    }
    
    .quick-reply-container {
        padding: 1rem;
        background: white;
        border-top: 1px solid #eee;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
</style>
""", unsafe_allow_html=True)

# --- 📊 데이터 로딩 ---
@st.cache_data
def load_product_data():
    """CSV 파일에서 제품 데이터 로드"""
    csv_path = 'product_data.csv'
    
    # CSV 파일이 있으면 로드, 없으면 샘플 데이터 사용
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # 데이터 정보 출력 (디버깅용)
            print(f"✅ 제품 데이터 로드 완료: {len(df)}개 제품")
            print(f"카테고리 종류: {df['카테고리'].unique()}")
            return df
        except Exception as e:
            st.warning(f"CSV 파일 로드 중 오류 발생: {e}")
    
    # 샘플 데이터
    return pd.DataFrame([
        {
            "브랜드명": "라네즈", "제품명": "수분크림 모이스처라이징", "카테고리": "크림",
            "효능": "보습,진정", "전성분": "히알루론산,세라마이드,알로에베라,나이아신아마이드,글리세린",
            "가격": 35000, "용량": "50ml", "유해성_점수": 2.1
        },
        {
            "브랜드명": "이니스프리", "제품명": "그린티 수분크림", "카테고리": "크림",
            "효능": "보습,진정", "전성분": "녹차추출물,히알루론산,세라마이드,알로에베라",
            "가격": 28000, "용량": "50ml", "유해성_점수": 1.8
        },
        {
            "브랜드명": "설화수", "제품명": "윤조에센스", "카테고리": "에센스/세럼/앰플",
            "효능": "보습,미백,주름/탄력", "전성분": "자초,당귀,작약,나이아신아마이드,히알루론산",
            "가격": 180000, "용량": "60ml", "유해성_점수": 3.2
        },
        {
            "브랜드명": "아이오페", "제품명": "레티놀 엑스퍼트", "카테고리": "에센스/세럼/앰플",
            "효능": "주름/탄력,미백", "전성분": "레티놀,펩타이드,나이아신아마이드,아데노신",
            "가격": 65000, "용량": "30ml", "유해성_점수": 2.5
        },
        {
            "브랜드명": "닥터자르트", "제품명": "시카페어 크림", "카테고리": "크림",
            "효능": "진정,보습", "전성분": "센텔라아시아티카,마데카소사이드,아시아티코사이드,판테놀",
            "가격": 42000, "용량": "50ml", "유해성_점수": 1.5
        }
    ])

# --- 🎯 상태 초기화 ---
if 'conversation' not in st.session_state:
    st.session_state.conversation = init_langchain()

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """안녕하세요! 🌿 **INGREVIA**입니다.
            
당신의 피부에 딱 맞는 화장품을 찾아드릴게요!

저는 AI 기술과 화장품 성분 데이터베이스를 활용해 맞춤 추천을 제공합니다.

먼저 **피부 타입**을 알려주세요:""",
            "quick_replies": ["🌸 민감성", "💧 지성", "🏜️ 건성", "🔴 아토피성"]
        }
    ]

if 'current_step' not in st.session_state:
    st.session_state.current_step = 'skin_type'
    st.session_state.user_data = {}

if 'typing' not in st.session_state:
    st.session_state.typing = False

# --- 🤖 핵심 성분 추출 함수 (LangChain 활용) ---
def get_key_ingredients_from_llm(skin_type, concerns):
    """LangChain을 활용하여 핵심 성분 추출"""
    prompt = f"""
    화장품 정보 사이트 '화해'의 정보를 참고하여 다음 조건에 맞는 화장품 핵심 성분을 찾아주세요.
    
    - 피부 타입: {skin_type}
    - 주요 고민: {', '.join(concerns)}
    
    이 조건에 가장 효과적인 성분 5-7개를 선정해주세요.
    각 성분의 효능도 간단히 설명해주세요.
    
    응답 형식:
    1. 성분명: 효능
    2. 성분명: 효능
    ...
    
    마지막에 성분명만 쉼표로 구분해서 한 줄로 정리해주세요.
    """
    
    try:
        response = st.session_state.conversation.predict(input=prompt)
        
        # 성분명만 추출 (마지막 줄에서)
        lines = response.strip().split('\n')
        for line in reversed(lines):
            if ',' in line and not ':' in line:
                ingredients = [ing.strip().lower() for ing in line.split(',')]
                return ingredients, response
        
        # 기본값
        return ["히알루론산", "세라마이드", "나이아신아마이드", "판테놀", "알로에베라"], response
        
    except Exception as e:
        st.error(f"AI 분석 중 오류: {e}")
        default_ingredients = ["히알루론산", "세라마이드", "나이아신아마이드"]
        return default_ingredients, "기본 성분을 사용합니다."

# --- 📊 제품 분석 및 추천 ---
def find_and_rank_products(user_data):
    """제품 필터링 및 순위 결정"""
    df = load_product_data()
    skin_type = user_data['skin_type']
    concerns = user_data['concerns']
    category = user_data['category']
    
    # 카테고리 필터링
    filtered_df = df[df['카테고리'] == category].copy()
    
    # 효능 필터링
    for concern in concerns:
        filtered_df = filtered_df[filtered_df['효능'].str.contains(concern, na=False)]
    
    if filtered_df.empty:
        # 조건을 완화하여 재시도
        filtered_df = df[df['카테고리'] == category].copy()
        if not filtered_df.empty:
            # 최소 하나의 고민이라도 매칭되는 제품 찾기
            mask = filtered_df['효능'].apply(
                lambda x: any(concern in str(x) for concern in concerns)
            )
            filtered_df = filtered_df[mask]
    
    if filtered_df.empty:
        return [], [], "조건에 맞는 제품을 찾을 수 없습니다."
    
    # AI로 핵심 성분 가져오기
    key_ingredients, ingredient_explanation = get_key_ingredients_from_llm(skin_type, concerns)
    
    # 제품 스코어링
    scored_products = []
    for _, row in filtered_df.iterrows():
        ingredients_str = str(row.get('전성분', '')).lower()
        
        # 매칭된 성분 찾기
        found_ingredients = [ing for ing in key_ingredients if ing in ingredients_str]
        match_count = len(found_ingredients)
        
        # 유해성 점수 처리
        try:
            harmfulness_score = float(row.get('유해성_점수', 999))
        except (ValueError, TypeError):
            harmfulness_score = 999.0
        
        if match_count > 0:  # 최소 1개 이상 매칭되는 제품만
            scored_products.append({
                "brand": row.get('브랜드명'),
                "name": row.get('제품명'),
                "match_count": match_count,
                "harmfulness_score": harmfulness_score,
                "found_ingredients": found_ingredients,
                "price": row.get('가격', 0),
                "volume": row.get('용량', 'N/A'),
                "efficacy": row.get('효능', '')
            })
    
    # 정렬: 매칭 성분 개수(내림차순) > 유해성 점수(오름차순)
    sorted_products = sorted(
        scored_products,
        key=lambda p: (-p['match_count'], p['harmfulness_score'])
    )
    
    return sorted_products[:3], key_ingredients, ingredient_explanation

# --- 💬 사용자 입력 처리 ---
def handle_user_input(user_input):
    """단계별 사용자 입력 처리"""
    step = st.session_state.current_step
    
    if step == 'skin_type':
        skin_types = {
            "민감성": "민감성", "지성": "지성", 
            "건성": "건성", "아토피성": "아토피성"
        }
        
        for key, value in skin_types.items():
            if key in user_input:
                st.session_state.user_data['skin_type'] = value
                st.session_state.current_step = 'concerns'
                st.session_state.user_data['concerns'] = []
                
                return {
                    "content": f"""좋아요! **{value}** 피부타입이시군요. 🎯

이제 **피부 고민**을 알려주세요!
여러 개를 선택할 수 있어요. 선택 후 '선택완료'를 눌러주세요.""",
                    "quick_replies": [
                        "💧 보습", "🌿 진정", "✨ 미백", 
                        "🔄 주름/탄력", "🎯 모공케어", "🛢️ 피지조절",
                        "✅ 선택완료"
                    ]
                }
    
    elif step == 'concerns':
        if '선택완료' in user_input:
            if st.session_state.user_data.get('concerns'):
                st.session_state.current_step = 'category'
                concerns_text = ', '.join(st.session_state.user_data['concerns'])
                
                return {
                    "content": f"""완벽해요! 📝

선택하신 피부 고민:
**{concerns_text}**

이제 **어떤 종류의 제품**을 찾으시나요?""",
                    "quick_replies": [
                        "🧴 스킨/토너", "🥛 로션/에멀젼", 
                        "💎 에센스/세럼/앰플", "🍯 크림",
                        "🎭 마스크/팩", "🧼 클렌징",
                        "☀️ 선크림", "🎯 올인원"
                    ]
                }
            else:
                return {
                    "content": "최소 하나의 피부 고민을 선택해주세요! 😊",
                    "quick_replies": [
                        "💧 보습", "🌿 진정", "✨ 미백", 
                        "🔄 주름/탄력", "🎯 모공케어", "🛢️ 피지조절",
                        "✅ 선택완료"
                    ]
                }
        else:
            # 고민 추가
            concerns_map = {
                "보습": "보습", "진정": "진정", "미백": "미백",
                "주름": "주름/탄력", "탄력": "주름/탄력",
                "모공": "모공케어", "피지": "피지조절"
            }
            
            added = []
            for key, value in concerns_map.items():
                if key in user_input and value not in st.session_state.user_data['concerns']:
                    st.session_state.user_data['concerns'].append(value)
                    added.append(value)
            
            current = st.session_state.user_data['concerns']
            if current:
                return {
                    "content": f"""현재 선택된 고민: **{', '.join(current)}**

더 선택하시거나 '선택완료'를 눌러주세요!""",
                    "quick_replies": [
                        "💧 보습", "🌿 진정", "✨ 미백", 
                        "🔄 주름/탄력", "🎯 모공케어", "🛢️ 피지조절",
                        "✅ 선택완료"
                    ]
                }
    
    elif step == 'category':
        categories = {
            "스킨": "스킨/토너", "토너": "스킨/토너",
            "로션": "로션/에멀젼", "에멀젼": "로션/에멀젼",
            "에센스": "에센스/세럼/앰플", "세럼": "에센스/세럼/앰플", "앰플": "에센스/세럼/앰플",
            "크림": "크림",
            "마스크": "마스크/팩", "팩": "마스크/팩",
            "클렌징": "클렌징",
            "선크림": "선크림/로션",
            "올인원": "올인원"
        }
        
        for key, value in categories.items():
            if key in user_input:
                st.session_state.user_data['category'] = value
                st.session_state.current_step = 'analyzing'
                return {
                    "content": f"""**{value}** 제품을 찾아드릴게요! ✨

🔍 AI가 성분을 분석하고 있습니다...
📊 데이터베이스에서 최적의 제품을 검색 중입니다...

잠시만 기다려주세요!""",
                    "analyzing": True
                }
    
    # 대화 모드
    return None

# --- 🎉 추천 결과 생성 ---
def generate_recommendation():
    """최종 추천 결과 생성"""
    user_data = st.session_state.user_data
    
    # 제품 찾기 및 순위 결정
    products, key_ingredients, ingredient_explanation = find_and_rank_products(user_data)
    
    if not products:
        st.session_state.current_step = 'complete'
        return {
            "content": """😅 죄송해요! 현재 조건에 정확히 맞는 제품을 찾지 못했습니다.

다른 카테고리나 조건으로 다시 시도해보시겠어요?""",
            "quick_replies": ["🔄 새로 시작하기", "📝 조건 변경하기"]
        }
    
    # 성공적인 추천 결과
    content = f"""🎉 **분석 완료! 맞춤 제품을 찾았습니다!**

📋 **고객님의 정보**
• 피부 타입: {user_data['skin_type']}
• 피부 고민: {', '.join(user_data['concerns'])}
• 제품 종류: {user_data['category']}

---

🧪 **AI 추천 핵심 성분**

{ingredient_explanation}

---

✨ **TOP {len(products)} 추천 제품**
"""
    
    # 제품 카드 생성
    for i, product in enumerate(products):
        rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}위"
        
        content += f"""

<div class="product-card">
<div class="product-rank">{rank_emoji} TOP {i+1}</div>
<div class="product-title">{product['brand']} - {product['name']}</div>
<div class="product-info">
📊 <b>성분 매칭도:</b> {product['match_count']}개 핵심 성분 포함<br>
🛡️ <b>안전성:</b> 유해성 점수 {product['harmfulness_score']:.1f}점 (낮을수록 안전)<br>
🧪 <b>포함 성분:</b> {', '.join(product['found_ingredients'])}<br>
💰 <b>가격:</b> {product['price']:,}원 ({product['volume']})<br>
✨ <b>효능:</b> {product['efficacy']}
</div>
</div>
"""
    
    content += """

---

🌿 **INGREVIA와 함께 건강한 피부를 만들어가세요!**

궁금한 점이 있으시면 언제든지 물어보세요! 😊"""
    
    st.session_state.current_step = 'complete'
    
    return {
        "content": content,
        "quick_replies": ["🔄 새로운 추천", "❓ 성분 설명", "💬 추가 질문"]
    }

# --- 🎨 메시지 렌더링 ---
def render_chat_message(message, is_user=False):
    """채팅 메시지 렌더링"""
    role = "user" if is_user else "assistant"
    avatar = "👤" if is_user else "🌿"
    css_class = "user-message" if is_user else "bot-message"
    avatar_class = "user-avatar" if is_user else "bot-avatar"
    
    return f"""
    <div class="chat-message {css_class}">
        {f'<div class="avatar {avatar_class}">{avatar}</div>' if not is_user else ''}
        <div class="message-content">{message}</div>
        {f'<div class="avatar {avatar_class}">{avatar}</div>' if is_user else ''}
    </div>
    """

# --- 🎮 메인 애플리케이션 ---
def main():
    # 헤더
    st.markdown("""
    <div class="chat-container">
        <div class="chat-header">
            <h1>🌿 INGREVIA</h1>
            <p>AI 기반 맞춤형 화장품 추천 서비스</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.markdown("### 📋 현재 선택 정보")
        if st.session_state.user_data:
            for key, value in st.session_state.user_data.items():
                if key == 'skin_type':
                    st.write(f"**피부 타입:** {value}")
                elif key == 'concerns':
                    if value:
                        st.write(f"**피부 고민:** {', '.join(value)}")
                elif key == 'category':
                    st.write(f"**제품 종류:** {value}")
        else:
            st.write("아직 선택된 정보가 없습니다.")
        
        st.markdown("---")
        
        if st.button("🔄 처음부터 다시 시작"):
            st.session_state.messages = [st.session_state.messages[0]]
            st.session_state.current_step = 'skin_type'
            st.session_state.user_data = {}
            st.session_state.conversation = init_langchain()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📌 사용 방법")
        st.markdown("""
        1. 피부 타입을 선택하세요
        2. 피부 고민을 선택하세요 (복수 선택 가능)
        3. 원하는 제품 종류를 선택하세요
        4. AI가 맞춤 제품을 추천해드립니다!
        """)
    
    # 채팅 영역
    chat_container = st.container()
    
    with chat_container:
        # 메시지 표시
        for message in st.session_state.messages:
            st.markdown(
                render_chat_message(message["content"], message["role"] == "user"),
                unsafe_allow_html=True
            )
        
        # 빠른 답변 버튼
        if st.session_state.messages and "quick_replies" in st.session_state.messages[-1]:
            st.markdown("### 빠른 선택")
            quick_replies = st.session_state.messages[-1]["quick_replies"]
            
            # 버튼을 행으로 배치
            cols = st.columns(min(len(quick_replies), 4))
            for i, reply in enumerate(quick_replies):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    if st.button(reply, key=f"quick_{i}"):
                        # 사용자 메시지 추가
                        st.session_state.messages.append({
                            "role": "user",
                            "content": reply
                        })
                        st.session_state.typing = True
                        st.rerun()
    
    # 입력 영역
    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "메시지를 입력하세요...",
            key="user_input",
            placeholder="궁금한 점을 자유롭게 물어보세요!"
        )
    
    with col2:
        send_button = st.button("📤 전송", use_container_width=True)
    
    if (send_button or user_input) and user_input:
        # 사용자 메시지 추가
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        st.session_state.typing = True
        st.rerun()
    
    # 봇 응답 처리
    if st.session_state.typing:
        with st.spinner("AI가 분석 중입니다... 🤔"):
            time.sleep(1)  # 자연스러운 딜레이
        
        st.session_state.typing = False
        last_user_message = st.session_state.messages[-1]["content"]
        
        # 분석 단계 처리
        if st.session_state.current_step == 'analyzing':
            response = generate_recommendation()
        # 재시작 처리
        elif '새로 시작' in last_user_message or '새로운 추천' in last_user_message:
            st.session_state.current_step = 'skin_type'
            st.session_state.user_data = {}
            response = {
                "content": """새로운 추천을 시작할게요! 🌟

다시 **피부 타입**을 알려주세요:""",
                "quick_replies": ["🌸 민감성", "💧 지성", "🏜️ 건성", "🔴 아토피성"]
            }
        # 성분 설명 요청
        elif '성분 설명' in last_user_message and st.session_state.current_step == 'complete':
            response = handle_ingredient_explanation()
        # 추가 질문 처리
        elif '추가 질문' in last_user_message or '질문' in last_user_message:
            response = handle_additional_question(last_user_message)
        # 단계별 입력 처리
        else:
            response = handle_user_input(last_user_message)
            
            # LangChain 대화 처리 (자유 대화 모드)
            if response is None and st.session_state.current_step == 'complete':
                try:
                    ai_response = st.session_state.conversation.predict(input=last_user_message)
                    response = {
                        "content": ai_response,
                        "quick_replies": ["🔄 새로운 추천", "❓ 성분 설명", "💬 추가 질문"]
                    }
                except Exception as e:
                    response = {
                        "content": "죄송해요, 잠시 문제가 발생했습니다. 다시 시도해주세요.",
                        "quick_replies": ["🔄 새로 시작하기"]
                    }
        
        # 분석 중 특별 처리
        if response and response.get("analyzing"):
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["content"]
            })
            st.rerun()
        
        # 봇 메시지 추가
        if response:
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["content"],
                "quick_replies": response.get("quick_replies", [])
            })
            st.rerun()

# --- 🧪 추가 기능 함수들 ---
def handle_ingredient_explanation():
    """성분에 대한 상세 설명 제공"""
    if 'user_data' not in st.session_state or not st.session_state.user_data:
        return {
            "content": "먼저 제품 추천을 받아보세요!",
            "quick_replies": ["🔄 새로 시작하기"]
        }
    
    prompt = f"""
    {st.session_state.user_data['skin_type']} 피부 타입과 
    {', '.join(st.session_state.user_data['concerns'])} 고민에 좋은 성분들에 대해 
    더 자세히 설명해주세요.
    
    각 성분이 어떻게 작용하는지, 어떤 효과가 있는지 설명해주세요.
    """
    
    try:
        response = st.session_state.conversation.predict(input=prompt)
        return {
            "content": response,
            "quick_replies": ["🔄 새로운 추천", "💬 추가 질문"]
        }
    except:
        return {
            "content": "성분 정보를 불러오는데 실패했습니다.",
            "quick_replies": ["🔄 새로 시작하기"]
        }

def handle_additional_question(question):
    """추가 질문 처리"""
    try:
        # 컨텍스트 포함하여 질문 처리
        context = f"""
        사용자 정보:
        - 피부 타입: {st.session_state.user_data.get('skin_type', '미선택')}
        - 피부 고민: {', '.join(st.session_state.user_data.get('concerns', []))}
        - 관심 제품: {st.session_state.user_data.get('category', '미선택')}
        
        사용자 질문: {question}
        """
        
        response = st.session_state.conversation.predict(input=context)
        return {
            "content": response,
            "quick_replies": ["🔄 새로운 추천", "❓ 성분 설명", "💬 계속 대화하기"]
        }
    except Exception as e:
        return {
            "content": f"질문 처리 중 오류가 발생했습니다: {str(e)}",
            "quick_replies": ["🔄 새로 시작하기"]
        }

# --- 📦 파일 업로드 기능 (선택사항) ---
def handle_csv_upload():
    """CSV 파일 업로드 처리"""
    uploaded_file = st.file_uploader(
        "제품 데이터베이스 CSV 파일 업로드 (선택사항)",
        type=['csv'],
        help="브랜드명, 제품명, 카테고리, 효능, 전성분, 가격, 용량, 유해성_점수 컬럼이 필요합니다."
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = ['브랜드명', '제품명', '카테고리', '효능', '전성분']
            
            if all(col in df.columns for col in required_columns):
                # 세션에 저장
                st.session_state.product_df = df
                st.success(f"✅ {len(df)}개의 제품 데이터를 성공적으로 로드했습니다!")
                
                # 데이터 미리보기
                with st.expander("데이터 미리보기"):
                    st.dataframe(df.head())
                
                return df
            else:
                st.error(f"❌ 필수 컬럼이 누락되었습니다: {required_columns}")
        except Exception as e:
            st.error(f"❌ 파일 로드 중 오류 발생: {e}")
    
    return None

# --- 🚀 앱 실행 ---
if __name__ == "__main__":
    # 페이지 하단에 정보 표시
    with st.container():
        main()
        
        # 푸터
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>💚 INGREVIA - AI 기반 맞춤형 화장품 추천 서비스</p>
            <p style='font-size: 0.8rem;'>OpenAI GPT-4 & LangChain 기반 | 화장품 성분 분석 전문</p>
        </div>
        """, unsafe_allow_html=True)