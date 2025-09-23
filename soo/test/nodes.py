import re
import json
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from utils import find_and_rank_products
from langchain_community.tools.tavily_search import TavilySearchResults

# ===== 모델 & 웹검색 =====
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
search = TavilySearchResults(k=3)

# ===== 동의어/정규화 =====
SKIN_TYPES = {"지성", "건성", "복합성", "민감성", "아토피성"}
CONCERNS = {"보습", "진정", "미백", "주름/탄력", "모공케어", "피지조절", "주름", "탄력"}
CATEGORY_SYNONYMS = {
    "토너": "스킨/토너", "스킨": "스킨/토너", "스킨/토너": "스킨/토너",
    "로션": "로션/에멀젼", "에멀전": "로션/에멀젼", "에멀젼": "로션/에멀젼", "로션/에멀젼": "로션/에멀젼",
    "세럼": "에센스/앰플/세럼", "앰플": "에센스/앰플/세럼", "에센스": "에센스/앰플/세럼", "에센스/앰플/세럼": "에센스/앰플/세럼",
    "크림": "크림",
    "밤": "밤/멀티밤", "멀티밤": "밤/멀티밤", "밤/멀티밤": "밤/멀티밤",
    "클렌징폼": "클렌징 폼", "클렌징": "클렌징 폼", "클렌징 폼": "클렌징 폼",
    "시트마스크": "시트마스크",
    "선크림": "선크림/로션", "선로션": "선크림/로션", "자외선차단제": "선크림/로션", "선크림/로션": "선크림/로션",
}

# ===== 공통 유틸 =====
def _coerce_to_text(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        parts = []
        for item in x:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(p for p in parts if p).strip()
    return str(x)

def _normalize_tokens(text: str) -> List[str]:
    if not isinstance(text, str):
        text = _coerce_to_text(text)
    cleaned = re.sub(r"[\"'`]+", "", text)
    raw = re.split(r"[,/]\s*|\s+", cleaned)
    return [t.strip() for t in raw if t.strip()]

def _messages_to_text(messages, limit=30) -> str:
    """최근 메시지들을 텍스트로 병합(컨텍스트 추론용)."""
    out = []
    for m in messages[-limit:]:
        role = "사용자" if isinstance(m, HumanMessage) else "도우미"
        out.append(f"{role}: {_coerce_to_text(m.content)}")
    return "\n".join(out)

# ===== 파싱 로직 =====
def _rule_based_parse(s: str) -> Dict[str, Any]:
    tokens = _normalize_tokens(s)
    skin = None
    concerns: List[str] = []
    category = None
    for t in tokens:
        if t in SKIN_TYPES:
            skin = t; continue
        if t in CONCERNS:
            concerns.append("주름/탄력" if t in {"주름", "탄력"} else t); continue
        if t in CATEGORY_SYNONYMS:
            category = CATEGORY_SYNONYMS[t]; continue
        if t.endswith("토너"):
            category = "스킨/토너"
        elif t in {"세럼", "앰플", "에센스"}:
            category = "에센스/앰플/세럼"

    return {
        "skin_type": skin or "알 수 없음",
        "concerns": concerns or ["알 수 없음"],
        "category": category or "알 수 없음",
    }

def _llm_json_parse(s: str) -> Dict[str, Any]:
    prompt = f"""
아래 문장에서 사용자 정보를 JSON으로만 추출하세요.
- 피부 타입: 민감성, 지성, 건성, 아토피성, 복합성
- 피부 고민: 보습, 진정, 미백, 주름/탄력, 모공케어, 피지조절
- 제품 종류: 스킨/토너, 로션/에멀젼, 에센스/앰플/세럼, 크림, 밤/멀티밤, 클렌징 폼, 시트마스크, 선크림/로션
못 찾으면 "알 수 없음"으로 채워주세요.

반드시 순수 JSON만:
{{
  "skin_type": "...",
  "concerns": ["..."],
  "category": "..."
}}

입력: {s}
"""
    resp = llm.invoke(prompt).content.strip()
    try:
        if "{" in resp and "}" in resp:
            resp = resp[resp.index("{"): resp.rindex("}") + 1]
        data = json.loads(resp)
    except Exception:
        data = {"skin_type": "알 수 없음", "concerns": ["알 수 없음"], "category": "알 수 없음"}

    if isinstance(data.get("concerns"), str):
        data["concerns"] = [data["concerns"]]
    cat = (data.get("category") or "").strip()
    data["category"] = CATEGORY_SYNONYMS.get(cat, cat if cat else "알 수 없음")
    return data

def _infer_prefs_from_history(messages) -> Dict[str, Any]:
    """이전 대화에서 가장 최근에 확정된 조건을 JSON으로 추출."""
    history_text = _messages_to_text(messages, limit=30)
    prompt = f"""
아래 대화 기록에서 가장 최근에 확정된 사용자 조건을 JSON으로만 추출하세요.
못 찾으면 "알 수 없음"으로 채워주세요.

반드시 순수 JSON만:
{{
  "skin_type": "...",
  "concerns": ["..."],
  "category": "..."
}}

대화 기록:
{history_text}
"""
    resp = llm.invoke(prompt).content.strip()
    try:
        if "{" in resp and "}" in resp:
            resp = resp[resp.index("{"): resp.rindex("}") + 1]
        data = json.loads(resp)
    except Exception:
        data = {"skin_type": "알 수 없음", "concerns": ["알 수 없음"], "category": "알 수 없음"}

    if isinstance(data.get("concerns"), str):
        data["concerns"] = [data["concerns"]]
    cat = (data.get("category") or "").strip()
    data["category"] = CATEGORY_SYNONYMS.get(cat, cat if cat else "알 수 없음")
    return data

# ===== 웹 검색(주의 성분) =====
def fetch_warnings_for_ingredients(ingredients: List[str]) -> str:
    """유해 가능 성분들에 대해 웹 검색 후 요약. (추가 제품 추천 X)"""
    if not ingredients:
        return ""
    query = f"{', '.join(ingredients)} 화장품 유해성 주의사항"
    try:
        web_results = search.run(query)
    except Exception as e:
        return f"(웹 검색 오류: {e})"
    prompt = f"""
다음은 화장품 성분에 대한 검색 결과입니다:
{web_results}

각 성분이 왜 주의해야 하는지, '성분명 - 한 줄 요약' 형태로 간단히 정리해 주세요.
추가 제품 추천은 금지합니다.
"""
    resp = llm.invoke(prompt)
    return resp.content.strip()

# ===== LangGraph 노드 =====
def parse_user_input(state: Dict[str, Any]):
    """현재 입력을 파싱하고, 비는 항목은 '이전 대화의 최근 조건'으로 백필."""
    # 마지막 사용자 메시지
    last = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last = _coerce_to_text(m.content)
            break

    # 1) 규칙 기반
    parsed = _rule_based_parse(last)

    # 2) '같은/이전 조건' 힌트가 있거나 비는 항목이 있으면, 과거 대화로 백필
    if (
        any(kw in last for kw in ["같은 조건", "이전 조건", "방금이랑", "아까랑"])
        or parsed["skin_type"] == "알 수 없음"
        or parsed["concerns"] == ["알 수 없음"]
        or parsed["category"] == "알 수 없음"
    ):
        defaults = _infer_prefs_from_history(state.get("messages", []))
        if parsed["skin_type"] == "알 수 없음":
            parsed["skin_type"] = defaults.get("skin_type", "알 수 없음")
        if parsed["concerns"] == ["알 수 없음"]:
            parsed["concerns"] = defaults.get("concerns", ["알 수 없음"])
        if parsed["category"] == "알 수 없음":
            parsed["category"] = defaults.get("category", "알 수 없음")

    # 3) 여전히 비면 LLM JSON으로 보정(현재 문장 기준)
    if (
        parsed["skin_type"] == "알 수 없음"
        or parsed["concerns"] == ["알 수 없음"]
        or parsed["category"] == "알 수 없음"
    ):
        fill = _llm_json_parse(last)
        if parsed["skin_type"] == "알 수 없음":
            parsed["skin_type"] = fill["skin_type"]
        if parsed["concerns"] == ["알 수 없음"]:
            parsed["concerns"] = fill["concerns"]
        if parsed["category"] == "알 수 없음":
            parsed["category"] = fill["category"]

    # 최종 안전장치
    if not parsed.get("skin_type"):
        parsed["skin_type"] = "알 수 없음"
    if not parsed.get("concerns"):
        parsed["concerns"] = ["알 수 없음"]
    if not parsed.get("category"):
        parsed["category"] = "알 수 없음"

    return {"user_selections": parsed}

def check_parsing_status(state: Dict[str, Any]):
    s = state["user_selections"]
    skin_ok = s.get("skin_type") and s["skin_type"] != "알 수 없음"
    con_ok = s.get("concerns") and any(c != "알 수 없음" for c in s["concerns"])
    cat_ok = s.get("category") and s["category"] != "알 수 없음"
    return "success" if (skin_ok and con_ok and cat_ok) else "clarification_needed"

def ask_for_clarification(state: Dict[str, Any]):
    s = state["user_selections"]
    missing = []
    if s.get("skin_type") == "알 수 없음": missing.append("피부 타입")
    if not s.get("concerns") or any(c == "알 수 없음" for c in s.get("concerns", [])): missing.append("피부 고민")
    if s.get("category") == "알 수 없음": missing.append("제품 종류")
    text = f"부족한 정보({', '.join(missing)})를 알려주세요. 예: '지성, 보습, 로션'"
    return {"messages": [AIMessage(content=text)]}

def get_ingredients(state: Dict[str, Any]):
    s = state["user_selections"]
    skin_type = s.get("skin_type", "알 수 없음")
    concerns = s.get("concerns", ["알 수 없음"])
    prompt_text = f"""
'{skin_type}' 피부의 '{', '.join(concerns)}' 고민에 좋은 핵심 성분 5가지를 뽑아 주세요.
쉼표(,)로만 구분된 목록으로 간단히 답변 (예: 히알루론산,세라마이드,판테놀,병풀추출물,스쿠알란)
"""
    resp = llm.invoke(prompt_text)
    key_ingredients_str = resp.content.strip()
    return {"key_ingredients": [ing.strip().lower() for ing in key_ingredients_str.split(",") if ing.strip()]}

def find_products(state: Dict[str, Any]):
    selections = state["user_selections"]
    key_ingredients = state["key_ingredients"]
    top_products = find_and_rank_products("product_data.csv", selections, key_ingredients)
    return {"top_products": top_products}

def create_recommendation_message(state: Dict[str, Any]):
    """
    고정 포맷으로 최종 메시지를 조립합니다.
    - 상단: 효능 성분(매칭 기준) 목록
    - 제품 리스트: (이모지 + 번호) 제품명 / 가격 / 용량 / [링크 열기]
                  + 각 제품별 "매칭 성분" 표시
    - 주의 성분: fetch_warnings_for_ingredients(...) 요약(웹검색)
    """
    top = state.get("top_products", [])
    if not top:
        text = "조건에 맞는 제품을 찾지 못했습니다. 다른 조건으로 다시 시도해 보실래요?"
        return {"messages": [AIMessage(content=text)], "recommendation_message": text}

    rank_emojis = ["🥇", "🥈", "🥉"]
    lines: List[str] = []

    # 1) 상단: 효능 성분(매칭 기준)
    key_ings = [k.strip() for k in state.get("key_ingredients", []) if k and k.strip()]
    if key_ings:
        lines.append("🧪 효능 성분(매칭 기준): " + ", ".join(key_ings))
        lines.append("")  # 한 줄 여백

    # 2) 제품 1~3위 카드
    for i, p in enumerate(top[:3]):
        emoji = rank_emojis[i] if i < len(rank_emojis) else f"{i+1}."
        brand = (p.get("brand") or "").strip()
        name = (p.get("name") or "").strip()
        price = p.get("price", "?")
        volume = p.get("volume", "?")
        link = (p.get("link") or "").strip()
        link_md = f"[열기]({link})" if link else "-"

        # 매칭 성분 (해당 제품 전성분에서 실제로 잡힌 것들)
        matched_raw = [m.strip() for m in p.get("found_ingredients", []) if m and str(m).strip()]
        matched_unique = ", ".join(sorted(set(matched_raw))) if matched_raw else "없음 (기준 성분과 직접 매칭 없음)"

        lines.append(f"{emoji} {i+1}. {brand} {name}")
        lines.append(f"   💰 가격: {price}")
        lines.append(f"   🫙 용량: {volume}")
        lines.append(f"   🔗 링크: {link_md}")
        lines.append(f"   🧪 효능 성분: {matched_unique}")
        if i < 2:  # 마지막 아이템 뒤에는 불필요한 공백 방지
            lines.append("")

    # 3) 주의 성분 (웹 검색 요약) — 불필요한 공백 줄 제거 처리 포함
    all_found = [ing for p in top for ing in p.get("found_ingredients", [])]
    unique_found = sorted({ing for ing in all_found if ing})
    warnings_raw = fetch_warnings_for_ingredients(unique_found) if unique_found else ""

    lines.append("")
    lines.append("⚠️ 주의해야 될 성분:")
    if warnings_raw:
        import re
        # 빈 줄 제거 + 글머리 기호 통일
        w_lines = [ln.strip() for ln in warnings_raw.splitlines() if ln.strip()]
        if len(w_lines) == 1:
            clean = re.sub(r"^[•\-\*\d\.\)\s]+", "", w_lines[0])
            lines.append(f"• {clean}")
        else:
            for ln in w_lines:
                clean = re.sub(r"^[•\-\*\d\.\)\s]+", "", ln)
                lines.append(f"• {clean}")
    else:
        lines.append("• 특별히 주의 성분은 발견되지 않았습니다.")

    final_text = "\n".join(lines)
    return {"messages": [AIMessage(content=final_text)], "recommendation_message": final_text}