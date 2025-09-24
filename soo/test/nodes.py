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
    # 스킨/토너
    "토너": "스킨/토너",
    "스킨": "스킨/토너",
    "스킨/토너": "스킨/토너",

    # 로션/에멀전(에멀젼 표기도 흡수)
    "로션": "로션/에멀전",
    "에멀전": "로션/에멀전",
    "에멀젼": "로션/에멀전",
    "로션/에멀전": "로션/에멀전",
    "로션/에멀젼": "로션/에멀전",

    # 에센스/앰플/세럼
    "세럼": "에센스/앰플/세럼",
    "앰플": "에센스/앰플/세럼",
    "에센스": "에센스/앰플/세럼",
    "에센스/앰플/세럼": "에센스/앰플/세럼",

    # 크림 & 밤
    "크림": "크림",
    "밤": "밤/멀티밤",
    "멀티밤": "밤/멀티밤",
    "밤/멀티밤": "밤/멀티밤",

    # 클렌징
    "클렌징폼": "클렌징 폼",
    "클렌징": "클렌징 폼",
    "클렌징 폼": "클렌징 폼",

    # 마스크
    "시트마스크": "시트마스크",

    # 선케어 (이제 '선크림'으로 통일)
    "선크림": "선크림",
    "선로션": "선크림",
    "자외선차단제": "선크림",
    "자차": "선크림",
}

ALL_CATEGORIES = ["스킨/토너","로션/에멀전","에센스/앰플/세럼","크림","밤/멀티밤","클렌징 폼","시트마스크","선크림"]

# 안전 성분 화이트리스트(겹침 방지용) 추가
SAFE_BENIGN_INGS = [
    "글리세린","히알루론산","소듐하이알루로네이트","하이알루로닉애씨드",
    "세라마이드","세라마이드엔피","판테놀","스쿠알란","베타인","알란토인",
    "토코페롤","잔탄검","프로판다이올","부틸렌글라이콜","펜틸렌글라이콜",
    "하이드록시아세토페논","다이소듐이디티에이"
]

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

PREFERRED_SITES = ["hwahae.co.kr"]

def search_prefer(query: str):
    """선호 도메인 → 결과 없으면 일반 검색으로 폴백."""
    for site in PREFERRED_SITES:
        try:
            r = search.run(f"site:{site} {query}")
            # 문자열/리스트 모두 커버: 내용 있으면 바로 반환
            if (isinstance(r, str) and r.strip()) or (isinstance(r, list) and len(r) > 0):
                return r
        except Exception:
            pass
    return search.run(query)

# ===== 웹 검색(주의 성분) =====
def fetch_warnings_for_ingredients(ingredients: List[str], efficacy_ings: List[str] = None) -> str:
    """유해 가능 성분들에 대해 웹 검색 후 요약 (효능 성분과의 겹침은 원칙적 제외/조건부 처리)."""
    if not ingredients:
        return ""
    efficacy_ings = efficacy_ings or []

    query = f"{', '.join(ingredients)} 화장품 유해성 주의사항"
    try:
        web_results = search_prefer(query)
    except Exception as e:
        return f"(웹 검색 오류: {e})"

    prompt = f"""
    역할: 당신은 화장품 안전성 요약가입니다.
    자료: 아래는 성분 위험성 관련 웹 검색 스니펫입니다.
    ---
    {web_results}
    ---
    입력 성분(후보): {', '.join(ingredients)}
    효능 성분(겹침 시 기본 제외): {', '.join(efficacy_ings)}
    일반 안전/보습 성분(특별 근거 없으면 제외): {', '.join(SAFE_BENIGN_INGS)}

    지침:
    1) '후보' 중 아래 기준에 해당하는 것만 경고 대상으로 채택:
    - 알레르기/자극 보고 빈도↑(향료/에센셜오일, 리모넨/리날룰/유제놀 등)
    - 산/레티노이드/벤조일퍼옥사이드 등 고농도·pH 의존 자극 가능
    - 논쟁성 UV 필터(옥시벤존/옥티녹세이트 등), 포름알데하이드 방출 방부제 등
    2) '효능 성분'과 겹치면 기본 제외. 단, 중등도 이상 위험 근거가 명확하면
    '조건부'로 표기하고 사유를 25자 이내로.
    3) '일반 안전/보습' 리스트는 특별한 근거가 없으면 제외.
    4) 중복 제거, 3~5개 이내, 중요도 순.
    출력 포맷(이 형식만):
    - 성분 — [위험|조건부] 한줄 이유
    - 성분 — [위험|조건부] 한줄 이유
    """
    resp = llm.invoke(prompt)
    txt = (resp.content or "").strip()
    if not txt:
        return ""
    # 과도한 공백/빈 줄 정리 + 5줄 이내로 제한
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if len(lines) > 5:
        lines = lines[:5]
    return "\n".join(lines)


# 제품별 '추천 이유' 웹 요약
def fetch_reasons_for_products(products: List[dict], selections: Dict[str, Any], key_ingredients: List[str]) -> List[str]:
    """각 제품에 대해 웹 스니펫을 바탕으로 '왜 추천하는지' 한 줄 요약을 만든다."""
    reasons: List[str] = []
    skin = selections.get("skin_type", "알 수 없음")
    concerns = ", ".join([c for c in selections.get("concerns", []) if c and c != "알 수 없음"]) or "알 수 없음"
    category = selections.get("category", "알 수 없음")

    import re
    for p in products[:3]:
        brand = (p.get("brand") or "").strip()
        name = (p.get("name") or "").strip()
        matched = ", ".join(sorted(set([m for m in p.get("found_ingredients", []) if m]))) or ", ".join(key_ingredients)

        # 1) 웹 검색 (부족하면 자동 폴백)
        query = f"{brand} {name} 성분 효과 리뷰 장단점 {category} {skin} {concerns}"
        try:
            web_results = search_prefer(query)
        except Exception as e:
            web_results = f"(웹 검색 오류: {e})"

        # 2) 매우 짧은 한 줄 요약
        prompt = f"""
        역할: 당신은 화장품 추천 근거 요약가입니다.
        상황: 사용자는 {skin} 피부, 고민은 {concerns}, 카테고리는 {category}입니다.
        제품: {brand} {name}
        매칭/핵심 성분: {matched}

        자료(웹 검색 스니펫):
        ---
        {web_results}
        ---

        규칙:
        - '왜 이 제품을 추천하는지' 한 줄(35~60자)로 한국어 요약
        - 가능한 근거: 매칭 성분 효능, 임상/보습/진정 지표, 저자극(무향/약산성), 논란 성분 무첨가 등
        - 자료에 없는 수치/사실 창작 금지
        - 자료 부족 시 매칭 성분 기반으로 작성
        - 출력: 한 줄만, 불릿/머리기호/따옴표 없이, 마침표 없이
        """
        try:
            resp = llm.invoke(prompt)
            reason = (resp.content or "").strip().splitlines()[0]
        except Exception:
            reason = ""

        # 과한 머리기호/숫자 제거 + 폴백
        reason = re.sub(r"^[•\-\*\d\.\)\s]+", "", reason) or "핵심 성분과 저자극 지표가 조건에 부합"
        reasons.append(reason)

    return reasons


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
    # ✅ 피부타입만 있어도 진행
    return "success" if skin_ok else "clarification_needed"

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

    if concerns == ["알 수 없음"]:
        prompt_text = f"""
        역할: 화장품 성분 큐레이터.
        목표: '{skin_type}' 피부에 보편적으로 안전하고 유효한 핵심 활성 성분 5개만 선정.
        지침: 자극 낮고 근거 기반. 보조/용매/가향/보존제 제외. UV필터 제외.
        출력: 쉼표로만 구분된 한 줄
        """
    else:
        prompt_text = f"""
        역할: 당신은 화장품 성분 큐레이터입니다.
        목표: '{skin_type}' 피부의 '{', '.join(concerns)}' 고민 개선에 '직접적으로' 기여하는 핵심 활성 성분 5개만 선정.
        지침:
        - 자극 위험이 낮고 해당 고민에 근거 기반(임상/메커니즘)이 있는 '활성 성분' 위주.
        - 다음 '일반 보조/저위험' 성분은 제외: 글리세린, 부틸렌글라이콜, 프로판다이올, 1,2-헥산다이올, 정제수, 카보머 등.
        - 자외선차단 제품 상황이 아니라면 UV 필터는 제외.
        - 향료/에센셜오일/보존제/용매 등 기능 보조 성분은 제외.
        출력: 쉼표로만 구분된 한 줄 (예: 히알루론산, 세라마이드, 판테놀, 병풀추출물, 나이아신아마이드)
        """

    resp = llm.invoke(prompt_text)
    key_ingredients_str = resp.content.strip()
    return {"key_ingredients": [ing.strip().lower() for ing in key_ingredients_str.split(",") if ing.strip()]}


def find_products(state: Dict[str, Any]):
    selections = state["user_selections"]
    key_ingredients = state["key_ingredients"]
    cat = selections.get("category")

    if not cat or cat == "알 수 없음":
        grouped = []
        for c in ALL_CATEGORIES:
            sel = {**selections, "category": c}
            top = find_and_rank_products("product_data.csv", sel, key_ingredients)[:2]  # 카테고리당 최대 2개
            if top:
                grouped.append({"category": c, "items": top})
        return {"top_products_by_cat": grouped}

    # 기존 단일 카테고리 경로
    top_products = find_and_rank_products("product_data.csv", selections, key_ingredients)
    return {"top_products": top_products}


def create_recommendation_message(state: Dict[str, Any]):
    """
    고정 포맷으로 최종 메시지를 조립합니다.
    - 상단: 효능 성분(분석 기준)
    - 제품 1~3위: (🥇/🥈/🥉 + 번호) 제품명
        - 브랜드 / 가격 / 용량 / 🔗 [링크]
        - 🧪 효능 매칭 성분
        - ✅ 추천 이유(웹 요약)
    - 하단: ⚠️ 주의해야 될 성분 (효능 성분은 제외하지 않음 → LLM이 [조건부] 판단)
    """
    import re
    from langchain_core.messages import AIMessage

    top = state.get("top_products", []) or []
    if not top:
        text = "조건에 맞는 제품을 찾지 못했습니다. 다른 조건으로 다시 시도해 보실래요?"
        return {"messages": [AIMessage(content=text)], "recommendation_message": text}

    rank_emojis = ["🥇", "🥈", "🥉"]
    lines: List[str] = []

    # 헤더
    lines.append("요청하신 조건에 맞춰 추천 제품을 정리했어요. 😊")

    s = state.get("user_selections", {}) or {}
    skin = s.get("skin_type", "알 수 없음")
    concerns_list = [c for c in s.get("concerns", []) if c and c != "알 수 없음"]
    category = s.get("category", "알 수 없음")

    # ✅ 사용자 조건: '알 수 없음'은 감추고, 점(·) 구분자로 깔끔하게
    cond_bits = []
    if skin != "알 수 없음":
        cond_bits.append(f"🧑‍🦰 피부: **{skin}**")
    if concerns_list:
        cond_bits.append(f"🌿 고민: **{', '.join(concerns_list)}**")
    if category != "알 수 없음":
        cond_bits.append(f"🧴 제품: **{category}**")

    if cond_bits:
        lines.append("**🎯 사용자 조건**")
        lines.append("   " + "   ·   ".join(cond_bits))
        lines.append("")

    # 1) 상단: 효능 성분(분석 기준)
    key_ings = [k.strip() for k in state.get("key_ingredients", []) if k and str(k).strip()]
    if key_ings:
        lines.append(f"**🧪 효능 성분(분석 기준):** {', '.join(key_ings)}")
        lines.append("")  # 한 줄 여백

    # ✅ 제품별 '추천 이유' 웹 요약
    selections = state.get("user_selections", {})
    web_reasons = fetch_reasons_for_products(top, selections, key_ings)

    # 2) 제품 카드 (상위 3개)
    for i, p in enumerate(top[:3]):
        emoji = rank_emojis[i] if i < len(rank_emojis) else f"{i+1}."
        name = (p.get("name") or "").strip()
        brand = (p.get("brand") or "").strip()
        price = p.get("price", "?")
        volume = p.get("volume", "?")
        link = (p.get("link") or "").strip()
        link_md = f"[링크]({link})" if link else "-"

        matched_raw = [m.strip() for m in p.get("found_ingredients", []) if m and str(m).strip()]
        matched_unique = ", ".join(sorted(set(matched_raw))) if matched_raw else "없음 (기준 성분과 직접 매칭 없음)"

        lines.append(f"{emoji} {i+1}. {name}")
        lines.append(f"   🏬 브랜드: {brand or '-'}")
        lines.append(f"   💰 가격: {price}")
        lines.append(f"   🫙 용량: {volume}")
        lines.append(f"   🔗 {link_md}")
        lines.append(f"   🧪 효능 매칭 성분: {matched_unique}")
        reason = web_reasons[i] if i < len(web_reasons) else "핵심 성분과 저자극 지표가 조건에 부합"
        lines.append(f"   ✅ 추천 이유: {reason}")
        if i < 2:
            lines.append("")

    # 3) 주의 성분 (효능 성분은 제외하지 않고 LLM이 [조건부]/[위험] 판단)
    all_found = [str(ing).strip().lower() for p in top for ing in p.get("found_ingredients", []) if str(ing).strip()]
    unique_found = sorted(set(all_found))

    key_set = {k.lower() for k in key_ings}
    safe_set = {s.lower() for s in SAFE_BENIGN_INGS}

    # ✅ 효능 성분은 필터링하지 않음(겹치면 프롬프트가 [조건부]로 표기)
    #    안전/보조 성분만 선제적으로 제외하여 노이즈 축소
    warn_candidates = [ing for ing in unique_found if ing not in safe_set]

    warnings_raw = fetch_warnings_for_ingredients(
        warn_candidates,
        efficacy_ings=sorted(list(key_set))
    ) if warn_candidates else ""

    # ✅ 폴백: 효능 성분과 겹치는 후보가 있는데도 요약이 비면, 최소한 [조건부] 한 줄 생성
    overlap = [ing for ing in warn_candidates if ing in key_set]
    if (not warnings_raw) and overlap:
        fallback_lines = [
            f"{ing} — [조건부] 핵심 성분과 겹침, 개인차에 따라 자극 가능"
            for ing in overlap[:3]
        ]
        warnings_raw = "\n".join(fallback_lines)

    lines.append("")
    lines.append("⚠️ 주의해야 될 성분:")
    if warnings_raw:
        w_lines = [ln.strip() for ln in warnings_raw.splitlines() if ln.strip()]
        for ln in w_lines:
            clean = re.sub(r"^[•\-\*\d\.\)\s]+", "", ln)
            lines.append(f"⚠️ {clean}")
    else:
        lines.append("• 특별히 주의해야 될 성분은 발견되지 않았습니다.")

    final_text = "\n".join(lines)
    return {"messages": [AIMessage(content=final_text)], "recommendation_message": final_text}
