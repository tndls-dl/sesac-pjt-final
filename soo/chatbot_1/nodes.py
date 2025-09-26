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
CONCERN_SYNONYMS = {
    # 보습 계열
    "보습감": "보습", "수분": "보습", "수분감": "보습", "유수분": "보습",
    # 진정 계열
    "진정": "진정", "쿨링": "진정", "붉음증": "진정", "홍조": "진정",
    # 미백/톤업
    "미백": "미백", "톤업": "미백", "잡티": "미백",
    # 주름/탄력
    "주름": "주름/탄력", "탄력": "주름/탄력", "탄력감": "주름/탄력", "리프팅": "주름/탄력",
    # 모공/피지
    "모공": "모공케어", "모공관리": "모공케어", "모공케어": "모공케어", "블랙헤드": "모공케어", "모공": "모공",
    "피지": "피지조절", "유분": "피지조절", "번들거림": "피지조절", "여드름": "피지조절", "트러블": "피지조절", "피지": "피지"
}

def _normalize_concern_label(c: str) -> str:
    c = (c or "").strip()
    if c in CONCERN_SYNONYMS:
        return CONCERN_SYNONYMS[c]
    if ("보습" in c) or ("수분" in c):
        return "보습"
    if "모공" in c:
        return "모공케어"
    if ("피지" in c) or ("번들" in c) or ("유분" in c) or ("여드름" in c) or ("트러블" in c):
        return "피지조절"
    if ("진정" in c) or ("붉" in c) or ("홍조" in c):
        return "진정"
    if ("미백" in c) or ("톤업" in c) or ("잡티" in c):
        return "미백"
    if ("주름" in c) or ("탄력" in c) or ("리프팅" in c):
        return "주름/탄력"
    return c or "알 수 없음"


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
    "마스크팩": "시트마스크",
    "팩": "시트마스크",

    # 선케어
    "선크림": "선크림",
    "선로션": "선크림",
    "자외선차단제": "선크림",
    "자차": "선크림",
}

ALL_CATEGORIES = ["스킨/토너","로션/에멀전","에센스/앰플/세럼","크림","밤/멀티밤","클렌징 폼","시트마스크","선크림"]

# 안전 성분 화이트리스트(겹침 방지용)
SAFE_BENIGN_INGS = [
    "글리세린","히알루론산","소듐하이알루로네이트","하이알루로닉애씨드",
    "세라마이드","세라마이드엔피","판테놀","스쿠알란","베타인","알란토인",
    "토코페롤","잔탄검","프로판다이올","부틸렌글라이콜","펜틸렌글라이콜",
    "하이드록시아세토페논","다이소듐이디티에이"
]

# 한국어 조사/부호 제거
JOSA_SUFFIXES = ("은","는","이","가","을","를","에","의","로","으로","과","와","랑","하고","에서","부터","까지","도","요")

def _strip_trailing_josa_punct(t: str) -> str:
    if not isinstance(t, str):
        return t
    t = re.sub(r"[!?.,~…·]+$", "", t)
    changed = True
    while changed and len(t) > 1:
        changed = False
        for suf in JOSA_SUFFIXES:
            if t.endswith(suf) and len(t) > len(suf):
                t = t[:-len(suf)]
                changed = True
                break
    return t

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
    tokens = []
    for t in raw:
        t = t.strip()
        if not t:
            continue
        t = _strip_trailing_josa_punct(t)
        if t:
            tokens.append(t)
    return tokens

def _messages_to_text(messages, limit=30) -> str:
    out = []
    for m in messages[-limit:]:
        role = "사용자" if isinstance(m, HumanMessage) else "도우미"
        out.append(f"{role}: {_coerce_to_text(m.content)}")
    return "\n".join(out)

# ===== 파싱 =====
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
        if t in CONCERN_SYNONYMS:
            concerns.append(CONCERN_SYNONYMS[t]); continue

        if ("보습" in t) or ("수분" in t):
            concerns.append("보습"); continue
        if ("모공" in t):
            concerns.append("모공케어"); continue
        if ("피지" in t) or ("번들" in t) or ("유분" in t) or ("여드름" in t) or ("트러블" in t):
            concerns.append("피지조절"); continue
        if ("진정" in t) or ("붉" in t) or ("홍조" in t):
            concerns.append("진정"); continue
        if ("미백" in t) or ("톤업" in t) or ("잡티" in t):
            concerns.append("미백"); continue
        if ("주름" in t) or ("탄력" in t) or ("리프팅" in t):
            concerns.append("주름/탄력"); continue

        if t in CATEGORY_SYNONYMS:
            category = CATEGORY_SYNONYMS[t]; continue
        if t.endswith("토너"):
            category = "스킨/토너"
        elif t in {"세럼", "앰플", "에센스"}:
            category = "에센스/앰플/세럼"

    concerns = list(dict.fromkeys(concerns))
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

    concerns = data.get("concerns", [])
    if isinstance(concerns, str):
        concerns = [concerns]
    concerns = [_normalize_concern_label(x) for x in concerns if x]
    data["concerns"] = list(dict.fromkeys([c for c in concerns if c and c != "알 수 없음"])) or ["알 수 없음"]
    return data

def _extract_category_intent(raw_text: str):
    tx = _coerce_to_text(raw_text or "")
    tokens = _normalize_tokens(tx)
    cats = []
    for t in tokens:
        if t in CATEGORY_SYNONYMS:
            cats.append(CATEGORY_SYNONYMS[t])
        elif t.endswith("토너"):
            cats.append("스킨/토너")
        elif t in {"세럼","앰플","에센스"}:
            cats.append("에센스/앰플/세럼")
    # 중복 제거
    uniq = []
    for c in cats:
        if c and c not in uniq:
            uniq.append(c)
    return uniq

# ===== 검색 유틸 =====
PREFERRED_SITES = ["hwahae.co.kr"]

def search_prefer(query: str):
    for site in PREFERRED_SITES:
        try:
            r = search.run(f"site:{site} {query}")
            if (isinstance(r, str) and r.strip()) or (isinstance(r, list) and len(r) > 0):
                return r
        except Exception:
            pass
    return search.run(query)

# ===== 웹 요약 유틸 =====
def fetch_warnings_for_ingredients(ingredients: List[str], efficacy_ings: List[str] = None) -> str:
    if not ingredients:
        return ""
    efficacy_ings = efficacy_ings or []
    web_results = search_prefer(f"{', '.join(ingredients)} 화장품 유해성 주의사항")
    prompt = f"""
    역할: 당신은 화장품 안전성 요약가입니다.
    자료:
    ---
    {web_results}
    ---
    입력 성분: {', '.join(ingredients)}
    효능 성분(겹치면 조건부 처리): {', '.join(efficacy_ings)}
    일반 보조 성분은 제외: {', '.join(SAFE_BENIGN_INGS)}
    규칙: 3~5개, 한 줄 요약, 중요도 순.
    출력:
    - 성분 — [위험|조건부] 사유
    """
    resp = llm.invoke(prompt)
    txt = (resp.content or "").strip()
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    return "\n".join(lines[:5])

def fetch_reasons_for_products(products: List[dict], selections: Dict[str, Any], key_ingredients: List[str]) -> List[str]:
    reasons: List[str] = []
    skin = selections.get("skin_type", "알 수 없음")
    concerns = ", ".join([c for c in selections.get("concerns", []) if c and c != "알 수 없음"]) or "알 수 없음"
    category = selections.get("category", "알 수 없음")
    import re
    for p in products[:3]:
        brand = (p.get("brand") or "").strip()
        name = (p.get("name") or "").strip()
        matched = ", ".join(sorted(set([m for m in p.get("found_ingredients", []) if m]))) or ", ".join(key_ingredients)
        query = f"{brand} {name} 성분 효과 리뷰 장단점 {category} {skin} {concerns}"
        try:
            web_results = search_prefer(query)
        except Exception as e:
            web_results = f"(웹 검색 오류: {e})"
        prompt = f"""
        역할: 화장품 추천 근거 요약가
        조건: 피부={skin}, 고민={concerns}, 카테고리={category}
        제품: {brand} {name}
        매칭 성분: {matched}
        자료:
        ---
        {web_results}
        ---
        규칙: 35~60자 한 줄, 근거 기반, 허위 금지, 자료 부족 시 매칭 성분 기반
        """
        try:
            resp = llm.invoke(prompt)
            reason = (resp.content or "").strip().splitlines()[0]
        except Exception:
            reason = ""
        reason = re.sub(r"^[•\-\*\d\.\)\s]+", "", reason) or "핵심 성분과 저자극 지표 기반 추천"
        reasons.append(reason)
    return reasons

# ===== LangGraph 노드 =====
def parse_user_input(state: Dict[str, Any]):
    """
    1) 현재 문장 파싱
    2) 이전 prefs와 머지 (현재 문장에 없는 항목은 유지)
    3) 병합 결과를 user_selections와 prefs에 동기화
    """
    last = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            last = _coerce_to_text(m.content)
            break

    # 현재 문장 파싱
    parsed = _rule_based_parse(last)

    # 카테고리만 말한 후속(예: "토너도", "선크림은?")
    if parsed.get("category") == "알 수 없음":
        cats = _extract_category_intent(last)
        if cats:
            parsed["category"] = cats[0]

    # 과거 prefs 불러와서 병합
    prev = state.get("prefs") or {"skin_type": "알 수 없음", "concerns": ["알 수 없음"], "category": "알 수 없음"}
    merged = {
        "skin_type": parsed["skin_type"] if parsed["skin_type"] != "알 수 없음" else prev.get("skin_type", "알 수 없음"),
        "concerns": parsed["concerns"] if parsed["concerns"] != ["알 수 없음"] else prev.get("concerns", ["알 수 없음"]),
        "category": parsed["category"] if parsed["category"] != "알 수 없음" else prev.get("category", "알 수 없음"),
    }

    # 안전 보정
    if not merged.get("skin_type"): merged["skin_type"] = "알 수 없음"
    if not merged.get("concerns"): merged["concerns"] = ["알 수 없음"]
    if not merged.get("category"): merged["category"] = "알 수 없음"

    # 상태에 저장(다음 턴에서 사용)
    state["prefs"] = merged
    return {"user_selections": merged, "prefs": merged}

def check_parsing_status(state: Dict[str, Any]):
    s = state["user_selections"]
    has_category = s.get("category") and s["category"] != "알 수 없음"
    has_skin = s.get("skin_type") and s["skin_type"] != "알 수 없음"
    has_concerns = s.get("concerns") and s["concerns"] != ["알 수 없음"]
    # ✅ 카테고리 + (피부타입 or 고민) 중 하나라도 있으면 진행
    return "success" if (has_category and (has_skin or has_concerns)) else "clarification_needed"

def ask_for_clarification(state: Dict[str, Any]):
    s = state["user_selections"]
    need_category = (s.get("category") in (None, "", "알 수 없음"))
    need_skin = (s.get("skin_type") in (None, "", "알 수 없음"))
    need_concerns = (not s.get("concerns")) or (s.get("concerns") == ["알 수 없음"])
    missing = []
    if need_category: missing.append("제품 종류")
    if need_skin and need_concerns: missing.append("피부 타입 또는 피부 고민")
    hint = "예: '지성, 보습, 로션' 또는 '지성, 로션' 또는 '보습, 로션'"
    text = f"부족한 정보({', '.join(missing)})를 알려주세요. {hint}"
    return {"messages": [AIMessage(content=text)]}

def get_ingredients(state: Dict[str, Any]):
    s = state["user_selections"]
    skin_type = s.get("skin_type", "알 수 없음")
    concerns = s.get("concerns", ["알 수 없음"])
    skin_label = skin_type if skin_type != "알 수 없음" else "일반적인"

    if concerns == ["알 수 없음"]:
        prompt_text = f"""
        역할: 화장품 성분 큐레이터.
        목표: '{skin_label}' 피부에 보편적으로 안전하고 유효한 핵심 활성 성분 5개만 선정.
        지침: 자극 낮고 근거 기반. 보조/용매/향/보존제/UV필터 제외.
        출력: 쉼표로만 구분된 한 줄
        """
    else:
        prompt_text = f"""
        역할: 화장품 성분 큐레이터.
        목표: '{skin_label}' 피부의 '{', '.join(concerns)}' 고민 개선에 기여하는 핵심 활성 성분 5개만 선정.
        지침: 근거 기반 활성 위주, 보조/용매/향/보존제/UV필터 제외.
        출력: 쉼표로만 구분된 한 줄
        """
    resp = llm.invoke(prompt_text)
    key_ingredients_str = resp.content.strip()
    return {"key_ingredients": [ing.strip().lower() for ing in key_ingredients_str.split(",") if ing.strip()]}

from pathlib import Path
DATA_PATH = Path(__file__).parent / "product_data.csv"

def find_products(state: Dict[str, Any]):
    selections = state["user_selections"]
    key_ingredients = state.get("key_ingredients", [])
    # 1차: 키성분 기반
    top_products = find_and_rank_products(str(DATA_PATH), selections, key_ingredients)
    # 폴백: 키성분 없이
    if not top_products:
        top_products = find_and_rank_products(str(DATA_PATH), selections, [])
    return {"top_products": top_products}

def create_recommendation_message(state: Dict[str, Any]):
    import re
    from langchain_core.messages import AIMessage

    top = state.get("top_products", []) or []
    if not top:
        text = "조건에 맞는 제품을 찾지 못했습니다. 카테고리와 피부타입/고민 중 하나 이상을 알려주세요."
        return {"messages": [AIMessage(content=text)], "recommendation_message": text}

    lines: List[str] = []
    lines.append("요청하신 조건에 맞춰 추천 제품을 정리했어요. 😊")

    s = state.get("user_selections", {}) or {}
    skin = s.get("skin_type", "알 수 없음")
    concerns_list = [c for c in s.get("concerns", []) if c and c != "알 수 없음"]
    category = s.get("category", "알 수 없음")

    cond_bits = []
    if skin != "알 수 없음": cond_bits.append(f"🧑‍🦰 피부: **{skin}**")
    if concerns_list: cond_bits.append(f"🌿 고민: **{', '.join(concerns_list)}**")
    if category != "알 수 없음": cond_bits.append(f"🧴 제품: **{category}**")
    if cond_bits:
        lines.append("**🎯 사용자 조건**")
        lines.append("   " + "   ·   ".join(cond_bits))
        lines.append("")

    key_ings = [k.strip() for k in state.get("key_ingredients", []) if k and str(k).strip()]
    if key_ings:
        lines.append(f"**🧪 효능 성분(분석 기준):** {', '.join(key_ings)}")
        lines.append("")

    web_reasons = fetch_reasons_for_products(top, s, key_ings)

    rank_emojis = ["🥇", "🥈", "🥉"]
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
        lines.append(f"   💰 가격: {price}원")
        lines.append(f"   🫙 용량: {volume}")
        lines.append(f"   🔗 {link_md}")
        lines.append(f"   🧪 효능 매칭 성분: {matched_unique}")
        reason = web_reasons[i] if i < len(web_reasons) else "핵심 성분과 저자극 지표가 조건에 부합"
        lines.append(f"   ✅ 추천 이유: {reason}")
        if i < 2:
            lines.append("")

    # 간단 경고 요약 (선택)
    all_found = [str(ing).strip().lower() for p in top for ing in p.get("found_ingredients", []) if str(ing).strip()]
    unique_found = sorted(set(all_found))
    safe_set = {s.lower() for s in SAFE_BENIGN_INGS}
    warn_candidates = [ing for ing in unique_found if ing not in safe_set]
    warnings_raw = fetch_warnings_for_ingredients(warn_candidates, efficacy_ings=[k.lower() for k in key_ings]) if warn_candidates else ""
    if warnings_raw:
        lines.append("")
        lines.append("⚠️ 주의해야 될 성분:")
        for ln in [ln.strip() for ln in warnings_raw.splitlines() if ln.strip()][:5]:
            clean = re.sub(r"^[•\-\*\d\.\)\s]+", "", ln)
            lines.append(f"⚠️ {clean}")

    final_text = "\n".join(lines)
    return {"messages": [AIMessage(content=final_text)], "recommendation_message": final_text}
