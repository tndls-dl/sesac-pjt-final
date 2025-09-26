import re
import json
from pathlib import Path
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from utils import find_and_rank_products

# [ADD] 웹 검색 (가능하면 사용, 실패 시 자동 폴백)
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    _search = TavilySearchResults(k=3)
except Exception:
    _search = None

SAFE_BENIGN_INGS = [
    "글리세린","히알루론산","소듐하이알루로네이트","하이알루로닉애씨드",
    "세라마이드","세라마이드엔피","판테놀","스쿠알란","베타인","알란토인",
    "토코페롤","잔탄검","프로판다이올","부틸렌글라이콜","펜틸렌글라이콜",
    "하이드록시아세토페논","다이소듐이디티에이"
]

_PREFERRED_SITES = ["hwahae.co.kr"]
def _search_prefer(query: str):
    if _search is None:
        return ""
    # 우선 선호 사이트
    for site in _PREFERRED_SITES:
        try:
            r = _search.run(f"site:{site} {query}")
            if (isinstance(r, str) and r.strip()) or (isinstance(r, list) and len(r) > 0):
                return r
        except Exception:
            pass
    # 일반 검색
    try:
        return _search.run(query)
    except Exception:
        return ""


# =========================
# 모델
# =========================
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# =========================
# 라벨/동의어
# =========================
SKIN_TYPES = {"민감성", "지성", "건성", "복합성", "아토피성", "중성"}

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
    "피지": "피지조절", "유분": "피지조절", "번들거림": "피지조절", "여드름": "피지조절", "트러블": "피지조절", "피지": "피지",
}

CATEGORY_SYNONYMS = {
    # 스킨/토너
    "토너": "스킨/토너",
    "스킨": "스킨/토너",
    "스킨/토너": "스킨/토너",

    # 로션/에멀전
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

# =========================
# 한국어 조사/부호 제거 & 토큰
# =========================
JOSA_SUFFIXES = ("은","는","이","가","을","를","에","의","로","으로","과","와","랑","하고","에서","부터","까지","도","요","인데")

def _strip_trailing_josa_punct(t: str) -> str:
    if not isinstance(t, str):
        return t
    t = re.sub(r"[!?.~…·]+$", "", t)
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
    tokens: List[str] = []
    for t in raw:
        t = (t or "").strip()
        if not t:
            continue
        t = _strip_trailing_josa_punct(t)  # ← 핵심: '건성인데' -> '건성'
        if t:
            tokens.append(t)
    return tokens

def _extract_category_intent(raw_text: str):
    """
    문장에서 카테고리 전환(add/switch) 의도를 추출.
    - '도/또/역시'가 포함되면 add 모드(이전 + 추가)
    - 카테고리 키워드를 토큰에서 추출해 표준화 리스트 반환
    반환: (mode, cats)
      mode: "add" 또는 "switch"
      cats: 표준화된 카테고리 리스트
    """
    tx = _coerce_to_text(raw_text or "")
    # '~도 / 또 / 역시'가 있으면 add 의도
    add_mode = bool(re.search(r"(?:^|\s)(도|또|역시)(?:\s|$)", tx))
    tokens = _normalize_tokens(tx)

    cats = []
    for t in tokens:
        if t in CATEGORY_SYNONYMS:
            cats.append(CATEGORY_SYNONYMS[t])
        elif t.endswith("토너"):
            cats.append("스킨/토너")
        elif t in {"세럼", "앰플", "에센스"}:
            cats.append("에센스/앰플/세럼")

    # 중복 제거(순서 유지)
    uniq = []
    for c in cats:
        if c and c not in uniq:
            uniq.append(c)

    mode = "add" if (add_mode and uniq) else "switch"
    return mode, uniq

def _looks_offtopic(text: str) -> bool:
    """스킨케어 의도 신호(피부타입/고민/카테고리/추천요청)가 전혀 없으면 True."""
    tx = _coerce_to_text(text or "")
    tokens = _normalize_tokens(tx)

    for t in tokens:
        if t in SKIN_TYPES:
            return False
        if (t in CONCERNS) or (t in CONCERN_SYNONYMS):
            return False
        if (t in CATEGORY_SYNONYMS) or t.endswith("토너") or (t in {"세럼","앰플","에센스"}):
            return False

    # “추천/찾아줘/골라줘” 같은 명시적 요청어도 스킨케어 의도로 간주
    if re.search(r"(추천|찾아줘|골라줘|추천해|추천해줘)", tx):
        return False

    return True


def _messages_to_text(messages, limit=30) -> str:
    out = []
    for m in messages[-limit:]:
        role = "사용자" if isinstance(m, HumanMessage) else "도우미"
        out.append(f"{role}: {_coerce_to_text(m.content)}")
    return "\n".join(out)

# =========================
# 파싱
# =========================
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
        if "모공" in t:
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

    # 중복 제거
    concerns = list(dict.fromkeys(concerns))

    return {
        "skin_type": skin or "알 수 없음",
        "concerns": concerns or ["알 수 없음"],
        "category": category or "알 수 없음",
    }

def _llm_json_parse(s: str) -> Dict[str, Any]:
    prompt = f"""
아래 문장에서 사용자 정보를 JSON으로만 추출하세요.
- 피부 타입: 민감성, 지성, 건성, 아토피성, 복합성, 중성
- 피부 고민: 보습, 진정, 미백, 주름/탄력, 모공케어, 피지조절
- 제품 종류: 스킨/토너, 로션/에멀전, 에센스/앰플/세럼, 크림, 밤/멀티밤, 클렌징 폼, 시트마스크, 선크림
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
    """이전 대화에서 가장 최근에 확정된 조건을 JSON으로만 추출."""
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

# =========================
# LangGraph 노드
# =========================
def parse_user_input(state: Dict[str, Any]):
    """
    1) 현재 문장 규칙 파싱
    2) 후속질문(같은 조건/도/또/역시 + 카테고리)일 때는 '이전 확정값'을 고정 유지하고 카테고리만 교체
    3) 부족하면 과거대화로 백필 → 그래도 부족하면 LLM JSON 보정
    4) 'prefs'에 이번 턴 선택값 저장 (다음 턴 후속질문에서 사용)
    """
    # 마지막 사용자 메시지
    last = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            last = _coerce_to_text(m.content)
            break

    # ⛔ 오프토픽(스킨케어 의도 전혀 없음)일 때는 곧바로 '정보 부족'으로 반환해서
    # ask_for_clarification 노드로 흐르게 만든다. (기존 prefs는 건드리지 않음)
    if _looks_offtopic(last):
        return {
            "user_selections": {
                "skin_type": "알 수 없음",
                "concerns": ["알 수 없음"],
                "category": "알 수 없음",
            }
        }  

    # 직전 확정 조건(있으면 최우선으로 사용)
    last_confirmed = state.get("last_confirmed_selections") or {}
    prev_skin = last_confirmed.get("skin_type")
    prev_concerns = last_confirmed.get("concerns", [])
    prev_category = last_confirmed.get("category")

    # 1) 규칙 기반 1차 파싱
    parsed = _rule_based_parse(last)

    # ---- 후속질문 의도 탐지: "같은 조건", "~도/또/역시" ----
    txt = last or ""
    followup_signal = bool(re.search(r"(같은\s*조건)|(?:^|\s)(도|또|역시)(?:\s|$)", txt))

    # 카테고리 의도 추출 (switch / add)
    mode, cats = _extract_category_intent(last)
    cat_from_intent = cats[0] if cats else (parsed.get("category") if parsed.get("category") != "알 수 없음" else None)

    # 사용자 입력에 '명시적' 피부타입/고민이 들어있는지 확인
    tokens = _normalize_tokens(txt)
    explicit_skin = any(t in SKIN_TYPES for t in tokens)
    explicit_concern = any(
        (t in CONCERNS) or (t in CONCERN_SYNONYMS)
        or ("보습" in t) or ("수분" in t) or ("진정" in t) or ("미백" in t)
        or ("주름" in t) or ("탄력" in t) or ("모공" in t) or ("피지" in t) or ("여드름" in t) or ("트러블" in t)
        for t in tokens
    )

    # 2) 후속질문: "같은 조건으로 ~도/또/역시" + 카테고리만 말한 경우 → 피부타입/고민은 유지, 카테고리만 교체
    if followup_signal and cat_from_intent:
        if last_confirmed:
            if not explicit_skin:
                parsed["skin_type"] = prev_skin or parsed.get("skin_type", "알 수 없음")
            if not explicit_concern:
                parsed["concerns"] = prev_concerns or parsed.get("concerns", ["알 수 없음"])
        parsed["category"] = cat_from_intent

    # 3) 누락값 백필(이전 대화 → LLM JSON 순)
    if (
        parsed["skin_type"] == "알 수 없음"
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

    # 안전 보정
    if not parsed.get("skin_type"):
        parsed["skin_type"] = "알 수 없음"
    if not parsed.get("concerns"):
        parsed["concerns"] = ["알 수 없음"]
    if not parsed.get("category"):
        parsed["category"] = "알 수 없음"

    # 다음 턴용 임시 메모
    state["prefs"] = parsed
    return {"user_selections": parsed, "prefs": parsed}


def check_parsing_status(state: Dict[str, Any]):
    """
    ✅ 완화 규칙: 카테고리가 없어도 '피부타입' 또는 '고민' 중 하나만 있으면 진행
    (카테고리 없는 경우 find_products가 전 카테고리 스캔으로 커버)
    """
    s = state["user_selections"]
    has_cat = s.get("category") and s["category"] != "알 수 없음"
    has_skin = s.get("skin_type") and s["skin_type"] != "알 수 없음"
    has_conc = s.get("concerns") and s["concerns"] != ["알 수 없음"]
    return "success" if (has_cat or has_skin or has_conc) else "clarification_needed"

def ask_for_clarification(state: Dict[str, Any]):
    s = state["user_selections"]
    need_cat = (s.get("category") in (None, "", "알 수 없음"))
    need_skin = (s.get("skin_type") in (None, "", "알 수 없음"))
    need_conc = (not s.get("concerns")) or (s.get("concerns") == ["알 수 없음"])
    missing = []
    if need_cat: missing.append("제품 종류")
    if need_skin and need_conc: missing.append("피부 타입 또는 피부 고민")
    text = "질문을 이해하지 못했습니다. 정확한 추천을 위해 제품 종류와 피부 타입 또는 고민 중 하나를 알려주세요. 🙂 예: 지성, 로션 / 건성, 보습, 크림"
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
    """
    - 카테고리가 있으면: 해당 카테고리 상위 결과
    - 카테고리가 없으면: 전 카테고리를 훑어 카테고리별 후보 1개씩 수집 → 상위 3개
    """
    sel = state["user_selections"]
    key_ings = state.get("key_ingredients", [])
    cat = sel.get("category")

    # 카테고리 지정 O → 단일 카테고리
    if cat and cat != "알 수 없음":
        top = find_and_rank_products(str(DATA_PATH), sel, key_ings) or []
        return {"top_products": top}

    # 카테고리 지정 X → 전 카테고리 스캔
    bucket = []
    for c in ALL_CATEGORIES:
        sub_sel = {**sel, "category": c}
        items = find_and_rank_products(str(DATA_PATH), sub_sel, key_ings) or []
        if items:
            bucket.append(items[0])
    # 상위 3개만 노출(없으면 빈 리스트)
    return {"top_products": bucket[:3]}

# [ADD] 제품별 '추천 이유' 웹 요약 (부족하면 성분 기반 폴백)
def _fetch_reasons_for_products(products: List[dict], selections: Dict[str, Any], key_ingredients: List[str]) -> List[str]:
    reasons: List[str] = []
    skin = selections.get("skin_type", "알 수 없음")
    concerns = ", ".join([c for c in selections.get("concerns", []) if c and c != "알 수 없음"]) or "알 수 없음"
    category = selections.get("category", "알 수 없음")

    for p in products[:3]:
        brand = (p.get("brand") or p.get("브랜드명") or "").strip()
        name = (p.get("name") or p.get("제품명") or "").strip()

        matched = ", ".join(sorted(set([m for m in (p.get("found_ingredients") or []) if m]))) \
                  or ", ".join([k for k in (key_ingredients or []) if k])

        query = f"{brand} {name} 성분 효과 리뷰 장단점 {category} {skin} {concerns}"
        web_results = _search_prefer(query)

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

        import re as _re
        reason = _re.sub(r"^[•\-\*\d\.\)\s]+", "", reason) or "핵심 성분과 저자극 지표가 조건에 부합"
        reasons.append(reason)

    return reasons

# --- [ADD] found_ingredients가 비었을 때, 웹 스니펫으로 효능 성분을 3~6개 추출하는 폴백 ---
def infer_beneficial_ings_via_web(brand: str, name: str, fallback_key_ings: List[str]) -> List[str]:
    import re
    query = f"{brand} {name} 전성분 효능 성분 성분표 성분 리스트"
    # ✅ 함수명 오타 수정: search_prefer → _search_prefer
    web_results = _search_prefer(query)

    prompt = f"""
    역할: 당신은 화장품 성분 큐레이터입니다.
    아래 자료(웹 스니펫)에서 '{brand} {name}' 제품의 피부에 이득이 되는 '핵심 효능 성분'만 3~6개 한국어 성분명으로 추출하세요.
    - 보습/진정/미백/주름/모공/피지 등과 직접 관련된 활성 성분 위주
    - 용매/보존제/향료/가교제 등 보조 성분 제외
    - 출력은 쉼표로만 나열 (예: 히알루론산, 세라마이드, 나이아신아마이드)

    자료:
    ---
    {web_results}
    ---
    """
    try:
        resp = llm.invoke(prompt)
        txt = (resp.content or "").strip()
        # 줄바꿈으로 오는 경우도 대비해서 쉼표/줄바꿈을 모두 분리
        raw = [x.strip() for x in re.split(r"[,\n]", txt) if x.strip()]
        filtered = []
        for x in raw:
            # ❌ 사과/설명 문구 제거
            if any(bad in x for bad in ["죄송", "추출할 수 없", "제공된 자료", "정보가 부족", "없습니다"]):
                continue
            # 너무 긴 문장/단락성 텍스트 제거 (성분명이 아닌 경우)
            if len(x) > 28 or len(x.split()) > 5:
                continue
            filtered.append(x)
        ings = list(dict.fromkeys(filtered))[:6]
    except Exception:
        ings = []

    # 완전 실패 시, 분석 기준 성분으로 폴백
    if not ings and fallback_key_ings:
        ings = list(dict.fromkeys([k.strip() for k in fallback_key_ings if k and str(k).strip()]))[:6]
    return ings


# [ADD] 하단 '주의 성분' 생성 (효능 성분과 겹치면 [조건부]로 표기)
def _fetch_warnings_for_ingredients(ingredients: List[str], efficacy_ings: List[str] = None) -> str:
    if not ingredients:
        return ""
    efficacy_ings = [str(i or "").lower().strip() for i in (efficacy_ings or []) if str(i or "").strip()]

    query = f"{', '.join(ingredients)} 화장품 유해성 주의사항"
    web_results = _search_prefer(query)

    prompt = f"""
역할: 당신은 화장품 안전성 요약가입니다.
자료: 아래는 성분 위험성 관련 웹 검색 스니펫입니다.
---
{web_results}
---
입력 성분(후보): {', '.join(ingredients)}
효능 성분(겹치면 기본 [조건부]): {', '.join(efficacy_ings)}
일반 안전/보습 성분(특별 근거 없으면 제외): {', '.join(SAFE_BENIGN_INGS)}

지침:
1) '후보' 중 다음 기준만 경고:
   - 알레르기/자극 보고 빈도↑(향료/에센셜오일, 리모넨/리날룰/유제놀 등)
   - 산/레티노이드/벤조일퍼옥사이드 등 고농도·pH 의존 자극 가능
   - 논쟁성 UV 필터(옥시벤존/옥티녹세이트 등), 포름알데하이드 방출 방부제 등
2) 효능 성분과 겹치면 기본 [조건부]로 표기하고 사유를 25자 이내로
3) '일반 안전/보습' 리스트는 특별한 근거 없으면 제외
4) 중복 제거, 3~5개 이내, 중요도 순
출력(이 형식만):
- 성분 — [위험|조건부] 한줄 이유
- 성분 — [위험|조건부] 한줄 이유
"""
    try:
        resp = llm.invoke(prompt)
        txt = (resp.content or "").strip()
    except Exception:
        txt = ""

    # 5줄 제한 & 불릿 정리
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if len(lines) > 5:
        lines = lines[:5]
    return "\n".join(lines)


def create_recommendation_message(state: Dict[str, Any]):
    """
    고정 포맷으로 최종 메시지를 조립합니다.
    - 상단: 효능 성분(분석 기준)
    - 제품 1~3위 카드:
        - 브랜드 / 가격 / 용량 / 🔗 [링크]
        - 🧪 효능 성분  ← (매칭 성분이 있으면 우선 사용, 없으면 전성분에서 key_ingredients 교차검출)
        - ✅ 추천 이유(웹 요약)
        - ⚠️ 주의 성분  ← (제품별, 효능 성분과 겹치면 제외, [조건부] 표기 없음)
    """
    import re
    from langchain_core.messages import AIMessage

    top = state.get("top_products", []) or []
    if not top:
        text = "조건에 맞는 제품을 찾지 못했습니다. 다른 조건으로 다시 시도해 보실래요?"
        return {"messages": [AIMessage(content=text)], "recommendation_message": text}

    # 헤더
    lines: List[str] = []
    lines.append("요청하신 조건에 맞춰 추천 제품을 정리했어요. 😊")

    s = state.get("user_selections", {}) or {}
    skin = s.get("skin_type", "알 수 없음")
    concerns_list = [c for c in s.get("concerns", []) if c and c != "알 수 없음"]
    category = s.get("category", "알 수 없음")

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

    # 효능 성분(분석 기준)
    key_ings = [k.strip() for k in state.get("key_ingredients", []) if k and str(k).strip()]
    if key_ings:
        lines.append(f"**🧪 효능 성분(분석 기준):** {', '.join(key_ings)}")
        lines.append("")

    # 제품별 '추천 이유'
    selections = state.get("user_selections", {})
    web_reasons = _fetch_reasons_for_products(top, selections, key_ings)

    # 제품 카드
    medals = ["🥇", "🥈", "🥉"]
    safe_set = {s.lower() for s in SAFE_BENIGN_INGS}

    # 2) 제품 카드 (상위 3개)
    for i, p in enumerate(top[:3]):
        emoji = medals[i] if i < len(medals) else f"{i+1}."
        name = (p.get("name") or "").strip()
        brand = (p.get("brand") or "").strip()
        price = p.get("price", "?")
        volume = p.get("volume", "?")
        link = (p.get("link") or "").strip()
        link_md = f"[링크]({link})" if link else "-"

        # --- 추천 이유 (이미 계산된 web_reasons 사용) ---
        reason = web_reasons[i] if i < len(web_reasons) else "핵심 성분과 저자극 지표가 조건에 부합"

        # --- 효능 성분: found_ingredients → 비었으면 웹 폴백 ---
        found = [m.strip() for m in p.get("found_ingredients", []) if m and str(m).strip()]
        if not found:
            # 웹에서 3~6개 추출 + 마지막 안전망으로 key_ingredients 사용
            found = infer_beneficial_ings_via_web(brand, name, state.get("key_ingredients", []))

        eff_unique_list = sorted(set([x for x in found if x]))
        eff_unique = ", ".join(eff_unique_list) if eff_unique_list else "정보 부족"

        # --- 제품별 주의 성분: 효능 성분과 겹치면 제외 ---
        caution_lines = _fetch_warnings_for_ingredients(eff_unique_list) if eff_unique_list else ""
        caution_items = []
        if caution_lines:
            for ln in [ln.strip() for ln in caution_lines.splitlines() if ln.strip()]:
                token = ln.split("—", 1)[0].replace("-", "").strip()
                if token and token not in eff_unique_list and token not in caution_items:
                    caution_items.append(token)
        caution_text = ", ".join(caution_items) if caution_items else "없음"

        # --- 출력 ---
        lines.append(f"{emoji} {i+1}. {name}")
        lines.append(f"   🏬 브랜드: {brand or '-'}")
        lines.append(f"   💰 가격: {price}원")
        lines.append(f"   🫙 용량: {volume}")
        lines.append(f"   🔗 {link_md}")
        lines.append(f"   ✅ 추천 이유: {reason}")
        lines.append(f"   🧪 효능 성분: {eff_unique}")
        lines.append(f"   ⚠️ 주의 성분: {caution_text}")
        if i < 2:
            lines.append("")

    final_text = "\n".join(lines)
    return {
    "messages": [AIMessage(content=final_text)],
    "recommendation_message": final_text,
    "last_confirmed_selections": s 
}

