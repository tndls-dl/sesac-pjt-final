import pandas as pd
from typing import Dict, Any, List

# 카테고리 정규화(데이터와 사용자 입력을 같은 축으로 맞춤)
def _normalize_category(cat: str) -> str:
    if not isinstance(cat, str):
        return "알 수 없음"
    c = cat.strip().lower()
    # 선크림은 크림과 절대 섞이면 안 됨
    if "선" in c and "크림" in c:
        return "선크림"
    # 로션/에멀전(젼/젼 표기 차이 통일)
    if ("로션" in c) or ("에멀" in c):
        return "로션/에멀전"
    if ("스킨" in c) or ("토너" in c):
        return "스킨/토너"
    if ("세럼" in c) or ("앰플" in c) or ("에센스" in c):
        return "에센스/앰플/세럼"
    if ("클렌징" in c):
        return "클렌징 폼"
    if ("마스크" in c):
        return "시트마스크"
    if "밤" in c:
        return "밤/멀티밤"
    if "크림" in c:
        return "크림"
    return "알 수 없음"

def find_and_rank_products(filepath, user_selections, key_ingredients):
    """CSV에서 조건/성분 기준으로 제품을 필터링하고 상위 3개를 점수화해 반환."""
    skin_type = user_selections.get("skin_type") or ""
    concerns = user_selections.get("concerns", []) or []
    category_in = (user_selections.get("category") or "").strip()

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ '{filepath}' 파일을 찾을 수 없습니다.")
        return []

    # 1) 데이터 카테고리 정규화 컬럼 생성
    df["카테고리_norm"] = df["카테고리"].fillna("").astype(str).map(_normalize_category)

    # 2) 사용자 카테고리도 정규화 후 **정확 일치**로 필터
    category_norm = _normalize_category(category_in)
    filtered = df[df["카테고리_norm"] == category_norm]

    # 고민: 모두 포함하도록 필터(부분일치)
    for c in concerns:
        if c and c != "알 수 없음":
            filtered = filtered[filtered["효능"].fillna("").astype(str).str.contains(c, na=False)]

    if filtered.empty:
        return []

    # 점수 계산: 매칭된 핵심성분 개수(내림차순) → 유해성_점수(오름차순)
    scored: List[Dict[str, Any]] = []
    key_lw = [str(k).lower() for k in key_ingredients if k]
    for _, row in filtered.iterrows():
        ingredients = str(row.get("전성분", "")).lower()
        found = [k for k in key_lw if k and k in ingredients]
        try:
            harm = float(row.get("유해성_점수", 999))
        except (TypeError, ValueError):
            harm = 999.0
        scored.append({
            "brand": row.get("브랜드명"),
            "name": row.get("제품명"),
            "price": row.get("가격"),
            "volume": row.get("용량"),
            "link": row.get("링크"),
            "match_count": len(found),
            "harmfulness_score": harm,
            "found_ingredients": found,
        })

    scored.sort(key=lambda p: (-p["match_count"], p["harmfulness_score"]))
    return scored[:3]
