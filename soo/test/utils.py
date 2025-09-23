import os
import pandas as pd

def _num(x, default=999.0):
    try:
        return float(x)
    except Exception:
        return default

def find_and_rank_products(filepath, user_selections, key_ingredients):
    """
    CSV에서 조건/성분 기준으로 제품을 필터링하고 상위 3개를 점수화해 반환.
    - 1차 필터: 카테고리(부분 일치, case-insensitive) + 고민(부분 일치, 모두 포함)
    - 2차 필터: 유해성 점수(HARM_THRESHOLD 이하만 통과)  ← 기본 3.5, .env로 조정 가능
    - 랭킹: 매칭된 핵심성분 개수(내림차순) → 유해성 점수(오름차순)
    - 반환 필드: brand, name, price, volume, link, match_count, harmfulness_score, found_ingredients, full_ingredients
    """
    # 입력 파라미터
    concerns = user_selections.get("concerns", []) or []
    category = (user_selections.get("category") or "").strip()

    # 유해성 임계값 (환경변수 → 기본 3.5)
    harm_threshold = _num(os.getenv("HARM_THRESHOLD", 3.5), default=3.5)

    # CSV 로드
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ '{filepath}' 파일을 찾을 수 없습니다.")
        return []

    # 필요한 컬럼 존재 보정
    for col in ["카테고리", "효능", "전성분", "유해성_점수", "브랜드명", "제품명", "가격", "용량", "링크"]:
        if col not in df.columns:
            # 없으면 비어있는 컬럼 추가(크래시 방지)
            df[col] = ""

    # 1) 카테고리 필터: 정확/부분 일치, 대소문자 무시
    if category:
        cat_key = category.split("/")[0]  # "스킨/토너" → "스킨"
        mask_cat = df["카테고리"].fillna("").astype(str).str.contains(cat_key, case=False, na=False)
        filtered = df[mask_cat].copy()
    else:
        filtered = df.copy()

    # 2) 고민(효능) 필터: 모두 포함(AND), 부분 일치, 대소문자 무시
    for c in concerns:
        if c and c != "알 수 없음":
            mask_c = filtered["효능"].fillna("").astype(str).str.contains(str(c), case=False, na=False)
            filtered = filtered[mask_c]

    if filtered.empty:
        return []

    # 3) 유해성 점수 2차 필터링
    filtered["__harm__"] = filtered["유해성_점수"].apply(_num)
    filtered = filtered[filtered["__harm__"] <= harm_threshold]
    if filtered.empty:
        return []

    # 4) 점수화 & 정렬
    scored = []
    key_lc = [str(k).lower() for k in key_ingredients if k]
    for _, row in filtered.iterrows():
        ingredients_text = str(row.get("전성분", "")).lower()
        found = [k for k in key_lc if k in ingredients_text]
        harm = _num(row.get("유해성_점수", 999))

        scored.append({
            "brand": row.get("브랜드명"),
            "name": row.get("제품명"),
            "price": row.get("가격"),
            "volume": row.get("용량"),
            "link": row.get("링크"),
            "match_count": len(found),
            "harmfulness_score": harm,
            "found_ingredients": found,
            "full_ingredients": row.get("전성분", ""),   # ✅ 주의성분 분석에 활용 가능
        })

    # 매칭 성분 많은 순 → 유해성 낮은 순
    scored.sort(key=lambda p: (-p["match_count"], p["harmfulness_score"]))

    # 상위 3개만
    return scored[:3]
