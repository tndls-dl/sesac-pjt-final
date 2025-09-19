# nodes.py

import pandas as pd
import numpy as np
from ingredients import EFFICACY_INGREDIENTS, CAUTION_INGREDIENTS

def calculate_final_scores(state):
    product_df = state.product_df
    user_skin_concerns = state.user_skin_concerns
    user_skin_type = state.user_skin_type
    ewg_dict = state.ewg_dict

    def get_score_for_product(row):
        ingredients_str = row['전성분']
        harmfulness_score = row['유해성_점수']
        if pd.isna(ingredients_str):
            return -np.inf
        ingredients = ingredients_str.split(';')
        ewg_scores = [11 - ewg_dict.get(ing, 3.0) for ing in ingredients]
        avg_ewg_score = np.mean(ewg_scores) if ewg_scores else 0
        efficacy_score = 0
        if user_skin_concerns:
            for concern in user_skin_concerns:
                if concern in EFFICACY_INGREDIENTS:
                    efficacy_score += sum(1 for ing in ingredients if ing in EFFICACY_INGREDIENTS[concern])
        caution_penalty = 0
        if user_skin_type in CAUTION_INGREDIENTS:
            caution_penalty = sum(2 for ing in ingredients if ing in CAUTION_INGREDIENTS[user_skin_type])
        final_score = avg_ewg_score + efficacy_score - harmfulness_score - caution_penalty
        return final_score

    df_copy = product_df.copy()
    df_copy['최종_점수'] = df_copy.apply(get_score_for_product, axis=1)
    state.scored_df = df_copy
    return state

def recommend_by_selection(state):
    scored_df = state.scored_df
    selected_category = state.selected_category
    category_products = scored_df[scored_df['카테고리'] == selected_category]
    top_5_products = category_products.sort_values(by='최종_점수', ascending=False).head(5)
    if top_5_products.empty:
        print(f"\n죄송합니다. '{selected_category}' 카테고리에서 적합한 추천 제품을 찾지 못했습니다.")
    else:
        print(f"\n--- 🕵️‍♀️ 당신을 위한 맞춤 '{selected_category}' 추천 Top 5 ---")
        print(top_5_products[['제품명', '브랜드명', '최종_점수']].to_string(index=False))
    return state

def recommend_by_chatbot(state):
    user_input = state.user_input
    keyword_to_category = {
        '선세럼': '선크림/로션', '선로션': '선크림/로션','선크림': '선크림/로션', '썬크림': '선크림/로션','자차': '선크림/로션',
        '클렌징폼': '클렌징 폼', '폼클렌징': '클렌징 폼', '클렌저': '클렌징 폼',
        '에멀젼': '로션/에멀전', '에멀전': '로션/에멀전','로션': '로션/에멀전',
        '닦토': '스킨/토너', '스킨': '스킨/토너','토너': '스킨/토너',
        '에센스': '에센스/앰플/세럼', '세럼': '에센스/앰플/세럼','앰플': '에센스/앰플/세럼',
        '크림': '크림', 
        '마스크팩': '시트마스크','시트마스크': '시트마스크', '마스크': '시트마스크',
        '멀티밤': '밤/멀티밤', '밤': '밤/멀티밤'
    }
    found_category = None
    for keyword, category in keyword_to_category.items():
        if keyword in user_input:
            found_category = category
            break
    if not found_category:
        print("\n죄송합니다. 어떤 종류의 제품을 찾으시는지 파악하기 어려워요.\n('크림', '로션', '선크림' 등) 제품 종류를 명확히 말씀해주시겠어요?")
        return state
    found_concerns = [concern for concern in EFFICACY_INGREDIENTS.keys() if concern in user_input]
    if found_concerns:
        print(f"\n✅ 챗봇이 이해한 당신의 고민: {', '.join(found_concerns)}")
    else:
        print("\n✅ 특정 피부 고민이 언급되지 않아, 기본 점수로 추천해드릴게요.")
    print("점수를 계산 중입니다...")
    state.user_skin_concerns = found_concerns
    state.user_skin_type = "미정(해당사항 없음)"
    state.selected_category = found_category
    state = calculate_final_scores(state)
    category_products = state.scored_df[state.scored_df['카테고리'] == found_category]
    top_3_products = category_products.sort_values(by='최종_점수', ascending=False).head(3)
    if top_3_products.empty:
        print(f"\n죄송합니다. '{found_category}' 카테고리에서 적합한 추천 제품을 찾지 못했습니다.")
    else:
        print(f"\n--- 🤖 챗봇이 추천하는 '{found_category}' Top 3 ---")
        print(top_3_products[['제품명', '브랜드명', '최종_점수']].to_string(index=False))
    return state
