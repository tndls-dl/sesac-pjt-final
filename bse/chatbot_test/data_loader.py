# data_loader.py

import pandas as pd

def load_data():
    try:
        product_df = pd.read_csv('product_with_score.csv')
        ingredient_df = pd.read_csv('ingredient_data.csv')
        product_df['카테고리'] = product_df['카테고리'].replace('로션/에멀젼', '로션/에멀전')
        ingredient_df.dropna(subset=['EWG등급'], inplace=True)
        ewg_dict = pd.Series(ingredient_df['EWG등급'].values, index=ingredient_df['한국어성분명']).to_dict()
        return product_df, ingredient_df, ewg_dict
    except FileNotFoundError:
        print("오류: 데이터 파일을 찾을 수 없습니다.")
        return None, None, {}
