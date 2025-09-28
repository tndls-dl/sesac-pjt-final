# 🌿 Ingrevia: 내 피부를 위한 현명한 질문의 시작



**Ingrevia**는 *Ingredient + Via*의 합성어로, 
화장품 **성분(Ingredient)을 통해 올바른 길(Via)을 제시하는** 맞춤형 추천 에이전트입니다.

<br>

## ✨ 프로젝트 소개
- **리뷰 평점이나 판매량이 아닌**, 제품의 **전성분 분석**을 기반으로 맞춤형 스킨케어 제품을 추천하는 것이 핵심입니다.  
- 전성분은 INCI 표준명으로 통일한 뒤, 외부 자료를 통해 확인된 **효능 성분·주의 성분**을 매핑하여 사용자의 피부 고민과 타입에 따라 **효능 성분 기준**으로 제품을 추천합니다.

<br>

### 🔑 주요 기능
- **맞춤형 질문 응답**: 피부 타입, 피부 고민, 원하는 제품 종류를 입력하면 AI가 조건을 이해
- **개인화 추천**: 조건에 맞는 제품을 찾아 추천하고 추천 이유를 함께 설명
- **설명 가능한 근거 제공**: 추천 제품의 핵심 효능 성분과 주의 성분을 정리해 투명하게 제시
- **확장성 있는 링크 제공**: 성분 배지를 클릭하면 해당 성분의 구글 검색 결과로 바로 이동
- **직관적인 인터페이스**: 버튼 선택과 자유 채팅 모두 지원, 추천 카드를 시각적으로 확인 가능

<br>

### ⚙️ 내부 구현
- 전성분을 INCI 표준명으로 정리
- 외부 자료 기반 효능/주의 성분 매핑
- 유해성 점수를 활용해 더 안전한 제품 우선 추천

<br>
---

## 📁 프로젝트 구조

```
sesac-pjt-final/
├── README.md
├── .gitignore
├── langgraph.json
├── .env
├── venv (숨김 폴더)
│    
├── src/
│    └── graph.py
│
├── bse/ (승은 작업 공간)
│    ├── data-gathering-b.ipynb
│    ├── data_separate.ipynb
│    ├── embedding.ipynb
│    ├── design_b.ipynb
│    ├── data.xlsx
│    ├── ingredient_data.csv
│    ├── product_data.csv
│    ├── product_data.json
│    ├── product_with_score.csv
│    │     
│    │          .
│    │          .
│    │          .
│    │ 
│    ├──   (데이터 파일)
│    └── ingrevia_chatbot/
│          ├── app-final.py
│          ├── test.py
│          ├── chatbot.ipynb
│          ├── requirements.txt
│          ├── product_data.csv
│          └── .env
│    
└── soo/ (수인 작업 공간)
     ├── chatbot_1/ (테스트용)
     ├── data-gathering-s.ipynb
     ├── project_mvp.ipynb
     ├── design_s.ipynb
     ├── data.xlsx
     ├── ingredient_data.csv
     ├── product_data.csv
     ├── product_data.json
     ├── product_with_score.csv
     │
     │          .
     │          .
     │          .
     │ 
     ├──   (데이터 파일)
     └── ingrevia/
           ├── main.py
           ├── state.py
           ├── utils.py
           ├── nodes.py
           ├── langgraph.json
           ├── requirements.txt
           ├── product_data.csv
           └── .env

```

<br>

### 🗂️ 데이터 구조
`제품명, 브랜드명, 카테고리, 가격, 용량, 전성분, 효능, 유해성_점수, 링크`

<br>
---
## 👯 팀원 구성

<div align="center">

| **배승은** | **임수인** |
| :------: |  :------: |
| [<img src="https://i.pinimg.com/736x/bd/05/6e/bd056e0ff7138b992464d96dfffe8ff7.jpg" height=150 width=150> <br/> @bse1120](https://github.com/bse1120) | [<img src="https://raw.githubusercontent.com/tndls-dl/TIL/refs/heads/main/images/4.jpg" height=150 width=150> <br/> @tndls-dl](https://github.com/tndls-dl) |

</div>

<br>

### 🧩 역할 분담
🤝 **함꼐 한 일**
- 제품 데이터 수집, 정제 및 데이터 보강
    - 💻 product_data.csv 및 json

🍌 **승은**
- 데이터분할 (csv to plain text)
    - 💻 date_separate.ipynb
- 임베딩
    - 💻 embedding.ipynb
- streamlit을 활용한 웹 애플리케이션 제작
    - 💻 app-final.py

🍊 **수인**
- 성분 매핑 및 유해성 점수 분석
    - 💻 product_with_score.csv
- 추천 로직 구성
- 랭체인을 통해 챗봇 구성하기 (랭스미스 연결)
    - 💻ingrevia/

<br>
---
## 🚀 확장 가능성
현재는 효능 성분 개수를 기준으로 한 단순 추천을 제공하고 있지만, 향후에는 다음과 같이 확장할 수 있습니다.

- **데이터셋 확장**: 더 많은 제품군과 글로벌 데이터셋을 연동
- **계절/시간대별 추천**: 예) 겨울에는 보습 강화, 여름에는 피지 조절 중심
- **안전성 정보 추가**: 알러지 유발 가능 성분이나 EWG 등급 등 보강

<br>



