# 🌿 Ingrevia: 내 피부를 위한 현명한 질문의 시작

![ingrevia_mockup](https://raw.githubusercontent.com/tndls-dl/sesac-pjt-final/refs/heads/main/src/ingrevia_mockup.gif)


**Ingrevia**는 *Ingredient + Via*의 합성어로, 

화장품 **성분(Ingredient)을 통해 올바른 길(Via)을 제시하는** 맞춤형 추천 에이전트입니다.


---

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


### 🗂️ 데이터 구조
`제품명, 브랜드명, 카테고리, 가격, 용량, 전성분, 효능, 유해성_점수, 링크`

---

## 👯 팀원 구성

<div align="center">

| **배승은** | **임수인** |
| :------: |  :------: |
| [<img src="https://i.pinimg.com/736x/bd/05/6e/bd056e0ff7138b992464d96dfffe8ff7.jpg" height=150 width=150> <br/> @bse1120](https://github.com/bse1120) | [<img src="https://raw.githubusercontent.com/tndls-dl/TIL/refs/heads/main/images/4.jpg" height=150 width=150> <br/> @tndls-dl](https://github.com/tndls-dl) |

</div>

### 🧩 역할 분담
🤝 **함께 한 일**
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

---

## 🌿 Langsmith
### 🧩 LangGraph 노드 구성
LangSmith를 통해 대화 흐름과 노드 실행을 추적하고, 다양한 입력 시나리오에 대한 반응을 시각적으로 검증했습니다.  
아래 그래프는 사용자의 입력이 어떤 단계를 거쳐 최종 추천으로 이어지는지를 보여줍니다. 

<div align="center">

![workflow](/src/랭스미스%20그래프.png)

</div>

### 🔎 주요 노드

- **Input Parser (parser)**  
  사용자 대화에서 피부 타입, 피부 고민, 제품 카테고리를 추출  

- **Profile Manager (profile_manager)**  
  추출된 정보를 프로필에 저장하고 누락된 정보는 보완  

- **Candidate Finder (search_candidates)**  
  제품 데이터셋에서 조건(타입·고민·카테고리)에 맞는 후보 제품 탐색  

- **Ingredient Analyzer (analyze_ingredients)** ✨ *LLM 사용*  
  후보 제품의 전성분을 LLM에 전달해 피부 타입·고민 기준 **핵심 효능 성분** 도출, **주의 성분** 탐지 및 추천 이유 생성 

- **Recommendation Generator (recommend)** ✨ *LLM 사용*  
  분석 결과를 기반으로 사용자 친화적인 추천 메시지 생성  

- **End (end)**  
  최종 분석 결과를 사용자에게 반환하고 대화를 마무리  

<br>

### 🎥 실행 화면
**1) 기본 질문**

사용자가 피부 타입/고민/제품 종류를 입력하면 조건에 맞는 제품을 추천합니다.  

<p align="center">
  <img src="/src/랭스미스%20기본%20질문.png" width="45%"/>
  <img src="/src/랭스미스%20기본%20질문%202.png" width="45%"/>
</p>


**2) 후속 질문**

사용자가 조건을 변경하거나 추가하면 실시간으로 다시 계산해 최적의 제품을 추천합니다.

![후속 질문](/src/랭스미스%20후속%20질문.png)


**3) 부분 질문**

피부 타입이나 고민 중 하나만 입력하고 카테고리를 선택해도 추천이 가능합니다.

<p align="center">
  <img src="/src/랭스미스%20부분%20질문1.png" width="45%"/>
  <img src="/src/랭스미스%20부분%20질문2.png" width="45%"/>
</p>


**4) 다중 고민 선택**

피부 고민을 2개 이상 선택해도 조건에 맞는 추천을 제공합니다. 

![다중 고민](/src/랭스미스%20조건%20질문.png)


**5) 무관한 질문**

관련 없는 질문에는 “이해하지 못했다”는 안내와 함께 가이드를 제공합니다.

![무관한 질문](/src/랭스미스%20다른%20질문.png)
  
---

## 🌿 Streamlit
사용자는 버튼 클릭과 채팅 입력으로 쉽게 제품 추천을 받을 수 있습니다.


### 🎥 실행 화면
다음은 주요 기능을 GIF로 정리한 실행 화면입니다.

**1) 버튼 기능 이용 (기본)**  
피부 타입, 고민, 제품 카테고리를 버튼으로 선택해 추천을 시작합니다.

![버튼기능_기본](/src/버튼%20기능%20이용_기본.gif)


**2) 다중 선택 가능**  
피부 고민이나 조건을 여러 개 선택해 맞춤형 추천을 받을 수 있습니다.  

![여러 개 선택 가능](/src/여러%20개%20선택%20가능.gif)


**3) 후속 질문 (예: 앰플도)**  
추천을 받은 뒤, 추가로 다른 카테고리를 요청하면 새로운 조건에 맞는 제품을 이어서 추천합니다.  

![후속 질문(채팅으로)](/src/후속%20질문%20가능(앰플도).gif)


**4) 이전 선택으로 + 선택 취소 가능**  
잘못 선택한 경우 이전 단계로 돌아가거나 선택 항목을 한 번 더 클릭해 취소할 수 있습니다.  

![이전 선택으로 + 선택 취소](/src/이전%20선택으로%20+%20선택%20취소%20가능.gif)


**5) 채팅 기능 이용**  
버튼 대신 자연어 채팅으로 피부 타입/고민/제품을 입력해도 추천이 가능합니다.  

![채팅 기능 이용](/src/채팅%20기능%20이용.gif)


**6) 왼쪽 탭 + 새 추천 시작하기**  
사이드바에서 현재 조건을 확인하거나 언제든지 “새 추천 시작하기”를 눌러 초기화할 수 있습니다.  

![왼쪽 탭 + 새 추천 시작](/src/왼쪽%20탭%20+%20새%20추천%20시작하기.gif)


**7) 기초라인 추천 가능**  
사용자가 *“기초라인 추천해줘”* 라고 입력하면,  
토너·로션·세럼·크림·선크림 등 **카테고리별로 한 제품씩** 묶어 부문별로 하나씩 추천합니다. 

![기초라인 추천 가능](/src/기초라인%20추천%20가능.gif)

**8) 링크 연결 가능**  
추천된 제품 이름을 클릭하면 외부 웹사이트로 연결되어 상세 정보를 확인할 수 있습니다.  

![링크 연결 가능](/src/링크%20연결%20가능.gif)


**9) 무관한 질문**  
관련 없는 질문에는 안내 메시지를 제공하고 대화를 이어갑니다.  
![무관한 질문](/src/ingrevia%20다른%20질문.png)




---
## 🚀 확장 가능성
현재는 효능 성분 개수를 기준으로 한 단순 추천을 제공하고 있지만, 향후에는 다음과 같이 확장할 수 있습니다.

- **데이터셋 확장**: 더 많은 제품군과 글로벌 데이터셋을 연동
- **계절/시간대별 추천**: 예) 겨울에는 보습 강화, 여름에는 피지 조절 중심
- **안전성 정보 추가**: 알러지 유발 가능 성분이나 EWG 등급 등 보강


