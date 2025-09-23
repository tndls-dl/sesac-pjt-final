# state.py
from typing import TypedDict, List, Dict, Any
from typing_extensions import Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    """
    LangGraph 실행 상태.
    - messages: LangChain 메시지 히스토리(채팅 UI와 대화 기억은 이 한 필드로 관리)
    - user_selections: 파싱된 사용자 조건 (skin_type, concerns, category)
    - key_ingredients: 조건에 맞는 핵심 성분 리스트
    - top_products: 랭킹된 추천 제품 리스트(이번 턴 결과)
    - recommendation_message: 최종 출력 문자열(로그/디버깅용)
    - history_products: 이전 턴들까지 포함한 누적 추천 제품 리스트
    """
    messages: Annotated[List[BaseMessage], add_messages]
    user_selections: Dict[str, Any]
    key_ingredients: List[str]
    top_products: List[dict]
    recommendation_message: str
    history_products: List[dict]
