# main.py
import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END

# ✅ state.py의 단일 소스 GraphState를 사용 (중복 정의 제거)
from state import GraphState

# 노드들
from nodes import (
    parse_user_input,
    check_parsing_status,
    ask_for_clarification,
    get_ingredients,
    find_products,
    create_recommendation_message,
    # router,            # ← 지금 흐름에서는 사용 안 함
    # handle_follow_up,  # ← 지금 흐름에서는 사용 안 함
)

load_dotenv()

# ---- 그래프 구성 ----
workflow = StateGraph(GraphState)

workflow.add_node("parse_user_input", parse_user_input)
workflow.add_node("ask_for_clarification", ask_for_clarification)
workflow.add_node("get_ingredients", get_ingredients)
workflow.add_node("find_products", find_products)
workflow.add_node("create_recommendation_message", create_recommendation_message)
# workflow.add_node("handle_follow_up", handle_follow_up)  # 사용 시에만 다시 추가

workflow.add_edge(START, "parse_user_input")

workflow.add_conditional_edges(
    "parse_user_input",
    check_parsing_status,
    {
        "success": "get_ingredients",
        "clarification_needed": "ask_for_clarification",
    },
)

# 질문 던진 뒤엔 사용자 입력을 기다리기 위해 종료
workflow.add_edge("ask_for_clarification", END)

workflow.add_edge("get_ingredients", "find_products")
workflow.add_edge("find_products", "create_recommendation_message")

# ✅ 추천 생성 후 이 턴(run) 종료 → 다음 질문은 새 run에서 과거 messages를 참고
workflow.add_edge("create_recommendation_message", END)

app = workflow.compile()
