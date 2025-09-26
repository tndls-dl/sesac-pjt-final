from langgraph.graph import StateGraph, START, END
from state import GraphState
from nodes import (
    parse_user_input,
    ask_for_clarification,
    get_ingredients,
    find_products,
    create_recommendation_message,
)

workflow = StateGraph(GraphState)

# 노드 등록
workflow.add_node("parse_user_input", parse_user_input)
workflow.add_node("ask_for_clarification", ask_for_clarification)
workflow.add_node("get_ingredients", get_ingredients)
workflow.add_node("find_products", find_products)
workflow.add_node("create_recommendation_message", create_recommendation_message)

# 1) START → parse
workflow.add_edge(START, "parse_user_input")

# 2) parse → 분기: 카테고리 있고 + (피부타입 or 고민) 있으면 성공
workflow.add_conditional_edges(
    "parse_user_input",
    lambda x: "success" if (
        (x.get("user_selections", {}).get("category") not in (None, "", "알 수 없음")) and
        (
            (x.get("user_selections", {}).get("skin_type") not in (None, "", "알 수 없음")) or
            (
                isinstance(x.get("user_selections", {}).get("concerns"), list) and
                x["user_selections"]["concerns"] and
                x["user_selections"]["concerns"] != ["알 수 없음"]
            )
        )
    ) else "clarification_needed",
    {"success": "get_ingredients", "clarification_needed": "ask_for_clarification"},
)

# 3) 직선 연결
workflow.add_edge("get_ingredients", "find_products")
workflow.add_edge("find_products", "create_recommendation_message")
workflow.add_edge("create_recommendation_message", END)

app = workflow.compile()
