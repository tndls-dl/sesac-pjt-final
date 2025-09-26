# state.py

from typing import Dict, Any, List
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class GraphState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    user_selections: Dict[str, Any]
    key_ingredients: List[str]
    top_products: List[Dict[str, Any]]
    top_products_by_cat: List[Dict[str, Any]]
    recommendation_message: str
    __reset__: bool
    # ✅ 후속 질의(“토너도/선크림도/기초”)를 전달하기 위한 상태 필드
    multi_categories: List[str]
