# state.py

from langgraph.graph import MessagesState
from typing_extensions import Any

class State(MessagesState):
    product_df: Any = None
    ingredient_df: Any = None
    ewg_dict: dict = {}
    user_skin_type: str = "미정(해당사항 없음)"
    user_skin_concerns: list = []
    selected_category: str = ""
    scored_df: Any = None
    user_input: str = ""
    recommendation_result: Any = None
