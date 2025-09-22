# graph.py

from langgraph.graph import StateGraph, START, END
from state import State
from nodes import calculate_final_scores, recommend_by_selection, recommend_by_chatbot
from router import route_by_user_choice

builder = StateGraph(State)

builder.add_node('calculate_final_scores', calculate_final_scores)
builder.add_node('recommend_by_selection', recommend_by_selection)
builder.add_node('recommend_by_chatbot', recommend_by_chatbot)

builder.add_edge(START, 'calculate_final_scores')
builder.add_conditional_edges('calculate_final_scores', route_by_user_choice,
                              {'recommend_by_selection': 'recommend_by_selection',
                               'recommend_by_chatbot': 'recommend_by_chatbot'})
builder.add_edge('recommend_by_selection', END)
builder.add_edge('recommend_by_chatbot', END)

graph = builder.compile()
