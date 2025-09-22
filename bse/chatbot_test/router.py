# router.py

def route_by_user_choice(state):
    """
    선택지 기반 추천과 챗봇 추천을 분기하는 예시 라우터 함수
    """
    if hasattr(state, 'user_input') and state.user_input:
        return 'recommend_by_chatbot'
    if hasattr(state, 'selected_category') and state.selected_category:
        return 'recommend_by_selection'
    return None
