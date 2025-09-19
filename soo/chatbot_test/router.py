# router.py

def route_by_user_choice(state):
    """
    선택지 기반 추천과 챗봇 추천을 분기하는 예시 라우터 함수
    """
    if getattr(state, 'user_input', None):
        return 'recommend_by_chatbot'
    if getattr(state, 'selected_category', None):
        return 'recommend_by_selection'
    # 기본 경로나 예외 처리
    return 'recommend_by_selection'
