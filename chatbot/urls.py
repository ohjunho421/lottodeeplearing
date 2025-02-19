# chatbot/urls.py

from django.urls import path
from .views import (
    ChatbotHomeView, 
    CSRFTokenView, 
    ChatAPIView, 
    DataStatusView  # views.py에 있는 이름과 일치하게 수정
)

app_name = 'chatbot'

urlpatterns = [
    path('', ChatbotHomeView.as_view(), name='home'),
    path('api/chatbot/csrf/', CSRFTokenView.as_view(), name='csrf-token'),
    path('api/chatbot/chat/', ChatAPIView.as_view(), name='chat'),
    path('api/chatbot/data-status/', DataStatusView.as_view(), name='data-status'),  # 기존 data-status URL
    path('status/', DataStatusView.as_view(), name='status'),  # 새로운 status URL
]