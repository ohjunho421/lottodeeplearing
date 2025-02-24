# chatbot/urls.py

from django.urls import path
from . import views
from .views import ChatbotHomeView, CSRFTokenView, ChatAPIView, DataStatusView, HistoryAPIView

app_name = 'chatbot'

urlpatterns = [
    path('', ChatbotHomeView.as_view(), name='chat'),
    path('chatbot/csrf/', CSRFTokenView.as_view(), name='csrf-token'),
    path('chatbot/chat/', ChatAPIView.as_view(), name='chat'),
    path('chatbot/status/', DataStatusView.as_view(), name='status'),
    path('history/', HistoryAPIView.as_view(), name='history'),
    path('metrics/', views.ModelMetricsView.as_view(), name='model-metrics'),

]