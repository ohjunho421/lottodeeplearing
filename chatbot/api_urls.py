# chatbot/api_urls.py

from django.urls import path
from . import views
from .views import CSRFTokenView, ChatAPIView, HistoryAPIView

app_name = 'chatbot_api'

urlpatterns = [
    path('csrf/', CSRFTokenView.as_view(), name='csrf-token'),
    path('chat/', ChatAPIView.as_view(), name='chat-api'),
    path('history/', HistoryAPIView.as_view(), name='history'),
    path('metrics/', views.ModelMetricsView.as_view(), name='model-metrics'),
]
