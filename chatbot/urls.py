# chatbot/urls.py

from django.urls import path
from . import views
from .views import ChatbotHomeView

app_name = 'chatbot'

urlpatterns = [
    path('', ChatbotHomeView.as_view(), name='chatbot-home'),
]