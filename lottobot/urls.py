# lottobot/urls.py

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth.decorators import login_required
from django.contrib.auth import views as auth_views
from . import views

# 미들웨어 설정을 위한 설정 파일 import
from django.apps import apps


urlpatterns = [
    path('admin/', admin.site.urls),
    # Health check endpoint for Railway
    path('', views.health_check, name='health_check'),
    # 로그인/로그아웃/회원가입 URL
    path('login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    path('register/', views.register_view, name='register'),  # 회원가입 URL 추가
    
    # accounts 앱 URLs
    path("api/accounts/", include("accounts.urls", namespace="accounts")),
    # 메인 페이지와 다른 기능들
    path('main/', login_required(views.main_view), name='main'),
    path('chatbot/', include('chatbot.urls')),  # Chatbot UI
    path('api/chatbot/', include('chatbot.api_urls')),  # API URLs
    path('mypage/', login_required(views.mypage_view), name='mypage'),
    
    # 구독 관련 URL
    path('subscription/', include('chatbot.subscription_urls')),  # 구독 시스템 URL
    
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
