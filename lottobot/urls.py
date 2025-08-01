# lottobot/urls.py

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth.decorators import login_required
from django.contrib.auth import views as auth_views
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import RedirectView
from . import views

# 미들웨어 설정을 위한 설정 파일 import
from django.apps import apps


urlpatterns = [
    path('admin/', admin.site.urls),
    # Health check endpoints for Railway
    path('', views.health_check, name='health_check'),
    path('health/', views.health_check, name='health_check_alt'),  # Railway가 /health/로 체크할 수 있도록
    # Login redirect
    path('home/', RedirectView.as_view(url='/login/', permanent=False), name='home'),
    # 로그인/로그아웃/회원가입 URL - 커스텀 뷰 강제 사용
    path('login/', csrf_exempt(views.custom_login_view), name='login'),  # CSRF 검증이 없는 커스텀 로그인 뷰 강제 사용
    path('custom-login/', csrf_exempt(views.custom_login_view), name='custom_login'),  # 대체 로그인 URL 테스트용
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
