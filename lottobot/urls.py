# lottobot/urls.py

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth.decorators import login_required
from django.contrib.auth import views as auth_views
from . import views


urlpatterns = [
    path('admin/', admin.site.urls),
    # 로그인/로그아웃/회원가입 URL
    path('', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    path('register/', views.register_view, name='register'),  # 회원가입 URL 추가
    
    # accounts 앱 URLs
    path("api/accounts/", include("accounts.urls", namespace="accounts")),
    # 메인 페이지와 다른 기능들
    path('main/', login_required(views.main_view), name='main'),
    path('chatbot/', include('chatbot.urls')),  # chatbot 기본 URL
    path('api/', include('chatbot.urls')),  # API URL 추가
    path('mypage/', login_required(views.mypage_view), name='mypage'),
    
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
