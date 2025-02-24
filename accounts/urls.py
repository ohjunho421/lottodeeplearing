from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views import SignupAPIView, LogoutAPIView, DeleteAPIView
from . import views

app_name = "accounts"

urlpatterns = [
    # 회원가입/탈퇴/로그인/로그아웃
    path("signup/", SignupAPIView.as_view(), name="signup"),
    path("register/", views.register_view, name="register"),
    path("login/", TokenObtainPairView.as_view(), name="TokenObtainPairView"),
    path("logout/", LogoutAPIView.as_view(), name="logout"),
    path("delete/", DeleteAPIView.as_view(), name="delete"),
    # 토큰 갱신 - 프론트에서 필요
    # path("token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
]
