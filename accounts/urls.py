from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views import SignupAPIView, DeleteAPIView, LogoutAPIView

app_name = "accounts"
urlpatterns = [
    # 토큰 갱신 - 프론트에서 필요
    path("api/token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    # 회원가입/탈퇴/로그인/로그아웃
    path("user/signup/", SignupAPIView.as_view(), name="signup"),
    path("user/delete/", DeleteAPIView.as_view(), name="delete"),
    path(
        "auth/login/", TokenObtainPairView.as_view(), name="login"
    ),  # 로그인 : JWT 기본제공
    path("auth/logout/", LogoutAPIView.as_view(), name="logout"),
]
