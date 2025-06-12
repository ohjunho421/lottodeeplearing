"""
로또봇 구독 확인 미들웨어
"""
from django.utils.deprecation import MiddlewareMixin
from django.urls import reverse
from django.shortcuts import redirect
from django.contrib import messages

from .models import UserProfile

class SubscriptionMiddleware(MiddlewareMixin):
    """구독 상태를 확인하는 미들웨어"""
    
    def process_request(self, request):
        """요청을 처리하기 전에 사용자의 구독 상태를 확인"""
        if request.user.is_authenticated:
            try:
                # 프로필이 없는 경우 생성
                profile, created = UserProfile.objects.get_or_create(user=request.user)
                
                # 요청 객체에 구독 상태 저장
                request.user.is_active_subscriber = profile.is_subscription_active()
                request.user.is_premium = profile.is_premium
                
            except Exception as e:
                # 오류 발생 시 기본값으로 설정
                request.user.is_active_subscriber = False
                request.user.is_premium = False
                
        return None  # 계속 진행
