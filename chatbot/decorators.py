"""
로또봇 구독 관련 데코레이터
"""
from functools import wraps
from django.shortcuts import redirect
from django.contrib import messages
from django.urls import reverse

def subscription_required(view_func):
    """
    구독이 필요한 뷰를 위한 데코레이터
    구독 중인 사용자 또는 프리미엄 계정만 접근 가능
    """
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        # 로그인하지 않은 경우
        if not request.user.is_authenticated:
            messages.warning(request, "이 기능을 이용하려면 로그인이 필요합니다.")
            return redirect('login')
        
        # 프리미엄 계정은 항상 접근 가능
        if hasattr(request.user, 'is_premium') and request.user.is_premium:
            return view_func(request, *args, **kwargs)
            
        # 일반 계정은 구독 상태 확인
        if hasattr(request.user, 'is_active_subscriber') and request.user.is_active_subscriber:
            return view_func(request, *args, **kwargs)
            
        # 구독이 없는 경우
        messages.warning(request, "이 기능을 이용하려면 구독이 필요합니다.")
        return redirect('subscription_needed')
        
    return _wrapped_view


def api_subscription_required(view_func):
    """
    API 뷰를 위한 구독 확인 데코레이터
    JSON 응답 반환
    """
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        from django.http import JsonResponse
        
        # 로그인하지 않은 경우
        if not request.user.is_authenticated:
            return JsonResponse({
                'status': 'error',
                'message': '로그인이 필요합니다.'
            }, status=401)
        
        # 프리미엄 계정은 항상 접근 가능
        if hasattr(request.user, 'is_premium') and request.user.is_premium:
            return view_func(request, *args, **kwargs)
            
        # 일반 계정은 구독 상태 확인
        if hasattr(request.user, 'is_active_subscriber') and request.user.is_active_subscriber:
            return view_func(request, *args, **kwargs)
            
        # 구독이 없는 경우
        return JsonResponse({
            'status': 'error',
            'message': '이 기능을 이용하려면 구독이 필요합니다.'
        }, status=403)
        
    return _wrapped_view
