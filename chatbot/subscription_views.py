"""
로또봇 구독 및 결제 관련 뷰
"""
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils import timezone
from datetime import timedelta
import json
import uuid
import logging

from .models import UserProfile, Payment
from .decorators import subscription_required

logger = logging.getLogger(__name__)

@login_required
def subscription_page(request):
    """구독 정보 및 결제 페이지"""
    try:
        profile, created = UserProfile.objects.get_or_create(user=request.user)
        
        # 귀하의 계정 확인 (귀하의 사용자명을 입력하세요)
        if request.user.username == 'admin':  # 관리자 계정을 여기에 지정
            profile.is_premium = True
            profile.save()
            messages.success(request, "프리미엄 계정으로 설정되었습니다. 모든 기능을 구독 없이 이용하실 수 있습니다.")
        
        context = {
            'profile': profile,
            'is_active': profile.is_subscription_active(),
            'end_date': profile.subscription_end.strftime('%Y년 %m월 %d일') if profile.subscription_end else None,
            'merchant_uid': f"order_{uuid.uuid4().hex[:10]}",  # 주문 ID 생성
            'amount': 30000,  # 월 구독료
        }
        return render(request, 'subscription.html', context)
        
    except Exception as e:
        logger.error(f"구독 페이지 오류: {str(e)}")
        messages.error(request, "구독 정보를 불러오는 중 오류가 발생했습니다.")
        return redirect('home')

@login_required
def subscription_needed(request):
    """구독이 필요함을 알리는 페이지"""
    return render(request, 'subscription_needed.html')

@csrf_exempt
@login_required
def process_payment(request):
    """결제 처리 API"""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': '잘못된 요청 방식입니다.'})
        
    try:
        data = json.loads(request.body)
        imp_uid = data.get('imp_uid')  # 결제 ID
        merchant_uid = data.get('merchant_uid')  # 주문 ID
        
        # TODO: 실제 구현시 아임포트 API로 결제 검증 필요
        # 여기서는 단순화를 위해 항상 성공으로 처리
        
        # 결제 정보 저장
        payment = Payment.objects.create(
            user=request.user,
            amount=30000,
            payment_id=imp_uid,
            payment_method='card',  # 기본값
            is_successful=True,
            subscription_period=1  # 1개월
        )
        
        # 사용자 프로필 업데이트
        profile, created = UserProfile.objects.get_or_create(user=request.user)
        profile.extend_subscription(months=1)  # 1개월 연장
        profile.last_payment_date = timezone.now()
        profile.save()
        
        return JsonResponse({
            'status': 'success',
            'message': '구독이 성공적으로 등록되었습니다.',
            'subscription_end': profile.subscription_end.strftime('%Y-%m-%d')
        })
        
    except Exception as e:
        logger.error(f"결제 처리 중 오류: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': f'결제 처리 중 오류가 발생했습니다: {str(e)}'
        }, status=500)

@login_required
def payment_history(request):
    """결제 내역 조회"""
    payments = Payment.objects.filter(user=request.user, is_successful=True).order_by('-payment_date')
    return render(request, 'payment_history.html', {'payments': payments})

@login_required
def cancel_subscription(request):
    """구독 취소"""
    if request.method != 'POST':
        return redirect('subscription_page')
        
    try:
        profile = UserProfile.objects.get(user=request.user)
        profile.is_subscribed = False
        profile.save()
        
        messages.success(request, "구독이 취소되었습니다. 구독 기간이 끝날 때까지 서비스를 이용하실 수 있습니다.")
        return redirect('subscription_page')
        
    except Exception as e:
        logger.error(f"구독 취소 중 오류: {str(e)}")
        messages.error(request, "구독 취소 중 오류가 발생했습니다.")
        return redirect('subscription_page')
