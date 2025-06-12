from django.conf import settings
from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
from datetime import timedelta

class LottoDraw(models.Model):
    round_no = models.IntegerField(unique=True)  # 회차 번호
    draw_date = models.DateField()  # 추첨 날짜
    winning_numbers = models.CharField(max_length=50)  # 당첨 번호 (쉼표로 구분)
    bonus_number = models.IntegerField()  # 보너스 번호

    def __str__(self):
        return f"회차: {self.round_no}, 날짜: {self.draw_date}"

class ChatHistory(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    user_message = models.TextField()
    bot_response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Chat by {self.user} at {self.created_at}"

class Recommendation(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)  # User 모델 참조 수정
    recommendation_date = models.DateField(auto_now_add=True)
    strategy = models.IntegerField()  # 1 또는 2
    numbers = models.CharField(max_length=20)  # "1,3,6,34,47,25" 형식으로 저장
    is_checked = models.BooleanField(default=False)  # 당첨 여부 확인 했는지
    is_won = models.BooleanField(default=False)  # 당첨 여부
    draw_round = models.IntegerField(null=True)  # 해당 회차
    draw_date = models.DateField(null=True)  # 추첨일
    matched_count = models.IntegerField(null=True)  # 맞춘 개수
    has_bonus = models.BooleanField(default=False)  # 보너스 번호 일치 여부
    rank = models.IntegerField(null=True)  # 당첨 순위 (1~5, 0은 낙첨)

    class Meta:
        ordering = ['-recommendation_date']


class Payment(models.Model):
    """결제 정보 모델"""
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    amount = models.IntegerField(default=30000)  # 월 3만원 고정
    payment_date = models.DateTimeField(auto_now_add=True)
    payment_id = models.CharField(max_length=100)  # 결제 대행사의 결제 ID
    payment_method = models.CharField(max_length=50)  # 카드, 계좌이체 등
    is_successful = models.BooleanField(default=False)
    subscription_period = models.IntegerField(default=1)  # 구독 개월 수
    
    def __str__(self):
        return f"{self.user.username}의 결제 - {self.payment_date.strftime('%Y-%m-%d')} ({self.amount}원)"


class UserProfile(models.Model):
    """사용자 프로필 모델"""
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='profile')
    subscription_start = models.DateTimeField(null=True, blank=True)
    subscription_end = models.DateTimeField(null=True, blank=True)
    is_subscribed = models.BooleanField(default=False)
    is_premium = models.BooleanField(default=False)  # 특별 계정 표시 (구독 없이도 이용 가능)
    last_payment_date = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        subscription_status = "구독 중" if self.is_subscription_active() else "구독 없음"
        if self.is_premium:
            subscription_status = "프리미엄 계정"
        return f"{self.user.username} - {subscription_status}"
    
    def is_subscription_active(self):
        """구독이 현재 유효한지 확인"""
        if self.is_premium:  # 프리미엄 계정은 항상 유효
            return True
        if not self.is_subscribed:
            return False
        now = timezone.now()
        return self.subscription_start <= now <= self.subscription_end
    
    def extend_subscription(self, months=1):
        """구독 기간 연장"""
        now = timezone.now()
        if self.is_subscription_active():
            self.subscription_end = self.subscription_end + timedelta(days=30*months)
        else:
            self.subscription_start = now
            self.subscription_end = now + timedelta(days=30*months)
            self.is_subscribed = True
        self.save()