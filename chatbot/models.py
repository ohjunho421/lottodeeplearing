from django.conf import settings
from django.db import models
from django.contrib.auth.models import User


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
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    recommendation_date = models.DateField(auto_now_add=True)
    created_at = models.DateTimeField(auto_now_add=True)  # 생성일시
    strategy = models.IntegerField()  # 1 또는 2
    numbers = models.CharField(max_length=20)  # "1,3,6,34,47,25" 형식으로 저장
    is_checked = models.BooleanField(default=False)  # 당첨 여부 확인 했는지
    is_won = models.BooleanField(default=False)  # 당첨 여부
    draw_round = models.IntegerField(null=True)  # 해당 회차
    draw_date = models.DateField(null=True)  # 추첨일

    class Meta:
        ordering = ['-recommendation_date']

