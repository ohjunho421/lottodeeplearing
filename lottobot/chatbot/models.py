from django.conf import settings
from django.db import models

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
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    numbers = models.JSONField()  # 추천된 번호들 저장
    strategy = models.IntegerField()  # 사용된 전략
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']