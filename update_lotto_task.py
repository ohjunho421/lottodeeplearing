# D:\lottobot\lottodeeplearing\update_lotto_task.py
import os
import sys
import django

# Django 설정 로드
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Lottobot.settings')
django.setup()

# 필요한 함수 import
from chatbot.cron import update_lotto_draws

if __name__ == "__main__":
    # 업데이트 함수 실행
    result = update_lotto_draws()
    print(f"Update result: {result}")