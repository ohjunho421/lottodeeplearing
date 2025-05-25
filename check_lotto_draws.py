# 로또 당첨 정보 확인 스크립트
import os
import django

# Django 설정 로드
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Lottobot.settings')
django.setup()

# 모델 import
from chatbot.models import LottoDraw

def main():
    print("===== 로또 당첨 정보 확인 =====")
    
    # 당첨 정보 총 개수
    total_count = LottoDraw.objects.all().count()
    print(f"등록된 당첨 정보: {total_count}개")
    
    # 당첨 정보가 있는 경우 최근 5개 출력
    if total_count > 0:
        recent_draws = LottoDraw.objects.all().order_by('-round_no')[:5]
        print("\n== 최근 당첨 정보 ==")
        for draw in recent_draws:
            print(f"회차: {draw.round_no}, 추첨일: {draw.draw_date.strftime('%Y-%m-%d')}")
            print(f"당첨번호: {draw.winning_numbers}, 보너스: {draw.bonus_number}")
            print("-" * 40)
    else:
        print("\n당첨 정보가 없습니다. 데이터를 먼저 크롤링해야 합니다.")
        
    print("\n== LottoDraw 모델 필드 확인 ==")
    for field in LottoDraw._meta.fields:
        print(f"필드명: {field.name}, 타입: {field.get_internal_type()}")

if __name__ == "__main__":
    main()
