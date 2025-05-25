# 모든 추천 번호의 당첨 여부를 강제로 다시 확인하는 스크립트
import os
import django
import sys

# Django 설정 로드
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Lottobot.settings')
django.setup()

# 필요한 함수 import
from chatbot.services import check_winning_numbers

def main():
    print("===== 모든 추천 번호의 당첨 여부 강제 확인 =====")
    
    # force_check_all=True로 호출하여 모든 추천 번호 확인
    success, message = check_winning_numbers(force_check_all=True)
    
    if success:
        print(f"성공: {message}")
    else:
        print(f"오류: {message}")
    
    print("처리 완료")

if __name__ == "__main__":
    main()
