# 로또 데이터 크롤링 및 추천 번호 당첨 여부 확인 스크립트
import os
import django

# Django 설정 로드
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Lottobot.settings')
django.setup()

import logging
from chatbot.services import LottoDataCollector, check_winning_numbers

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("===== 로또 데이터 크롤링 및 추천 번호 당첨 여부 확인 =====")
    
    try:
        # 1. 로또 데이터 크롤링
        print("\n[1] 로또 당첨 데이터 크롤링 시작...")
        collector = LottoDataCollector()
        update_result = collector.update_latest_data()  # 로또 데이터 업데이트
        
        if update_result:
            print("로또 데이터 크롤링 성공!")
        else:
            print("로또 데이터 크롤링 실패 또는 변경사항 없음")
        
        # 크롤링된 데이터 확인
        from chatbot.models import LottoDraw
        draws_count = LottoDraw.objects.all().count()
        print(f"현재 저장된 로또 당첨 정보: {draws_count}개")
        
        if draws_count > 0:
            # 최근 5개 당첨 정보 출력
            recent_draws = LottoDraw.objects.all().order_by('-round_no')[:5]
            print("\n== 최근 당첨 정보 ==")
            for draw in recent_draws:
                print(f"회차: {draw.round_no}, 추첨일: {draw.draw_date.strftime('%Y-%m-%d')}")
                print(f"당첨번호: {draw.winning_numbers}, 보너스: {draw.bonus_number}")
                print("-" * 40)
            
            # 2. 추천 번호 당첨 여부 확인
            print("\n[2] 모든 추천 번호의 당첨 여부 확인 시작...")
            success, message = check_winning_numbers(force_check_all=True)
            print(f"결과: {message}")
            
            # 3. 결과 확인
            from chatbot.models import Recommendation
            checked_count = Recommendation.objects.filter(is_checked=True).count()
            won_count = Recommendation.objects.filter(is_won=True).count()
            
            print("\n===== 업데이트 결과 =====")
            print(f"확인된 추천 기록: {checked_count}개")
            print(f"당첨된 추천 기록: {won_count}개")
            
            # 등수별 통계
            for i in range(1, 6):
                count = Recommendation.objects.filter(rank=i).count()
                print(f"{i}등 당첨: {count}개")
                
            # 맞춘 개수별 통계
            for i in range(7):
                count = Recommendation.objects.filter(matched_count=i, is_checked=True).count()
                if count > 0:
                    print(f"{i}개 맞춘 추천: {count}개")
        else:
            print("당첨 정보가 없습니다. 크롤링에 실패했습니다.")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
