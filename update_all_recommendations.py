# 모든 추천 번호의 당첨 여부를 직접 업데이트하는 스크립트
import os
import django
import sys
from datetime import datetime

# Django 설정 로드
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Lottobot.settings')
django.setup()

# 필요한 모델과 함수 import
from chatbot.models import Recommendation, LottoDraw
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_all_recommendations():
    """모든 추천 기록의 당첨 여부를 직접 업데이트"""
    print("===== 모든 추천 번호의 당첨 여부 강제 업데이트 =====")
    
    # 모든 추천 기록 조회
    recommendations = Recommendation.objects.all()
    total_count = recommendations.count()
    print(f"총 {total_count}개의 추천 기록을 처리합니다.")
    
    # 모든 당첨 정보 조회 (최신순)
    draws = LottoDraw.objects.all().order_by('-round_no')
    
    if not draws.exists():
        print("당첨 정보가 없습니다.")
        return
    
    # 각 추천 기록 처리
    updated_count = 0
    for rec in recommendations:
        try:
            # 추천일 이후의 첫 번째 당첨 회차 찾기
            matching_draw = None
            for draw in draws:
                # 당첨일이 추천일 이후인지 확인 (날짜만 비교)
                rec_date = rec.recommendation_date.date()
                draw_date = draw.draw_date.date()
                
                if draw_date >= rec_date:
                    matching_draw = draw
                    break
            
            if not matching_draw:
                print(f"ID: {rec.id}, 추천일: {rec.recommendation_date.strftime('%Y-%m-%d')}의 추천 번호에 해당하는 추첨 결과가 없습니다.")
                continue
            
            # 추천 번호와 당첨 번호 비교
            winning_numbers = [int(n) for n in matching_draw.winning_numbers.split(',')]
            recommended_numbers = [int(n) for n in rec.numbers.split(',')]
            
            # 일치하는 번호 개수 계산
            matched_count = len(set(winning_numbers) & set(recommended_numbers))
            
            # 보너스 번호 일치 여부 확인
            has_bonus = matching_draw.bonus_number in recommended_numbers
            
            # 당첨 등수 계산
            rank = 0  # 기본값 (낙첨)
            is_won = False
            
            if matched_count == 6:
                rank = 1
                is_won = True
            elif matched_count == 5 and has_bonus:
                rank = 2
                is_won = True
            elif matched_count == 5:
                rank = 3
                is_won = True
            elif matched_count == 4:
                rank = 4
                is_won = True
            elif matched_count == 3:
                rank = 5
                is_won = True
            
            # 추천 기록 업데이트
            rec.is_checked = True
            rec.is_won = is_won
            rec.draw_round = matching_draw.round_no
            rec.draw_date = matching_draw.draw_date
            rec.matched_count = matched_count
            rec.has_bonus = has_bonus
            rec.rank = rank
            rec.save()
            
            print(f"ID: {rec.id}, 추천일: {rec.recommendation_date.strftime('%Y-%m-%d')}, "
                  f"당첨회차: {matching_draw.round_no}, 맞춘개수: {matched_count}, "
                  f"등수: {rank if rank > 0 else '낙첨'}")
            
            updated_count += 1
        
        except Exception as e:
            print(f"ID: {rec.id} 처리 중 오류 발생: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    print(f"총 {updated_count}/{total_count}개의 추천 기록이 업데이트되었습니다.")

if __name__ == "__main__":
    update_all_recommendations()
    
    # 결과 확인
    checked_count = Recommendation.objects.filter(is_checked=True).count()
    won_count = Recommendation.objects.filter(is_won=True).count()
    
    print("\n===== 업데이트 결과 =====")
    print(f"확인된 추천 기록: {checked_count}개")
    print(f"당첨된 추천 기록: {won_count}개")
    
    # 등수별 통계
    for i in range(1, 6):
        count = Recommendation.objects.filter(rank=i).count()
        print(f"{i}등 당첨: {count}개")
