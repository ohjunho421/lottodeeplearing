# 추천 번호의 당첨 정보 조회 스크립트
import os
import django

# Django 설정 로드
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Lottobot.settings')
django.setup()

# 모델 import
from chatbot.models import Recommendation

def main():
    print("===== 추천 번호 당첨 정보 확인 =====")
    
    # 확인된 추천 기록 카운트
    checked_count = Recommendation.objects.filter(is_checked=True).count()
    total_count = Recommendation.objects.all().count()
    
    print(f"전체 추천 기록: {total_count}개")
    print(f"확인된 추천 기록: {checked_count}개")
    print(f"확인 비율: {checked_count/total_count*100:.2f}%")
    
    # 당첨 정보 요약
    wins_by_rank = {}
    for i in range(1, 6):
        rank_count = Recommendation.objects.filter(rank=i).count()
        wins_by_rank[i] = rank_count
    
    print("\n== 당첨 통계 ==")
    print(f"1등 당첨: {wins_by_rank[1]}개")
    print(f"2등 당첨: {wins_by_rank[2]}개")
    print(f"3등 당첨: {wins_by_rank[3]}개")
    print(f"4등 당첨: {wins_by_rank[4]}개")
    print(f"5등 당첨: {wins_by_rank[5]}개")
    
    # 맞춘 개수별 통계
    matches = {}
    for i in range(7):
        match_count = Recommendation.objects.filter(matched_count=i, is_checked=True).count()
        matches[i] = match_count
    
    print("\n== 맞춘 개수 통계 ==")
    for i in range(7):
        print(f"{i}개 맞춘 추천: {matches[i]}개")
    
    # 최근 확인된 추천 10개 출력
    print("\n== 최근 확인된 추천 10개 ==")
    recent_checked = Recommendation.objects.filter(is_checked=True).order_by('-recommendation_date')[:10]
    for rec in recent_checked:
        print(f"ID: {rec.id}, 추천일: {rec.recommendation_date.strftime('%Y-%m-%d')}, " +
              f"번호: {rec.numbers}, 맞춘개수: {rec.matched_count}, 당첨: {'O' if rec.is_won else 'X'}")

if __name__ == "__main__":
    main()
