# CSV 파일의 로또 데이터를 DB로 가져오는 스크립트
import os
import django
import pandas as pd
from datetime import datetime

# Django 설정 로드
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Lottobot.settings')
django.setup()

from chatbot.models import LottoDraw
from django.conf import settings
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def import_csv_to_db():
    """CSV 파일의 로또 데이터를 DB로 가져오기"""
    csv_file = os.path.join(settings.BASE_DIR, 'data', 'lotto_history.csv')
    
    if not os.path.exists(csv_file):
        logger.error(f"CSV 파일이 존재하지 않습니다: {csv_file}")
        return False
    
    try:
        # CSV 파일 읽기
        df = pd.read_csv(csv_file)
        logger.info(f"CSV 파일 로드 완료: 총 {len(df)}개 회차 데이터")
        
        # 데이터 정렬 (최신 회차가 위로)
        df = df.sort_values('회차', ascending=False).reset_index(drop=True)
        
        # 기존 DB 데이터 확인
        existing_rounds = set(LottoDraw.objects.values_list('round_no', flat=True))
        logger.info(f"기존 DB 데이터: {len(existing_rounds)}개 회차")
        
        # 저장된 데이터 수
        imported_count = 0
        
        # 각 회차 데이터를 DB에 저장
        for _, row in df.iterrows():
            round_no = int(row['회차'])
            
            # 이미 있는 회차는 건너뜀
            if round_no in existing_rounds:
                continue
            
            # 날짜 형식 변환
            draw_date_str = row['추첨일']
            draw_date = datetime.strptime(draw_date_str, '%Y.%m.%d') if '.' in draw_date_str else datetime.strptime(draw_date_str, '%Y-%m-%d')
            
            # 당첨 번호 추출
            winning_numbers = ','.join([str(int(row[str(i)])) for i in range(1, 7)])
            bonus_number = int(row['보너스'])
            
            # DB에 저장
            lotto_draw = LottoDraw(
                round_no=round_no,
                draw_date=draw_date,
                winning_numbers=winning_numbers,
                bonus_number=bonus_number
            )
            lotto_draw.save()
            
            imported_count += 1
            logger.info(f"회차 {round_no} 데이터 저장 완료")
        
        logger.info(f"총 {imported_count}개 회차 데이터 DB 저장 완료")
        return True
    
    except Exception as e:
        logger.error(f"데이터 가져오기 중 오류: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def check_recommendations_after_import():
    """데이터 가져오기 후 추천 번호 당첨 여부 확인"""
    from chatbot.services import check_winning_numbers
    
    try:
        # 모든 추천 번호 강제 확인
        success, message = check_winning_numbers(force_check_all=True)
        logger.info(f"추천 번호 당첨 확인 결과: {message}")
        
        # 확인 결과 출력
        from chatbot.models import Recommendation
        checked_count = Recommendation.objects.filter(is_checked=True).count()
        won_count = Recommendation.objects.filter(is_won=True).count()
        
        logger.info(f"확인된 추천 기록: {checked_count}개")
        logger.info(f"당첨된 추천 기록: {won_count}개")
        
        # 등수별 통계
        for i in range(1, 6):
            count = Recommendation.objects.filter(rank=i).count()
            logger.info(f"{i}등 당첨: {count}개")
        
        return success
    
    except Exception as e:
        logger.error(f"추천 번호 확인 중 오류: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("===== CSV 파일의 로또 데이터를 DB로 가져오기 =====")
    import_success = import_csv_to_db()
    
    if import_success:
        print("\n===== 추천 번호 당첨 여부 확인 =====")
        check_recommendations_after_import()
    else:
        print("CSV 데이터 가져오기 실패. 추천 번호 확인을 건너뜁니다.")
