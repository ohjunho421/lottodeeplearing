# chatbot/cron.py
import logging
from .services import LottoDataCollector, AdvancedLottoPredictor, check_winning_numbers

logger = logging.getLogger(__name__)

def update_lotto_draws():
    try:
        # 데이터 수집
        collector = LottoDataCollector()
        update_result = collector.update_latest_data()
        logger.info(f"데이터 수집 결과: {update_result}")
        
        # 새로운 데이터가 있을 경우에만 모델 재학습 및 당첨 확인 실행
        if update_result:
            # 모델 재학습
            predictor = AdvancedLottoPredictor()
            train_result = predictor.train_models()
            logger.info(f"모델 재학습 결과: {train_result}")
            
            # 추천 번호 당첨 여부 확인
            winning_check_result, message = check_winning_numbers()
            logger.info(f"추천 번호 당첨 확인 결과: {message}")
        else:
            logger.info("새로운 데이터가 없어 모델 재학습 및 당첨 확인을 건너뜁니다.")
        
        logger.info("크롤링 및 관련 작업 완료.")
        return True
    except Exception as e:
        logger.error(f"크론 작업 오류: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False