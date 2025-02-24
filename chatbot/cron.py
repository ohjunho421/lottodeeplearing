# chatbot/cron.py
import logging
from .services import LottoDataCollector, LottoPredictor

logger = logging.getLogger(__name__)

def update_lotto_draws():
    try:
        # 데이터 수집
        collector = LottoDataCollector()
        update_result = collector.update_latest_data()
        logger.info(f"데이터 수집 결과: {update_result}")
        
        # 모델 재학습
        predictor = LottoPredictor()
        train_result = predictor.train_model()
        logger.info(f"모델 재학습 결과: {train_result}")
        
        logger.info("크롤링 및 모델 재학습 완료.")
        return True
    except Exception as e:
        logger.error(f"크론 작업 오류: {str(e)}")
        return False