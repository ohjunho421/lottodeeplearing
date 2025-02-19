# chatbot/cron.py
from .services import LottoDataCollector, LottoPredictor

def update_lotto_draws():
    try:
        # 데이터 수집
        collector = LottoDataCollector()
        collector.update_latest_data()
        
        # 모델 재학습
        predictor = LottoPredictor()
        predictor.train_model()
        
        print("크롤링 및 모델 재학습 완료.")
    except Exception as e:
        print("크론 작업 오류:", str(e))