# chatbot/management/commands/update_lotto.py

from django.core.management.base import BaseCommand
from chatbot.services import LottoDataCollector, AdvancedLottoPredictor
import logging
import pandas as pd
from django.conf import settings

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = '로또 데이터 수집 및 머신러닝 모델 학습 실행'

    def handle(self, *args, **options):
        try:
            # 1. 데이터 수집
            collector = LottoDataCollector()
            if pd.io.common.file_exists(settings.LOTTO_DATA_FILE):
                df = pd.read_csv(settings.LOTTO_DATA_FILE)
                self.stdout.write(f"기존 데이터 로드 완료: {len(df)}개의 데이터")
            else:
                self.stdout.write("기존 데이터 파일이 없습니다.")
                df = None

            updated = collector.update_latest_data()
            if updated:
                self.stdout.write(self.style.SUCCESS('새로운 데이터가 추가되었습니다.'))
                df = pd.read_csv(settings.LOTTO_DATA_FILE)
                self.stdout.write(f"갱신된 데이터: 총 {len(df)}개의 데이터")
            else:
                self.stdout.write(self.style.WARNING('새로운 데이터가 없거나 이미 최신 상태입니다.'))

            # 2. 머신러닝 모델 학습 (AdvancedLottoPredictor 사용)
            if df is not None and len(df) >= 6:
                self.stdout.write("머신러닝 모델 학습 시작...")
                predictor = AdvancedLottoPredictor()
                success, results = predictor.train_models()
                if success:
                    self.stdout.write(self.style.SUCCESS('머신러닝 모델 학습 완료'))
                    if results:
                        self.stdout.write(f"XGBoost 모델 R2 (Train): {results['xgb_train_r2']:.4f}")
                        self.stdout.write(f"XGBoost 모델 R2 (Test): {results['xgb_test_r2']:.4f}")
                        self.stdout.write(f"RandomForest 모델 R2 (Train): {results['rf_train_r2']:.4f}")
                        self.stdout.write(f"RandomForest 모델 R2 (Test): {results['rf_test_r2']:.4f}")
                else:
                    self.stdout.write(self.style.ERROR('머신러닝 모델 학습 실패'))
            else:
                self.stdout.write(self.style.ERROR('학습에 필요한 충분한 데이터가 없습니다.'))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'오류 발생: {str(e)}'))
            import traceback
            self.stdout.write(traceback.format_exc())