# chatbot/management/commands/update_lotto.py

from django.core.management.base import BaseCommand
from chatbot.services import LottoDataCollector
from chatbot import lotto_ml  # 새로 만든 머신러닝 모듈
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

            # 2. 머신러닝 모델 학습 (충분한 데이터가 있는 경우)
            if df is not None and len(df) >= 6:
                self.stdout.write("머신러닝 모델 학습 시작...")
                model, scaler, df_model = lotto_ml.train_lotto_model(window_size=5)
                self.stdout.write(self.style.SUCCESS('머신러닝 모델 학습 완료'))
            else:
                self.stdout.write(self.style.ERROR('학습에 필요한 충분한 데이터가 없습니다.'))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'오류 발생: {str(e)}'))
