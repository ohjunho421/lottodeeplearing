import os
from datetime import datetime
import pandas as pd
from django.core.management.base import BaseCommand
from chatbot.models import LottoDraw

class Command(BaseCommand):
    help = "Load lotto data from lotto_history.csv into the LottoDraw model"

    def handle(self, *args, **kwargs):
        # CSV 파일 경로
        csv_file_path = os.path.join('data', 'lotto_history.csv')
        if not os.path.exists(csv_file_path):
            self.stderr.write(self.style.ERROR(f"CSV 파일을 찾을 수 없습니다: {csv_file_path}"))
            return

        # CSV 파일 읽기
        df = pd.read_csv(csv_file_path)

        # 날짜 변환
        df['추첨일'] = df['추첨일'].apply(lambda x: datetime.strptime(x, "%Y.%m.%d").strftime("%Y-%m-%d"))

        # 데이터 추가
        for _, row in df.iterrows():
            if not LottoDraw.objects.filter(round_no=int(row['회차'])).exists():
                LottoDraw.objects.create(
                    round_no=int(row['회차']),
                    draw_date=row['추첨일'],
                    winning_numbers=",".join(map(str, [row['1'], row['2'], row['3'], row['4'], row['5'], row['6']])),
                    bonus_number=int(row['보너스'])
                )
                self.stdout.write(self.style.SUCCESS(f"회차 {row['회차']} 데이터 추가 완료"))
            else:
                self.stdout.write(self.style.WARNING(f"회차 {row['회차']} 데이터가 이미 존재합니다."))

        total = LottoDraw.objects.count()
        self.stdout.write(self.style.SUCCESS(f"총 데이터 개수: {total}"))
