# 예측 결과 확인 스크립트
import os
import django
import numpy as np

# Django 설정 로드
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Lottobot.settings')
django.setup()

# 필요한 함수 및 클래스 import
from chatbot.services import AdvancedLottoPredictor, shared_predictor

def check_predictions():
    print("===== 로또봇 예측 모델 결과 확인 =====")
    
    # 이미 로드된 모델 사용
    predictor = shared_predictor
    
    # 예측 확률 가져오기
    probabilities = predictor.predict_numbers()
    
    # 상위 10개 번호와 확률 출력
    top_indices = np.argsort(probabilities)[-10:][::-1]
    print("\n상위 10개 번호와 확률:")
    for idx in top_indices:
        number = idx + 1  # 인덱스는 0부터 시작하므로 1 더함
        probability = probabilities[idx] * 100  # 퍼센트로 변환
        print(f"번호 {number:2d}: {probability:.2f}%")
    
    # 번호별 확률 분포 출력
    print("\n전체 번호 확률 분포:")
    for i in range(45):
        number = i + 1
        prob = probabilities[i] * 100
        bar = "#" * int(prob * 2)  # 확률에 비례한 시각적 표현
        print(f"번호 {number:2d}: {prob:.2f}% {bar}")
    
    # 추천 번호 조합 생성
    print("\n추천 번호 조합 (5개):")
    for i in range(5):
        # 가중치에 따른 번호 선택
        selected = np.random.choice(
            range(1, 46),
            size=6,
            replace=False,
            p=probabilities
        )
        selected_list = sorted([int(num) for num in selected])
        print(f"조합 {i+1}: {selected_list}")
    
    # 시계열 패턴 분석 결과 요약
    if hasattr(predictor, 'temporal_patterns') and predictor.temporal_patterns:
        print("\n시계열 트렌드 상위 번호:")
        trends = []
        for num, pattern in predictor.temporal_patterns.items():
            if 'trend' in pattern and not pattern['trend'].empty:
                trend_value = pattern['trend'].iloc[-1]
                if not np.isnan(trend_value):
                    trends.append((num, trend_value))
        
        # 트렌드 값 기준 상위 10개 출력
        top_trends = sorted(trends, key=lambda x: x[1], reverse=True)[:10]
        for num, trend in top_trends:
            print(f"번호 {num:2d}: 트렌드 값 {trend:.3f}")

if __name__ == "__main__":
    check_predictions()
