# services.py

import os
import logging
import numpy as np
import pandas as pd
import requests
import json
from bs4 import BeautifulSoup
from django.conf import settings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from joblib import dump, load

logger = logging.getLogger(__name__)

class LottoDataCollector:
    def __init__(self):
        self.base_url = "https://www.dhlottery.co.kr/gameResult.do?method=byWin"
        self.data_file = settings.LOTTO_DATA_FILE

    def collect_initial_data(self):
        """초기 데이터 수집"""
        try:
            logger.info("초기 데이터 수집 시작")
            response = requests.get(self.base_url)
            if response.status_code != 200:
                logger.error(f"HTTP 오류: {response.status_code}")
                return None

            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 당첨번호 추출
            win_numbers = soup.select('div.num.win span.ball_645')
            if not win_numbers or len(win_numbers) != 6:
                logger.error("당첨번호를 찾을 수 없습니다")
                return None

            # 보너스 번호 추출
            bonus_ball = soup.select('div.num.bonus span.ball_645')
            if not bonus_ball or len(bonus_ball) != 1:
                logger.error("보너스 번호를 찾을 수 없습니다")
                return None

            # 회차 정보 추출
            draw_result = soup.select('div.win_result h4')
            if not draw_result:
                logger.error("회차 정보를 찾을 수 없습니다")
                return None

            # 추첨일 추출
            draw_date = soup.select('p.desc')
            if not draw_date:
                logger.error("추첨일을 찾을 수 없습니다")
                return None

            try:
                # 데이터 파싱
                numbers = [int(n.text.strip()) for n in win_numbers]
                bonus = int(bonus_ball[0].text.strip())
                draw_no = int(''.join(filter(str.isdigit, draw_result[0].text)))
                date_text = draw_date[0].text.strip()
                drawn_date = date_text[date_text.find('(')+1:date_text.find(')')]

                logger.info(f"추출된 데이터: 회차={draw_no}, 번호={numbers}, 보너스={bonus}, 날짜={drawn_date}")

                # 데이터프레임 생성
                df = pd.DataFrame([{
                    '회차': draw_no,
                    '추첨일': drawn_date,
                    '1': numbers[0],
                    '2': numbers[1],
                    '3': numbers[2],
                    '4': numbers[3],
                    '5': numbers[4],
                    '6': numbers[5],
                    '보너스': bonus
                }])

                # 데이터 저장
                os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
                df.to_csv(self.data_file, index=False)
                logger.info(f"데이터 파일 저장 완료: {self.data_file}")
                return df

            except Exception as e:
                logger.error(f"데이터 파싱 오류: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"초기 데이터 수집 중 오류 발생: {str(e)}")
            return None

            if all_data:
                df = pd.DataFrame(all_data)
                df = df.sort_values('회차', ascending=False)
                logger.info(f"데이터프레임 생성 완료. 총 {len(df)}개의 데이터")
                
                # 데이터 저장 전에 디렉토리 확인
                os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
                df.to_csv(self.data_file, index=False)
                logger.info(f"데이터 파일 저장 완료: {self.data_file}")
                return df
            else:
                logger.error("수집된 데이터가 없습니다.")
                return None

        except Exception as e:
            logger.error(f"초기 데이터 수집 중 오류 발생: {str(e)}")
            return None

    def _parse_date(self, date_text):
        """크롤링한 날짜를 YYYY.MM.DD 형식으로 변환"""
        try:
            # '2025년 02월 15일' 형식에서 숫자만 추출
            date_parts = ''.join(filter(str.isdigit, date_text))
            year = date_parts[:4]
            month = date_parts[4:6]
            day = date_parts[6:8]
            return f"{year}.{month}.{day}"
        except Exception as e:
            logger.error(f"날짜 파싱 오류: {str(e)}")
            return date_text

    def update_latest_data(self):
        """최신 데이터 업데이트 (CSV 형식을 첫번째 파일과 동일하게 통일)"""
        try:
            logger.info("최신 데이터 업데이트 시작")
            response = requests.get(self.base_url)
            if response.status_code != 200:
                logger.error(f"HTTP 오류: {response.status_code}")
                return False

            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 당첨번호 추출 (ball_645 클래스 사용)
            win_numbers = soup.select('div.num.win span.ball_645')
            if not win_numbers or len(win_numbers) != 6:
                logger.error("당첨번호를 찾을 수 없습니다")
                return False

            # 보너스 번호 추출
            bonus_ball = soup.select('div.num.bonus span.ball_645')
            if not bonus_ball or len(bonus_ball) != 1:
                logger.error("보너스 번호를 찾을 수 없습니다")
                return False

            # 회차 정보 추출
            draw_result = soup.select('div.win_result h4')
            if not draw_result:
                logger.error("회차 정보를 찾을 수 없습니다")
                return False

            # 추첨일 추출
            draw_date = soup.select('p.desc')
            if not draw_date:
                logger.error("추첨일을 찾을 수 없습니다")
                return False

            try:
                numbers = [int(n.text.strip()) for n in win_numbers]
                bonus = int(bonus_ball[0].text.strip())
                draw_no = int(''.join(filter(str.isdigit, draw_result[0].text)))
                date_text = draw_date[0].text.strip()
                # 괄호 안의 날짜를 추출하여 전처리
                draw_date_extracted = date_text[date_text.find('(')+1:date_text.find(')')]
                draw_date_formatted = self._parse_date(draw_date_extracted)

                logger.info(f"추출된 데이터: 회차={draw_no}, 번호={numbers}, 보너스={bonus}, 날짜={draw_date_formatted}")
            except Exception as e:
                logger.error(f"데이터 파싱 오류: {str(e)}")
                return False

            # 기존 CSV 파일 로드 (없으면 모든 컬럼을 포함한 빈 DataFrame 생성)
            if os.path.exists(self.data_file):
                df = pd.read_csv(self.data_file)
            else:
                df = pd.DataFrame(columns=[
                    '회차', '추첨일', '1', '2', '3', '4', '5', '6', '보너스',
                    '날짜', '번호1', '번호2', '번호3', '번호4', '번호5', '번호6'
                ])

            # 신규 회차가 존재하지 않을 때만 추가
            if draw_no not in df['회차'].values:
                new_row = pd.DataFrame([{
                    '회차': draw_no,
                    '추첨일': draw_date_formatted,
                    '1': numbers[0],
                    '2': numbers[1],
                    '3': numbers[2],
                    '4': numbers[3],
                    '5': numbers[4],
                    '6': numbers[5],
                    '보너스': bonus
                }])
                
                # 신규 데이터는 원본 형식(회차,추첨일,1~6,보너스)만 가지고 있음.
                # 기존 데이터와 병합
                updated_df = pd.concat([new_row, df], ignore_index=True)
                
                # **형식 통일 작업 시작**
                # 1. '추첨일'이 비어있는 행은 '날짜' 컬럼 값으로 채우기
                if '날짜' in updated_df.columns:
                    updated_df['추첨일'] = updated_df['추첨일'].fillna(updated_df['날짜'])
                
                # 2. 번호 컬럼(1~6)이 비어있는 경우 기존 '번호1'~'번호6' 컬럼 값을 채우기
                old_number_cols = ['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']
                new_number_cols = ['1', '2', '3', '4', '5', '6']
                for new_col, old_col in zip(new_number_cols, old_number_cols):
                    if old_col in updated_df.columns:
                        updated_df[new_col] = updated_df[new_col].fillna(updated_df[old_col])
                
                # 3. 최종적으로 필요한 컬럼만 선택: 회차, 추첨일, 1,2,3,4,5,6,보너스
                final_columns = ['회차', '추첨일'] + new_number_cols + ['보너스']
                updated_df = updated_df[final_columns]
                
                # 4. 번호 컬럼들을 숫자형으로 변환 (필요 시 소수점 제거)
                for col in new_number_cols + ['보너스']:
                    updated_df[col] = pd.to_numeric(updated_df[col], errors='coerce')
                # **형식 통일 작업 끝**
                
                os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
                updated_df.to_csv(self.data_file, index=False)
                logger.info(f"신규 회차 {draw_no} 추가 및 형식 재정렬 완료")
                return True
            
            logger.info(f"회차 {draw_no}는 이미 존재함")
            return False

        except Exception as e:
            logger.error(f"데이터 업데이트 중 오류 발생: {str(e)}")
            return False



class LottoPredictor:
    def __init__(self):
        self.model = LogisticRegression(multi_class='ovr', max_iter=1000)
        self.scaler = StandardScaler()
        self.model_file = os.path.join(settings.BASE_DIR, 'data', 'lotto_model.pkl')
        self.scaler_file = os.path.join(settings.BASE_DIR, 'data', 'lotto_scaler.pkl')
        self.stats_file = os.path.join(settings.BASE_DIR, 'data', 'model_stats.json')
        self.recent_data = None
        self.recent_features = None

    def prepare_features(self, df):
        """특성 데이터 준비"""
        try:
            # 데이터프레임을 회차 기준으로 정렬
            df = df.sort_values('회차', ascending=False).reset_index(drop=True)
            features = []
            
            for i in range(len(df) - 5):  # 최근 5회차 데이터 사용
                recent_numbers = []
                for j in range(5):
                    row = df.iloc[i + j]
                    numbers = [row[str(k)] for k in range(1, 7)]  # 1~6번 번호
                    numbers.append(row['보너스'])  # 보너스 번호 추가
                    recent_numbers.extend(numbers)
                features.append(recent_numbers)
            
            return np.array(features)
        except Exception as e:
            logger.error(f"특성 데이터 준비 중 오류: {str(e)}")
            raise  # 오류를 상위로 전파하여 train_model에서 처리

    def save_model(self):
        """학습된 모델 저장"""
        try:
            os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
            dump(self.model, self.model_file)
            dump(self.scaler, self.scaler_file)
            
            stats = {
                'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_size': len(self.recent_data) if self.recent_data is not None else 0,
                'feature_size': self.recent_features.shape[1] if self.recent_features is not None else 0
            }
            
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
                
            logger.info("모델 저장 완료")
            return True
        except Exception as e:
            logger.error(f"모델 저장 중 오류: {str(e)}")
            return False

    def load_model(self):
        """저장된 모델 로드"""
        try:
            if os.path.exists(self.model_file) and os.path.exists(self.scaler_file):
                self.model = load(self.model_file)
                self.scaler = load(self.scaler_file)
                logger.info("저장된 모델 로드 완료")
                return True
            return False
        except Exception as e:
            logger.error(f"모델 로드 중 오류: {str(e)}")
            return False

    def train_model(self):
        """모델 학습 및 R2 점수 계산"""
        try:
            if not os.path.exists(settings.LOTTO_DATA_FILE):
                logger.error("데이터 파일이 존재하지 않습니다")
                return False

            df = pd.read_csv(settings.LOTTO_DATA_FILE)
            if len(df) < 6:
                logger.error("학습에 필요한 최소 데이터가 부족합니다")
                return False

            self.recent_data = df
            logger.info(f"데이터 로드 완료: {len(df)}개의 데이터")

            X = self.prepare_features(df)
            if len(X) == 0:
                logger.error("특성 데이터 준비 실패")
                return False

            self.recent_features = X
            logger.info(f"특성 데이터 준비 완료: {X.shape}")

            y = []
            for i in range(len(df) - 5):
                next_number = df.iloc[i]['1']
                y.append(next_number)
            y = np.array(y)

            # 데이터를 학습용과 테스트용으로 분할 (예: 80% 학습, 20% 테스트)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 데이터 스케일링
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # 모델 학습
            self.model.fit(X_train_scaled, y_train)

            # R2 점수 계산
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            logger.info(f"학습 데이터 R2 점수: {train_score:.4f}")
            logger.info(f"테스트 데이터 R2 점수: {test_score:.4f}")

            self.save_model()
            logger.info("모델 학습 및 저장 완료")
            return True

        except Exception as e:
            logger.error(f"모델 학습 중 오류 발생: {str(e)}")
            return False

    def predict_probabilities(self):
        """각 번호의 출현 확률 예측"""
        try:
            if not self.load_model():
                if not self.train_model():
                    return np.ones(45) / 45  # 45개의 번호에 대한 균등 확률 반환

            df = pd.read_csv(settings.LOTTO_DATA_FILE)
            latest_features = self.prepare_features(df)[-1:]
            latest_features = self.scaler.transform(latest_features)
            
            # 각 클래스(번호)에 대한 확률 계산
            probabilities = np.zeros(45)  # 1~45까지의 번호에 대한 확률 배열
            probs = self.model.predict_proba(latest_features)
            
            # 모델이 예측한 클래스들에 대해서만 확률 할당
            for i, class_idx in enumerate(self.model.classes_):
                if 1 <= class_idx <= 45:  # 유효한 로또 번호 범위 확인
                    probabilities[class_idx-1] = probs[0][i]
            
            # 확률 정규화
            probabilities = probabilities / np.sum(probabilities)
            
            return probabilities
            
        except Exception as e:
            logger.error(f"예측 중 오류 발생: {str(e)}")
            return np.ones(45) / 45  # 오류 발생 시 균등 확률 반환

def get_recommendation(strategy_counts):
    """전략별 로또 번호 추천"""
    try:
        if not os.path.exists(settings.LOTTO_DATA_FILE):
            logger.info("데이터 파일이 없습니다. 초기 데이터를 수집합니다.")
            collector = LottoDataCollector()
            collector.collect_initial_data()

        df = pd.read_csv(settings.LOTTO_DATA_FILE)
        predictor = LottoPredictor()
        predictor.train_model()
        
        # 번호별 출현 빈도 분석
        all_numbers = []
        for col in ['1', '2', '3', '4', '5', '6']:
            all_numbers.extend(df[col].tolist())
        number_counts = pd.Series(all_numbers).value_counts()
        
        # 평균과 표준편차 계산
        mean_freq = number_counts.mean()
        std_freq = number_counts.std() 
        
        # 머신러닝 예측 확률 가져오기
        predicted_probs = predictor.predict_probabilities()
        
        recommendations = []
        previous_selections = set()  # 이전 추천 번호들 저장

        for strategy, count in strategy_counts.items():
            strategy = int(strategy)
            for _ in range(count):
                candidates = []
                if strategy == 1:
                    # 전략 1: 자주 출현하는 번호들 (상위 60% 범위로 확대)
                    threshold = np.percentile(list(number_counts.values), 40)  # 상위 60%
                    candidates = [n for n in range(1, 46) if number_counts.get(n, 0) >= threshold]
                else:
                    # 전략 2: 평균 주변 구간의 번호들
                    candidates = [n for n in range(1, 46) if (mean_freq + 0.5 * std_freq) > number_counts.get(n, 0) >= (mean_freq - std_freq)]
                
                if not candidates:  # 후보 번호가 부족한 경우
                    candidates = list(range(1, 46))  # 모든 번호를 후보로 사용
                
                # 가중치 계산
                if strategy == 1:
                    freq_weights = [(number_counts.get(n, 0) - threshold + std_freq) / (number_counts.max() - threshold + std_freq) for n in candidates]
                else:
                    freq_weights = [(mean_freq - number_counts.get(n, 0) + std_freq) / (mean_freq + std_freq) for n in candidates]
                
                ml_weights = [predicted_probs[n-1] for n in candidates]
                
                # 이전 선택된 번호에 대한 페널티 적용
                penalty = [0.7 if n in previous_selections else 1.0 for n in candidates]
                
                # 최종 가중치 계산 (빈도:ML예측:랜덤성 = 4:4:2 비율)
                weights = [0.4 * fw + 0.4 * mw + 0.2 * np.random.random() for fw, mw in zip(freq_weights, ml_weights)]
                weights = np.multiply(weights, penalty)  # 페널티 적용
                weights = np.array(weights) / np.sum(weights)  # 정규화

                # 구간별 번호 선택
                selected = []
                ranges = [(1,15), (16,30), (31,45)]
                numbers_per_range = [2, 2, 2]
                
                # 구간별 선택 수 랜덤 조정
                while sum(numbers_per_range) > 6:
                    idx = np.random.randint(0, 3)
                    if numbers_per_range[idx] > 1:
                        numbers_per_range[idx] -= 1

                for (start, end), n_select in zip(ranges, numbers_per_range):
                    range_candidates = [n for n in candidates if start <= n <= end]
                    if range_candidates:
                        range_weights = [weights[candidates.index(n)] for n in range_candidates]
                        range_weights = np.array(range_weights) / np.sum(range_weights)
                        try:
                            selected.extend(np.random.choice(range_candidates, 
                                                            min(n_select, len(range_candidates)), 
                                                            replace=False, 
                                                            p=range_weights))
                        except ValueError:  # 구간에 충분한 번호가 없는 경우
                            selected.extend(np.random.choice(range_candidates, 
                                                            min(n_select, len(range_candidates)), 
                                                            replace=True))

                # 남은 자리 채우기
                remaining = 6 - len(selected)
                if remaining > 0:
                    remaining_candidates = [n for n in candidates if n not in selected]
                    if not remaining_candidates:  # 남은 후보가 없는 경우
                        remaining_candidates = [n for n in range(1, 46) if n not in selected]
                    
                    if remaining_candidates:
                        remaining_weights = [weights[candidates.index(n)] if n in candidates else 1.0/len(candidates) 
                                               for n in remaining_candidates]
                        remaining_weights = np.array(remaining_weights) / np.sum(remaining_weights)
                        selected.extend(np.random.choice(remaining_candidates, 
                                                        remaining, 
                                                        replace=False, 
                                                        p=remaining_weights))
                
                # 선택된 번호들을 이전 선택 목록에 추가
                previous_selections.update(selected)
                recommendations.append((strategy, sorted(selected)))
        
        return recommendations, None
        
    except Exception as e:
        logger.error(f"Error in get_recommendation: {str(e)}")
        return None, str(e)
    
def check_data_status():
    """데이터 상태 확인"""
    try:
        if not os.path.exists(settings.LOTTO_DATA_FILE):
            logger.warning("Lotto data file not found")
            return False, "데이터 파일이 없습니다. 초기 데이터를 수집해야 합니다."

        df = pd.read_csv(settings.LOTTO_DATA_FILE)
        if len(df) == 0:
            logger.warning("Empty data file")
            return False, "데이터 파일이 비어있습니다."

        # 최신 데이터 확인
        latest_date = pd.to_datetime(df['추첨일'].iloc[0])
        current_date = pd.Timestamp.now()
        days_diff = (current_date - latest_date).days

        if days_diff > 7:
            logger.warning(f"Data might be outdated. Last update: {latest_date}")
            return True, f"마지막 업데이트: {latest_date.strftime('%Y-%m-%d')}\n{days_diff}일 전에 업데이트되었습니다."
        
        return True, f"데이터가 최신 상태입니다.\n마지막 업데이트: {latest_date.strftime('%Y-%m-%d')}"

    except Exception as e:
        logger.error(f"Error checking data status: {str(e)}")
        return False, f"데이터 상태 확인 중 오류 발생: {str(e)}"