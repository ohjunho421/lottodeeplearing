import os
import logging
import numpy as np
import pandas as pd
import requests
import json
from bs4 import BeautifulSoup
from django.conf import settings
import tensorflow as tf
from tensorflow import keras
import keras
from keras import layers, models, optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
from joblib import dump, load
import xgboost as xgb
from statsmodels.tsa.seasonal import seasonal_decompose
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

logger = logging.getLogger(__name__)

class LottoDataCollector:
    def __init__(self):
        self.base_url = "https://www.dhlottery.co.kr/gameResult.do?method=byWin"
        self.data_file = settings.LOTTO_DATA_FILE

    def collect_initial_data(self):
        """초기 데이터 수집 (파일이 있으면 최신 데이터만 읽어옴)"""
        if os.path.exists(self.data_file):
            df = pd.read_csv(self.data_file)
            logger.info("기존 데이터 파일 발견. 최신 데이터만 읽어옵니다.")
            return df.iloc[[0]]

        try:
            logger.info("초기 데이터 수집 시작")
            response = requests.get(self.base_url)
            if response.status_code != 200:
                logger.error(f"HTTP 오류: {response.status_code}")
                return None

            soup = BeautifulSoup(response.text, 'html.parser')
            
            win_numbers = soup.select('div.num.win span.ball_645')
            bonus_ball = soup.select('div.num.bonus span.ball_645')
            draw_result = soup.select('div.win_result h4')
            draw_date = soup.select('p.desc')

            if not all([win_numbers, bonus_ball, draw_result, draw_date]):
                logger.error("필요한 데이터를 찾을 수 없습니다")
                return None

            try:
                numbers = [int(n.text.strip()) for n in win_numbers]
                bonus = int(bonus_ball[0].text.strip())
                draw_no = int(''.join(filter(str.isdigit, draw_result[0].text)))
                date_text = draw_date[0].text.strip()
                drawn_date = date_text[date_text.find('(')+1:date_text.find(')')]

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

                os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
                df.to_csv(self.data_file, index=False)
                return df

            except Exception as e:
                logger.error(f"데이터 파싱 오류: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"초기 데이터 수집 중 오류 발생: {str(e)}")
            return None

    def _parse_date(self, date_text):
        """크롤링한 날짜를 YYYY.MM.DD 형식으로 변환"""
        try:
            date_parts = ''.join(filter(str.isdigit, date_text))
            year = date_parts[:4]
            month = date_parts[4:6]
            day = date_parts[6:8]
            return f"{year}.{month}.{day}"
        except Exception as e:
            logger.error(f"날짜 파싱 오류: {str(e)}")
            return date_text

    def update_latest_data(self):
        """최신 데이터 업데이트"""
        try:
            logger.info("최신 데이터 업데이트 시작")
            response = requests.get(self.base_url)
            if response.status_code != 200:
                return False

            soup = BeautifulSoup(response.text, 'html.parser')
            
            win_numbers = soup.select('div.num.win span.ball_645')
            bonus_ball = soup.select('div.num.bonus span.ball_645')
            draw_result = soup.select('div.win_result h4')
            draw_date = soup.select('p.desc')

            if not all([win_numbers, bonus_ball, draw_result, draw_date]):
                return False

            try:
                numbers = [int(n.text.strip()) for n in win_numbers]
                bonus = int(bonus_ball[0].text.strip())
                draw_no = int(''.join(filter(str.isdigit, draw_result[0].text)))
                date_text = draw_date[0].text.strip()
                drawn_date = date_text[date_text.find('(')+1:date_text.find(')')]
                drawn_date_formatted = self._parse_date(drawn_date)

                df = pd.read_csv(self.data_file)
                if draw_no in df['회차'].values:
                    return False

                new_row = pd.DataFrame([{
                    '회차': draw_no,
                    '추첨일': drawn_date_formatted,
                    '1': numbers[0],
                    '2': numbers[1],
                    '3': numbers[2],
                    '4': numbers[3],
                    '5': numbers[4],
                    '6': numbers[5],
                    '보너스': bonus
                }])
                
                updated_df = pd.concat([df, new_row], ignore_index=True)
                updated_df = updated_df.sort_values('회차', ascending=False).reset_index(drop=True)
                updated_df.to_csv(self.data_file, index=False)
                return True

            except Exception as e:
                logger.error(f"데이터 파싱 오류: {str(e)}")
                return False

        except Exception as e:
            logger.error(f"데이터 업데이트 중 오류 발생: {str(e)}")
            return False

class AdvancedLottoPredictor:
    def __init__(self):
        self.xgb_model = None
        self.lstm_model = None
        self.rl_model = None
        self.scaler = StandardScaler()
        self.model_dir = os.path.join(settings.BASE_DIR, 'data', 'models')
        self.xgb_file = os.path.join(self.model_dir, 'xgb_model.json')
        self.lstm_file = os.path.join(self.model_dir, 'lstm_model.h5')
        self.scaler_file = os.path.join(self.model_dir, 'scaler.pkl')
        self.rl_file = os.path.join(self.model_dir, 'rl_model.zip')
        self.stats_file = os.path.join(self.model_dir, 'model_stats.json')
        self.recent_data = None
        self.temporal_patterns = None

    def prepare_features(self, df):
        """향상된 특성 데이터 준비"""
        try:
            # 날짜 컬럼을 datetime으로 변환
            df['추첨일'] = pd.to_datetime(df['추첨일'])
            
            # 시간적 특성 추출
            df['month'] = df['추첨일'].dt.month
            df['day_of_week'] = df['추첨일'].dt.dayofweek
            df['week_of_year'] = df['추첨일'].dt.isocalendar().week
            
            features = []
            
            # 최근 5회차 데이터를 사용하여 특성 생성
            for i in range(len(df) - 5):  # 마지막 5회차는 제외
                recent_numbers = []
                for j in range(5):
                    row = df.iloc[i + j]
                    numbers = [row[str(k)] for k in range(1, 7)]  # 1~6번 번호
                    numbers.append(row['보너스'])  # 보너스 번호 추가
                    recent_numbers.extend(numbers)
                
                # 시간적 특성 추가
                current_row = df.iloc[i]
                time_features = [
                    current_row['month'],
                    current_row['day_of_week'],
                    current_row['week_of_year']
                ]
                recent_numbers.extend(time_features)
                features.append(recent_numbers)

            features = np.array(features)
            logger.info(f"Generated features shape: {features.shape}")
            return features

        except Exception as e:
            logger.error(f"특성 데이터 준비 중 오류: {str(e)}")
            raise

    def analyze_temporal_patterns(self, df):
        """시계열 패턴 분석"""
        patterns = {}
        try:
            df['추첨일'] = pd.to_datetime(df['추첨일'])
            
            for num in range(1, 46):
                number_series = pd.Series(
                    index=pd.DatetimeIndex(df['추첨일']),
                    data=[1 if num in row[['1','2','3','4','5','6']].values else 0 
                          for _, row in df.iterrows()]
                )
                
                monthly_pattern = number_series.resample('ME').sum()
                
                if len(monthly_pattern) >= 24:
                    try:
                        decomposition = seasonal_decompose(monthly_pattern, period=12)
                        patterns[num] = {
                            'trend': decomposition.trend,
                            'seasonal': decomposition.seasonal,
                            'monthly_freq': monthly_pattern
                        }
                    except Exception as e:
                        patterns[num] = {
                            'trend': pd.Series([1.0] * len(monthly_pattern), index=monthly_pattern.index),
                            'seasonal': pd.Series([0.0] * len(monthly_pattern), index=monthly_pattern.index),
                            'monthly_freq': monthly_pattern
                        }
                else:
                    patterns[num] = {
                        'trend': pd.Series([1.0] * len(monthly_pattern), index=monthly_pattern.index),
                        'seasonal': pd.Series([0.0] * len(monthly_pattern), index=monthly_pattern.index),
                        'monthly_freq': monthly_pattern
                    }
            
            self.temporal_patterns = patterns
            return patterns
            
        except Exception as e:
            logger.error(f"시계열 패턴 분석 중 오류: {str(e)}")
            return {}

    def build_lstm_model(self, input_shape):
        """LSTM 모델 구축"""
        model = keras.Sequential([
            layers.LSTM(128, input_shape=input_shape, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(64),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(6, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model

    def save_models(self):
        """모든 모델 저장"""
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            
            # XGBoost 모델 저장
            if self.xgb_model:
                self.xgb_model.save_model(self.xgb_file)
            
            # RandomForest 모델 저장
            if hasattr(self, 'rf_model') and self.rf_model:
                rf_file = os.path.join(self.model_dir, 'rf_model.pkl')
                dump(self.rf_model, rf_file)
            
            # 스케일러 저장
            dump(self.scaler, self.scaler_file)
            
            # 모델 통계 저장
            stats = {
                'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_size': len(self.recent_data) if self.recent_data is not None else 0
            }
            
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            logger.info("모든 모델 저장 완료")
            return True
            
        except Exception as e:
            logger.error(f"모델 저장 중 오류: {str(e)}")
            return False

    def load_models(self):
        """모든 모델 로드"""
        try:
            if not all(os.path.exists(f) for f in [self.xgb_file, self.scaler_file]):
                return False
                
            # XGBoost 모델 로드
            self.xgb_model = xgb.XGBRegressor()
            self.xgb_model.load_model(self.xgb_file)
            
            # RandomForest 모델 로드
            rf_file = os.path.join(self.model_dir, 'rf_model.pkl')
            if os.path.exists(rf_file):
                from sklearn.ensemble import RandomForestRegressor
                self.rf_model = load(rf_file)
                logger.info("RandomForest 모델 로드 완료")
            
            # 스케일러 로드
            self.scaler = load(self.scaler_file)
            
            logger.info("모든 모델 로드 완료")
            return True
            
        except Exception as e:
            logger.error(f"모델 로드 중 오류: {str(e)}")
            return False
        
    def load_models(self):
        """모든 모델 로드"""
        try:
            if all(os.path.exists(f) for f in [self.xgb_file, self.lstm_file, self.scaler_file]):
                self.xgb_model = xgb.XGBRegressor()
                self.xgb_model.load_model(self.xgb_file)
                self.lstm_model = models.load_model(self.lstm_file)
                self.scaler = load(self.scaler_file)
                
                if os.path.exists(self.rl_file):
                    self.rl_model = PPO.load(self.rl_file)
                
                return True
            return False
            
        except Exception as e:
            logger.error(f"모델 로드 중 오류: {str(e)}")
            return False

    def train_models(self):
        """모든 모델 학습"""
        try:
            if not os.path.exists(settings.LOTTO_DATA_FILE):
                logger.error("데이터 파일이 존재하지 않습니다")
                return False, None

            df = pd.read_csv(settings.LOTTO_DATA_FILE)
            df = df.sort_values('회차', ascending=False).reset_index(drop=True)
            
            if len(df) < 6:
                logger.error("학습에 필요한 최소 데이터가 부족합니다")
                return False, None

            self.recent_data = df
            
            # 시간적 특성 추가
            df['추첨일'] = pd.to_datetime(df['추첨일'])
            df['year'] = df['추첨일'].dt.year
            df['month'] = df['추첨일'].dt.month
            df['day_of_week'] = df['추첨일'].dt.dayofweek
            df['week_of_year'] = df['추첨일'].dt.isocalendar().week
            
            # 최근 3년 데이터만 선택
            cutoff_date = df['추첨일'].max() - pd.DateOffset(years=3)
            recent_df = df[df['추첨일'] >= cutoff_date].copy()
            
            # X 데이터 준비
            X = self.prepare_features(recent_df)
            logger.info(f"X shape after prepare_features: {X.shape}")
            
            # y 데이터 준비 - 명시적으로 float 타입으로 변환
            y = []
            for i in range(len(X)):
                if i < len(recent_df):
                    next_numbers = recent_df.iloc[i][['1','2','3','4','5','6']].values.astype(float)
                    y.append(next_numbers)
            y = np.array(y, dtype=float)
            
            logger.info(f"X shape: {X.shape}, y shape: {y.shape}, X dtype: {X.dtype}, y dtype: {y.dtype}")
            
            if len(X) != len(y):
                logger.error(f"X ({len(X)}) and y ({len(y)}) have different lengths!")
                return False, None

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 데이터 타입 확인
            logger.info(f"X_train dtype: {X_train.dtype}, y_train dtype: {y_train.dtype}")
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # 타입 확인
            logger.info(f"X_train_scaled dtype: {X_train_scaled.dtype}")

            # XGBoost 모델 학습 (과적합 방지 파라미터 강화)
            self.xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                max_depth=1,               # 더 낮게 조정
                learning_rate=0.01,        # 더 낮게 조정
                n_estimators=200,
                subsample=0.6,             # 더 낮게 조정
                colsample_bytree=0.6,      # 더 낮게 조정
                reg_alpha=0.1,             # 더 높게 조정
                reg_lambda=2.0,            # 더 높게 조정
                min_child_weight=3,        # 추가
                gamma=0.2,                 # 추가 (노드 분할에 필요한 최소 손실 감소)
                random_state=42
            )
            self.xgb_model.fit(X_train_scaled, y_train)

            # XGBoost R2 점수 계산
            xgb_train_score = self.xgb_model.score(X_train_scaled, y_train)
            xgb_test_score = self.xgb_model.score(X_test_scaled, y_test)
            logger.info(f"XGBoost Train R2 Score: {xgb_train_score:.4f}")
            logger.info(f"XGBoost Test R2 Score: {xgb_test_score:.4f}")

            # RandomForest 모델 학습 (LSTM 대체)
            from sklearn.ensemble import RandomForestRegressor
            self.rf_model = RandomForestRegressor(
                n_estimators=100,   # 트리 개수
                max_depth=10,       # 최대 깊이
                min_samples_split=5,  # 분할에 필요한 최소 샘플 수
                min_samples_leaf=2,   # 리프 노드에 필요한 최소 샘플 수
                max_features='sqrt',  # 특성 선택 방법
                n_jobs=-1,            # 모든 CPU 사용
                random_state=42
            )
            self.rf_model.fit(X_train_scaled, y_train)

            # RandomForest R2 점수 계산
            rf_train_score = self.rf_model.score(X_train_scaled, y_train)
            rf_test_score = self.rf_model.score(X_test_scaled, y_test)
            logger.info(f"RandomForest Train R2 Score: {rf_train_score:.4f}")
            logger.info(f"RandomForest Test R2 Score: {rf_test_score:.4f}")

            # 특성 중요도 확인
            feature_importances = self.rf_model.feature_importances_
            logger.info(f"RandomForest Feature Importances: {feature_importances}")

            # 시계열 패턴 분석
            self.analyze_temporal_patterns(recent_df)

            # 결과 저장
            train_results = {
                'xgb_train_r2': xgb_train_score,
                'xgb_test_r2': xgb_test_score,
                'rf_train_r2': rf_train_score,
                'rf_test_r2': rf_test_score,
            }

            # 모델 저장용 메서드도 수정 필요
            self.save_models()
            
            # 결과를 JSON 파일로 저장
            results_file = os.path.join(self.model_dir, 'training_results.json')
            with open(results_file, 'w') as f:
                json.dump({
                    'xgb_train_r2': float(xgb_train_score),
                    'xgb_test_r2': float(xgb_test_score),
                    'rf_train_r2': float(rf_train_score),
                    'rf_test_r2': float(rf_test_score)
                }, f, indent=4)
            
            logger.info("모든 모델 학습 완료")
            return True, train_results

        except Exception as e:
            logger.error(f"모델 학습 중 오류 발생: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False, None

    def predict_numbers(self):
        """번호 예측"""
        try:
            if not self.load_models():
                if not self.train_models():
                    return np.ones(45) / 45  # 균등 확률 반환

            df = pd.read_csv(settings.LOTTO_DATA_FILE)
            df = df.sort_values('회차', ascending=False).reset_index(drop=True)
            
            # 시간적 특성 추가
            df['추첨일'] = pd.to_datetime(df['추첨일'])
            df['year'] = df['추첨일'].dt.year
            df['month'] = df['추첨일'].dt.month
            df['day_of_week'] = df['추첨일'].dt.dayofweek
            df['week_of_year'] = df['추첨일'].dt.isocalendar().week
            
            # 최근 3년 데이터만 선택
            cutoff_date = df['추첨일'].max() - pd.DateOffset(years=3)
            recent_df = df[df['추첨일'] >= cutoff_date].copy()
            
            # 특성 생성
            latest_features = self.prepare_features(recent_df)[:1]
            latest_features_scaled = self.scaler.transform(latest_features)

            # XGBoost 예측
            xgb_pred = self.xgb_model.predict(latest_features_scaled)

            # RandomForest 예측
            rf_pred = self.rf_model.predict(latest_features_scaled)

            # 시계열 패턴 가중치 계산
            if self.temporal_patterns is None:
                self.analyze_temporal_patterns(recent_df)

            weights = np.ones(45)
            for num in range(1, 46):
                if num in self.temporal_patterns:
                    pattern = self.temporal_patterns[num]
                    recent_trend = pattern['trend'].iloc[-1] if not pd.isna(pattern['trend'].iloc[-1]) else 1
                    weights[num-1] *= (1 + recent_trend/10)

            # 모델 예측 결합
            combined_pred = 0.3 * xgb_pred + 0.5 * rf_pred.flatten() + 0.2 * weights
            
            # 확률 정규화
            probabilities = combined_pred / np.sum(combined_pred)

            return probabilities

        except Exception as e:
            logger.error(f"예측 중 오류 발생: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return np.ones(45) / 45

class LottoEnvironment(gym.Env):
    """강화학습을 위한 로또 환경"""
    def __init__(self, historical_data):
        super().__init__()
        self.historical_data = historical_data
        self.action_space = spaces.Box(low=1, high=45, shape=(6,), dtype=np.int32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(45,), dtype=np.float32)
        self.current_step = 0
        
    def reset(self):
        self.current_step = 0
        return self._get_observation()
        
    def step(self, action):
        reward = self._calculate_reward(action)
        self.current_step += 1
        done = self.current_step >= len(self.historical_data) - 1
        return self._get_observation(), reward, done, {}
        
    def _get_observation(self):
        if self.current_step >= len(self.historical_data):
            return np.zeros(45)
            
        recent_numbers = self.historical_data.iloc[self.current_step][['1','2','3','4','5','6']].values
        obs = np.zeros(45)
        for num in recent_numbers:
            obs[int(num)-1] += 1
        return obs / np.sum(obs) if np.sum(obs) > 0 else obs
        
    def _calculate_reward(self, action):
        if self.current_step >= len(self.historical_data) - 1:
            return 0
            
        next_numbers = set(self.historical_data.iloc[self.current_step + 1][['1','2','3','4','5','6']].values)
        selected_numbers = set(action)
        matches = len(next_numbers.intersection(selected_numbers))
        
        rewards = {6: 1000, 5: 100, 4: 10, 3: 1, 2: 0.1, 1: 0.01, 0: -0.1}
        return rewards.get(matches, 0)

def get_recommendation(strategy_counts):
    """전략별 로또 번호 추천"""
    try:
        if not os.path.exists(settings.LOTTO_DATA_FILE):
            logger.info("데이터 파일이 없습니다. 초기 데이터를 수집합니다.")
            collector = LottoDataCollector()
            collector.collect_initial_data()

        df = pd.read_csv(settings.LOTTO_DATA_FILE)
        predictor = AdvancedLottoPredictor()
        
        # 고급 예측 확률 계산
        predicted_probs = predictor.predict_numbers()
        
        # 번호별 출현 빈도 분석
        all_numbers = []
        for col in ['1', '2', '3', '4', '5', '6']:
            all_numbers.extend(df[col].tolist())
        number_counts = pd.Series(all_numbers).value_counts()
        
        # 시계열 패턴 분석
        temporal_patterns = predictor.analyze_temporal_patterns(df)
        
        recommendations = []
        previous_selections = set()

        for strategy, count in strategy_counts.items():
            strategy = int(strategy)
            for _ in range(count):
                if strategy == 1:
                    weights = np.array([
                        number_counts.get(n, 0) / number_counts.max() for n in range(1, 46)
                    ])
                else:
                    weights = np.zeros(45)
                    for num in range(1, 46):
                        if num in temporal_patterns:
                            pattern = temporal_patterns[num]
                            trend = pattern['trend'].iloc[-1] if not pd.isna(pattern['trend'].iloc[-1]) else 1
                            seasonal = pattern['seasonal'].iloc[-1] if not pd.isna(pattern['seasonal'].iloc[-1]) else 0
                            weights[num-1] = (1 + trend/10) * (1 + seasonal/5)
                
                # ML 예측 확률과 결합
                final_weights = 0.3 * weights + 0.4 * predicted_probs
                
                # 이전 선택에 대한 페널티
                penalty = np.array([0.7 if i+1 in previous_selections else 1.0 for i in range(45)])
                final_weights *= penalty
                
                # 랜덤성 추가
                random_weights = np.random.random(45)
                final_weights = 0.7 * final_weights + 0.3 * random_weights
                
                # 정규화
                final_weights = final_weights / np.sum(final_weights)
                
                try:
                    selected = np.random.choice(
                        range(1, 46),
                        size=6,
                        replace=False,
                        p=final_weights
                    )
                    previous_selections.update(selected)
                    recommendations.append((strategy, sorted(selected)))
                except ValueError as e:
                    logger.error(f"번호 선택 중 오류: {str(e)}")
                    selected = np.random.choice(range(1, 46), size=6, replace=False)
                    recommendations.append((strategy, sorted(selected)))
        
        return recommendations, None
        
    except Exception as e:
        logger.error(f"번호 추천 중 오류 발생: {str(e)}")
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

        latest_date = pd.to_datetime(df['추첨일'].iloc[0])
        current_date = pd.Timestamp.now()
        days_diff = (current_date - latest_date).days

        if days_diff > 7:
            logger.warning(f"Data might be outdated. Last update: {latest_date}")
            return True, f"마지막 업데이트: {latest_date.strftime('%Y-%m-%d')}\n{days_diff}일 전에 업데이트되었습니다."
        
        return True, f"데이터가 최신 상태입니다.\n마지막 업데이트: {latest_date.strftime('%Y-%m-%d')}"

    except Exception as e:
        logger.error(f"데이터 상태 확인 중 오류 발생: {str(e)}")
        return False, f"데이터 상태 확인 중 오류 발생: {str(e)}"