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

# LLM API 관련 설정
LLM_API_KEY = os.environ.get("OPENAI_API_KEY")  # OpenAI API 키 가져오기
LLM_API_URL = "https://api.openai.com/v1/chat/completions"  # OpenAI API URL

class LLMCache:
    def __init__(self, cache_file=None, max_age_hours=24):
        self.cache_file = cache_file or os.path.join(settings.BASE_DIR, 'data', 'llm_cache.json')
        self.max_age_hours = max_age_hours
        self.cache = self._load_cache()
        
    def _load_cache(self):
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                return cache
            return {}
        except Exception as e:
            logger.error(f"캐시 로드 오류: {str(e)}")
            return {}
            
    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.error(f"캐시 저장 오류: {str(e)}")
            
    def get(self, key):
        if key in self.cache:
            entry = self.cache[key]
            timestamp = entry['timestamp']
            current_time = datetime.now().timestamp()
            
            # 캐시 만료 확인
            if (current_time - timestamp) / 3600 <= self.max_age_hours:
                return entry['data']
                
        return None
        
    def set(self, key, data):
        self.cache[key] = {
            'timestamp': datetime.now().timestamp(),
            'data': data
        }
        self._save_cache()

# 전역 LLM 캐시 인스턴스 생성
llm_cache = LLMCache()

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

            # 모델 예측을 가중치로 변환
            # XGBoost 예측을 가중치로 변환
            xgb_weights = np.ones(45) * 0.01  # 기본 낮은 가중치 설정
            if len(xgb_pred.shape) > 1 and xgb_pred.shape[1] == 6:
                # 특정 번호를 예측하는 경우
                for i in range(xgb_pred.shape[1]):
                    predicted_num = int(round(xgb_pred[0, i]))
                    if 1 <= predicted_num <= 45:
                        xgb_weights[predicted_num-1] = 1.0
            else:
                # 이미 확률 형태라면 그대로 사용
                xgb_weights = xgb_pred

            # RandomForest 예측을 가중치로 변환
            rf_weights = np.ones(45) * 0.01  # 기본 낮은 가중치 설정
            if len(rf_pred.shape) == 1 and rf_pred.shape[0] == 6:
                # 특정 번호를 예측하는 경우
                for i in range(rf_pred.shape[0]):
                    predicted_num = int(round(rf_pred[i]))
                    if 1 <= predicted_num <= 45:
                        rf_weights[predicted_num-1] = 1.0
            elif len(rf_pred.shape) > 1 and rf_pred.shape[1] == 6:
                # 2차원 배열인 경우
                for i in range(rf_pred.shape[1]):
                    predicted_num = int(round(rf_pred[0, i]))
                    if 1 <= predicted_num <= 45:
                        rf_weights[predicted_num-1] = 1.0
            else:
                # 이미 확률 형태라면 그대로 사용
                rf_weights = rf_pred.flatten() if rf_pred.size > 6 else np.ones(45) / 45

            # 정규화
            xgb_weights = xgb_weights / np.sum(xgb_weights)
            rf_weights = rf_weights / np.sum(rf_weights)
            weights = weights / np.sum(weights)
            
            # 가중치 결합
            combined_pred = 0.3 * xgb_weights + 0.5 * rf_weights + 0.2 * weights
            
            # 확률 정규화
            probabilities = combined_pred / np.sum(combined_pred)

            return probabilities

        except Exception as e:
            logger.error(f"예측 중 오류 발생: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return np.ones(45) / 45

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

def get_llm_weights(df, predicted_probs, frequency_data, temporal_patterns, previous_selections, strategy):
    """LLM을 사용하여 로또 번호에 대한 가중치 생성"""
    try:
        if not LLM_API_KEY:
            logger.warning("OpenAI API 키가 설정되지 않았습니다. 랜덤 가중치를 반환합니다.")
            return np.random.random(45)
        
        # 캐시 키 생성 (최근 회차 번호, 이전 선택, 전략으로 구성)
        latest_draw = df.iloc[0]['회차'] if not df.empty else 0
        cache_key = f"{latest_draw}-{sorted(list(previous_selections))}-{strategy}"
        
        # 캐시 확인
        cached_weights = llm_cache.get(cache_key)
        if cached_weights is not None:
            logger.info("캐시된 LLM 가중치 사용")
            return np.array(cached_weights)
        
        # 최근 10회 당첨 번호 추출
        recent_draws = df.head(10)[['회차', '1', '2', '3', '4', '5', '6', '보너스']].to_dict('records')
        
        # 빈도 데이터 정리
        freq_list = []
        for num, freq in frequency_data.items():
            try:
                num_int = int(num)
                freq_int = int(freq)
                freq_list.append({"number": num_int, "frequency": freq_int})
            except (ValueError, TypeError):
                continue
        
        # 빈도 통계 계산
        freqs = np.array([frequency_data.get(n, 0) for n in range(1, 46)])
        mean_freq = np.mean(freqs)
        std_freq = np.std(freqs)
        
        # 시계열 패턴 데이터 정리
        patterns_simplified = {}
        for num in range(1, 46):
            if num in temporal_patterns:
                pattern = temporal_patterns[num]
                try:
                    trend_value = float(pattern['trend'].iloc[-1]) if not pd.isna(pattern['trend'].iloc[-1]) else 1.0
                    seasonal_value = float(pattern['seasonal'].iloc[-1]) if not pd.isna(pattern['seasonal'].iloc[-1]) else 0.0
                    patterns_simplified[str(num)] = {
                        "trend": trend_value,
                        "seasonal": seasonal_value
                    }
                except (IndexError, AttributeError, TypeError):
                    patterns_simplified[str(num)] = {
                        "trend": 1.0,
                        "seasonal": 0.0
                    }
        
        # 이전 선택 번호
        prev_selected = [int(num) for num in previous_selections]
        
        # ML 모델 예측 확률
        ml_probs = [float(prob) for prob in predicted_probs]
        
        # 전략 정보
        strategy_int = int(strategy)
        
        # 전략별 설명 추가
        strategy_description = ""
        if strategy_int == 1:
            strategy_description = """
            전략 1은 '핫 넘버(Hot Number)' 전략으로, 과거에 평균보다 더 자주 나온 번호를 선호합니다.
            평균 빈도는 {:.2f}이며, 이보다 높은 빈도를 가진 번호에 더 높은 가중치를 부여해야 합니다.
            """.format(mean_freq)
        else:
            strategy_description = """
            전략 2는 '쿨링 다운(Cooling Down)' 전략으로, 평균~평균-표준편차 범위의 번호 중 상승 추세를 보이는 번호를 선호합니다.
            평균 빈도는 {:.2f}, 표준편차는 {:.2f}이며, 평균-표준편차({:.2f})와 평균 사이의 빈도를 가진 번호를 선호합니다.
            특히 상승 추세(trend > 1)를 보이는 번호에 더 높은 가중치를 부여해야 합니다.
            """.format(mean_freq, std_freq, mean_freq - std_freq)
        
        # LLM에 전송할 컨텍스트 생성 - 전략별 맞춤형 프롬프트
        prompt = {
            "model": "gpt-4",  # 또는 다른 모델
            "messages": [
                {"role": "system", "content": f"""
                당신은 로또 번호 예측 전문가입니다. 과거 데이터, 빈도 분석, 시계열 패턴, 그리고 머신러닝 예측을 기반으로 
                1부터 45까지의 번호에 대한 가중치를 생성해주세요. 가중치는 각 번호가 다음 추첨에 나올 가능성을 나타냅니다.
                
                {strategy_description}
                
                응답은 JSON 형식의 배열로, 1부터 45까지 각 번호에 대한 가중치 값만 포함해야 합니다.
                가중치 값은 0과 1 사이의 숫자여야 하며, 전체 합은 1에 가까워야 합니다.
                """},
                {"role": "user", "content": json.dumps({
                    "strategy": strategy_int,
                    "recent_draws": recent_draws,
                    "frequency_data": freq_list,
                    "mean_frequency": float(mean_freq),
                    "std_frequency": float(std_freq),
                    "temporal_patterns": patterns_simplified,
                    "ml_predictions": ml_probs,
                    "previous_selections": prev_selected
                }, ensure_ascii=False)}
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        # API 요청
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_KEY}"
        }
        
        try:
            response = requests.post(LLM_API_URL, headers=headers, json=prompt, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"LLM API 오류: {response.status_code} - {response.text}")
                return np.random.random(45)
            
            # 응답 처리
            result = response.json()
            llm_content = result["choices"][0]["message"]["content"]
            
            logger.info(f"LLM 응답 수신 (길이: {len(llm_content)})")
            
            try:
                # JSON 응답 파싱
                weights_data = json.loads(llm_content)
                logger.info(f"LLM 응답 파싱 성공, 타입: {type(weights_data)}")
                
                # 배열 형태로 반환되었는지 확인
                if isinstance(weights_data, list) and len(weights_data) == 45:
                    weights = np.array(weights_data, dtype=float)
                    llm_cache.set(cache_key, weights.tolist())
                    return weights
                else:
                    # 다른 형식으로 반환된 경우 처리
                    weights = np.ones(45)
                    if isinstance(weights_data, dict):
                        for key, value in weights_data.items():
                            try:
                                if key.isdigit() and 1 <= int(key) <= 45:
                                    weights[int(key)-1] = float(value)
                            except (ValueError, TypeError, IndexError):
                                pass
                    llm_cache.set(cache_key, weights.tolist())
                    return weights
                    
            except json.JSONDecodeError as e:
                logger.error(f"LLM 응답 파싱 오류: {e}")
                logger.error(f"원본 응답: {llm_content[:100]}...")
                # 파싱 실패 시 텍스트에서 숫자 추출 시도
                try:
                    import re
                    numbers = re.findall(r"[\d\.]+", llm_content)
                    if len(numbers) >= 45:
                        weights = np.array([float(num) for num in numbers[:45]])
                        llm_cache.set(cache_key, weights.tolist())
                        return weights
                except Exception as parsing_e:
                    logger.error(f"숫자 추출 시도 오류: {parsing_e}")
                
                return np.random.random(45)
        
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API 요청 오류: {e}")
            return np.random.random(45)
            
    except Exception as e:
        logger.error(f"LLM 가중치 생성 중 오류: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return np.random.random(45)

# 공유 인스턴스 생성
# 공유 인스턴스는 그대로 유지
shared_predictor = AdvancedLottoPredictor() 
shared_predictor.load_models()

# 캐시 변수 추가
import time
predicted_probs_cache = None
strategy_weights_cache = {}
temporal_patterns_cache = None
last_updated = None

def get_recommendation(strategy_counts):
    """전략별 로또 번호 추천 (LLM 통합)"""
    # 전역 변수 참조
    global shared_predictor, predicted_probs_cache, strategy_weights_cache
    global temporal_patterns_cache, last_updated
    
    try:
        if not os.path.exists(settings.LOTTO_DATA_FILE):
            logger.info("데이터 파일이 없습니다. 초기 데이터를 수집합니다.")
            collector = LottoDataCollector()
            collector.collect_initial_data()

        df = pd.read_csv(settings.LOTTO_DATA_FILE)
        
        # 데이터 파일 마지막 수정 시간 확인
        data_mtime = os.path.getmtime(settings.LOTTO_DATA_FILE)
        
        # 캐시 초기화 또는 업데이트 필요한지 확인
        cache_valid = (last_updated is not None and data_mtime <= last_updated)
        
        if not cache_valid:
            logger.info("캐시가 없거나 데이터가 업데이트되어 가중치를 새로 계산합니다.")
            
            # 번호별 출현 빈도 분석
            all_numbers = []
            for col in ['1', '2', '3', '4', '5', '6']:
                all_numbers.extend(df[col].tolist())
            number_counts = pd.Series(all_numbers).value_counts()
            
            # ML 예측 확률 캐싱
            predicted_probs_cache = shared_predictor.predict_numbers()
            
            # 시계열 패턴 분석 캐싱
            temporal_patterns_cache = shared_predictor.analyze_temporal_patterns(df)
            
            # 전략별 기본 가중치 캐싱
            strategy_weights_cache = {}
            
            # 전략 1: 평균 이상 많이 나온 번호 가중치
            freqs = np.array([number_counts.get(n, 0) for n in range(1, 46)])
            mean_freq = np.mean(freqs)
            weights1 = np.array([
                (number_counts.get(n, 0) - mean_freq) if number_counts.get(n, 0) > mean_freq else 0.0001
                for n in range(1, 46)
            ])
            min_weight = np.min(weights1)
            if min_weight < 0:
                weights1 = weights1 - min_weight
            weights1 = weights1 / np.sum(weights1)
            strategy_weights_cache[1] = weights1
            
            # 전략 2: 평균~평균-표준편차 범위 + 상승 추세
            std_freq = np.std(freqs)
            weights2 = np.array([
                1.0 if (mean_freq - std_freq <= number_counts.get(n, 0) <= mean_freq) else 0.0001
                for n in range(1, 46)
            ])
            for num in range(1, 46):
                if num in temporal_patterns_cache:
                    pattern = temporal_patterns_cache[num]
                    trend = pattern['trend'].iloc[-1] if not pd.isna(pattern['trend'].iloc[-1]) else 1
                    if trend > 1:
                        weights2[num-1] *= (1 + trend/5)
            weights2 = weights2 / np.sum(weights2)
            strategy_weights_cache[2] = weights2
            
            last_updated = time.time()  # 현재 시간으로 업데이트
        else:
            logger.info("캐시된 예측 가중치를 사용합니다.")
        
        recommendations = []
        previous_selections = set()

        # 전략 타입을 문자열에서 정수로 변환
        strategy_counts_int = {}
        for strategy, count in strategy_counts.items():
            try:
                strategy_key = int(strategy)
                strategy_counts_int[strategy_key] = int(count)
            except (ValueError, TypeError):
                logger.error(f"전략 변환 오류: {strategy}:{count}")
                strategy_counts_int[1] = 1  # 기본값 설정
        
        logger.info(f"전략 카운트: {strategy_counts_int}")

        for strategy, count in strategy_counts_int.items():
            logger.info(f"전략 {strategy} 처리, {count}개 번호 조합 생성")
            
            for i in range(count):
                logger.info(f"전략 {strategy}, 조합 {i+1}/{count} 생성 중")
                
                try:
                    # 캐시된 가중치 사용
                    if strategy in strategy_weights_cache:
                        weights = strategy_weights_cache[strategy]
                        if strategy == 1:
                            logger.info(f"전략 1 (핫 넘버) 가중치 사용: {weights[:5]}...")
                        else:
                            logger.info(f"전략 2 (쿨링 다운 + 트렌드) 가중치 사용: {weights[:5]}...")
                    else:
                        # 캐시 없는 경우 기본값
                        weights = np.ones(45) / 45
                    
                    # ML 예측 확률과 결합
                    stat_ml_weights = 0.3 * weights + 0.4 * predicted_probs_cache
                    
                    # 이전 선택에 대한 페널티
                    penalty = np.array([0.7 if i+1 in previous_selections else 1.0 for i in range(45)])
                    stat_ml_weights *= penalty
                    
                    # LLM 기반 가중치 생성 시도
                    try:
                        llm_weights = get_llm_weights(
                            df, 
                            predicted_probs_cache, 
                            None,  # number_counts는 함수 내에서 처리됨
                            temporal_patterns_cache, 
                            previous_selections,
                            strategy
                        )
                        
                        # 유효한 가중치인지 확인
                        if np.any(np.isnan(llm_weights)) or np.any(np.isinf(llm_weights)):
                            logger.warning("LLM 가중치에 NaN 또는 무한값 발견, 랜덤 가중치로 대체")
                            llm_weights = np.random.random(45)
                        
                        # 정규화
                        llm_sum = np.sum(llm_weights)
                        if llm_sum > 0:
                            llm_weights = llm_weights / llm_sum
                        else:
                            llm_weights = np.ones(45) / 45
                        
                        logger.info(f"LLM 가중치 생성 완료: {llm_weights[:5]}...")
                        
                    except Exception as e:
                        logger.error(f"LLM 가중치 생성 실패, 랜덤 가중치 사용: {str(e)}")
                        llm_weights = np.random.random(45)
                        llm_weights = llm_weights / np.sum(llm_weights)
                    
                    # 최종 가중치 계산
                    final_weights = 0.7 * stat_ml_weights + 0.3 * llm_weights
                    
                    # 정규화
                    weight_sum = np.sum(final_weights)
                    if weight_sum > 0:
                        final_weights = final_weights / weight_sum
                    else:
                        final_weights = np.ones(45) / 45
                    
                    logger.info(f"최종 가중치 생성 완료: {final_weights[:5]}...")
                    
                    # 번호 선택
                    selected = np.random.choice(
                        range(1, 46),
                        size=6,
                        replace=False,
                        p=final_weights
                    )
                    selected_list = sorted([int(num) for num in selected])
                    logger.info(f"선택된 번호: {selected_list}")
                    
                    previous_selections.update(selected)
                    recommendations.append((strategy, selected_list))
                    
                except Exception as e:
                    logger.error(f"번호 추천 과정 중 오류: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
                    # 오류 발생 시 안전한 대체 방법으로 번호 선택
                    selected = np.random.choice(range(1, 46), size=6, replace=False)
                    selected_list = sorted([int(num) for num in selected])
                    recommendations.append((strategy, selected_list))
                    logger.info(f"오류 후 대체 번호: {selected_list}")
        
        logger.info(f"최종 추천 번호 개수: {len(recommendations)}")
        return recommendations, None
        
    except Exception as e:
        logger.error(f"번호 추천 중 전체 오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return [], str(e)
    
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

def check_llm_status():
    """LLM 연결 상태 확인"""
    if not LLM_API_KEY:
        return False, "LLM API 키가 설정되지 않았습니다. 환경 변수 LLM_API_KEY를 설정해주세요."
        
    try:
        # 간단한 테스트 요청
        prompt = {
            "model": "gpt-3.5-turbo",  # 가장 저렴한 모델로 테스트
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 5
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_KEY}"
        }
        
        response = requests.post(LLM_API_URL, headers=headers, json=prompt, timeout=5)
        
        if response.status_code == 200:
            return True, "LLM API 연결이 정상입니다."
        else:
            return False, f"LLM API 연결 테스트 실패: {response.status_code} - {response.text}"
            
    except requests.exceptions.RequestException as e:
        return False, f"LLM API 연결 오류: {str(e)}"
    except Exception as e:
        return False, f"LLM 상태 확인 중 오류 발생: {str(e)}"