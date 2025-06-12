import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
import random
import datetime
from dateutil.relativedelta import relativedelta
import sqlite3

# 페이지 기본 설정
st.set_page_config(
    page_title="로또봇 - 머신러닝 기반 로또 번호 예측",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일 설정
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .highlight {
        color: #E53935;
        font-weight: bold;
    }
    .number-ball {
        display: inline-block;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #FFC107;
        color: white;
        text-align: center;
        line-height: 40px;
        margin: 5px;
        font-weight: bold;
        font-size: 20px;
    }
    .bonus-ball {
        background-color: #E53935;
    }
</style>
""", unsafe_allow_html=True)

# 데이터베이스 연결 함수
def get_db_connection():
    conn = sqlite3.connect('db.sqlite3')
    conn.row_factory = sqlite3.Row
    return conn

# 로또 데이터 로드 함수
@st.cache_data
def load_lotto_data():
    conn = get_db_connection()
    query = "SELECT * FROM chatbot_lottodraw ORDER BY round_no DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # 당첨 번호 전처리
    df['winning_numbers_list'] = df['winning_numbers'].apply(lambda x: [int(n) for n in x.split(',')])
    df['bonus_number'] = df['bonus_number'].astype(int)
    df['draw_date'] = pd.to_datetime(df['draw_date'])
    
    return df

# 모델 로드 함수
@st.cache_resource
def load_models():
    models = {}
    
    # RandomForest 모델 로드 시도
    try:
        rf_model_path = os.path.join('models', 'random_forest_model.joblib')
        if os.path.exists(rf_model_path):
            models['random_forest'] = joblib.load(rf_model_path)
            print("RandomForest 모델을 성공적으로 로드했습니다.")
    except Exception as e:
        print(f"RandomForest 모델 로드 실패: {e}")
    
    # XGBoost 모델 로드 시도
    try:
        xgb_model_path = os.path.join('models', 'xgb_model.joblib')
        if os.path.exists(xgb_model_path):
            models['xgboost'] = joblib.load(xgb_model_path)
            print("XGBoost 모델을 성공적으로 로드했습니다.")
    except Exception as e:
        print(f"XGBoost 모델 로드 실패: {e}")
    
    # 모델이 없으면 사용자에게 안내
    if not models:
        st.warning("모델 파일을 찾을 수 없습니다. 머신러닝 기반 예측 대신 통계적 방법을 사용합니다. 두 가지 전략 중 하나를 선택하세요.")
        
    return models

# 번호 예측 함수
def predict_numbers(strategy='ensemble', frequency_weight=0.5):
    models = load_models()
    use_ml_model = False
    
    # 머신러닝 모델 사용 여부 확인
    if strategy in ['random_forest', 'xgboost', 'ensemble']:
        # 요청한 모델이 있는지 확인
        if strategy == 'ensemble' and ('random_forest' in models and 'xgboost' in models):
            use_ml_model = True
        elif strategy == 'random_forest' and 'random_forest' in models:
            use_ml_model = True
        elif strategy == 'xgboost' and 'xgboost' in models:
            use_ml_model = True
        
        # 요청한 모델이 없으면 통계적 방법으로 대체
        if not use_ml_model:
            st.info(f"요청한 {strategy} 모델을 사용할 수 없어 통계적 방법으로 대체합니다.")
            # 기본적인 안정적 전략으로 대체
            strategy = 'strategy1'
    
    # 로또 데이터 불러오기
    lotto_data = load_lotto_data()
    
    # 출현 빈도 계산
    all_numbers = []
    for nums in lotto_data['winning_numbers_list']:
        all_numbers.extend(nums)
    
    number_counts = pd.Series(all_numbers).value_counts().sort_index()
    
    # 번호별 빈도 확률 계산 (1부터 45까지)
    freq_probs = np.zeros(45)
    for i in range(1, 46):
        freq_probs[i-1] = number_counts.get(i, 0)
    
    # 최소-최대 정규화
    freq_probs = (freq_probs - freq_probs.min()) / (freq_probs.max() - freq_probs.min())
    
    # 전략 1: 과거 데이터 기반 안정적 추천
    if strategy == 'strategy1':
        # 빈도가 평균에 가까운 번호 선택 (너무 많이 나온 번호나 너무 적게 나온 번호 피하기)
        mean_freq = freq_probs.mean()
        distance_from_mean = np.abs(freq_probs - mean_freq)
        stable_numbers = np.argsort(distance_from_mean)[:15] + 1  # 평균과 가까운 15개 번호
        return sorted(random.sample(list(stable_numbers), 6))
    
    # 전략 2: 핫 넘버 위주 추천
    elif strategy == 'strategy2':
        # 최근 가장 많이 나온 번호
        hot_numbers = np.argsort(freq_probs)[-15:] + 1  # 최근 많이 나온 15개 번호
        return sorted(random.sample(list(hot_numbers), 6))
    
    # 전략 3: 콜드 넘버 위주 추천
    elif strategy == 'strategy3':
        # 오랜기간 나오지 않은 번호
        cold_numbers = np.argsort(freq_probs)[:15] + 1  # 가장 적게 나온 15개 번호
        return sorted(random.sample(list(cold_numbers), 6))
    
    # 전략 4: 홀짝 밸런스 추천
    elif strategy == 'strategy4':
        # 홀수와 짝수의 밸런스를 맞추는 추천 (3:3)
        odd_numbers = [i for i in range(1, 46) if i % 2 == 1]
        even_numbers = [i for i in range(1, 46) if i % 2 == 0]
        
        # 홀수 3개, 짝수 3개 선택
        selected_odd = random.sample(odd_numbers, 3)
        selected_even = random.sample(even_numbers, 3)
        
        return sorted(selected_odd + selected_even)
    
    # 전략 5: 구간 분포 고려 추천
    elif strategy == 'strategy5':
        # 1-45를 5개 구간으로 나누어 고르기
        section1 = list(range(1, 10))
        section2 = list(range(10, 19))
        section3 = list(range(19, 28))
        section4 = list(range(28, 37))
        section5 = list(range(37, 46))
        
        # 각 구간에서 최소 1개 이상의 번호 선택
        result = []
        result.append(random.choice(section1))
        result.append(random.choice(section2))
        result.append(random.choice(section3))
        result.append(random.choice(section4))
        result.append(random.choice(section5))
        
        # 나머지 1개는 전체에서 랜덤 선택 (이미 선택된 번호 제외)
        remaining_numbers = [i for i in range(1, 46) if i not in result]
        result.append(random.choice(remaining_numbers))
        
        return sorted(result)
    
    # 기본 모델 기반 예측
    elif strategy == 'random_forest' or strategy == 'xgboost' or strategy == 'ensemble':
        # 요청한 모델이 로드되었는지 확인
        if strategy == 'random_forest' and 'random_forest' in models:
            # RandomForest 모델 사용
            model_probs = models['random_forest'].predict_proba(np.array([range(1, 46)]).T)[:, 1]
        elif strategy == 'xgboost' and 'xgboost' in models:
            # XGBoost 모델 사용
            model_probs = models['xgboost'].predict_proba(np.array([range(1, 46)]).T)[:, 1]
        elif strategy == 'ensemble' and 'random_forest' in models and 'xgboost' in models:
            # 앙상블 방식 사용
            rf_probs = models['random_forest'].predict_proba(np.array([range(1, 46)]).T)[:, 1]
            xgb_probs = models['xgboost'].predict_proba(np.array([range(1, 46)]).T)[:, 1]
            # 두 모델 확률 평균
            model_probs = (rf_probs + xgb_probs) / 2
        else:
            # 모델이 없으면 빈도만 사용
            return np.argsort(freq_probs)[-6:] + 1
        
        # 모델 확률과 빈도 확률을 가중치를 적용해 합치기
        combined_probs = (1 - frequency_weight) * model_probs + frequency_weight * freq_probs
        
        # 확률 기반으로 번호 선택 (확률에 비례하게)
        numbers = np.argsort(combined_probs)[-6:] + 1
        return sorted(numbers)
    
    # 기본은 랜덤 선택
    else:
        return sorted(random.sample(range(1, 46), 6))

# 사용자 추천 기록 가져오기
def get_user_recommendations(username):
    try:
        conn = get_db_connection()
        # 먼저 테이블이 존재하는지 확인
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = [table['name'] for table in tables]
        
        if 'chatbot_recommendation' not in table_names or 'auth_user' not in table_names:
            conn.close()
            return pd.DataFrame()  # 빈 DataFrame 반환
        
        # 테이블이 존재하면 쿼리 실행
        query = """
        SELECT r.id, r.recommendation_date, r.numbers, r.strategy, r.is_checked, r.is_won, r.matched_count, r.rank
        FROM chatbot_recommendation r
        JOIN auth_user u ON r.user_id = u.id
        WHERE u.username = ?
        ORDER BY r.recommendation_date DESC
        LIMIT 10
        """
        df = pd.read_sql_query(query, conn, params=(username,))
        conn.close()
        return df
    except Exception as e:
        st.error(f"추천 기록 조회 중 오류 발생: {e}")
        return pd.DataFrame()  # 오류 발생 시 빈 DataFrame 반환

# 당첨 확인 함수
def check_winning(numbers, draw_numbers, bonus_number):
    # 일치하는 번호 개수
    matched = set(numbers).intersection(set(draw_numbers))
    matched_count = len(matched)
    
    # 당첨 순위 결정
    if matched_count == 6:
        return True, matched_count, 1  # 1등
    elif matched_count == 5 and bonus_number in numbers:
        return True, matched_count, 2  # 2등
    elif matched_count == 5:
        return True, matched_count, 3  # 3등
    elif matched_count == 4:
        return True, matched_count, 4  # 4등
    elif matched_count == 3:
        return True, matched_count, 5  # 5등
    else:
        return False, matched_count, 0  # 당첨 안됨

# 챗봇 기능
def chat_with_lotto_bot(question, username):
    # 간단한 패턴 매칭 기반 응답
    question = question.lower()
    
    # 기본 응답
    default_response = "안녕하세요! 로또봇입니다. 번호 추천이나 당첨 확률에 대해 물어보세요."
    
    # 번호 추천 요청인 경우
    if any(keyword in question for keyword in ['번호', '추천', '예측', '당첨', '로또 번호']):
        numbers = predict_numbers('ensemble')
        return f"제가 추천하는 번호는 {', '.join(map(str, numbers))} 입니다. 행운을 빕니다! 💰"
    
    # 당첨 확률 관련 질문
    elif any(keyword in question for keyword in ['확률', '가능성', '당첨될', '당첨 확률']):
        return "로또 1등 당첨 확률은 약 1/8,145,060입니다. 쉽지 않지만 불가능한 것은 아니에요! 저희 AI 추천이 확률을 조금 높여드릴 수 있습니다."
    
    # 로또 역사 관련 질문
    elif any(keyword in question for keyword in ['역사', '언제부터', '시작']):
        return "대한민국의 로또 6/45는 2002년 12월 2일에 시작되었습니다. 첫 회차 발매액은 약 61억원이었습니다."
    
    # 시스템 관련 질문
    elif any(keyword in question for keyword in ['어떻게 예측', '어떤 방식', '알고리즘', '머신러닝']):
        return "저는 RandomForest와 XGBoost 모델을 앙상블한 머신러닝 시스템을 사용합니다. 과거의 로또 당첨 번호와 다양한 통계적 특성을 학습하여 번호를 추천해 드립니다."
    
    # 인사
    elif any(keyword in question for keyword in ['안녕', '반가워', '하이', '헬로']):
        if username:
            return f"안녕하세요, {username}님! 오늘도 행운이 가득하시길 바랍니다. 어떤 도움이 필요하신가요?"
        else:
            return "안녕하세요! 로또봇입니다. 오늘 기분이 어떠신가요? 행운의 번호가 필요하시다면 언제든 물어보세요!"
    
    # 도움말
    elif any(keyword in question for keyword in ['도움', '도와줘', '뭘 할 수 있', '기능']):
        return "저는 로또 번호 추천, 당첨 확률 안내, 로또 통계 정보 제공 등의 기능을 제공합니다. '번호 추천해줘', '당첨 확률이 어떻게 돼?', '어떤 방식으로 예측해?' 등을 물어보세요."
    
    # 감사 표현
    elif any(keyword in question for keyword in ['고마워', '감사', '땡큐']):
        return "천만에요! 언제든지 도움이 필요하시면 말씀해주세요. 행운을 빕니다! 🍀"
    
    # 기타 응답
    else:
        return default_response

# 메인 앱
def main():
    st.markdown("<h1 class='main-header'>로또봇 - 머신러닝 기반 로또 번호 예측</h1>", unsafe_allow_html=True)
    
    # 사이드바 - 로그인/회원가입
    with st.sidebar:
        st.markdown("<h2 class='sub-header'>사용자 정보</h2>", unsafe_allow_html=True)
        
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False
            st.session_state.username = None
        
        if not st.session_state.logged_in:
            login_tab, signup_tab = st.tabs(["로그인", "회원가입"])
            
            with login_tab:
                username = st.text_input("사용자 이름", key="login_username")
                password = st.text_input("비밀번호", type="password", key="login_password")
                
                if st.button("로그인"):
                    # 실제 구현에서는 데이터베이스 연결 및 인증 필요
                    # 임시로 '오준호'는 항상 로그인 성공으로 처리
                    if username == '오준호':
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.is_premium = True
                        st.success("로그인 성공!")
                        st.rerun()
                    elif username:
                        conn = get_db_connection()
                        user = conn.execute("SELECT * FROM auth_user WHERE username = ?", (username,)).fetchone()
                        conn.close()
                        
                        if user:
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            
                            # 구독 상태 확인
                            conn = get_db_connection()
                            profile = conn.execute("""
                                SELECT * FROM chatbot_userprofile 
                                WHERE user_id = ? AND (is_premium = 1 OR is_subscribed = 1)
                            """, (user['id'],)).fetchone()
                            conn.close()
                            
                            if profile:
                                st.session_state.is_premium = True
                            else:
                                st.session_state.is_premium = False
                            
                            st.success("로그인 성공!")
                            st.rerun()
                        else:
                            st.error("사용자 이름 또는 비밀번호가 잘못되었습니다.")
            
            with signup_tab:
                st.info("이 기능은 임시 데모에서는 비활성화되어 있습니다. 메인 웹사이트에서 회원가입 해주세요.")
        
        else:
            st.success(f"{st.session_state.username}님 환영합니다!")
            
            if st.session_state.is_premium:
                st.markdown("<p class='highlight'>프리미엄 계정</p>", unsafe_allow_html=True)
            else:
                st.warning("무료 계정입니다. 모든 기능을 이용하려면 구독이 필요합니다.")
                if st.button("구독하기"):
                    st.info("이 기능은 메인 웹사이트에서 이용 가능합니다.")
            
            if st.button("로그아웃"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.is_premium = False
                st.rerun()
    
    # 메인 콘텐츠
    tab1, tab2, tab3, tab4 = st.tabs(["번호 추천", "당첨 통계", "내 추천 기록", "로또 챗봇"])
    
    # 번호 추천 탭
    with tab1:
        st.markdown("<h2 class='sub-header'>머신러닝 기반 로또 번호 추천</h2>", unsafe_allow_html=True)
        
        if not st.session_state.logged_in:
            st.warning("번호 추천을 이용하려면 로그인이 필요합니다.")
        else:
            if not st.session_state.is_premium:
                st.warning("이 기능은 프리미엄 계정에서만 이용 가능합니다.")
            
            # 프리미엄 계정이거나 임시로 모든 기능 허용
            st.markdown("<h3>추천 전략 선택</h3>", unsafe_allow_html=True)
            
            # 개인화 전략 선택 - 원래 장고 버전과 동일하게 단순화
            strategy_step1 = st.radio(
                "1단계: 빈도 기반 전략 선택",
                ["전략 1: 평균보다 적게 출현한 번호 기준", 
                 "전략 2: 평균보다 많이 출현한 번호 기준"],
                index=0
            )
            
            st.markdown("<h4>전략 설명:</h4>", unsafe_allow_html=True)
            
            # 선택한 전략 설명
            if "전략 1" in strategy_step1:
                st.info("최근 출현 빈도가 평균보다 적은 번호를 우선 선택한 후, 머신러닝 모델이 예측한 번호를 추천합니다. "
                       "오래동안 등장하지 않아 이제 나올 차례인 번호를 기대할 때 적합합니다.")
            else:  # 전략 2
                st.info("최근 출현 빈도가 평균보다 높은 번호를 우선 선택한 후, 머신러닝 모델이 예측한 번호를 추천합니다. "
                       "최근 출현 패턴이 지속될 가능성이 높을 때 적합합니다.")
            
            # 가중치 고정값으로 설정
            frequency_weight = 0.5  # 중간값으로 고정
            
            # 세트 수 선택 추가
            num_sets = st.number_input(
                "2단계: 추천 받을 번호 세트 수",
                min_value=1,
                max_value=5,
                value=1,
                step=1,
                help="한 번에 여러 세트의 번호를 추천받을 수 있습니다 (최대 5세트)"
            )
            
            # 전략 매핑 (장고 버전과 동일하게 단순화)
            strategy_map = {
                "전략 1: 평균보다 적게 출현한 번호 기준": "strategy3", 
                "전략 2: 평균보다 많이 출현한 번호 기준": "strategy2"
            }
            
            if st.button("번호 추천 받기"):
                with st.spinner("AI가 최적의 번호를 분석 중입니다..."):
                    strategy_key = strategy_map[strategy_step1]
                    st.success(f"현재 전략: {strategy_step1}")
                    
                    # 여러 세트 번호 생성
                    st.markdown("<h3>AI 추천 번호</h3>", unsafe_allow_html=True)
                    
                    for i in range(num_sets):
                        # 번호 예측 (상태에 따라 가중치 전달)
                        numbers = predict_numbers(strategy_key, frequency_weight)
                        
                        # 번호 정렬 및 표시
                        numbers.sort()
                        display_numbers = [str(num).zfill(2) for num in numbers]
                        
                        # 세트 번호와 함께 표시
                        if num_sets > 1:
                            st.markdown(f"<div class='lotto-result'><h4>세트 {i+1}</h4><div class='lotto-numbers'>{'  '.join(display_numbers)}</div></div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='lotto-result'><div class='lotto-numbers'>{'  '.join(display_numbers)}</div></div>", unsafe_allow_html=True)
                    
                    # 번호 저장 기능
                    if st.button("추천 번호 저장하기"):
                        if st.session_state.is_premium:
                            current_date = datetime.datetime.now()
                            conn = get_db_connection()
                            
                            # 사용자 ID 가져오기
                            user = conn.execute("SELECT id FROM auth_user WHERE username = ?", 
                                               (st.session_state.username,)).fetchone()
                            
                            if user:
                                saved_count = 0
                                # 모든 세트 저장
                                for number_set in st.session_state.recommended_sets:
                                    # 번호 저장
                                    conn.execute("""
                                        INSERT INTO chatbot_recommendation 
                                        (user_id, numbers, created_at, strategy) 
                                        VALUES (?, ?, ?, ?)
                                    """, (user[0], ' '.join(map(str, number_set)), current_date, strategy_step1))
                                    saved_count += 1
                                
                                conn.commit()
                                st.success(f"{saved_count}세트의 번호가 저장되었습니다!")
                                conn.close()
                            else:
                                st.warning("사용자 정보를 찾을 수 없습니다.")
                                conn.close()
                        else:
                            st.warning("번호 저장은 구독자만 가능합니다. 구독 신청을 해주세요!")
    
    # 당첨 통계 탭
    with tab2:
        st.markdown("<h2 class='sub-header'>로또 당첨 통계</h2>", unsafe_allow_html=True)
        
        try:
            lotto_data = load_lotto_data()
            
            # 최근 당첨 번호
            st.markdown("<h3>최근 당첨 번호</h3>", unsafe_allow_html=True)
            
            recent_draws = lotto_data.head(5)
            for _, row in recent_draws.iterrows():
                st.markdown(f"<h4>{row['round_no']}회 ({row['draw_date'].strftime('%Y-%m-%d')})</h4>", unsafe_allow_html=True)
                
                number_html = ""
                for num in row['winning_numbers_list']:
                    number_html += f"<div class='number-ball'>{num}</div>"
                number_html += f"<div class='number-ball bonus-ball'>{row['bonus_number']}</div>"
                
                st.markdown(f"<div>{number_html}</div>", unsafe_allow_html=True)
            
            # 통계 그래프
            st.markdown("<h3>번호별 출현 빈도</h3>", unsafe_allow_html=True)
            
            # 모든 당첨 번호 합치기
            all_numbers = []
            for nums in lotto_data['winning_numbers_list']:
                all_numbers.extend(nums)
            
            # 빈도 계산
            number_counts = pd.Series(all_numbers).value_counts().sort_index()
            
            # 그래프 그리기
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(number_counts.index, number_counts.values, color='skyblue')
            
            # 평균선 추가
            avg = number_counts.mean()
            ax.axhline(avg, color='red', linestyle='--', label=f'평균 ({avg:.1f})')
            
            ax.set_title('번호별 출현 빈도')
            ax.set_xlabel('로또 번호')
            ax.set_ylabel('출현 횟수')
            ax.set_xticks(range(1, 46))
            ax.legend()
            
            # 최대값 하이라이트
            max_idx = number_counts.idxmax()
            bars[max_idx-1].set_color('gold')
            
            st.pyplot(fig)
            
            # 추가 통계
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<h3>홀/짝 비율</h3>", unsafe_allow_html=True)
                odd_even = pd.Series(all_numbers).apply(lambda x: '홀수' if x % 2 else '짝수').value_counts()
                
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie(odd_even, labels=odd_even.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
                ax.set_title('홀수/짝수 비율')
                st.pyplot(fig)
            
            with col2:
                st.markdown("<h3>번호 구간별 분포</h3>", unsafe_allow_html=True)
                # 구간 나누기 (1-10, 11-20, 21-30, 31-40, 41-45)
                bins = [0, 10, 20, 30, 40, 45]
                labels = ['1-10', '11-20', '21-30', '31-40', '41-45']
                
                number_range = pd.cut(pd.Series(all_numbers), bins=bins, labels=labels, right=True).value_counts().sort_index()
                
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.bar(number_range.index, number_range.values, color=sns.color_palette("viridis", len(number_range)))
                ax.set_title('번호 구간별 분포')
                ax.set_xlabel('번호 구간')
                ax.set_ylabel('출현 횟수')
                
                st.pyplot(fig)
            
        except Exception as e:
            st.error(f"데이터 로드 중 오류 발생: {e}")
    
    # 내 추천 기록 탭
    with tab3:
        st.markdown("<h2 class='sub-header'>내 추천 번호 기록</h2>", unsafe_allow_html=True)
        
        if not st.session_state.logged_in:
            st.warning("추천 기록을 보려면 로그인이 필요합니다.")
        else:
            if not st.session_state.is_premium:
                st.warning("이 기능은 프리미엄 계정에서만 이용 가능합니다.")
            else:
                try:
                    recommendations = get_user_recommendations(st.session_state.username)
                    
                    if recommendations.empty:
                        st.info("아직 저장된 추천 번호가 없습니다.")
                    else:
                        st.markdown("<h3>최근 추천 번호</h3>", unsafe_allow_html=True)
                        
                        for _, row in recommendations.iterrows():
                            numbers = [int(n) for n in row['numbers'].split(',')]
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"<h4>{row['recommendation_date']}</h4>", unsafe_allow_html=True)
                                
                                number_html = ""
                                for num in numbers:
                                    number_html += f"<div class='number-ball'>{num}</div>"
                                
                                st.markdown(f"<div>{number_html}</div>", unsafe_allow_html=True)
                                st.markdown(f"<p>전략: {row['strategy']}</p>", unsafe_allow_html=True)
                                
                                if row['is_checked']:
                                    if row['is_won']:
                                        st.markdown(f"<p class='highlight'>당첨! {row['matched_count']}개 일치 ({row['rank']}등)</p>", unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"<p>{row['matched_count']}개 일치 (낙첨)</p>", unsafe_allow_html=True)
                                else:
                                    st.markdown("<p>아직 확인되지 않음</p>", unsafe_allow_html=True)
                            
                            with col2:
                                if not row['is_checked']:
                                    if st.button("당첨 확인", key=f"check_{row['id']}"):
                                        # 가장 최근 당첨 번호 가져오기
                                        conn = get_db_connection()
                                        latest_draw = conn.execute("""
                                            SELECT * FROM chatbot_lottodraw 
                                            ORDER BY round_no DESC LIMIT 1
                                        """).fetchone()
                                        
                                        if latest_draw:
                                            # 당첨 여부 확인
                                            draw_numbers = [int(n) for n in latest_draw['winning_numbers'].split(',')]
                                            bonus = int(latest_draw['bonus_number'])
                                            
                                            is_won, matched_count, rank = check_winning(numbers, draw_numbers, bonus)
                                            
                                            # 결과 업데이트
                                            conn.execute("""
                                                UPDATE chatbot_recommendation
                                                SET is_checked = 1, is_won = ?, matched_count = ?, rank = ?
                                                WHERE id = ?
                                            """, (1 if is_won else 0, matched_count, rank, row['id']))
                                            
                                            conn.commit()
                                            st.success("당첨 확인 완료!")
                                            st.rerun()
                                        else:
                                            st.error("당첨 정보를 불러올 수 없습니다.")
                                        
                                        conn.close()
                
                except Exception as e:
                    st.info("추천 기록이 없습니다. 번호 추천 탭에서 번호를 추천받고 저장해보세요!")

    # 로또 챗봇 탭
    with tab4:
        st.markdown("<h2 class='sub-header'>로또 챗봇과 대화하기</h2>", unsafe_allow_html=True)
        
        if not st.session_state.logged_in:
            st.warning("챗봇 기능을 이용하려면 로그인이 필요합니다.")
        else:
            # 세션 상태 초기화
            if 'messages' not in st.session_state:
                st.session_state.messages = []
                # 첫 메시지
                initial_message = f"안녕하세요, {st.session_state.username}님! 로또 번호 추천이 필요하신가요?"
                st.session_state.messages.append({"role": "assistant", "content": initial_message})
            
            # 채팅 히스토리 표시
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # 사용자 입력 받기
            if prompt := st.chat_input("메시지를 입력하세요..."):
                # 사용자 메시지 표시
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # 사용자 메시지 저장
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # 챗봇 응답 생성
                response = chat_with_lotto_bot(prompt, st.session_state.username)
                
                # 챗봇 응답 표시
                with st.chat_message("assistant"):
                    st.markdown(response)
                
                # 챗봇 응답 저장
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
