# Lotto_Bot
# Lotto Bot 📊🎯

Lotto Bot은 머신러닝 알고리즘과 통계 분석을 활용하여 로또 번호를 추천해주는 Django 기반 웹 애플리케이션입니다. 두 가지 전략을 통해 로또 번호를 추천받고 당첨 여부를 확인할 수 있습니다.

## 주요 기능

- **두 가지 번호 추천 전략**:
  - 전략 1: 평균적으로 자주 당첨된 번호 기반 추천
  - 전략 2: 앞으로 많이 나올 잠재력 있는 번호 기반 추천
  
- **사용자 계정 관리**: 회원가입, 로그인, 로그아웃 기능

- **챗봇 인터페이스**: 사용자 친화적인 챗봇 UI를 통한 로또 번호 추천

- **추천 내역 저장**: 사용자별 번호 추천 내역 저장 및 조회

- **당첨 확인**: 추천받은 번호의 당첨 여부 자동 확인

## 기술 스택

- **Backend**: Django, Django REST Framework
- **Frontend**: HTML, CSS, JavaScript, TailwindCSS
- **데이터 분석**: Pandas, NumPy, XGBoost, scikit-learn
- **머신러닝**: XGBoost, RandomForest
- **인증**: JWT (JSON Web Tokens)
- **자연어 처리**: OpenAI API

## 설치 및 실행 방법

### 사전 요구사항

- Python 3.9 이상
- pip (Python 패키지 관리자)
- 가상환경 (선택사항이지만 권장)

### 설치 단계

1. 저장소 클론:
```bash
git clone https://github.com/yourusername/lotto-bot.git
cd lotto-bot
```

2. 가상환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Windows의 경우: venv\Scripts\activate
```

3. 의존성 설치:
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정:
.env 파일을 프로젝트 루트에 생성하고 다음 내용을 추가:
```
DJANGO_SECRET_KEY=your_secret_key_here
OPEN_API_KEY=your_openai_api_key_here
```

5. 데이터베이스 마이그레이션:
```bash
python manage.py migrate
```

6. data 폴더 확인:
프로젝트 루트에 `data` 폴더가 있고 `lotto_history.csv` 파일이 존재하는지 확인합니다. 이 파일은 10년치의 로또 당첨 번호 데이터를 포함하고 있어야 합니다.

7. 데이터 로드:
```bash
python manage.py load_lotto_data
```

8. 서버 실행:
```bash
python manage.py runserver
```

9. 브라우저에서 `http://127.0.0.1:8000/`으로 접속하여 애플리케이션 사용

## 프로젝트 구조

```
lotto-bot/
├── accounts/             # 사용자 계정 관리 앱
├── chatbot/              # 챗봇 및 로또 예측 앱
│   ├── management/       # Django 커스텀 명령어
│   ├── migrations/       # 데이터베이스 마이그레이션
│   ├── templates/        # 챗봇 템플릿
│   ├── cron.py           # 예약 작업
│   ├── models.py         # 데이터 모델
│   ├── services.py       # 비즈니스 로직
│   ├── urls.py           # URL 라우팅
│   └── views.py          # 뷰 로직
├── data/                 # 로또 데이터 저장 폴더
│   └── lotto_history.csv # 10년치 로또 당첨 번호 데이터
├── lottobot/             # 프로젝트 메인 앱
│   ├── settings.py       # 프로젝트 설정
│   ├── urls.py           # 메인 URL 설정
│   └── views.py          # 메인 뷰
├── templates/            # 공통 템플릿
├── .env                  # 환경 변수 파일
├── .gitignore
├── manage.py
└── requirements.txt      # 의존성 목록
```

## 머신러닝 모델

Lotto Bot은 다음과 같은 머신러닝 모델을 사용하여 로또 번호를 예측합니다:

1. **XGBoost**: 과거 당첨 번호의 패턴을 분석하여 예측
2. **RandomForest**: 다양한 특성을 기반으로 번호 예측
3. **시계열 분석**: 번호별 등장 빈도의 시간적 패턴 분석

각 전략은 이러한 모델들의 예측을 다른 가중치로 조합하여 최종 번호를 추천합니다.

## API 엔드포인트

- `/api/accounts/`: 사용자 계정 관리 API
- `/api/chatbot/chat/`: 챗봇 대화 API
- `/api/chatbot/status/`: 데이터 상태 확인 API
- `/api/history/`: 사용자별 추천 내역 API
- `/api/metrics/`: 모델 성능 지표 API

## 정기적인 데이터 업데이트

매주 토요일 21:00에 cron 작업이 실행되어 최신 로또 당첨 번호를 수집하고 모델을 재학습합니다.

```python
# chatbot/cron.py
def update_lotto_draws():
    # 데이터 수집 및 모델 재학습 로직
```

## 기여 방법

1. 저장소를 포크합니다.
2. 새로운 기능 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`).
3. 변경사항을 커밋합니다 (`git commit -m 'Add some amazing feature'`).
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`).
5. Pull Request를 생성합니다.

## 라이센스

MIT License

## 연락처

프로젝트 관리자: [이메일주소]

---

Lotto Bot은 로또 당첨을 보장하지 않으며, 단순한 재미와 참고용으로만 사용해 주세요.
행운을 빕니다! 🍀