from django.apps import AppConfig

class ChatbotConfig(AppConfig):
    name = 'chatbot'

    def ready(self):
        # 크롤링 및 머신러닝 모델 업데이트는 Celery Beat의 예약 태스크에서 실행됩니다.
        # 따라서 앱 초기화 시에는 실행하지 않도록 합니다.
        pass
