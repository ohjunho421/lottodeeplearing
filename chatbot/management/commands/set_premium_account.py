from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from chatbot.models import UserProfile

User = get_user_model()

class Command(BaseCommand):
    help = '특정 사용자를 프리미엄 계정으로 설정'

    def add_arguments(self, parser):
        parser.add_argument('username', type=str, help='프리미엄으로 설정할 사용자 이름')

    def handle(self, *args, **kwargs):
        username = kwargs['username']
        
        try:
            user = User.objects.get(username=username)
            profile, created = UserProfile.objects.get_or_create(user=user)
            profile.is_premium = True
            profile.save()
            
            self.stdout.write(self.style.SUCCESS(f'사용자 "{username}"가 프리미엄 계정으로 설정되었습니다.'))
            
        except User.DoesNotExist:
            self.stdout.write(self.style.ERROR(f'사용자 "{username}"를 찾을 수 없습니다.'))
