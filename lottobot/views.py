from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from chatbot.services import LottoDataCollector
from chatbot.models import Recommendation
from django.shortcuts import render, redirect
# from django.contrib.auth.forms import UserCreationForm  # 이 줄 제거 또는 주석 처리
from django.contrib import messages
from django.contrib.auth import login, authenticate
from accounts.forms import CustomUserCreationForm
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.forms import AuthenticationForm

@csrf_exempt
def custom_login_view(request):
    """CSRF 검증이 없는 커스텀 로그인 뷰"""
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        if username and password:
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('/main/')  # 로그인 성공 시 메인 페이지로 리다이렉트
            else:
                messages.error(request, '사용자명 또는 비밀번호가 잘못되었습니다.')
        else:
            messages.error(request, '사용자명과 비밀번호를 모두 입력해주세요.')
    
    # GET 요청이거나 로그인 실패 시 로그인 페이지 렌더링
    return render(request, 'registration/login.html', {
        'no_csrf': True  # 템플릿에서 CSRF 토큰을 렌더링하지 않도록 플래그 설정
    })

def register_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)  # 여기 변경
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'계정이 생성되었습니다. 이제 로그인할 수 있습니다.')
            return redirect('login')
    else:
        form = CustomUserCreationForm()  # 여기도 변경
    return render(request, 'registration/register.html', {'form': form})

@login_required
def main_view(request):
    # 최신 당첨 번호 가져오기
    collector = LottoDataCollector()
    latest_numbers = None
    try:
        df = collector.collect_initial_data()
        if df is not None and not df.empty:
            latest_numbers = {
                'round': df.iloc[0]['회차'],
                'date': df.iloc[0]['추첨일'],
                'numbers': [
                    df.iloc[0]['1'],
                    df.iloc[0]['2'],
                    df.iloc[0]['3'],
                    df.iloc[0]['4'],
                    df.iloc[0]['5'],
                    df.iloc[0]['6']
                ],
                'bonus': df.iloc[0]['보너스']
            }
    except Exception as e:
        print(f"Error fetching latest numbers: {e}")
    
    context = {
        'latest_numbers': latest_numbers
    }
    return render(request, 'lottobot/main.html', context)

@login_required
def mypage_view(request):
    # 사용자의 추천 번호 기록 가져오기
    recommendations = Recommendation.objects.filter(
        user=request.user
    ).order_by('-recommendation_date')[:100]
    
    context = {
        'recommendations': recommendations
    }
    return render(request, 'chatbot/history.html', context)

def health_check(request):
    """Simple health check endpoint for Railway deployment"""
    return HttpResponse("OK", status=200)