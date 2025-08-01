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
    # 강력한 디버그 로깅 추가
    import logging
    logger = logging.getLogger(__name__)
    print(f"[CUSTOM LOGIN VIEW] Called - Method: {request.method}, Path: {request.path}")
    logger.error(f"[CUSTOM LOGIN VIEW] Called - Method: {request.method}, Path: {request.path}")
    
    if request.method == 'POST':
        logger.info(f"POST request received with data: {request.POST}")
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        if username and password:
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('/main/')  # 로그인 성공 시 메인 페이지로 리다이렉트
            else:
                logger.info(f"Authentication failed for username: {username}")
                messages.error(request, '사용자명 또는 비밀번호가 잘못되었습니다.')
        else:
            logger.info("Missing username or password in POST request")
            messages.error(request, '사용자명과 비밀번호를 모두 입력해주세요.')
    
    # GET 요청이거나 로그인 실패 시 로그인 페이지 렌더링
    logger.info("Rendering custom login page with direct HTML response")
    # 완전히 CSRF 토큰 없는 HTML 직접 반환
    error_message = ""
    for message in messages.get_messages(request):
        if message.tags == 'error':
            error_message = f'<div class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-6"><p>{message}</p></div>'
    
    html_content = f'''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lotto Bot</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100">
    <div class="flex items-center justify-center min-h-screen p-6">
        <div class="w-full max-w-md">
            <h1 class="text-3xl font-bold text-center mb-8">로또 봇 로그인 (CUSTOM VIEW)</h1>
            <p class="text-center text-sm text-green-600 mb-4">CSRF-Free Custom Login View Active</p>
            <div class="bg-white rounded-lg shadow-lg p-8">
                {error_message}
                <form method="post" class="space-y-6">
                    <div>
                        <label for="username" class="block text-sm font-medium text-gray-700">아이디</label>
                        <input type="text" name="username" id="username" required 
                               class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 p-2 border">
                    </div>
                    <div>
                        <label for="password" class="block text-sm font-medium text-gray-700">비밀번호</label>
                        <input type="password" name="password" id="password" required 
                               class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 p-2 border">
                    </div>
                    <div>
                        <button type="submit" 
                                class="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
                            로그인
                        </button>
                    </div>
                </form>
                <div class="mt-4 text-center">
                    <p class="text-sm text-gray-600">
                        계정이 없으신가요? 
                        <a href="/register/" class="font-medium text-blue-600 hover:text-blue-500">
                            회원가입
                        </a>
                    </p>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
    '''
    return HttpResponse(html_content)

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