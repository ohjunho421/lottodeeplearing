from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from chatbot.services import LottoDataCollector
from chatbot.models import Recommendation
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages

def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'계정이 생성되었습니다. 이제 로그인할 수 있습니다.')
            return redirect('login')
    else:
        form = UserCreationForm()
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