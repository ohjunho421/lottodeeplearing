"""
로또 추천 결과 관련 뷰
"""
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.core.paginator import Paginator

from .models import Recommendation
from .decorators import subscription_required

@login_required
@subscription_required
def recommendation_history(request):
    """사용자의 추천 번호 히스토리 보기"""
    # 로그인한 사용자의 추천 결과만 조회
    recommendations = Recommendation.objects.filter(user=request.user).order_by('-recommendation_date')
    
    # 페이지네이션 적용
    paginator = Paginator(recommendations, 20)  # 페이지당 20개
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'total_count': recommendations.count(),
    }
    return render(request, 'recommendation_history.html', context)

@login_required
@subscription_required
def recommendation_stats(request):
    """추천 번호 통계 보기"""
    # 사용자의 추천 결과 통계
    user_recommendations = Recommendation.objects.filter(user=request.user)
    
    # 당첨 통계
    win_stats = {
        'total': user_recommendations.count(),
        'checked': user_recommendations.filter(is_checked=True).count(),
        'won': user_recommendations.filter(is_won=True).count(),
    }
    
    # 등수별 통계
    rank_stats = {}
    for i in range(1, 6):
        rank_stats[i] = user_recommendations.filter(rank=i).count()
    
    # 맞춘 개수별 통계
    match_stats = {}
    for i in range(7):  # 0부터 6까지
        match_stats[i] = user_recommendations.filter(matched_count=i).count()
    
    context = {
        'win_stats': win_stats,
        'rank_stats': rank_stats,
        'match_stats': match_stats,
    }
    return render(request, 'recommendation_stats.html', context)

@login_required
def recommendation_api(request):
    """추천 번호 API (JSON 응답)"""
    try:
        if not hasattr(request.user, 'is_active_subscriber') or not request.user.is_active_subscriber:
            if not hasattr(request.user, 'is_premium') or not request.user.is_premium:
                return JsonResponse({
                    'status': 'error',
                    'message': '이 기능을 이용하려면 구독이 필요합니다.'
                }, status=403)
        
        # 사용자의 추천 결과 (최근 10개)
        recommendations = Recommendation.objects.filter(
            user=request.user
        ).order_by('-recommendation_date')[:10]
        
        # JSON 응답 형태로 변환
        result = []
        for rec in recommendations:
            result.append({
                'id': rec.id,
                'date': rec.recommendation_date.strftime('%Y-%m-%d'),
                'numbers': rec.numbers,
                'strategy': rec.strategy,
                'is_checked': rec.is_checked,
                'is_won': rec.is_won,
                'matched_count': rec.matched_count if rec.matched_count is not None else 0,
                'rank': rec.rank if rec.rank is not None else 0,
            })
        
        return JsonResponse({
            'status': 'success',
            'data': result
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)
