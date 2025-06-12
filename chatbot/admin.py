from django.contrib import admin
from .models import ChatHistory, Recommendation, LottoDraw, UserProfile, Payment

@admin.register(ChatHistory)
class ChatHistoryAdmin(admin.ModelAdmin):
    list_display = ('user', 'user_message', 'bot_response', 'created_at')

@admin.register(Recommendation)
class RecommendationAdmin(admin.ModelAdmin):
    list_display = ('user', 'recommendation_date', 'strategy', 'numbers', 'is_checked', 'is_won', 'matched_count', 'rank')
    list_filter = ('is_checked', 'is_won', 'strategy', 'rank')
    search_fields = ('user__username', 'numbers')
    date_hierarchy = 'recommendation_date'

@admin.register(LottoDraw)
class LottoDrawAdmin(admin.ModelAdmin):
    list_display = ('round_no', 'draw_date', 'winning_numbers', 'bonus_number')
    search_fields = ('round_no', 'winning_numbers')
    date_hierarchy = 'draw_date'

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'is_subscribed', 'is_premium', 'subscription_start', 'subscription_end', 'is_subscription_active')
    list_filter = ('is_subscribed', 'is_premium')
    search_fields = ('user__username', 'user__email')
    actions = ['make_premium', 'cancel_premium']
    
    def is_subscription_active(self, obj):
        return obj.is_subscription_active()
    is_subscription_active.boolean = True
    is_subscription_active.short_description = '구독 상태'
    
    def make_premium(self, request, queryset):
        queryset.update(is_premium=True)
    make_premium.short_description = '프리미엄 계정으로 설정'
    
    def cancel_premium(self, request, queryset):
        queryset.update(is_premium=False)
    cancel_premium.short_description = '일반 계정으로 설정'

@admin.register(Payment)
class PaymentAdmin(admin.ModelAdmin):
    list_display = ('user', 'amount', 'payment_date', 'payment_method', 'is_successful', 'subscription_period')
    list_filter = ('is_successful', 'payment_method')
    search_fields = ('user__username', 'payment_id')
    date_hierarchy = 'payment_date'
