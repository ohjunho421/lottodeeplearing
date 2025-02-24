# chatbot/serializers.py
from rest_framework import serializers
from .models import LottoDraw, Recommendation

class LottoDrawSerializer(serializers.ModelSerializer):
    class Meta:
        model = LottoDraw
        fields = ['id', 'round_no', 'draw_date', 'winning_numbers', 'bonus_number']

class RecommendationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Recommendation
        fields = [
            'recommendation_date', 'strategy', 'numbers', 
            'is_checked', 'is_won', 'draw_round', 'draw_date',
            'matched_count', 'has_bonus', 'rank'
        ]