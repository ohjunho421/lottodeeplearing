# chatbot/serializers.py
from rest_framework import serializers
from .models import LottoDraw, RecommendationHistory

class LottoDrawSerializer(serializers.ModelSerializer):
    class Meta:
        model = LottoDraw
        fields = ['id', 'round_no', 'draw_date', 'winning_numbers', 'bonus_number']

class RecommendationHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = RecommendationHistory
        fields = ['id', 'user', 'strategy', 'recommended_numbers', 'created_at', 'draw_no', 'match_count']
        read_only_fields = ['user', 'created_at', 'draw_no', 'match_count']