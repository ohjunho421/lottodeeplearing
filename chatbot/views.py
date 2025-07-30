# views.py
import json
import logging
import random
import google.generativeai as genai
from django.http import JsonResponse
from django.views import View
from django.views.generic import TemplateView
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.middleware.csrf import get_token
from django.conf import settings
from django.contrib.auth.decorators import login_required
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated

from .models import Recommendation, LottoDraw
from .serializers import RecommendationSerializer
from .services import get_recommendation, AdvancedLottoPredictor

logger = logging.getLogger(__name__)

# views.py
class ModelMetricsView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        try:
            predictor = AdvancedLottoPredictor()
            success, results = predictor.train_models()
            
            if success and results:
                return Response({
                    'xgb_train_r2': results['xgb_train_r2'],
                    'xgb_test_r2': results['xgb_test_r2'],
                    'rf_train_r2': results['rf_train_r2'],
                    'rf_test_r2': results['rf_test_r2']
                })
            return Response({'error': 'Failed to train models'}, status=400)
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return Response({
                'error': str(e),
                'details': error_details
            }, status=400)

class ChatbotHomeView(TemplateView):
    """View for rendering the chatbot home page"""
    template_name = 'chatbot/home.html'

class CSRFTokenView(View):
    """View for getting CSRF token"""
    def get(self, request, *args, **kwargs):
        csrf_token = get_token(request)
        return JsonResponse({'csrfToken': csrf_token})



@method_decorator(csrf_exempt, name='dispatch')
class ChatAPIView(View):
    """Main API view for handling chat interactions"""
    
    def __init__(self):
        super().__init__()
        self.conversation_history = []
        self.lucky_messages = [
            "행운이 함께하길 바랍니다! ✨",
            "이번에는 좋은 결과가 있기를 기원합니다! 🍀",
            "당신의 꿈이 이루어지길 바라며 이 번호들을 선택했습니다! 🌟",
            "이 번호들과 함께 큰 행운이 찾아오길 바랍니다! 🎯",
            "당첨의 기쁨을 누리실 수 있기를 진심으로 응원합니다! ⭐",
            "이번 주는 특별한 행운이 함께하길 바랍니다! 🌈",
            "당신의 성공을 기원하며 이 번호들을 추천해드립니다! 💫",
            "모든 세트에 행운이 가득하길 기원합니다! 🌠",
            "이 번호들이 당신에게 좋은 기운을 가져다주길 바랍니다! 🎊",
            "당첨의 행운이 함께하시길 진심으로 바랍니다! 💫"
        ]

    def _get_gpt_response(self, user_message):
        """Get response from Gemini 2.5 Pro API"""
        try:
            # Gemini API 설정
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-2.5-pro')
            
            system_prompt = """
안녕하세요! 로또 번호 추천 챗봇입니다.

세 가지 전략으로 번호를 추천해드릴 수 있습니다:

전략 1. 평균적으로 자주 당첨된 번호 기반 추천
전략 2. 앞으로 많이 나올 잠재력 있는 번호 기반 추천
전략 3. ML(머신러닝) 모델 기반 추천

원하시는 전략을 선택해주세요! 

최대 5세트까지 추천가능합니다.
(예: "전략1로 3세트 추천해주세요" 또는 "전략1 2세트, 전략2 2세트, 전략3 1세트 추천해주세요")
"""
            
            # 대화 기록을 포함한 전체 프롬프트 구성
            conversation_text = ""
            for msg in self.conversation_history:
                role = "사용자" if msg["role"] == "user" else "챗봇"
                conversation_text += f"{role}: {msg['content']}\n"
            
            full_prompt = f"{system_prompt}\n\n대화 기록:\n{conversation_text}\n사용자: {user_message}"
            
            response = model.generate_content(full_prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini Error: {str(e)}")
            raise Exception("죄송합니다. 서버 연결에 문제가 발생했습니다.")

    def _process_strategy_counts(self, user_message):
        """Parse strategy counts from user message with enhanced flexibility"""
        strategy_counts = {'1': 0, '2': 0, '3': 0}
        
        try:
            message = user_message.lower()
            exceed_limit = False  # 제한 초과 여부를 추적하는 플래그
            
            # 전처리
            processed_msg = message.replace(',', ' ').replace('그리고', ' ').replace('와', ' ').replace('과', ' ')
            
            # 전략1 패턴들
            strategy1_patterns = ['전략1', '1번전략', '1번 전략', '1전략', '전략 1', '1 전략', '1번']
            # 전략2 패턴들  
            strategy2_patterns = ['전략2', '2번전략', '2번 전략', '2전략', '전략 2', '2 전략', '2번']
            # 전략3 패턴들
            strategy3_patterns = ['전략3', '3번전략', '3번 전략', '3전략', '전략 3', '3 전략', '3번']
            
            # 메시지 전체를 검색하여 모든 전략 패턴 찾기
            # 전략1 검색
            for pattern in strategy1_patterns:
                if pattern in processed_msg:
                    # 패턴 주변에서 숫자 찾기
                    idx = processed_msg.find(pattern)
                    # 패턴 뒤쪽 문자열 추출
                    after_pattern = processed_msg[idx + len(pattern):].strip()
                    
                    # 패턴에 붙어있는 숫자 확인 (예: "1번전략2개")
                    next_char_idx = idx + len(pattern)
                    if next_char_idx < len(processed_msg) and processed_msg[next_char_idx].isdigit():
                        # 숫자가 붙어있는 경우
                        digit_part = ""
                        i = next_char_idx
                        while i < len(processed_msg) and (processed_msg[i].isdigit() or processed_msg[i] in [' ', '개', '세', '트', 's', 'e', 't']):
                            digit_part += processed_msg[i]
                            i += 1
                        
                        digit = ''.join(filter(str.isdigit, digit_part))
                        if digit:
                            count = int(digit)
                            if count > 5:
                                exceed_limit = True
                            else:
                                strategy_counts['1'] = count
                    else:
                        # 패턴 뒤에 공백으로 분리된 숫자 찾기
                        for word in after_pattern.split():
                            if word.isdigit():
                                count = int(word)
                                if count > 5:
                                    exceed_limit = True
                                else:
                                    strategy_counts['1'] = count
                                break
                            elif any(unit in word for unit in ['개', '세트', '셋트', 'set']):
                                digit = ''.join(filter(str.isdigit, word))
                                if digit:
                                    count = int(digit)
                                    if count > 5:
                                        exceed_limit = True
                                    else:
                                        strategy_counts['1'] = count
                                    break
            
            # 전략2 검색 - 위와 동일한 로직
            for pattern in strategy2_patterns:
                if pattern in processed_msg:
                    # 패턴 주변에서 숫자 찾기
                    idx = processed_msg.find(pattern)
                    # 패턴 뒤쪽 문자열 추출
                    after_pattern = processed_msg[idx + len(pattern):].strip()
                    
                    # 패턴에 붙어있는 숫자 확인 (예: "2번전략3개")
                    next_char_idx = idx + len(pattern)
                    if next_char_idx < len(processed_msg) and processed_msg[next_char_idx].isdigit():
                        # 숫자가 붙어있는 경우
                        digit_part = ""
                        i = next_char_idx
                        while i < len(processed_msg) and (processed_msg[i].isdigit() or processed_msg[i] in [' ', '개', '세', '트', 's', 'e', 't']):
                            digit_part += processed_msg[i]
                            i += 1
                        
                        digit = ''.join(filter(str.isdigit, digit_part))
                        if digit:
                            count = int(digit)
                            if count > 5:
                                exceed_limit = True
                            else:
                                strategy_counts['2'] = count
                    else:
                        # 패턴 뒤에 공백으로 분리된 숫자 찾기
                        for word in after_pattern.split():
                            if word.isdigit():
                                count = int(word)
                                if count > 5:
                                    exceed_limit = True
                                else:
                                    strategy_counts['2'] = count
                                break
                            elif any(unit in word for unit in ['개', '세트', '셋트', 'set']):
                                digit = ''.join(filter(str.isdigit, word))
                                if digit:
                                    count = int(digit)
                                    if count > 5:
                                        exceed_limit = True
                                    else:
                                        strategy_counts['2'] = count
                                    break
            
            # 전략3 검색 - 위와 동일한 로직
            for pattern in strategy3_patterns:
                if pattern in processed_msg:
                    # 패턴 주변에서 숫자 찾기
                    idx = processed_msg.find(pattern)
                    # 패턴 뒤쪽 문자열 추출
                    after_pattern = processed_msg[idx + len(pattern):].strip()
                    
                    # 패턴에 붙어있는 숫자 확인 (예: "3번전략2개")
                    next_char_idx = idx + len(pattern)
                    if next_char_idx < len(processed_msg) and processed_msg[next_char_idx].isdigit():
                        # 숫자가 붙어있는 경우
                        digit_part = ""
                        i = next_char_idx
                        while i < len(processed_msg) and (processed_msg[i].isdigit() or processed_msg[i] in [' ', '개', '세', '트', 's', 'e', 't']):
                            digit_part += processed_msg[i]
                            i += 1
                        
                        digit = ''.join(filter(str.isdigit, digit_part))
                        if digit:
                            count = int(digit)
                            if count > 5:
                                exceed_limit = True
                            else:
                                strategy_counts['3'] = count
                    else:
                        # 패턴 뒤에 공백으로 분리된 숫자 찾기
                        for word in after_pattern.split():
                            if word.isdigit():
                                count = int(word)
                                if count > 5:
                                    exceed_limit = True
                                else:
                                    strategy_counts['3'] = count
                                break
                            elif any(unit in word for unit in ['개', '세트', '셋트', 'set']):
                                digit = ''.join(filter(str.isdigit, word))
                                if digit:
                                    count = int(digit)
                                    if count > 5:
                                        exceed_limit = True
                                    else:
                                        strategy_counts['3'] = count
                                    break
            
            # 단순히 "N개 추천해줘" 형태의 요청 처리 (기본값 전략1)
            if sum(strategy_counts.values()) == 0 and not exceed_limit:
                for word in message.split():
                    if any(unit in word for unit in ['개', '세트', '셋트', 'set']):
                        digit = ''.join(filter(str.isdigit, word))
                        if digit:
                            count = int(digit)
                            if count > 5:
                                exceed_limit = True
                            else:
                                strategy_counts['1'] = count
                            break
            
            # 합계 계산 및 검증
            total_sets = sum(int(count) for count in strategy_counts.values())
            if total_sets > 5:
                exceed_limit = True
                logger.warning(f"Total sets {total_sets} exceeds limit")
            
            # 제한 초과 시 빈 딕셔너리 반환하여 오류 메시지 표시
            if exceed_limit:
                return {'1': 0, '2': 0}
            
            logger.info(f"Processed strategy counts: {strategy_counts}")
            logger.info(f"Total sets requested: {total_sets}")
            
            return strategy_counts
        
        except Exception as e:
            logger.error(f"Error processing strategy counts: {e}")
            return strategy_counts

    def _format_recommendations(self, recommendations, strategy_num=None, num_sets=None):
        """Format lottery number recommendations with better readability"""
        # 전략별로 번호를 분류
        strategy1_sets = []
        strategy2_sets = []
        strategy3_sets = []
        
        for strategy, numbers in recommendations:
            if strategy == 1:
                strategy1_sets.append(f"□ {len(strategy1_sets)+1}세트: {', '.join(map(str, numbers))}")
            elif strategy == 2:
                strategy2_sets.append(f"□ {len(strategy2_sets)+1}세트: {', '.join(map(str, numbers))}")
            elif strategy == 3:
                strategy3_sets.append(f"□ {len(strategy3_sets)+1}세트: {', '.join(map(str, numbers))}")
        
        formatted_message = ""
        
        # 전략 1 결과가 있으면 추가
        if strategy1_sets:
            formatted_message += """[전략 1: 자주 당첨된 번호 기반 추천]

====================================

{}

====================================""".format('\n'.join(strategy1_sets))

        # 전략 1과 다른 전략들 사이 구분선 추가
        if strategy1_sets and (strategy2_sets or strategy3_sets):
            formatted_message += "\n\n"

        # 전략 2 결과가 있으면 추가
        if strategy2_sets:
            formatted_message += """[전략 2: 잠재력 있는 번호 기반 추천]

====================================

{}

====================================""".format('\n'.join(strategy2_sets))
        
        # 전략 2와 전략 3 사이 구분선 추가
        if strategy2_sets and strategy3_sets:
            formatted_message += "\n\n"
            
        # 전략 3 결과가 있으면 추가
        if strategy3_sets:
            formatted_message += """[전략 3: ML 모델 기반 추천]

====================================

{}

====================================""".format('\n'.join(strategy3_sets))

        # 행운 메시지 추가
        lucky_message = random.choice(self.lucky_messages)
        formatted_message += f"\n\n▶ {lucky_message}"
        
        return formatted_message

    def post(self, request, *args, **kwargs):
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '').strip()
            logger.info(f"Received message: {user_message}")

            if not user_message:
                return JsonResponse({'response': '메시지를 입력해주세요.'}, status=400)

            # 전략 키워드가 있는 경우 GPT 응답 스킵하고 바로 번호 추천 처리
            if "전략" in user_message.lower():
                try:
                    strategy_counts = self._process_strategy_counts(user_message)
                    total_sets = sum(strategy_counts.values())

                    if total_sets == 0:
                        return JsonResponse({
                            'response': '최대 5세트까지 추천가능합니다.\n세트 수를 정확히 입력해주세요.\n(예: "전략1로 3세트 추천해주세요")'
                        }, status=400)
                    
                    if total_sets > 5:
                        return JsonResponse({
                            'response': '죄송합니다. 최대 5세트까지만 추천 가능합니다.\n전략1, 전략2, 전략3을 조합해서 5세트를 추천해드릴까요?\n(예: "전략1 2세트, 전략2 2세트, 전략3 1세트")'
                        }, status=200)

                    recommendations, error = get_recommendation(strategy_counts)
                    
                    if error:
                        return JsonResponse({'response': error}, status=400)

                    if not recommendations:
                        return JsonResponse({'response': '번호 추천 중 오류가 발생했습니다.'}, status=400)

                    #사용자 인증확인 추가
                    if recommendations and request.user.is_authenticated:
                        try:
                            # 추천 번호 저장
                            for strategy, numbers in recommendations:
                                Recommendation.objects.create(
                                    user=request.user,
                                    strategy=strategy,
                                    numbers=','.join(map(str, sorted(numbers))),
                                    is_checked=False, #당첨 여부확인, 아직확인안함
                                    is_won=False, #당첨 여부표시, 아직 모름
                                    draw_round=None, #비교할 추첨회차, 아직 정보없음 
                                    draw_date=None #비교할 추첨일, 아직정보없음
                                )
                            
                            # 번호 추천 결과 반환
                            response_message = self._format_recommendations(recommendations)
                            return JsonResponse({'response': response_message}, status=200)
                            
                        except Exception as e:
                            logger.error(f"Error saving recommendations: {str(e)}")
                            return JsonResponse({'response': '번호 저장 중 오류가 발생했습니다.'}, status=400)
                    
                except Exception as e:
                    logger.error(f"Error in processing strategy: {str(e)}")
                    return JsonResponse({
                        'response': '번호 추천 처리 중 오류가 발생했습니다.'
                    }, status=400)
            
            # 전략 키워드가 없는 경우만 GPT 응답 처리
            else:
                try:
                    assistant_message = self._get_gpt_response(user_message)
                    return JsonResponse({'response': assistant_message}, status=200)
                except Exception as e:
                    logger.error(f"GPT Error: {str(e)}")
                    return JsonResponse({'response': str(e)}, status=500)

        except json.JSONDecodeError:
            return JsonResponse({'response': '잘못된 요청 형식입니다.'}, status=400)
        except Exception as e:
            logger.error(f"Error in ChatAPIView: {str(e)}")
            return JsonResponse({
                'response': '서버 에러가 발생했습니다. 잠시 후 다시 시도해주세요.'
            }, status=500)
        
# ChatAPIView 클래스 다음에 추가
class HistoryAPIView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get_winning_rank(self, matched_count, has_bonus):
        """당첨 순위 확인"""
        if matched_count == 6:
            return 1  # 1등: 6개 번호 일치
        elif matched_count == 5 and has_bonus:
            return 2  # 2등: 5개 번호 + 보너스 번호 일치
        elif matched_count == 5:
            return 3  # 3등: 5개 번호 일치
        elif matched_count == 4:
            return 4  # 4등: 4개 번호 일치
        elif matched_count == 3:
            return 5  # 5등: 3개 번호 일치
        else:
            return 0  # 낙첨: 2개 이하 일치
    
    def get(self, request):
        from .services import check_winning_numbers
        try:
            # 추천 번호 당첨 여부 확인 및 업데이트
            check_winning_numbers()

            recommendations = Recommendation.objects.filter(user=request.user).order_by('-recommendation_date')
            latest_draw = LottoDraw.objects.order_by('-round_no').first()

            serializer = RecommendationSerializer(recommendations, many=True)
            
            response_data = {
                'recommendations': serializer.data,
                'latest_draw': {
                    'round': latest_draw.round_no if latest_draw else None,
                    'date': latest_draw.draw_date if latest_draw else None,
                    'numbers': latest_draw.winning_numbers if latest_draw else None,
                    'bonus': latest_draw.bonus_number if latest_draw else None
                } if latest_draw else None
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )