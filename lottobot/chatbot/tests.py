from django.test import TestCase

   def post(self, request, *args, **kwargs):
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '').strip()

            if not user_message:
                return JsonResponse({'response': '메시지를 입력해주세요.'}, status=400)

            # GPT 시스템 프롬프트 설정
            system_prompt = """
            당신은 로또 번호 추천 챗봇입니다.

            사용자에게 처음 인사할 때는 다음과 같이 안내해주세요:

            안녕하세요! 로또 번호 추천 챗봇입니다.

            두 가지 전략으로 번호를 추천해드릴 수 있습니다:
            1. 자주 당첨된 번호 기반 추천
            2. 잠재력 있는 번호 기반 추천

            원하시는 전략을 선택해주세요!

            번호 추천시에는 인사말 없이 바로 전략 설명과 번호를 알려주세요.
            번호는 반드시 오름차순으로 정렬하여 제시합니다.
            """

            # 대화 히스토리에 사용자 메시지 추가
            self.conversation_history.append({"role": "user", "content": user_message})

            # GPT에 대화 요청
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *self.conversation_history
                ],
                temperature=0.7
            )

            # GPT 응답 분석
            assistant_message = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": assistant_message})

            # 전략 선택 확인 및 번호 추천
            strategy = None
            if "전략 1" in assistant_message or "전략1" in assistant_message or "첫 번째 전략" in assistant_message:
                strategy = 1
                recommended_numbers = get_recommendation(strategy=strategy)
                sorted_numbers = sorted(recommended_numbers)
                lucky_message = self.get_random_lucky_message()
                assistant_message = f"""자주 당첨된 번호 기반 추천 전략을 선택하셨군요.
이 전략에 따라서 번호를 추천해드리겠습니다.

추천 번호: {', '.join(map(str, sorted_numbers))} 입니다.

{lucky_message}"""

            elif "전략 2" in assistant_message or "전략2" in assistant_message or "두 번째 전략" in assistant_message:
                strategy = 2
                recommended_numbers = get_recommendation(strategy=strategy)
                sorted_numbers = sorted(recommended_numbers)
                lucky_message = self.get_random_lucky_message()
                assistant_message = f"""잠재력 있는 번호 기반 추천 전략을 선택하셨군요.
이 전략에 따라서 번호를 추천해드리겠습니다.

추천 번호: {', '.join(map(str, sorted_numbers))} 입니다.

{lucky_message}"""

            return JsonResponse({'response': assistant_message}, status=200)

        except openai.OpenAIError as e:
            return JsonResponse({'response': f'OpenAI API 호출 중 문제가 발생했습니다: {str(e)}'}, status=500)
        except Exception as e:
            return JsonResponse({'response': f'서버 에러: {str(e)}'}, status=500)