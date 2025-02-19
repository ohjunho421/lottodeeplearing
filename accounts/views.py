from rest_framework.views import APIView
from rest_framework import status
from rest_framework.response import Response
from .serializers import UserSerializer
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.permissions import IsAuthenticated


class SignupAPIView(APIView):
    def post(self, request):
        data = request.data
        serializer = UserSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(
                {"message": "가입이 완료되었습니다🎉"}, status=status.HTTP_201_CREATED
            )
        return Response({"message": "가입 에러"}, status=status.HTTP_400_BAD_REQUEST)


class DeleteAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request):
        user = request.user
        user.delete()
        return Response(
            {"message": "회원탈퇴가 완료되었습니다."}, status=status.HTTP_204_NO_CONTENT
        )


class LogoutAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        refresh_token = request.data.get("refresh")
        if refresh_token:
            try:
                token = RefreshToken(refresh_token)
                token.blacklist()
            except Exception as e:
                return Response(
                    {
                        "message": "로그아웃에 실패했습니다",
                        "error": str(e),
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )
        return Response(
            {
                "message": "로그아웃이 완료되었습니다!",
            },
            status=status.HTTP_204_NO_CONTENT,
        )
