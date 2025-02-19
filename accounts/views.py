from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import SignupSerializer
from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.contrib import messages
from .forms import CustomUserCreationForm



def register_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('main')
        else:
            messages.error(request, '회원가입 중 오류가 발생했습니다.')
    else:
        form = CustomUserCreationForm()
    return render(request, 'registration/register.html', {'form': form})




class SignupAPIView(APIView):
    permission_classes = [AllowAny]
    def post(self, request):
        serializer = SignupSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            user = serializer.save()
            return Response(
                {
                    "msg": "signup 성공",
                    "ID": user.username,
                },
                status=status.HTTP_201_CREATED
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
                        "message": "로그아웃 실패",
                        "error": str(e),
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
        return Response(
            {
                "message": "로그아웃 완료!",
            },
            status=status.HTTP_204_NO_CONTENT
        )


class DeleteAPIView(APIView):
    permission_classes = [IsAuthenticated]
    
    def delete(self, request):
        request.user.delete()
        return Response(
            {
                "msg": "회원탈퇴 완료"
            },
            status=status.HTTP_204_NO_CONTENT
        )



