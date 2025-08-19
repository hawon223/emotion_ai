from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Diary

# 회원가입 폼
class SignUpForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")

# 일기 작성 폼
class DiaryForm(forms.ModelForm):
    class Meta:
        model = Diary
        fields = ("text",)  # Diary 모델에서 작성할 필드만 넣기
