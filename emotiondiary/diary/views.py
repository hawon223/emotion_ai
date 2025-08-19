from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from .models import Diary  # 일기 모델

# 기본 페이지
def home(request):
    return render(request, "diary/home.html")

def diary_list(request):
    return render(request, "diary/diary_list.html")

def diary_create(request):
    return render(request, "diary/diary_create.html")



# 회원가입
def signup(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("login")
    else:
        form = UserCreationForm()
    return render(request, "signup.html", {"form": form})

# 일기 작성
@login_required
def diary_create(request):
    if request.method == "POST":
        title = request.POST.get("title")
        content = request.POST.get("content")
        Diary.objects.create(user=request.user, title=title, content=content)
        return redirect("diary_list")
    return render(request, "diary/diary_create.html")

# 일기 목록
@login_required
def diary_list(request):
    diaries = Diary.objects.filter(user=request.user).order_by("-date")
    return render(request, "diary/diary_list.html", {"diaries": diaries})

# 감정 그래프
@login_required
def emotion_distribution(request):
    # 예시: emotion 필드가 있다고 가정
    diaries = Diary.objects.filter(user=request.user)
    emotion_count = {}
    for d in diaries:
        emotion_count[d.emotion] = emotion_count.get(d.emotion, 0) + 1
    return render(request, "emotion_graph.html", {"emotion_count": emotion_count})

# 일기 검색
@login_required
def search_diary(request):
    query = request.GET.get("q")
    if query:  # query가 None이 아닌 경우만 필터링
        diaries = Diary.objects.filter(user=request.user, content__icontains=query)
    else:
        diaries = Diary.objects.none()  # 검색어 없으면 빈 쿼리셋

    return render(request, "diary/diary_list.html", {"diaries": diaries, "query": query})

