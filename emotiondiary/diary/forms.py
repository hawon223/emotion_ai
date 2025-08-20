from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .forms import DiaryForm

@login_required
def diary_create(request):
    if request.method == "POST":
        form = DiaryForm(request.POST)
        if form.is_valid():
            diary = form.save(commit=False)
            diary.user = request.user  # 작성자 연결
            diary.save()
            return redirect("diary:list")  # 일기 목록 페이지로 이동
    else:
        form = DiaryForm()
    return render(request, "diary/create.html", {"form": form})
