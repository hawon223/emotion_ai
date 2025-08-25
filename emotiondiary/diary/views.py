from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.db.models import Q
from .models import Diary  # 일기 모델
import matplotlib.pyplot as plt
import io
import urllib, base64
import matplotlib
matplotlib.use('Agg')  # 서버에서 GUI 없이 이미지 생성
from matplotlib import font_manager, rc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import pipeline
from openai import OpenAI
import os
from dotenv import load_dotenv
from functools import lru_cache


# 환경변수 로드 및 OpenAI 클라이언트 생성
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 한글 폰트 설정 (Windows 기준)
font_path = "C:/Users/user/Downloads/malgun-gothic/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 기본 페이지
def home(request):
    return render(request, "diary/home.html")

def diary_list(request):
    return render(request, "diary/diary_list.html")



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


# Hugging Face 감정 분석 모델
pipe = pipeline("text-classification", model="hun3359/klue-bert-base-sentiment")

# 환경변수 로드 및 OpenAI 클라이언트 생성
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 감정 점수 딕셔너리
emotion_scores = {
    "분노": 0, "툴툴대는": 5, "좌절한": 8, "짜증내는": 4,
    "방어적인": 10, "악의적인": 3, "안달하는": 12, "구역질 나는": 1,
    "노여워하는": 2, "성가신": 15, "슬픔": 20, "실망한": 22,
    "비통한": 18, "후회되는": 19, "우울한": 25, "마비된": 30,
    "염세적인": 35, "눈물이 나는": 28, "낙담한": 26, "환멸을 느끼는": 33,
    "불안": 40, "두려운": 42, "스트레스 받는": 38, "취약한": 45,
    "혼란스러운": 48, "당혹스러운": 44, "회의적인": 46, "걱정스러운": 41,
    "조심스러운": 47, "초조한": 43, "상처": 37, "질투하는": 39,
    "배신당한": 36, "고립된": 34, "충격 받은": 32, "가난한 불우한": 31,
    "희생된": 29, "억울한": 27, "괴로워하는": 23, "버려진": 24,
    "당황": 21, "고립된(당황한)": 16, "남의 시선을 의식하는": 17,
    "외로운": 13, "열등감": 11, "죄책감의": 14, "부끄러운": 6,
    "혐오스러운": 7, "한심한": 9, "혼란스러운(당황한)": 49,
    "기쁨": 60, "감사하는": 70, "신뢰하는": 65, "편안한": 68,
    "만족스러운": 75, "흥분": 80, "느긋": 72, "안도": 77,
    "신이 난": 90, "자신하는": 100
}

# LABEL -> 대표 감정 매핑
label_mapping = {
    "LABEL_0": ["분노", "슬픔", "불안", "걱정스러운", "우울한"],  # 부정
    "LABEL_1": ["혼란스러운", "당황", "부끄러운", "죄책감의"],   # 중립
    "LABEL_2": ["기쁨", "자신하는", "편안한", "감사하는"]        # 긍정
}

@login_required
def diary_create(request):
    if request.method == "POST":
        title = request.POST.get("title")
        text = request.POST.get("text")

        # 감정 분석
        result = pipe(text)[0]
        label = result['label'].strip()
        score = emotion_scores.get(label, 50)

        # OpenAI GPT 요약 생성
        summary_prompt = f"다음 일기를 한 문장으로 요약해줘:\n{text}"
        summary = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": summary_prompt}]
        ).choices[0].message.content.strip()

        # OpenAI GPT 공감 생성
        empathy_prompt = f"사용자의 하루: {text}\n너는 따뜻하게 공감해주는 AI 친구야. 한 문장으로 공감해줘."
        empathy = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": empathy_prompt}]
        ).choices[0].message.content.strip()

        # 일기 생성 후 저장
        Diary.objects.create(
            user=request.user,
            title=title,
            text=text,
            label=label,
            score=score,
            summary=summary,
            empathy=empathy
        )

        # 저장 후 diary_list로 redirect
        return redirect("diary_list")

    return render(request, "diary/diary_create.html")

# 일기 목록
@login_required
def diary_list(request):
    if request.user.is_staff:  # 관리자는 모든 일기 조회
        diaries = Diary.objects.all().order_by('-date')
    else:  # 일반 사용자는 자신이 쓴 일기만 조회
        diaries = Diary.objects.filter(user=request.user).order_by('-date')
    return render(request, "diary/diary_list.html", {"diaries": diaries})


@login_required
def diary_edit(request, diary_id):
    diary = get_object_or_404(Diary, pk=diary_id)

    if request.method == "POST":
        diary.title = request.POST.get("title")
        diary.text = request.POST.get("text")# text 필드에 저장
        diary.save()
        return redirect("diary_list")  # 상세페이지 대신 목록으로 이동

    return render(request, "diary/diary_edit.html", {"diary": diary})

@login_required
def diary_delete_confirm(request, diary_id):
    diary = get_object_or_404(Diary, id=diary_id, user=request.user)
    return render(request, "diary/diary_delete_confirm.html", {"diary": diary})

@login_required
def diary_delete(request, diary_id):
    diary = get_object_or_404(Diary, id=diary_id, user=request.user)
    if request.method == "POST":   # 확인 눌렀을 때만 삭제
        diary.delete()
        return redirect("diary_list")
    return redirect("diary_delete_confirm", diary_id=diary.id)


@login_required
def emotion_distribution(request):
    diaries = Diary.objects.filter(user=request.user).order_by("date")

    dates = [d.date for d in diaries]
    scores = [d.score for d in diaries]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, scores, marker='o')
    ax.set_title("감정 점수 변화")
    ax.set_xlabel("날짜")
    ax.set_ylabel("감정 점수(0=부정, 100=긍정)")
    plt.xticks(rotation=45)
    ax.grid(True)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    plt.close()

    return render(request, "diary/emotion_graph.html", {"graph": uri})

# 일기 검색 (제목 + 내용)
def search_diary(request):
    query = request.GET.get('q', '')
    diaries = Diary.objects.filter(text__icontains=query) if query else []

    similar_texts = []
    if diaries:
        # 모든 일기 텍스트 가져오기
        all_texts = [diary.text for diary in Diary.objects.all()]

        if all_texts:
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(all_texts + [query])
            cosine_sim = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
            top_indices = np.argsort(cosine_sim)[::-1][:5]  # 상위 5개 유사 일기
            similar_texts = [all_texts[i] for i in top_indices if cosine_sim[i] > 0]

    context = {
        'query': query,
        'diaries': diaries,
        'similar_texts': similar_texts,
    }
    return render(request, 'search_diary.html', context)


@lru_cache(maxsize=1)
def get_pipeline():
    from transformers import pipeline
    return pipeline("text-classification", model="hun3359/klue-bert-base-sentiment")