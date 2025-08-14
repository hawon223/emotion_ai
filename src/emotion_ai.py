from transformers import pipeline
import json
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# 기본 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
csv_path = os.path.join(BASE_DIR, "data", "sample.csv")

    
# OPENAI API
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    
# Hugging Face pipeline 모델 불러오기 (감정분석용)
pipe = pipeline("text-classification", model="hun3359/klue-bert-base-sentiment")

# 감정 점수 정의
emotion_scores = {
    "분노": 0,          # 가장 부정적
    "툴툴대는": 5,
    "좌절한": 8,
    "짜증내는": 4,
    "방어적인": 10,
    "악의적인": 3,
    "안달하는": 12,
    "구역질 나는": 1,
    "노여워하는": 2,
    "성가신": 15,
    "슬픔": 20,
    "실망한": 22,
    "비통한": 18,
    "후회되는": 19,
    "우울한": 25,
    "마비된": 30,
    "염세적인": 35,
    "눈물이 나는": 28,
    "낙담한": 26,
    "환멸을 느끼는": 33,
    "불안": 40,
    "두려운": 42,
    "스트레스 받는": 38,
    "취약한": 45,
    "혼란스러운": 48,
    "당혹스러운": 44,
    "회의적인": 46,
    "걱정스러운": 41,
    "조심스러운": 47,
    "초조한": 43,
    "상처": 37,
    "질투하는": 39,
    "배신당한": 36,
    "고립된": 34,
    "충격 받은": 32,
    "가난한 불우한": 31,
    "희생된": 29,
    "억울한": 27,
    "괴로워하는": 23,
    "버려진": 24,
    "당황": 21,
    "고립된(당황한)": 16,
    "남의 시선을 의식하는": 17,
    "외로운": 13,
    "열등감": 11,
    "죄책감의": 14,
    "부끄러운": 6,
    "혐오스러운": 7,
    "한심한": 9,
    "혼란스러운(당황한)": 49,   # 최소 부정

    "기쁨": 60,          # 긍정 시작
    "감사하는": 70,
    "신뢰하는": 65,
    "편안한": 68,
    "만족스러운": 75,
    "흥분": 80,
    "느긋": 72,
    "안도": 77,
    "신이 난": 90,
    "자신하는": 100      # 가장 긍정적
}

# CSV 로드 + 날짜 변환 + 정렬 함수
def load_and_prepare_data():
    if os.path.exists(csv_path) and os.stat(csv_path).st_size > 0:
        df = pd.read_csv(csv_path, encoding="utf-8")
        df["date"] = pd.to_datetime(df["date"])  # 날짜 컬럼 변환
        df["score"] = pd.to_numeric(df["score"], errors='coerce')  # 숫자 변환
        df = df.sort_values(by="date")
        return df
    return pd.DataFrame()

# 일기 분석 함수
def analyze_diary(text):
    ## 감정분석
    result = pipe(text)[0]
    label = result['label'].strip()
    score = emotion_scores.get(label, 50)

    ## 요약 생성 (GPT API 호출)
    summary_prompt = f"다음 일기를 한 문장으로 요약해줘: \n{text}"

    ## GPT 호출
    summary = client.chat.completions.create(
        model= "gpt-4o-mini",
        messages=[{"role":"user","content":summary_prompt}]
    ).choices[0].message.content.strip()


    ## 공감 생성 (GPT API 호출)
    empathy_prompt = f"사용자의 하루: {text}\n너는 따뜻하게 공감해주는 AI 친구야. 한 문장으로 공감해줘."

    ## GPT 호출
    empathy = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [{"role":"user","content":empathy_prompt}]
    ).choices[0].message.content.strip()

    ## csv에 저장
    new_entry = {
        "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "text": text,
        "label": label,
        "score": score,
        "summary": summary,
        "empathy": empathy
    }
 
    try:
        if os.path.exists(csv_path) and os.stat(csv_path).st_size > 0:
            df_existing = pd.read_csv(csv_path, encoding="utf-8")
        else:
            df_existing = pd.DataFrame()
    except pd.errors.EmptyDataError:
        df_existing = pd.DataFrame()

    # 새 데이터 추가
    df_existing = pd.concat([df_existing, pd.DataFrame([new_entry])], ignore_index=True)

    # CSV 저장
    df_existing.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    plot_emotion_distribution()
    plot_emotion_trend()
    
    return f"감정: {label}\n점수: {score}\n요약: {summary}\n공감: {empathy}"


# 감정 분포 그래프
def plot_emotion_distribution():
    df = load_and_prepare_data()
    if df.empty:
        return None
    
    font_path = r"C:\Users\user\Downloads\malgun-gothic\malgun.ttf"
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)

    fig, ax = plt.subplots(figsize=(10, 5))
    df["label"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("감정 분포")
    ax.set_xlabel("감정")
    ax.set_ylabel("빈도")
    plt.tight_layout()
    
    return fig  # Figure 객체 반환

def plot_emotion_trend():
    df = load_and_prepare_data()
    if df.empty:
        return None
    
    font_path = r"C:\Users\user\Downloads\malgun-gothic\malgun.ttf"
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["date"], df["score"], marker='o')
    ax.set_title("감정 점수 변화")
    ax.set_xlabel("날짜")
    ax.set_ylabel("감정 점수(0=부정, 100=긍정)")
    plt.xticks(rotation=45)
    ax.grid(True)
    plt.tight_layout()
    
    return fig  # Figure 객체 반환


# 임베딩 모델 로드
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

## 임베딩 생성 함수
def get_embedding(text):
    return embedder.encode(text, convert_to_numpy=True)

## FAISS 인덱스 구축
def build_faiss_index(texts):
    embeddings = [get_embedding(t) for t in texts]
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

def find_similar(text, index, texts, top_k = 3):
    query_vec = get_embedding(text).reshape(1, -1)
    distances, indices = index.search(query_vec, top_k)
    return [texts[i] for i in indices[0]]

def search_similar_diary(text):
    df = load_and_prepare_data()
    if df.empty:
        return "일기가 없습니다."
    texts = df["text"].tolist()
    index = build_faiss_index(texts)
    similar_texts = find_similar(text, index, texts)
    return "\n\n".join(similar_texts)

st.title("📓 감정일기 분석 및 시각화")

# 입력
diary_input = st.text_area("오늘의 일기를 입력하세요:")

# 분석
if st.button("일기 분석 및 저장"):
    if diary_input.strip():
        result = analyze_diary(diary_input)
        st.success(result)
    else:
        st.warning("일기를 입력하세요.")

# 감정 분포
if st.button("감정 분포 그래프 보기"):
    fig = plot_emotion_distribution()
    if fig is not None:
        st.pyplot(fig)
    else:
        st.warning("데이터가 없습니다.")

# 감정 추이
if st.button("감정 점수 추이 그래프 보기"):
    fig = plot_emotion_trend()
    if fig is not None:
        st.pyplot(fig)
    else:
        st.warning("데이터가 없습니다.")


# 유사 일기 검색
if st.button("유사한 과거 일기 찾기"):
    if diary_input.strip():
        similar = search_similar_diary(diary_input)
        st.info(similar)
    else:
        st.warning("일기를 입력하세요.")