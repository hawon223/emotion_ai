from transformers import pipeline
from openai import OpenAI
import os
from dotenv import load_dotenv

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
    
    return label, score, summary, empathy