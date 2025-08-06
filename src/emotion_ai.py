from transformers import pipeline
import json
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

## 현재 파일 위치 기준으로 한 단계 위 폴더에 있는 data/sample_diary.json 경로를 만드는 코드(기본경로 설정)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(BASE_DIR, "data", "sample_diary.json")
csv_path = os.path.join(BASE_DIR, "data", "sample.csv")
json_path = os.path.join(BASE_DIR, "data", "week3_summary.json")
csv_summary_path = os.path.join(BASE_DIR, "data", "week3_summary.csv")

## 원본 json 불러오기
try: 
    with open(data_path, "r", encoding="UTF-8")as f:
        diary_data = json.load(f)
        
except FileNotFoundError:
    print("파일이 존재 하지 않습니다")
except json.JSONDecodeError:
    print("형식이 잘못되었습니다")
    
    
## pipeline으로 모델 불러오기
pipe = pipeline("text-classification", model="hun3359/klue-bert-base-sentiment")

## 일기 하나씩 꺼내서 -> 감정 분석 -> 결과 라벨을 JSON에 저장 
for diary in diary_data:
    result = pipe(diary["text"])[0]
    diary["label"] = result["label"]
    
## csv파일로 변환 및 저장
df = pd.DataFrame(diary_data)
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

## csv 불러오기
df = pd.read_csv(csv_path, encoding="utf-8-sig")
## 최신 csv 가져오기
today = df.iloc[-1]
print(today["date"], today["text"])

text = today["text"]
date = today["date"]

## openai api
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

## 요약 프롬프트
summary_prompt = f"다음 일기를 한 문장으로 요약해줘: \n{text}"

## GPT 호출
summary = client.chat.completions.create(
    model= "gpt-4o-mini",
    messages=[{"role":"user","content":summary_prompt}]
).choices[0].message.content.strip()


## 공감 프롬프트
empathy_prompt = f"사용자의 하루: {text}\n너는 따뜻하게 공감해주는 AI 친구야. 한 문장으로 공감해줘."

## GPT 호출
empathy = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [{"role":"user","content":empathy_prompt}]
).choices[0].message.content.strip()

print("요약: ", summary)
print("공감: ", empathy)


## json 불러오기
if os.path.exists(json_path):
    with open(json_path, 'r', encoding="utf-8") as f:
        try:
            summaries = json.load(f)
        except json.JSONDecodeError:
            summaries = []
        
else:
    summaries = []
    
# 오늘 기록이 없으면 추가
if not any(entry["date"] == date for entry in summaries):
    summaries.append({
        "date": date,
        "summary": summary,
        "empathy": empathy
    })
    
## json에 저장
with open(json_path, 'w', encoding="utf-8") as f:
    json.dump(summaries, f, ensure_ascii=False, indent=2)
    
## csv 저장
pd.DataFrame(summaries).to_csv(csv_summary_path, index=False, encoding="utf-8-sig")

print(f"[{date}] 요약 & 공감 저장 완료")


## 점수 
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

df["label"] = df["label"].str.strip()
df["score"] = df["label"].map(emotion_scores)

if "labe" in df.columns:
    df = df.drop(columns=["labe"])

print(df.to_string(index=False))

daily_score = df[["date", "score"]]
print(daily_score)


## 폰트 설정
font_path = r"C:\Users\user\Downloads\malgun-gothic\malgun.ttf"
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

## 시각화
plt.figure(figsize = (10, 5))
plt.plot(daily_score["date"], daily_score["score"], marker = 'o')
plt.title("일별 평균 감정 점수 변화")
plt.xlabel("날짜")
plt.ylabel("감정 점수(0=부정, 100=긍정)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
