from transformers import pipeline
import json
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv

## 현재 파일 위치 기준으로 한 단계 위 폴더에 있는 data/sample_diary.json 경로를 만드는 코드
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(BASE_DIR, "data", "sample_diary.json")
csv_path = os.path.join(BASE_DIR, "data", "sample.csv")

## 예외처리
try:
    ## json 파일열어서 읽기 
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


# print("csv 파일 저장")


## openai api
load_dotenv()

## csv 불러오기
df = pd.read_csv(csv_path, encoding="utf-8-sig")
## 최신 csv 가져오기
today = df.iloc[-1]
print(today["date"], today["text"])

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

text = today["text"]
## 요약
summary_prompt = f"다음 일기를 한 문장으로 요약해줘: \n{text}"

## GPT 호출
summary = client.chat.completions.create(
    model= "gpt-4o-mini",
    messages=[{"role":"user","content":summary_prompt}]
).choices[0].message.content.strip()


## 공감
empathy_prompt = f"사용자의 하루: {text}\n너는 따뜻하게 공감해주는 AI 친구야. 한 문장으로 공감해줘."

## GPT 호출
empathy = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [{"role":"user","content":empathy_prompt}]
).choices[0].message.content.strip()

print("요약: ", summary)
print("공감: ", empathy)