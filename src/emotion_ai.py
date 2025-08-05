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
# df = pd.DataFrame(diary_data)
# df.to_csv(csv_path, index=False, encoding="utf-8-sig")


# print("csv 파일 저장")


## openai api
load_dotenv()

## csv 불러오기
df = pd.read_csv(csv_path, encoding="utf-8-sig")
## 최신 csv 가져오기
one_day_text = df["text"].iloc[-1]
print("오늘의 일기:", one_day_text)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt = f"""오늘 일기를 한 문장으로 요약하고, 공감 답변 만들어줘.
일기:\n{one_day_text}

요약:
공감 답변:
"""

##GPT 호출
response = client.chat.completions.create(
    model= "gpt-4o-mini",
    messages=[{"role":"user","content":prompt}]
)
print(response.choices[0].message.content)