from transformers import pipeline
import json
import pandas as pd
import os

## 현재 파일 위치 기준으로 한 단계 위 폴더에 있는 data/sample_diary.json 경로를 만드는 코드
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(BASE_DIR, "data", "sample_diary.json")

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
df.to_csv("data/sample.csv", index=False, encoding="utf-8-sig")


print("csv 파일 저장")

## csv 출력하기
df_csv = pd.read_csv("data/sample.csv")
print(df_csv)