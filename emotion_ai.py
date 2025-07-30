from transformers import pipeline
import json
import pandas as pd

with open("data/sample_diary.json", "r", encoding="UTF-8")as f:
    diary_data = json.load(f)

pipe = pipeline("text-classification", model="hun3359/klue-bert-base-sentiment")

for diary in diary_data:
    result = pipe(diary["text"])[0]
    diary["label"] = result["label"]
    
df = pd.DataFrame(diary_data)
df.to_csv("data/sample.csv", index=False, encoding="utf-8 sig")


print("csv 파일 저장")