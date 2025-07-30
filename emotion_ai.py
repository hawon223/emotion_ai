from transformers import pipeline
import json

with open("data/sample_diary.json", "r", encoding="UTF-8")as f:
    diary_data = json.load(f)

pipe = pipeline("text-classification", model="hun3359/klue-bert-base-sentiment")

for diary in diary_data:
    result = pipe(diary["text"])[0]
    diary["label"] = result["label"]

print(diary_data)