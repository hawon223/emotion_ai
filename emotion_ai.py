from transformers import pipeline

pipe = pipeline("text-classification", model="hun3359/klue-bert-base-sentiment")

text = "오늘은 기분이 너무 좋아"

result = pipe(text)

print(result)