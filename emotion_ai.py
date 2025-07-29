from transformers import pipeline

pipe = pipeline("text-classification", model="hun3359/klue-bert-base-sentiment")

print("오늘의 기분은?: ")
text = str(input())

result = pipe(text)

print(result)