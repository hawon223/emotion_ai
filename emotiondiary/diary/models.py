from django.db import models
from django.contrib.auth.models import User

class Diary(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=100, default="일기 제목 없음")
    text = models.TextField()
    date = models.DateField(auto_now_add=True)
    label = models.CharField(max_length=50, blank=True)
    score = models.FloatField(null=True, blank=True)
    summary = models.TextField(blank=True)
    empathy = models.TextField(blank=True)
    emotion_scores = models.JSONField(null=True, blank=True)
    emotion = models.CharField(max_length=50, blank=True)  # 새로 추가

    def save(self, *args, **kwargs):
        # emotion_scores가 있으면 가장 높은 점수의 감정으로 emotion 채우기
        if self.emotion_scores:
            self.emotion = max(self.emotion_scores, key=self.emotion_scores.get)
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.user.username} - {self.date}"
