from django.db import models
from django.contrib.auth.models import User

class Diary(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    text = models.TextField()
    date = models.DateField(auto_now_add=True)
    label = models.CharField(max_length=50, blank=True)
    score = models.FloatField(null=True, blank=True)
    summary = models.TextField(blank=True)
    empathy = models.TextField(blank=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.date}"
