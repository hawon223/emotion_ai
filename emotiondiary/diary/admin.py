from django.contrib import admin
from .models import Diary

@admin.register(Diary)
class DiaryAdmin(admin.ModelAdmin):
    list_display = ('user', 'title', 'date', 'label', 'score')  # 관리자 화면에 보여줄 필드
    list_filter = ('user', 'label', 'date')  # 필터 기능 추가
    search_fields = ('title', 'text', 'user__username')  # 검색 기능 추가
