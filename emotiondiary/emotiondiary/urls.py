from django.contrib import admin
from django.urls import path, include
from diary import views
from django.contrib.auth import views as auth_views 

urlpatterns = [
    path("admin/", admin.site.urls),

    # 기본 페이지
    path("", views.home, name="home"),

    # 일기 관련
    path("diary/create/", views.diary_create, name="diary_create"),
    path("diary/list/", views.diary_list, name="diary_list"),
    path("diary/graph/", views.emotion_distribution, name="emotion_graph"),
    path("diary/search/", views.search_diary, name="search_diary"),
    path("diary/edit/<int:diary_id>/", views.diary_edit, name="diary_edit"),
    path("diary/delete/<int:diary_id>/", views.diary_delete_confirm, name="diary_delete_confirm"),
    path("diary/delete/<int:diary_id>/confirm/", views.diary_delete, name="diary_delete"),






    # 회원가입
    path("signup/", views.signup, name="signup"),

    # Django 기본 인증 URL 사용 (로그인, 로그아웃, 비밀번호 변경 등)
    path("accounts/", include("django.contrib.auth.urls")),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
]
