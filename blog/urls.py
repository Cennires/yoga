from django.conf import settings
from django.urls import path
from django.contrib.auth import views as auth_views
from . import views
from users import views as users_views
from .views import PostList, PostLatestList, PostDetailView,JuniorList,PostCreateView,post_correct,MiddleList,SeniorList,HistoryList
from django.conf.urls.static import static


urlpatterns = [
    path('', JuniorList.as_view(), name='index'),
    path('junior/', JuniorList.as_view(), name='junior'),
    path('middle/', MiddleList.as_view(), name='middle'),
    path('senior/', SeniorList.as_view(), name='senior'),
    path('history/', HistoryList.as_view(), name='history'),
    path('latest-post/', PostLatestList.as_view(), name='latest_post'),
    path('post/<int:pk>/', PostDetailView.as_view(), name='post-detail'),
    path('post/correct/', post_correct, name='post-correct'),
    path('post/create/', PostCreateView.as_view(), name='post-create'),
    path('register/', users_views.register, name='register'),
    path('profile/', users_views.profile, name='profile'),
    path('profile/<str:post_author>/', users_views.profile_detail, name='show_profile'),
    path('login/', auth_views.LoginView.as_view(template_name='blog/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(template_name='blog/logout.html'), name='logout'),
    path('password-reset/', auth_views.PasswordResetView.as_view(template_name='blog/password_reset.html'), name='password_reset'),
    path('password-reset/done/', auth_views.PasswordResetDoneView.as_view(template_name='blog/password_reset_done.html'), name='password_reset_done'),
    path('password-reset-confirm/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(template_name='blog/password_reset_confirm.html'), name='password_reset_confirm'),
    path('password-reset-complete/', auth_views.PasswordResetCompleteView.as_view(template_name='blog/password_reset_complete.html'), name='password_reset_complete'),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)