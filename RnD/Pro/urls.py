from django.urls import path
from . import views

urlpatterns = [
    path('',views.index,name="index"),
    path('home',views.home,name="home"),
    path('ad/',views.ad,name="ad"),
    path('visual/',views.visualize_data,name="visual"),
    path('signup/',views.signup,name="signup"),
    path('signin/',views.signin,name="signin"),
    path('index/',views.index,name="index"),
]
