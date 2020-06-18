from django.urls import path
from . import views

urlpatterns = [
    path('',views.home,name="home"),
    path('ad/',views.ad,name="ad"),
    path('visual/',views.visualize_data,name="visual"),
]
