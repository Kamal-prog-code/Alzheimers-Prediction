from django import forms
from .models import Client
from .models import Med_file

class ClientForm(forms.ModelForm):
    class Meta:
        model = Client
        fields = ['User_Name','Email','Password']

class MediaForm(forms.ModelForm):
    class Meta:
        model = Med_file
        fields = '__all__'