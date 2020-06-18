from django import forms
from .models import Med_file

class MediaForm(forms.ModelForm):
    class Meta:
        model = Med_file
        fields = '__all__'