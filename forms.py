from django import forms
from .models import *

class UploadFileForm(forms.Form):
    video = forms.FileField()
    CHOICES = [
        ('101', 'Option 1'),
        ('102', 'Option 2'),
        ('103', 'Option 3'),
        ('104', 'Option 4'),
    ]
    modelChoice = [
        ('accident', 'Option 1'),
        ('traffic', 'Option 2'),
    ]
    cam = forms.ChoiceField(
        widget=forms.RadioSelect,
        choices=CHOICES, 
    )
    model = forms.ChoiceField(
        widget=forms.RadioSelect,
        choices=modelChoice, 
    )

class CriminalForm(forms.ModelForm):
    class Meta:
        model = criminal_details
        fields = ['Name',"images"]
    def clean_images(self):
        return self.cleaned_data['images'] or None