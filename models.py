from django.db import models

# Create your models here.
class criminal_details(models.Model):
   
    Name = models.CharField(max_length = 20, default="")
    images = models.CharField(max_length = 100000000, default="", null = True)