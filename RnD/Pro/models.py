from django.db import models
from .storage import OverwriteStorage

class Client(models.Model):  
    User_Name = models.CharField(max_length=20) 
    Email = models.CharField(max_length=40) 
    Password  = models.CharField(max_length=20)  
    
    def __str__(self):
        return self.User_Name

class Med_file(models.Model):
    filename = models.CharField(max_length=200)
    mediafile = models.FileField(storage=OverwriteStorage())

    def __str__(self):
        return self.filename
