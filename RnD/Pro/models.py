from django.db import models
from .storage import OverwriteStorage

class Med_file(models.Model):
    filename = models.CharField(max_length=200)
    mediafile = models.FileField(storage=OverwriteStorage())

    def __str__(self):
        return self.filename
