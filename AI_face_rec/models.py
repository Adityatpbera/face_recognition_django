from django.db import models

# Create your models here.
class FaceRecognization(models.Model):
    name=models.CharField(max_length=200,null=False,default=False)
    pan=models.ImageField(upload_to='pan',null=False)
    selfie=models.ImageField(upload_to="selfie",null=False)

    def __str__(self):
        return self.name
