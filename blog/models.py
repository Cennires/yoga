from django.db import models
from django.http import request
from django.utils import timezone
from django.urls import reverse
from django.contrib.auth.models import User
from PIL import Image


class Post(models.Model):
    title = models.CharField(max_length=150)
    image = models.ImageField(upload_to='upload_pics', default='')
    date_published = models.DateTimeField(default=timezone.now)
    author = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return f'{self.author.username} photo'
    # 自动跳转！
    def get_absolute_url(self):
        return reverse('post-detail', kwargs={'pk' : self.pk})


CHOICE=((1,"Junior"),(2,'Middle'),(3,'Senior'))


class Yoga(models.Model):
    img = models.ImageField(upload_to='yoga', default='default.jpg')
    level = models.IntegerField(choices=CHOICE)
    title = models.CharField(max_length=128,default='shushi')

    def get_absolute_url(self):
        return reverse('post-detail',kwargs={'pk':self.pk})


class DownYoga(models.Model):
    image = models.ImageField(upload_to='reback',default='default.jpg')
    jsonFile = models.FileField(upload_to='file',default='')
    upPost = models.ForeignKey(Post,on_delete=models.CASCADE)

