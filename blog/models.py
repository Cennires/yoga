from django.db import models
from django.http import request
from django.utils import timezone
from django.urls import reverse
from django.contrib.auth.models import User
from PIL import Image
import numpy as np

CHOICE=((1,"Junior"),(2,'Middle'),(3,'Senior'))


class Yoga(models.Model):
    img = models.ImageField(upload_to='yoga', default='default.jpg')
    level = models.IntegerField(choices=CHOICE)
    title = models.CharField(max_length=128,default='shushi')

    def get_absolute_url(self):
        return reverse('post-detail',kwargs={'pk':self.pk})

    def __str__(self):
        return self.title


goods = ['seat','mountain','standing_start','treebywind','walking_stick',
         'buttocks','crocodile','door','down_dog','forward_bending',
         'grinding','half_moon3','low_bow_stance','shushi','half_moon1',
         'half_mooon2','boat','dance','desk','monkey','single_bridge',
         'up_dog','upside_angle.jpg','Vbalance','warrior_back','wheel','warrior1','warrior2','warrior3']
tmp = zip(goods,goods)


class Post(models.Model):
    title = models.CharField(max_length=150,choices = list(tmp))
    image = models.ImageField(upload_to='upload_pics', default='')
    date_published = models.DateTimeField(default=timezone.now)
    author = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return f'{self.author.username} photo'
    # 自动跳转！
    def get_absolute_url(self):
        return reverse('post-detail', kwargs={'pk' : self.pk})


class DownYoga(models.Model):
    image = models.ImageField(upload_to='reback',default='default.jpg')
    jsonFile = models.FileField(upload_to='file',default='')
    upPost = models.ForeignKey(Post,on_delete=models.CASCADE)
    content = models.TextField(max_length=1024,default='perfect posture!')
    author = models.ForeignKey(User, on_delete=models.CASCADE)