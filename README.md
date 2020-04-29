# yoga
视频版本
一、数据库设计：

class Yoga(models.Model):
    img = models.ImageField(upload_to='yoga', default='default.jpg')
    level = models.IntegerField(choices=CHOICE)
    title = models.CharField(max_length=128,default='shushi')

    def get_absolute_url(self):
        return reverse('post-detail',kwargs={'pk':self.pk})
        
        
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
        
二、主要功能：

用户登录
用户注册
更换头像
上传照片
保存照片
显示返回图片信息
显示返回建议信息
