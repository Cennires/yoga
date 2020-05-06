import os
from PIL import Image
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.urls import reverse
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.messages.views import SuccessMessageMixin
from django.utils import timezone
from .models import Post,Yoga,DownYoga
from .func import func
import time
import cv2
import requests
import base64

class Joint(object):
    __circle_list = []

    def __init__(self, dic):
        self.dic = dic

    def draw_line(self, img):
        # nose ---> neck
        cv2.line(img, (int(self.dic['nose']['x']), int(self.dic['nose']['y'])),
                 (int(self.dic['neck']['x']), int(self.dic['neck']['y'])), (0, 255, 0), 2)
        # nose ---> top_head
        cv2.line(img, (int(self.dic['nose']['x']), int(self.dic['nose']['y'])),
                 (int(self.dic['top_head']['x']), int(self.dic['top_head']['y'])), (0, 255, 0), 2)
        # left_eye ---> left_ear
        cv2.line(img, (int(self.dic['left_eye']['x']), int(self.dic['left_eye']['y'])),
                 (int(self.dic['left_ear']['x']), int(self.dic['left_ear']['y'])), (0, 255, 0), 2)
        # right_eye ---> right_ear
        cv2.line(img, (int(self.dic['right_eye']['x']), int(self.dic['right_eye']['y'])),
                 (int(self.dic['right_ear']['x']), int(self.dic['right_ear']['y'])), (0, 255, 0), 2)
        # right_mouth_corner ---> nose
        cv2.line(img, (int(self.dic['right_mouth_corner']['x']), int(self.dic['right_mouth_corner']['y'])),
                 (int(self.dic['nose']['x']), int(self.dic['nose']['y'])), (0, 255, 0), 2)
        # left_mouth_corner ---> nose
        cv2.line(img, (int(self.dic['left_mouth_corner']['x']), int(self.dic['left_mouth_corner']['y'])),
                 (int(self.dic['nose']['x']), int(self.dic['nose']['y'])), (0, 255, 0), 2)
        # neck --> left_shoulder
        cv2.line(img, (int(self.dic['neck']['x']), int(self.dic['neck']['y'])),
                 (int(self.dic['left_shoulder']['x']), int(self.dic['left_shoulder']['y'])), (0, 255, 0), 2)
        # neck --> right_shoulder
        cv2.line(img, (int(self.dic['neck']['x']), int(self.dic['neck']['y'])),
                 (int(self.dic['right_shoulder']['x']), int(self.dic['right_shoulder']['y'])), (0, 255, 0), 2)
        # left_shoulder --> left_elbow
        cv2.line(img, (int(self.dic['left_shoulder']['x']), int(self.dic['left_shoulder']['y'])),
                 (int(self.dic['left_elbow']['x']), int(self.dic['left_elbow']['y'])), (0, 255, 0), 2)
        # left_elbow --> left_wrist
        cv2.line(img, (int(self.dic['left_elbow']['x']), int(self.dic['left_elbow']['y'])),
                 (int(self.dic['left_wrist']['x']), int(self.dic['left_wrist']['y'])), (0, 255, 0), 2)
        # right_shoulder --> right_elbow
        cv2.line(img, (int(self.dic['right_shoulder']['x']), int(self.dic['right_shoulder']['y'])),
                 (int(self.dic['right_elbow']['x']), int(self.dic['right_elbow']['y'])), (0, 255, 0), 2)
        # right_elbow --> right_wrist
        cv2.line(img, (int(self.dic['right_elbow']['x']), int(self.dic['right_elbow']['y'])),
                 (int(self.dic['right_wrist']['x']), int(self.dic['right_wrist']['y'])), (0, 255, 0), 2)
        # neck --> left_hip
        cv2.line(img, (int(self.dic['neck']['x']), int(self.dic['neck']['y'])),
                 (int(self.dic['left_hip']['x']), int(self.dic['left_hip']['y'])), (0, 255, 0), 2)
        # neck --> right_hip
        cv2.line(img, (int(self.dic['neck']['x']), int(self.dic['neck']['y'])),
                 (int(self.dic['right_hip']['x']), int(self.dic['right_hip']['y'])), (0, 255, 0), 2)
        # left_hip --> left_knee
        cv2.line(img, (int(self.dic['left_hip']['x']), int(self.dic['left_hip']['y'])),
                 (int(self.dic['left_knee']['x']), int(self.dic['left_knee']['y'])), (0, 255, 0), 2)
        # right_hip --> right_knee
        cv2.line(img, (int(self.dic['right_hip']['x']), int(self.dic['right_hip']['y'])),
                 (int(self.dic['right_knee']['x']), int(self.dic['right_knee']['y'])), (0, 255, 0), 2)
        # left_knee --> left_ankle
        cv2.line(img, (int(self.dic['left_knee']['x']), int(self.dic['left_knee']['y'])),
                 (int(self.dic['left_ankle']['x']), int(self.dic['left_ankle']['y'])), (0, 255, 0), 2)
        # right_knee --> right_ankle
        cv2.line(img, (int(self.dic['right_knee']['x']), int(self.dic['right_knee']['y'])),
                 (int(self.dic['right_ankle']['x']), int(self.dic['right_ankle']['y'])), (0, 255, 0), 2)

    def xunhun(self, img):
        im1 = cv2.imread(img, cv2.IMREAD_COLOR)
        # im2 = cv2.resize(im1, (500,900), interpolation=cv2.INTER_CUBIC)

        for i in self.dic:
            cv2.circle(im1, (int(self.dic[i]['x']), int(self.dic[i]['y'])), 5, (0, 255, 0), -1)

        self.draw_line(im1)
        return im1

import json
def post_correct(request):
    # 表示最近上传的图片
    post = Post.objects.last()
    request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/body_analysis"
    filename = post.image.path
    # 二进制方式打开图片文件
    f = open(filename, 'rb')
    img = base64.b64encode(f.read())
    params = {"image": img}
    access_token = "24.85ff7b63540069f746a3a4710c353a88.2592000.1591279549.282335-19733771"
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)
    # 描绘肢体节点向量
    jo = Joint(response.json()['person_info'][0]['body_parts'])
    im1 = jo.xunhun(filename)
    # 保存图片
    cv2.imwrite('media/reback/{}.jpg'.format(post.title), im1)
    print('-------------------------------')
    with open("media/file/{}.json".format(post.title), "w") as fp:
        fp.write(json.dumps(response.json(), indent=4))
    # 生成结果图片,指明路径
    downYoga = DownYoga.objects.create(
        image = 'reback/{}.jpg'.format(post.title),
        jsonFile = 'file/{}.json'.format(post.title),
        upPost = post
    )
    # 评价算法
    feedback = func(post.title)
    return render(request, 'blog/post_correct.html',{'feedback': feedback,'path': downYoga.image.url})
