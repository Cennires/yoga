from django import forms
from .models import Post,Yoga,DownYoga
from django.forms import models as models_fields


CHOICE=((1,"Junior"),(2,'Middle'),(3,'Senior'))


class YogaCreategForm(forms.Form):
    level_form = models_fields.ChoiceField(choices=[(1,"Junior"),(2,'Middle'),(3,'Senior')])
    title_form = models_fields.ModelChoiceField(queryset=Yoga.objects.all(),
                                                empty_label='请选择对应瑜伽动作名称',
                                                to_field_name='title')