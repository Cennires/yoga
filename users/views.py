from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.generic import ListView
from django.core.paginator import Paginator
from django.contrib.auth.models import User
from .forms import UserRegistrationForm, UserUpdateForm, ProfileUpdateForm
from blog.models import Post

def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account has been created {username}')
            return redirect('login')
    else:
        form = UserRegistrationForm()
    return render(request, 'blog/register.html', {'form' : form})


@login_required
def profile(request):
    if request.method == 'POST':
        u_form = UserUpdateForm(request.POST, instance=request.user)
        p_form = ProfileUpdateForm(request.POST, request.FILES, instance=request.user.profile)
        if u_form.is_valid() and p_form.is_valid():
            u_form.save()
            p_form.save()
            username = u_form.cleaned_data.get('username')
            messages.success(request, f'Account has been updated {username}!')
            return redirect('profile')
    else:
        u_form = UserUpdateForm(instance=request.user)
        p_form = ProfileUpdateForm(instance=request.user.profile)
    return render(request, 'blog/profile.html', {'u_form' : u_form, 'p_form' : p_form})


@login_required
def profile_detail(request, post_author):
    post_author = User.objects.get(username=post_author)
    post_user = Post.objects.filter(author=post_author).order_by('-date_published')
    paginator = Paginator(post_user, 5)
    page = request.GET.get('page')
    page_obj = paginator.get_page(page)
    return render(request, 'blog/show_profile.html', {'page_obj' : page_obj, 'post_author' : post_author, 'post_count' : post_user.count()})