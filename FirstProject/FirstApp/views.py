from django.shortcuts import render,HttpResponse

# Create your views here.
def HelloWorld(requests):
    return HttpResponse('Hello World')

