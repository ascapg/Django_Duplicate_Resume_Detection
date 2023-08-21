from django.shortcuts import render,HttpResponse
from django.shortcuts import render


# Create your views here.
def Hello_World(requests):
    print('Hi')
    return render(requests,"Form.html") 


def Company_Details(request):
    print (request.GET)
    return HttpResponse('Company_Details')

