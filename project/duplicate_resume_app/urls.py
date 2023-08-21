from django.urls import path
from duplicate_resume_app import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("", views.index, name="index"),
    path('details/<str>', views.details, name='details'),
    path('check_validity', views.check_validity, name='check_validity'),
    # path("scan",views.scan_resume,name="scan_resume"),
    # path('open_link/<string1>', views.open_linkedin_link,name='open_link'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)