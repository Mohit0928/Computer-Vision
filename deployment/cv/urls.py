"""deployment URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path

from . import views

urlpatterns = [
    path('', views.base, name='base'),
    path('classification', views.classification, name='classification'),
	path('semantic_segmentation', views.semantic_segmentation, name='semantic_segmentation'),
    path('panoptic_segmentation', views.panoptic_segmentation, name='panoptic_segmentation'),
    path('object_detection', views.object_detection, name='object_detection'),
    path('license_plate',views.license_plate,name='license_plate'),
]