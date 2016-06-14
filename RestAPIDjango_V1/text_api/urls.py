from django.conf.urls import url

from text_api import views
urlpatterns = [
    url(r'bulk' ,views.text_api, name= 'text_api'),
    url(r'text_api_doc' ,views.text_api_doc, name= 'text_api_doc'),
]