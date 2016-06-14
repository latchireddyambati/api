from django.conf.urls import include, url
from django.contrib import admin

admin.autodiscover()

urlpatterns = [
    url(r'^admin/', include(admin.site.urls)),
    url(r'^text_api/', include('text_api.urls')),
    #url(r'^text_api_doc/', include('text_api.urls')),
    #url(r'^sentiment_api/', include('sentiment_api.urls')),
    #url(r'^ruleengine_api/', include('ruleengine_api.urls')),

]
