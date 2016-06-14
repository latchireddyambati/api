"""
WSGI config for RestAPIDjango project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/1.8/howto/deployment/wsgi/
"""

import os
import sys
from django.core.wsgi import get_wsgi_application


path = "D:/LatchiReddy/RestAPIDjango_V1"
#print sys.path
if path not in sys.path:
    sys.path.append(path)

#os.environ.setdefault("DJANGO_SETTINGS_MODULE", "RestAPIDjango.settings")

application = get_wsgi_application()
