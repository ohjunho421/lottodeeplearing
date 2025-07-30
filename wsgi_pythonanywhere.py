#!/usr/bin/env python
import os
import sys

# Add your project directory to the sys.path
path = '/home/yourusername/lottodeeplearing'
if path not in sys.path:
    sys.path.append(path)

os.environ['DJANGO_SETTINGS_MODULE'] = 'lottobot.settings'

from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()