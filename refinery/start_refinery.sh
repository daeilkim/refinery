#!/bin/bash
# redis-server &
#. venv/bin/activate
gunicorn --debug --log-level=debug --timeout 120 -w 4 -b 0.0.0.0:8080 refinery.webapp.main_menu:app
