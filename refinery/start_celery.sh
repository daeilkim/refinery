#!/bin/bash
celery --loglevel=debug --concurrency=2 -A refinery.celery worker
