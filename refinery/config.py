#!/usr/bin/env python
# encoding: utf-8
import os
import sys
basedir = os.path.abspath(os.path.dirname(__file__))

# Must turn this off when in devlopment
DEBUG = True

# Flask WTF module requires these two settings
CSRF_ENABLED = True
SECRET_KEY = 'bulgogi'
# Path of our database file, required by flask-SQLAlchemy
#SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')
SQLALCHEMY_DATABASE_URI = "postgresql:///refinery"

#SQLALCHEMY_DATABASE_URI = 'postgresql://refinery_admin@localhost/refinery'
# Folder that stores our SQLAlchemy-migrate data files
SQLALCHEMY_MIGRATE_REPO = os.path.join(basedir, 'db_repository')

UPLOAD_FOLDER = 'refinery/static/datasets/'
USER_DIRECTORY = 'refinery/static/users/'
RANDOM_IMG_DIRECTORY = 'refinery/static/assets/images/random/'


