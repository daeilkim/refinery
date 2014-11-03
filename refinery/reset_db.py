#!/usr/bin/env python
# encoding: utf-8
import os

from config import basedir
from refinery import app,db
from refinery.data.models import *
import shutil

def create_db_entries():

    print "Creating new DB"

    userdir = app.config['USER_DIRECTORY']
    
    try:
        os.stat(userdir)
    except:
        os.mkdir(userdir)
        
    # remove all folders within the users directory
    remove_dir(userdir)
    
    # recreate database structure
    db.drop_all()
    db.create_all()
    
    # username and passwords for mock db fill
    usernames = ['doc']
    passwords = ['refinery']

    # Create a bunch of users
    for i in xrange(len(usernames)):
        add_user(usernames[i],passwords[i])
        
    # Create a bunch of datasets
    
    # Create a bunch of experiments

    # create a bunch of reports
    
    check_db()

def check_db():
    query_users = User.query.all()
    query_exp = Experiment.query.all()
    query_data = Dataset.query.all()
    
    #print query_users
    #print query_exp
    #print query_data

def remove_dir(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        if os.path.isdir(file_path):
            print "Deleting: " + file_path
            shutil.rmtree(file_path)

def create_user_dir(username):
    newdir = app.config['USER_DIRECTORY'] +  username
    if os.path.exists(newdir):
        print "Directory already exists for " + username
    else:
        datadir = newdir + "/documents"
        imdir = newdir + "/images"
        procdir = newdir + "/processed"
        os.makedirs(newdir)
        os.makedirs(procdir)
        os.makedirs(datadir)
        os.makedirs(imdir)
        
        print "Creating directory structure for: " + newdir

def add_user(username, password):
    ''' When we add a new user, we first check if this user exists. If not,
    we create this users directory structure.
    '''
    
    create_user_dir(username)
    email = username + "@refinery.com"
    u = User(username = username, password=password, email=email)
    
    if(username == "doc"):
        u.email = "refinery@docrefinery.com"
        shutil.copyfile("reset_db_files/default.jpg","refinery/static/users/doc/images/default.jpg")
        u.image = "default.jpg"

    db.session.add(u)
    db.session.commit()

if __name__ == '__main__':
    create_db_entries()
