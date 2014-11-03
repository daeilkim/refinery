import os
from flask import Response, request, g
import zipfile
import shutil
from refinery import app, db
from refinery.data.models import Dataset, Document, DataDoc, Folder
from refinery.webapp.pubsub import msgServer

def add_txt(path, fid, filename, dataset):
    
    doc = Document(filename, path)
    db.session.add(doc)
    db.session.commit() #sets the primary key
    target = file(path, "wb")
    shutil.copyfileobj(fid, target)

    data_doc = DataDoc(dataset.id, doc.id)
    db.session.add(data_doc)
    
    doc_ids = dataset.folders[0].docIDs

    doc_ids[doc.id] = 0 #how do i make a hash set?
    db.session.commit()
    folder = Folder.query.get(dataset.folders[0].id)

    folder.docIDs = doc_ids
    
    db.session.commit()

    return doc.id

# Initiates the upload process once files have been dropped in
@app.route('/<username>/upload/drop/', methods=['POST'])
def upload_drop(username=None):

    name = "My New Dataset"
    summary = "A summary of the dataset"
    user_id = g.user.id
    
    dset = Dataset(user_id,name,summary)
    db.session.add(dset)
    db.session.commit()
    did = dset.id

    main_folder = Folder(dset.id,"Main Folder",dict())
    db.session.add(main_folder)
    db.session.commit()

    main_folder.initialize()
    db.session.commit()
    
    print "DROP"

    dset = Dataset.query.get(did)

    ufilename = request.form['filename']
    fid = request.files.getlist('file')[0]  #grab only a single file

    fn,ext = os.path.splitext(ufilename)

    userpath = "refinery/static/users/" + username + "/documents"
    channel = username + "Xmenus"

    if ext == ".zip":
        zip_file = zipfile.ZipFile(fid)
        files = zip_file.namelist()
        nFiles = len(files)
        lastProg = 0
        count = 0.0
        for member in files:
            filename = os.path.basename(member)
            if filename:
                fn,ext = os.path.splitext(filename)
                if ext == ".txt" or ext == ".pdf":
                    add_txt(os.path.join(userpath,filename),zip_file.open(member),filename,dset)
            count += 1.0
            update = str(int(count / float(nFiles) * 100))
            if update != lastProg:
                lastProg = update
                s = 'uprog,' + update
                msgServer.publish(channel, "%s" % s)
            
    elif ext == ".txt" or ext == ".pdf":
        add_txt(os.path.join(userpath,filename),fid,filename,dset)

    else:
        print "unknown file format",ext,filename

    dset.dirty = "dirty"

    db.session.commit()

    print "GOT ",len(dset.folders[0].docIDs), "Documents"

    msgServer.publish(channel, "ucomplete," + ufilename) 

    return Response(status="200")
