// getElementById using DOM model
function $id(id) {
    return document.getElementById(id);
}

// file drag hover
function FileDragHover(e) {
    e.stopPropagation();
    e.preventDefault();
    e.target.className = (e.type == "dragover" ? "filedrag_box_hover" : "filedrag_box");
}

// file selection
function FileSelectHandlerData(e) {    
    // cancel event and hover styling
    FileDragHover(e);
    
    // fetch FileList object
    var files = e.target.files || e.dataTransfer.files;

    // process all File objects
    if (files.length == 1) {
        UploadFileData(files[0]);			
    }
}

// file selection
function FileSelectHandlerImage(e) {

    // cancel event and hover styling
    FileDragHover(e);
    
    // fetch FileList object
    var files = e.target.files || e.dataTransfer.files;
    
    if (files.length == 1 && files[0].type.indexOf("image") == 0) {
        var file = files[0];
        var reader = new FileReader();
        reader.onload = function(e) {
            $("#img_box").html('<img id="img_frame" src="' + e.target.result + '" />');
        }
        reader.readAsDataURL(file);

    } else {
        //TODO : do something to indicate failure
    }
}

/**
function UpdateAndGo(u) {

    var xhr = new XMLHttpRequest();

    var formData = new FormData($id("upload_img"));
    if(imgFile != null) {
        alert("adding image file!");
        formData.append('file', imgFile);
    }
    xhr.open("GET", u, true);
    xhr.send(formData);
    xhr.onreadystatechange = function(e) {
        if (xhr.readyState == 4) {
            alert(
        }
    };
}
*/

function refresh_doclist() {
    $.get($id("doclist_target").value,function(data) {
            $("#listofiles").html(data);
        });
}

function UploadFileData(file) {
    
    var xhr = new XMLHttpRequest();
    
    if (xhr.upload && file.size <= $id("MAX_FILE_SIZE").value) {
        
        
        
        // progress bar
        xhr.upload.addEventListener("progress", function(e) {
				var pc = parseInt(e.loaded / e.total * 100);

				$( "#upload_progress" ).progressbar( "option", "value", pc );
                $("#progress_text").html("Uploading " + file.name + " - " + pc + "%");
			}, false);
        
        // file received/failed
        xhr.onreadystatechange = function(e) {
            if (xhr.readyState == 4) {
                
            }
        };
		
        var formData = new FormData();
        formData.append('file', file);
        formData.append('filename', file.name);
        xhr.timeout = 100000;
        xhr.open("POST", $id("filedrag_target").value, true);
        xhr.send(formData);
    }
    else {
        alert(xhr.upload);
        alert(file.size);
        alert('no success');
    }
	
}



function finished() {
    
}

function Init() {
    var filedrag_data = $id("filedrag_data");
    filedrag_data.addEventListener("dragover", FileDragHover, false);
    filedrag_data.addEventListener("dragleave", FileDragHover, false);
    filedrag_data.addEventListener("drop", FileSelectHandlerData, false);

}

if (window.File && window.FileList && window.FileReader) {
    Init();
}

var evtSrc = new EventSource(pubsub_url);

evtSrc.onmessage = function(e) {
    //alert(e.data);
    //alert(e.data.indexOf('prog'));
    if (e.data=='NONE') {
        //do nothing
    } else if (e.data=='exit') {
        //give a link to data page
        evtSrc.close();
    } else if (e.data.indexOf('prog') == 0) {
        var p = e.data.split(",");
        var cur = parseFloat(p[1]);
        var tot = parseFloat(p[2]);
        var pc = parseInt(cur/tot * 100);
        //alert(cur + "," + tot + "," + pc);
        $("#upload_progress" ).progressbar( "option", "value", pc );
        $("#progress_text").html("Processing files - " + pc + "%");
    } else if (e.data.indexOf('complete') == 0) {
        var p = e.data.split(",");
        var filename = p[1];
        $("#progress_text").html("Sucessfully uploaded " + filename);
        refresh_doclist();
    } else {
        if (e.data!='1') {
            
        } 
    }
};







function viewFile(username,path,name,spath) {

    if(path.indexOf(".pdf") > 0) {
        $("#inline_viewer").html("<embed src=" + spath + " style='width:100%;height:100%;'></embed>");
        var fname = name;
        $.colorbox({inline : true, href : "#inline_viewer", title : fname}); 
    } else {
        var get_data_url = $id("get_data_url").value;
        $.post( get_data_url, { filename: path }, function(data) {
                $("#inline_viewer").html(data);
                var fname = name;
                $.colorbox({inline : true, href : "#inline_viewer", title : fname}); 
            });
    }
}
//$.colorbox({width: 600, height: 400, inline:true, href:$("#inline_upload"), closeButton:false});


function try_delete_document(username,data_id,doc_id,dname) {
    $("#deldocname").html(dname);
    $("#deldocyes").attr("onclick","delete_document('"+username+"',"+data_id+","+doc_id+")");
    $.colorbox({width: 600, height: 400, inline:true, href:$("#inline_delete_doc")});
}

function delete_document(username,data_id,doc_id) {
    $.post($id("deldoc_target").value,{docid : doc_id}, function(data) {
            $.colorbox.close();
            $("#listofiles").html(data);
        });
}

function preproc_and_go() {

    $.get(preproc_url);
    window.location = main_url;
    
}

$("#upload_progress").progressbar({'value':0});
