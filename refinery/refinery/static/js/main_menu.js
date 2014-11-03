var MAX_FILE_SIZE = 500000000; //maximum file upload size

$(".settings_btn").colorbox({inline:true, width:"350px", height:"350px"});
$(".dataset_btn").colorbox({inline:true, width:"600px"});
$(".colorbox_btn").colorbox({inline:true, width:"600px"});
$(".knob").knob();


function set_tm_status(fID,status) {

    /**
     *
     *  Alters layout based on topic model status for a folder
     *
     */
    
    if(status == "finish") {
        $("#tm_posttrain_" + fID).css("display","inline-block");
        $("#tm_pretrain_" + fID).css("display","none");
        $("#tm_bar_" + fID).css("display","block").css("background","#AADDAA");
        $("#tm_prog_" + fID).css("display","none");
    }
    if(status == "inprogress") {
        $("#tm_prog_" + fID).css("display","block");
        $("#tm_bar_" + fID).css("display","none");
        progress("Learning",$("#tm_statusbar_"+fID));
    }
    if(status == "idle") {
        $("#tm_posttrain_" + fID).css("display","none");
        $("#tm_pretrain_" + fID).css("display","inline-block");
        $("#tm_bar_" + fID).css("display","block").css("background","#FF4444");
        $("#tm_prog_" + fID).css("display","none");
    }
}

function set_sum_status(fID,status) {
    /**
     *
     *  Alters layout based on summarize model status for a folder
     *
     */ 
    
    $("#sum_statusbar_" + fID).progressbar();
    if(status == "finish") {
        $("#sum_posttrain_" + fID).css("display","block");
        $("#sum_pretrain_" + fID).css("display","none");
        $("#sum_bar_" + fID).css("display","block").css("background","#AADDAA");
        $("#sum_prog_" + fID).css("display","none");
        $("#sum_statusbar_" +fID).progressbar("destroy");
        $("#sum_status_" +fID).html("Ready to go, just press Run");
    }
    if(status == "inprogress") {
        $("#sum_bar_" + fID).css("display","none")
        $("#sum_prog_" + fID).css("display","block");
        $("#sum_statusbar_" +fID).progressbar({'value':0});
        $("#sum_status_" +fID).html("Learning...");
        progress("Learning",$("#sum_statusbar_"+fID));
    }
    if(status == "idle") {
        $("#sum_posttrain_" + fID).css("display","none");
        $("#sum_pretrain_" + fID).css("display","block");
        $("#sum_bar_" + fID).css("display","block").css("background","#FF4444");
        $("#sum_prog_" + fID).css("display","none");
        $("#sum_statusbar_" +fID).progressbar("destroy");
        $("#sum_status_" +fID).html("Press Run to build a model");
    }
}

function setDirty(folder_id, dirty) {
    /**
     *
     *   Called on page load for each folder, and on pubsub msgs
     *   indicating transition from dirty -> clean
     * 
     */

    console.log("SET DIRTY - " + dirty);
    
    if(dirty == "clean") {
        $("#prog_row_" + folder_id).css("display","none");
        $("#tm_row_" + folder_id).css("display","block");
        $("#sum_row_" + folder_id).css("display","block");
    } else {
        $("#prog_row_" + folder_id).css("display","block");
        $("#tm_row_" + folder_id).css("display","none");
        $("#sum_row_" + folder_id).css("display","none");
        
        //start preprocessing if dirty
        $.get("/" + username + "/start_preproc/" + folder_id);
    }
}


function show_info(show) {
    /**
     *
     *   Show/Hide the information page
     * 
     */ 
    if(show) {
        $("#datalist").css("visibility","hidden");
        $("#infopage").css("visibility","visible");
    } else {
        $("#datalist").css("visibility","visible");
        $("#infopage").css("visibility","hidden");
    }
}


function browse_folder(fID) {
    /**
     *
     *  Fetch the document list for a folder, populate the
     *  browsing colorbox and show it
     *
     */ 
    var url = "/docjson/" + fID;

    $.get(url,function(data) {
            
            // one document is [title,id,text]
            
            d3.select("#browse_list").selectAll("div").remove();
            
            var divs = d3.select("#browse_list").selectAll("div").data(data['documents']);
            
            divs.enter().append("div").attr("class","browse_list_item")
                .on("click",function(d,i) {
                        
                        d3.select("#browse_list").selectAll("div").style("background","white").style("color","black");
                        d3.select(this).style("background","blue").style("color","white");
                        
                        $.post( viewdoc_url , { filename: d[0] }, function(data) {
                                $("#browse_window").html(data);
                            });
                        
                    })
                .text(function(d) {return d[0]});
            
            $.colorbox({width: 1000, height: 800, inline:true, href:$("#inline_browse")});
            
        });
    
}

function setTopics(fID) {
    /**
     *
     *  Triggered by the save changes button in topic model settings,
     *  changes the number of topics and resets modeling
     * 
     */ 
    var v = $('#tm_knob_' + fID).val();
    var u = "/" + username + "/set_num_topics/" + fID;
    $.post(u, {'v' : v}, function(d) {
            //$('#ntopics_' + fID).html("Number of Topics : " + v);
            set_tm_status(fID,"idle");
            $.colorbox.close();
        });
}


function deleteFolder(username,fID,dID) {

    /**
     *
     *  Delete a folder
     * 
     */
    
    var u = "/delete_folder/" + fID;
    $.get(u,function(d) {
            $.colorbox.close();           
            $("#folder_" + dID + "_" + fID).remove();
        });
}

function edit_dataset(dID) {

    /**
     *
     *  Post the edits for a datasets title and summary
     *
     *  reloads the main menu on completion
     * 
     */ 
    
    var u = "data/edit_dataset/" + dID;
    var name = $("#edit_data_name_" + dID).val();
    var sum = $("#edit_data_sum_" + dID).val();
    
    $.post(u,{'name' :name, 'sum' : sum },function(d) {
            $.colorbox.close();
            location.reload();
        });        
}

function delete_dataset(dID) {

    /**
     *
     *  Delete a dataset, reload main menu on completion
     *
     */ 

    var u = "/delete_dataset/" + dID;
    $.get(u,function(d) {
            $.colorbox.close();
            location.reload();
        });
    
}

function UploadFileData(file) {

    /**
     *
     *  Upload a zip file, creating a new dataset
     *
     */ 
    
    var xhr = new XMLHttpRequest();
    
    if (xhr.upload && file.size <= MAX_FILE_SIZE) {
        
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
        xhr.timeout = 0;
        xhr.open("POST", upload_url, true);
        xhr.send(formData);
    }
    else {
        alert('Zorry, an error occured when uploading your file');
    }
	
}

var fileSelector = $('<input type="file" multiple />');
fileSelector.on("change",function(evt) {
        UploadFileData(evt.target.files[0]);
    });

function start_sum(fID,uname) {
    var u1 = "/" + uname + "/start_sum/" + fID;
    var u2 = "/viz_sum/" + fID;
    check_go(u1,u2);
}

function start_tm(fID,uname) {
    var u1 = "/" + uname + "/start_tm/" + fID;
    var u2 = "/" + uname + "/viz_tm/" + fID;
    check_go(u1,u2);
}

function check_go(u1,u2) {
    $.get(u1,function(data) {
            if(data['command'] == "go") {
                window.location = u2;
            }
        });
}


var evtSrc = new EventSource(pubsub_url);

evtSrc.onmessage = function(e) {

    if (e.data=='NONE') {
        //do nothing
    } else if (e.data=='exit') {
        evtSrc.close();
        
        /**
         * PREPROC PUBSUB
         */
        
    } else if (e.data.indexOf('proc') == 0) {
        var p = e.data.split(",");
        var dID = p[1];
        var fID = p[2]
        var dirty = p[3];
        if(p[3] == "clean") {
            setDirty(fID,"clean");
        }
    } else if (e.data.indexOf('pprog') == 0) {
        var p = e.data.split(",");
        var msg = p[1];
        var dID = p[2];
        var pc = parseInt(p[3]);
        progressPC(pc,$("#statusbar_"+dID));

        /**
         * TOPIC MODEL PUBSUB
         */
        
    } else if (e.data.indexOf('tm_prog') == 0) {
        var p = e.data.split(",");
        var fID = p[1];
        var pc = parseInt(p[2]);
        progressPC(pc,$("#tm_statusbar_"+fID));
    } else if (e.data.indexOf('tmstatus') == 0) {
        var p = e.data.split(",");
        var fID = p[1];
        var status = p[2];
        set_tm_status(fID,status);

        /**
         * SUMMMARY MODEL PUBSUB
         */
        
    } else if (e.data.indexOf('sum_prog') == 0) {
        var p = e.data.split(",");
        var fID = p[1];
        var pc = parseInt(p[2]);
        progressPC(pc,$("#sum_statusbar_"+fID));

    } else if (e.data.indexOf('sumstatus') == 0) {
        var p = e.data.split(",");
        var fID = p[1];
        var status = p[2];
        set_sum_status(fID,status);
        
        /**
         * UPLOAD PUBSUB
         */
    } else if (e.data.indexOf('uprog') == 0) {
        var p = e.data.split(",");
        var pc = p[1];
        progressPC(pc,$("#upload_prog"));

    } else if (e.data.indexOf('ucomplete') == 0) {
        var p = e.data.split(",");
        var fname = p[1];
        $("#upload_prog").find('div').html("Uploaded " + fname);
        location.reload();

    } else {
        if (e.data!='1') {
            
        } 
    }
};




function progress(msg, $element) {
    /**
     *    set the progress bar text to msg
     */ 
    $element.find('div').find('div').html(msg);
}
    
function progressPC(percent, $element) {
    /**
     *    set the progress bar text to some percent
     */ 
    var progressBarWidth = Math.floor(percent * Math.floor($element.width()) / 100);
    $element.find('div').css("width",progressBarWidth);
    $element.find('div').find('div').html(percent + "%&nbsp;");
}



