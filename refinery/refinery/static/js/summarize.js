var CIRCLE_WIDTH = 65;
var selected = {};

//Set up the three buttons


var buttons = d3.select("#menubarbuttons").selectAll("div").data(["Similar","Different","Delete"]);
var svgs = buttons.enter().append("svg")
    .attr("width",CIRCLE_WIDTH+2)
    .attr("height",CIRCLE_WIDTH+2)
    .style("margin","10px")
    .on("click",function(d,i) {
            var selected = get_selected(d3.select("#cur_sum"));
            if(i == 0) {
                get_sentences(selected,"SIMILAR");
            } else if (i == 1) {
                get_sentences(selected,"VARIETY");
            } else {
                get_sentences(selected,"DELETE");
            }
        });

var gradient = svgs.append("svg:defs")
    .append("svg:linearGradient")
    .attr("id", "gradient1")
    .attr("x1", "0%")
    .attr("y1", "0%")
    .attr("x2", "100%")
    .attr("y2", "100%")
    .attr("spreadMethod", "pad");

gradient.append("svg:stop")
    .attr("offset", "0%")
    .attr("stop-color", "#AACCAA")
    .attr("stop-opacity", 1);

gradient.append("svg:stop")
    .attr("offset", "100%")
    .attr("stop-color", "#116611")
    .attr("stop-opacity", 1);

var gradient2 = svgs.append("svg:defs")
    .append("svg:linearGradient")
    .attr("id", "gradient2")
    .attr("x1", "0%")
    .attr("y1", "0%")
    .attr("x2", "100%")
    .attr("y2", "100%")
    .attr("spreadMethod", "pad");

gradient2.append("svg:stop")
    .attr("offset", "0%")
    .attr("stop-color", "#339944")
    .attr("stop-opacity", 1);

gradient2.append("svg:stop")
    .attr("offset", "100%")
    .attr("stop-color", "#DDDDFF")
    .attr("stop-opacity", 1);


svgs.append("circle").attr("r",CIRCLE_WIDTH/2)
.attr("cx",CIRCLE_WIDTH/2+1).attr("cy",CIRCLE_WIDTH/2+1).attr("fill","url(#gradient1)").style("z-index","10")
.attr("stroke","black").attr("stroke-width",2)
.on("mouseover",function(d,i) { //mouseover highlight behavior
        d3.select(this).attr("fill","url(#gradient2)")
    }).on("mouseout",function(d,i) {
            d3.select(this).attr("fill","url(#gradient1)")
        });
svgs.append("text").style("text-anchor","middle").style("pointer-events","none").text(function(d) {return d;}).attr("y",CIRCLE_WIDTH/2+4).attr("x",CIRCLE_WIDTH/2+1).style("z-index",-10);


function get_selected(element) {

    /**
     *
     *  Get the selected sentences from a fact container
     *
     *  this is either the main page collection of facts or
     *  the suggestion colorbox
     *
     */ 
    
    var res = [];

    element.selectAll(".selcirc").each(function (d,i) {
            var cur = d3.select(this).attr("fill");            
            if (cur == "black")
                res.push(d);
        });

    if(res.length == 0) { //selecting nothing is the same as selecting all
        element.selectAll(".selcirc").each(function (d,i) {
                res.push(d);
            });
    }    
    return res;
}


function keep() {
    var selected = get_selected(d3.select("#results"));
    
    $.post(add_url,{'sents':JSON.stringify(selected)},function (d) {
            fill_with_facts(d3.select("#cur_sum"),d['data'],true);
            $.colorbox.close();
        });
}


function get_sentences(selected,mechanism) {

    var sel_json = JSON.stringify(selected);

    if(mechanism == "SIMILAR") {

        $.post(similar_url, {'sents':sel_json}, function (d) {

                data = d['results'];
                fill_with_facts(d3.select("#results"),data,false);
                $.colorbox({width: 1000, height: 800, inline:true, href:$("#results_outer")});
                
            });
        
    } else if (mechanism == "VARIETY") {

        $.post(variety_url, {'sents':sel_json}, function (d) {            
            data = d['results'];
            fill_with_facts(d3.select("#results"),data,false);
            $.colorbox({width: 1000, height: 800, inline:true, href:$("#results_outer")});
        });
        
    } else if (mechanism == "DELETE") {
        
        $.post(delete_url, {'sents':sel_json}, function (d) {
                fill_with_facts(d3.select("#cur_sum"),d['data'],true);
            });
        
    }
}




function fill_with_facts(element,data,can_view) {

    /**
     *
     *   fill the main screen with the facts in the current summary
     *
     *   data is a list of [sentence, word dict, file index, sentence index]
     *
     *
     */    

    //if there are no facts, launch "Variety" and return
    if(data.length == 0) {
        get_sentences([],"VARIETY");
        return;
    }
    
    element.selectAll(".sumrow").remove();
    
    var p = element
        .selectAll(".sumrow")
        .data(data);

    var outer = p.enter().append("div").attr("class","onerow sumrow");

    var cRad = 18;
    
    var svg = outer.append("div").attr("class","selbox").append("svg").attr("width",2*cRad+2).attr("height",2*cRad+2);
        
    svg.append("circle").attr("class","selcirc").attr("r",cRad).attr("fill","white").attr("stroke-width","2px").attr("stroke","black").attr("cx",cRad+1).attr("cy",cRad+1)
        .on("mouseover",function(d,i) {
                d3.select(this).attr("stroke","red");
            }).on("mouseout",function(d,i) {
                    d3.select(this).attr("stroke","black");
                }).on("click",function(d,i) {
                        
                        var cur = d3.select(this);
                        if(cur.attr("fill") == "white")
                            cur.attr("fill","black");
                        else
                            cur.attr("fill","white");
                        
                    });
    
    var txt = outer.append("div").attr("class","sumtext").text(function (d) {return d[0];})
        .style("background",function(d,i){
                if (i % 2 == 0)
                    return "#BBEEBB";
                else
                    return "#BBBBEE";
            });
    
    if(can_view) {
        txt.on("mouseover",function(d,i) {
                    
                    d3.select(this).style("border","solid 2px red");
                    
                }).on("mouseout",function(d,i) {
                        
                        d3.select(this).style("border","solid 2px black");
                        
                    }).on("click",function(d,i) {
                            
                            $.post( viewdoc_url , { didx: d[2] }, function(data) {

                                    var full = data['fulltext'];
                                    var sent = d[0];

                                    //highlight the search sentence in the document
                                    var startInd = full.indexOf(sent);
                                    var endInd = startInd + sent.length;                                    
                                    var newfull = full.slice(0,startInd) + "<span class='hilite'>" + sent + "</span>" + full.slice(endInd);
                                    
                                    $("#inline_viewer").html(newfull);
                                    $.colorbox({inline : true, href : "#inline_viewer", title : "TITLE"});
                                    
                                });
                        });
    }    
}


