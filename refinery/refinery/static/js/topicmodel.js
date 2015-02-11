var search_tops = [[1,1.0]];
var docT = [];
d3.refinery = {};
var canvas_w = 900; // Actual size of the full canvas 
var viz_cur = 0;
var donut,wc;

$.get(load_url,function(d) {
        
    topWI=d['topW'];
    numT = topWI.length;
    topic_probs=d['topic_probs'];
    docT=d['doc_tops'];

    var topW = [];
    for(var i=0;i<numT;++i) {
        var myTopW = [];
        var tot = 0.0;
        var numW = topWI[i].length;
        for(var j=0;j<numW;++j) {
            var wI = jsvocab[topWI[i][j][0]];
            var wW = topWI[i][j][1];
            tot += wW;
            var o = {'w' : wI, 'p' : wW} 
            myTopW.push(o);
        }
        for(var j=0;j<numW;++j) {
            myTopW[j]['p'] /= tot;
            myTopW[j]['p'] *= 400;
        }
        topW.push(myTopW);
    }

    $("#loadingmsg").remove();
    donut = d3.refinery.donut(600,'#donutviz');
    wc = d3.refinery.word_cloud(600,600,"#donutviz");
    cur_words = topW;
    update_pi(topic_probs);
    update_words(cur_words[viz_cur]);

    $("#searchbar").css("visibility","visible");
        
    
    });


var fillB = d3.scale.category20();

/**
 *
 *   Making the topic blocks
 *
 */ 

 var blockSz = 25;
 var blockW = 430;
 var blockH = 160;
 var rowLen = 5;
 var pad = 0;
 var svg = d3.select("#topicsearch").append("svg").attr("id","topicblocks")
 .attr("width", blockW)
 .attr("height", blockH);

 var initData = [];

 for(var i=0;i<8;++i) {
    initData.push(-1);
}

var blocks = svg.selectAll("g").data(initData);

blocks.attr("fill", function(d, i) {
    if(d >= 0)
        return fillB(d);
    else
        return 0;
})

var offy = 0;
var rects = blocks.enter().append("g").attr("transform", function(d,i) {
            if (i == 0) {offy = 0;}
            else { offy = blockSz + pad;
                pad = pad + blockSz;}
                return "translate(" +  0 + ","+offy+")";
            });

rects.append("rect").attr("height", blockSz).attr("width", blockW).attr("stroke","grey").attr("stroke-width",1);

blocks.on('click',function(d,i) {

        var blocksD = d3.select("#topicblocks").selectAll("g").data();
        blocksD.splice(i,1);
        blocksD.push(-1);

        refresh_blocks(blocksD);
});


function refresh_blocks(data) {
    var blocks = d3.select("#topicblocks").selectAll("g").data(data);
    blocks.attr("fill", function(d, i) {
            if(d >= 0)
                return fillB(d);
            else
                return 0;
        }).attr("stroke","grey").attr("stroke-width",1);                        
    
    d3.selectAll(".topic_text").remove();

    blocks.append('text').text( function(d, i) {
            var topic_str = '';
            if(d >= 0){    
                for(var word_id=0; word_id<6; ++word_id) {
                    topic_str = topic_str + cur_words[d][word_id].w +', ';
                }
                topic_str = topic_str + ',...';
                return topic_str;
            }
            else{
                return '';
            }
        })
        .attr('class','topic_text')
        .attr('x', 10)
        .attr('y', 18)
        .attr('fill', '#000')
        .attr('stroke-width',0);
}

d3.refinery.word_cloud = function module(width, height) {

    var fill = d3.scale.category20();
	var svg;
    var NUM_WORDS = 40;
    var SCALE_FAC = 2.0;
    
	function chart(_data) {

        /**
         *   _data should be a list of word/size tuples,
         *
         *   if so, this will generate wordcloud layout data and call draw
         *   
         */
        d3.layout.cloud().size([height, height])
            .words(_data.map(function(d) {

                        /**
                         *
                         *  first filter out user stopwords
                         *
                         */
                        
                        var stopwords = $("#stopword_text").val().split(" ").map(function(d) {return d.toLowerCase();});

                        var sz = d['p']*SCALE_FAC;
                        var w = d['w'];
                        if($.inArray(w.toLowerCase(),stopwords) > -1){
                            sz = 0.0;
                        }
                        return {text: w, size: sz};
                    }).slice(0,NUM_WORDS))
            .padding(5)
            .rotate(function() {return 0.0;})//return ~~(Math.random() * 2) * Math.random()*10; })
            .font("Courier")
            .fontSize(function(d) { return d.size; })
            .on("end", draw)
            .start();
                
        function draw(words) {

            console.log("WORDCLOUD DRAW");
            console.log(words[0]);
            
            if (!svg) {
                svg = d3.select("#d3_svg") //use the same svg as the topic ring
                    .append("g")
                    .attr("width", width)
                    .attr("height", height)
                    .attr("transform", "translate("+width/2+","+height/2+")")
                    }
                    
            var wrds = svg.selectAll("text").attr("class",'wc_text').data(words);
                    
            wrds.enter().append("text")
                .style("font-size", function(d) { return d.size + "px"; })
                .style("font-family", "Courier")
                //.style("fill", function(d, i) { return fill(i); })
                .attr("text-anchor", "middle")
                .attr("transform", function(d) {
                        return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
                    })
                .text(function(d) {
                        return d.text;
                    });
                    
            wrds.text(function(d) {
                        return d.text;
                    }).transition().attr("transform", function(d) {
                        return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
                    })
                .style("font-size", function(d) { return d.size + "px"; })
                .ease("linear")
                .duration(350);
            
            wrds.exit().remove();
        }
    }
    
    return chart;
}

///////////////////////////////////////////////////////////////////////
// Global topic distribution generated by pi
d3.refinery.donut = function module(height, id) {
	var width = height;
	var radius = height / 2;
	var color = fillB;//d3.scale.category20();
	var svg;

   //var dispatch = d3.dispatch("customHover");
   function chart(_selection) {
       _selection.each(function(_data) {
               var pie = d3.layout.pie()
                   .value(function(d) {return d; })
                   .sort(null);

           var arc = d3.svg.arc()
               .innerRadius(radius - 60)
               .outerRadius(radius - 30);

           if (!svg){
            svg = d3.select(id).append("svg")
            .attr("id", 'd3_svg')
            .style("background",'white')
            .attr("width", width)
            .attr("height", height)
            .append("g")
            .attr("transform", "translate(" + width/2 + "," + height/2 +")");
        }

        var path = svg.selectAll("path")
        .data(pie(_data));

        path.enter().append("path")
        .attr("fill", function(d, i) { return color(i); })
        .attr("d", arc)
        .style("cursor", "pointer")
        .each(function(d) {this._current = d;} )
        .on("mouseover", function(d,i) {
                console.log("CURRENT VIZ : " + i);
                viz_cur = i;
                update_words(cur_words[i]); //this is what makes the display change!
                d3.select(this)
                    .style("opacity", .5)
                    .style("stroke","black")
                    .style("stroke-width", 2);
            })
            .on("mouseout", function(){
                    d3.select(this)
                        .style("opacity", 1)
                        .style("stroke","none")
                        })
        .on("click", function(d,i){

                var blocksD = d3.select("#topicblocks").selectAll("g").data();
                console.log(blocksD);
                
                blocksD.splice(0,0,i); //add clicked topic
                blocksD.splice(10,1);
                console.log(blocksD);
                
                refresh_blocks(blocksD);
                
            });
        
        path.transition()
            .duration(2000)
            .attrTween("d", arcTween);
        
        path.exit().remove();
        
        function arcTween(a) {
            var i = d3.interpolate(this._current, a);
            this._current = i(0);
            return function(t) {
                return arc(i(t));
            };
        }
           });
   }
    chart.radius = function(_x) {
        if (!arguments.length) return radius;
        radius = _x;
        return this;
    };
    
    chart.width= function(_x) {
        if (!arguments.length) return width;
        width = _x;
        return this;
    };
    
    chart.height = function(_x) {
        if (!arguments.length) return height;
        height = _x;
        return this;
    };
    
    return chart;
}





function update_words(words) {
     console.log("UPDATE WORDS");
     console.log(words[0]);
     //d3.selectAll(".wc_text").remove();
     wc(words);
     //d3.select("#visual_content").datum(words).transition().call(wc);
}

function update_pi(data) {
    d3.select("#visual_content").datum(data).transition().ease("linear").call(donut);
}

function updateDox(data) {

    //now data is a list of tuples, ([topic post, filename, kl to search_tops])
    //sort by KL


    console.log("UPDATING " + data.length + " docs");

    d3.select("#browseviz")
        .selectAll("div").remove();
    
    var p = d3.select("#browseviz")
        .selectAll("div")
        .data(data);
        
    p.enter().append("div");
    p.attr("class","doc sel").style("margin-bottom","5px").style("display:relative");
    //d3.select(this).selectAll("div").remove();
    
    var color = d3.scale.category20();

    p.append("div").style("display","relative").append("img")
        .attr("class","viewer_btn")
        .attr("src",select_button_url)
        .style("width","25px")
        .style("margin-right","10px")
        .style("float","left").on("click",function(d,i) {
            
            console.log("click " + i);

            p.classed("sel",function(d2,i2) {
                    if(i2 <= i) {
                        return true;
                    } else {
                        return false;
                    }
                });
        });
    
    var titleD = p.append("div").style("display","inline-block").style("position","relative");
    
    titleD.append("div").attr("class","filename").text(function (d) {
            var fname = /([^\/]+)$/.exec(d[1])[1];
            if(fname.length > 35) {
                fname = fname.substring(0,32) + "...";
            }
            return fname;
        });

    
    
    p.append("div").style("display","inline-block").append("a")
        .style("float","left")
        .attr("class","viewer_btn")
        .on("click", function(d,i){                
                $.post( get_data_url, { filename: d[1] }, function(data) {
                        $("#inline_viewer").html(data);
                        console.log(data);
                        var fname = d[1];
                        $.colorbox({inline : true, href : "#inline_viewer", title : fname}); 
                    });
            }).append("img").attr("src",view_url).style("width","40px").style("margin-left","10px");
        

    
    var svgWID = 300;
    var svgHGT = 30;
    
    var svg = titleD.append("div").append("svg")
        .attr("width", svgWID)
        .attr("height", svgHGT)
        .style("background",'white');

    svg.each(function(d,i) {

            var accum = 0;
            
            var g = d3.select(this).selectAll("g").data(d[2]).enter().append("g");
                
            g.attr("fill", function(d, i) { return color(i); })
                 .attr("transform", function(d, i) {
                         var myW = d*svgWID;
                         var off = accum;
                         accum += myW;
                         return "translate(" +  off + ",0)";
                     });
            
            g.append("rect")
                .attr("height", svgHGT)
                .attr("width", function(d) {return d*svgWID;});
        });
    
}


function makeSubset() {
    
    var prc = $('#knob_percent_subset').val();
    
    console.log("Percent dataset - " + prc);
    
    var nn = prc / 100 * totalDox;
    var nD = Math.floor(nn);
    
    var sTerms = $("#keyword_box").val();
    
    console.log("terms : " + sTerms);

    var blocksD = d3.select("#topicblocks").selectAll("g").data();

    /**
     *  Swap to subset view
     */ 
    $("#searchbar").css("visibility","hidden");
    $("#browsevizouter").css("visibility","visible");
    $("#browseresults").css("visibility","hidden");
    $("#browsewait").css("visibility","visible");

    
    $.post( makesubset_url, { nDox : nD, searchwords : sTerms, 'blocks[]' : blocksD }, function(data) {
            
            //update doc with results
            
            console.log(data['targDist']);
            console.log(data['docs']);
            
            jsondocs = data['docs']; 
            
            updateDox(data['docs']);
            $("#searchbar").css("visibility","hidden");
            $("#browsevizouter").css("visibility","visible");
            $("#browsewait").css("visibility","hidden");
            $("#browseresults").css("visibility","visible");
        });
}


function subsetInfo() {
    $.colorbox({width: 600, height: 260, inline:true, href:$("#inline_accept_creation")});
}


function createSubset(goback) {
    
    idxs = new Array();
    
    var dox = d3.select("#browseviz")
        .selectAll("div .doc");
    
    console.log(dox.length + " selections");
    
    dox.each(function (d) {
            if(d3.select(this).classed("sel")) {
                console.log(d[0]);
                idxs.push(d[0]);
            }
        });
    
    console.log("IDXS " + idxs);

    name = $("#foldername").val();
        
    $.post( createsubset_url, { 'name' : name, 'docIdxs[]' : idxs}, function(data) {
                
            if(goback)
                window.location = "/";
            else
                $.colorbox.close();
            backToSearch();
        });
    
}

function backToSearch() {
    
    $("#searchbar").css("visibility","visible");
    $("#browsevizouter").css("visibility","hidden");
    $("#browseresults").css("visibility","hidden");
    
    
}

function change_stopwords() {

    var stopwords = $("#stopword_text").val();

    $.post(stopword_url,{'stopwords':stopwords},function(data) {
            update_words(cur_words[viz_cur]);
            $.colorbox.close();
        });

}


function show_stopwords() {
    $.colorbox({inline : true, href : "#stopwords", title : "Edit Stopwords", width: 600, height : 330}); 
}
