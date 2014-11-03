
var selected = {};

console.log(lines);

var ls = d3.select("#survey").selectAll("div").data(lines);

ls.enter().append("div").attr("class","sentex").on("click",function(d,i) {
        d3.select(this).style("background",function(d2,i2) {
                console.log("clicked " + i);
                if(i in selected) {
                    delete selected[i];
                    return "#ffffff";
                } else {
                    selected[i] = 0;
                    return "#aaeeff";
                }
                
            });
    }).html(function(d) { 
            return d;
        });



function submitSurvey() {

    var linse = d3.select("#survey").selectAll("div").data();

    var labels = [];

    for (i in selected){
        console.log("!" + i);
        labels.push(i);
    }

    $.post(ann_url,{'labels[]' : labels, 'lines[]' : lines},function(d) {

            window.location = "/annotate";

        });

}

