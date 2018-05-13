function getCurrGramId() {
    return $('#input-grammar')[0].value;
}


function getCurrGrammar() {
    return RESOURCES[getCurrGramId()];
}

// Using underscore.js/lodash.js Templates
var Templates = {};

Templates.result = [
    '<div data-result="<%= resultId %>" class="result">',
        '<div class="result-inner">',
            '<div class="num"><%= resultId %></div>',
        '</div>',
    '</div>'].join("\n");


Templates.viz = [
    '<div class="viz <%= vizType %>" data-viz="<%= vizType %>">',
        '<div class="tools hidden">',
            '<div title="Save as PNG" class="save" data-img="png"><div class="icon"></div><div>PNG</div></div>',
            '<div title="Save as SVG" class="save" data-img="svg"><div class="icon"></div><div>SVG</div></div>',
            '<div title="Save as LateX" class="save" data-img=latex><div class="icon"></div><div>LaTeX</div></div>',
        '</div>',
    '</div>'
].join("\n");

Templates.edm = [
    '<div name="test">',
      '<div><a href="#" id="edm_option" class="btn btn-primary btn-sm" role="button" aria-pressed="true"></a></div>',
        '<div><span class="common">Common</span>/<span class="gold">Gold Only</span>/<span class="system">System Only</span></div>',
        '<div><table class="edm edm_table edm_stats">',
          '<tr><th>Total</th><td class="stats_value"><%= total %></td></tr>',
          '<tr><th>Common </th><td class="stats_value"><%= common %></td></tr>',
          '<tr><th>Gold Unique</th><td class="stats_value"><%= gold %></td></tr>',
          '<tr><th>System Unique</th><td class="stats_value"><%= system %></td></tr>',
          '<tr><th>&nbsp;&nbsp;name</th><td class="stats_value"><%= name %></td></tr>',
          '<tr><th>&nbsp;&nbsp;unknown</th><td class="stats_value"><%= unknown %></td></tr>',
          '<tr><th>&nbsp;&nbsp;compound</th><td class="stats_value"><%= compound %></td></tr>',
          '<tr><th>&nbsp;&nbsp;udef_q</th><td class="stats_value"><%= udef_q %></td></tr>',
          '<tr><th>&nbsp;&nbsp;proper_q</th><td class="stats_value"><%= proper_q %></td></tr>',
          '<tr><th>&nbsp;&nbsp;yofc</th><td class="stats_value"><%= yofc %></td></tr>',
          '<tr><th>Predicate not in Surface</th><td class="stats_value"><%= predicate %></td></tr>',
          '<tr><th>Predicate Arg Mismatch</th><td class="stats_value"><%= predicate_arg %></td></tr>',
          '</table></div>',
        '<table class="edm edm_table">',
          '<tr>',
            '<th>span</th>',
            '<th>span string</th>',
            '<th>gold predicates</th>',
            '<th>system predicates</th>',
          '</tr>',
        '</table>',
    '</div>'
].join("\n");

Templates.edm_entry = [
    '<tr class="<%=span_class%>">',
      '<td><%= span_start %>:<%= span_end %></td>',
      '<td align="right"><%= span_text %></td>',
      '<td><%= gold %></td>',
      '<td><%= system %></td>',
    '</tr>',
].join("\n");

Templates.successStatus = [
    '<p id="parse-status">Showing <%= numResults %> of <%= readings %> analyses.</p>',
    '<div id="text-input"><%= input %></div>'
].join("\n");

function EDM(parentElement, edm) {
  var self = { };
  var predicates = [];
  $.each(edm.predicates, function(i, d) {
    predicates.push(d);
  });
  predicates = predicates.sort(function(a, b) {
    return b.len - a.len;
  });
  $.each(predicates, function(i, d) {
    var gold = "",
        system = "",
        gold_set = new Set([]),
        system_set = new Set([]);
    if (typeof d.predicate.gold !== 'undefined') {
      gold_set = new Set(d.predicate.gold);
    }
    if (typeof d.predicate.system !== 'undefined') {
      system_set = new Set(d.predicate.system);
    }
    var intersection = new Set([...gold_set].filter(x => system_set.has(x)));
    var gold_unique = new Set([...gold_set].filter(x => !system_set.has(x)));
    var system_unique = new Set([...system_set].filter(x => !gold_set.has(x)));
    for (let item of intersection) {
      gold += '<span class="common">' + item + '</span>';
      system += '<span class="common">' + item + '</span>';
    }
    for (let item of gold_unique) {
      gold += '<span class="gold">' + item + '</span>';
    }
    for (let item of system_unique) {
      system += '<span class="system">' + item + '</span>';
    }
    span_class = "";
    if (gold_unique.size === 0 && system_unique.size === 0) {
      span_class = "common";
    } else {
      span_class = "unique";
    }
    $(parentElement).find("tr:last").after(
      $(Templates.edm_entry({
        span_start: d.start,
        span_end: d.end,
        span_text: d.span,
        gold: gold,
        system: system,
        span_class: span_class,
      }))
    );
  });
  return self;
}

// Precompile the templates
for (var template in Templates) {
    if (Templates.hasOwnProperty(template)) {
        Templates[template] = _.template(Templates[template]);
    }
}


function setInlineStyles(svg, emptySvgDeclarationComputed) {
    // Applies computed CSS styles for an SVG to the element as inline
    // styles. This allows SVG elements to be saved as SVG and PNG images that
    // display as viewed in the browser.
    // This function taken from the svg-crowbar tool:
    // https://github.com/NYTimes/svg-crowbar/blob/gh-pages/svg-crowbar-2.js

    function explicitlySetStyle (element) {
        var cSSStyleDeclarationComputed = getComputedStyle(element);
        var i, len, key, value;
        var computedStyleStr = "";
        for (i=0, len=cSSStyleDeclarationComputed.length; i<len; i++) {
            key=cSSStyleDeclarationComputed[i];
            value=cSSStyleDeclarationComputed.getPropertyValue(key);
            if (value!==emptySvgDeclarationComputed.getPropertyValue(key)) {
                computedStyleStr+=key+":"+value+";";
            }
        }
        element.setAttribute('style', computedStyleStr);
    }
    function traverse(obj){
        var tree = [];
        tree.push(obj);
        visit(obj);
        function visit(node) {
            if (node && node.hasChildNodes()) {
                var child = node.firstChild;
                while (child) {
                    if (child.nodeType === 1 && child.nodeName != 'SCRIPT'){
                        tree.push(child);
                        visit(child);
                    }
                    child = child.nextSibling;
                }
            }
        }
        return tree;
    }
    // hardcode computed css styles inside svg
    var allElements = traverse(svg);
    var i = allElements.length;
    while (i--){
        explicitlySetStyle(allElements[i]);
    }
}


function Result(result, parent) {
    var resultId = result['result-id'];
    var $path = window.location.pathname;
    var EDM_ONLY = false;
    if ($path.endsWith("edm.html")) {
      EDM_ONLY = true;
    }

    if (EDM_ONLY) {
      if (result.derivation || result.dmrs || result.mrs) {
        return;
      }
    }

    // Create and attach the DOM element that will contain the Result
    var $result = $(Templates.result({'resultId': resultId})).appendTo(parent);

    // Create this object
    var self = {
        data : result,
        num : resultId,
        element: $result[0],
        saveVizSvg : function(vizType) {
            var svg = self[vizType].element;
            setInlineStyles(svg, emptySvgDeclarationComputed);
            var svgData = self[vizType].element.outerHTML;
            var svgBlob = new Blob([svgData], {type: 'image/svg+xml;charset=utf-8'});
            var DOMURL = window.URL || window.webkitURL || window;
            var url = DOMURL.createObjectURL(svgBlob);
            triggerDownload(url, vizType+'.svg');
        },
        saveVizPng : function(vizType) {
            var svg = self[vizType].element;
            setInlineStyles(svg, emptySvgDeclarationComputed);
            var height = svg.getBoundingClientRect().height;

            // Save SVG to a canvas
            var canvas = $('<canvas>')[0];
            var ctx = canvas.getContext('2d');
            var bbox = svg.getBBox();
            ctx.canvas.height = bbox.height;
            ctx.canvas.width = bbox.width;
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);

            // convert SVG to dataUrl
            var data = (new XMLSerializer()).serializeToString(svg);
            var svgBlob = new Blob([data], {type: 'image/svg+xml;charset=utf-8'});
            var DOMURL = window.URL || window.webkitURL || window;
            var url = DOMURL.createObjectURL(svgBlob);

            var img = new Image();
            img.onload = function () {
                ctx.drawImage(img, 0, 0);
                DOMURL.revokeObjectURL(url);

                var imgURI = canvas
                        .toDataURL('image/png')
                        .replace('image/png', 'image/octet-stream');

                // on the image load, actually download it
                triggerDownload(imgURI, vizType+'.png');
            };
            img.src = url;
        },
        saveVizLatex : function(vizType, resultId) {
            var data = {
                input: $('#input-text').val(),
                results: $('#input-results').val()
            };
            data[vizType] = 'latex';

            $.ajax({
                url: CURR_GRAMMAR.url,
                dataType: 'json',
                data: data,
                success: function(data){
                    var latex = data.results[resultId][vizType];
                    var textBlob = new Blob([latex], {type: "text/plain;charset=utf-8"});
                    var DOMURL = window.URL || window.webkitURL || window;
                    var url = DOMURL.createObjectURL(textBlob);
                    triggerDownload(url, vizType+'.tex');
                },
                error: function(data){
                    // TODO: better error handling
                    alert("Sorry, something went wrong saving LaTex.");
                }
        });
        }
    };

    var $inner = $result.find('.result-inner');

    // Add data structures as per the available data
    if (self.data.derivation) {
        var $viz = $(Templates.viz({vizType:'tree'})).appendTo($inner);
        self.tree = {element: drawTree($viz[0], self.data.derivation)};
    }
    if (self.data.mrs) {
        var $viz = $(Templates.viz({vizType:'mrs'})).appendTo($inner);
        self.mrs = MRS($viz[0], self.data.mrs);
    }
    if (self.data.dmrs) {
        var $viz = $(Templates.viz({vizType:'dmrs'})).appendTo($inner);
        self.dmrs = DMRS($viz[0], self.data.dmrs);
    }
    if (self.data.edm) {
        var $stat = self.data.edm.stats;
        var $edm = $(Templates.edm({total: $stat.total,
                                    gold: $stat.gold,
                                    system: $stat.system,
                                    common: $stat.common,
                                    named: $stat.named,
                                    compound: $stat.compound,
                                    unknown: $stat.unknown,
                                    udef_q: $stat.udef_q,
                                    proper_q: $stat.proper_q,
                                    yofc: $stat.yofc,
                                    predicate: $stat.predicate,
                                    predicate_arg: $stat.predicate_arg})).appendTo($inner);
        self.edm = EDM($edm, self.data.edm);
    }


    //Add various event bindings to things in the visualisations
    $result.find('.viz').hover(
        function(event) {
            $(this).find('.tools').removeClass('hidden');
        },
        function(event) {
            $(this).find('.tools').addClass('hidden');
        }
    ).each(function(index) {
        var vizType = this.dataset.viz;
        $(this).find('.save').click(function(event){
            if (this.dataset.img == 'svg') {
                self.saveVizSvg(vizType);
            } else if (this.dataset.img == 'png') {
                self.saveVizPng(vizType);
            } else if (this.dataset.img == 'latex') {
                var resultId = $(this).closest('.result').data('result');
                self.saveVizLatex(vizType, resultId);
            }
        });
    });

    // Return this object
    return self;
}


function triggerDownload (uri, filename) {
    var evt = new MouseEvent('click', {
        view: window,
        bubbles: false,
        cancelable: true
    });

    var a = document.createElement('a');
    a.setAttribute('download', filename);
    a.setAttribute('href', uri);
    a.setAttribute('target', '_blank');
    a.dispatchEvent(evt);
}


function getQueryVariable(variable) {
    var query = window.location.search.substring(1);
    var vars = query.split("&");
    for (var i=0;i<vars.length;i++) {
        var pair = vars[i].split("=");
        if (pair[0] == variable)
            return pair[1];
    }

    return false;
}


function doResults(data) {
    // Update the status
    $(Templates.successStatus({
        'input': data.input,
        'readings': data.readings,
        'numResults': data.results.length}))
        .appendTo($('#results-info').empty());

    //Create and add the results
    var parent = $('#results-container').empty()[0];

    RESULTLIST = [];

    for (var i=0; i < data.results.length; i++) {
        var result = Result(data.results[i], parent);
        RESULTLIST.push(result);
    }
}


var MAX = 1769;
function updateLinks(index, edm_diff_only) {
  var previous = parseInt(index) - 1;
  var next = parseInt(index) + 1;
  var current = index;
  if (previous === -1) {
      previous = MAX;
  }
  if (next >= MAX) {
      next = 0;
  }
  href_common = "?edm_diff_only=";
  if (edm_diff_only === false || edm_diff_only === "0") {
    href_common += "0";
  } else {
    href_common += "1";
  }

  $('a.page-link').each(function(l) {

      if ($(this).text() === "Previous") {
        $(this).attr("href", href_common + "&index=" + previous.toString());
      } else if ($(this).text() === "Next") {
        $(this).attr("href", href_common + "&index=" + next.toString());
      } else if ($(this).text() === "EDM") {
        $(this).attr("href", "edm.html" + href_common + "&index=" + current.toString());
      } else if ($(this).text() === "Everything") {
        $(this).attr("href", "index.html" + href_common + "&index=" + current.toString());
      }
  });
}

function edm_show(edm_diff_only, index) {
    if (edm_diff_only === true) {
        $("tr.common").removeClass('hidden');
        $('#edm_option').text("Difference Only");
        $('#edm_option').attr('href', "?edm_diff_only=1&index="+index);
    } else {
        $("tr.common").addClass('hidden');
        $('#edm_option').text("Show All");
        $('#edm_option').attr('href', "?edm_diff_only=0&index="+index);
    }
}
$(document).ready(function(){
    var index = getQueryVariable('index'),
        edm_diff_only = getQueryVariable('edm_diff_only');
    if (index === false) {
        index = 0;
    }
    updateLinks(index, edm_diff_only);
    $.getJSON("./data/n" + index.toString() + ".json", function(data) {
        doResults(data);
        if (edm_diff_only === false || edm_diff_only === 0 || edm_diff_only === "0") {
            edm_show(true, index);
        } else {
            edm_show(false, index);
        }
    });

    // add empty svg element for use in saving SVGs as SVGs and PNGs
    var emptySvg = window.document.createElementNS("http://www.w3.org/2000/svg", 'svg');
    window.document.body.appendChild(emptySvg);
    emptySvgDeclarationComputed = getComputedStyle(emptySvg);

});
