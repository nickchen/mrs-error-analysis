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
        '<div><table id="all_stats" class="edm edm_table edm_stats">',
          '<tr><th>Total</th><td class="stats_value"><%= total %></td></tr>',
          '<tr><th>Common </th><td class="stats_value"><%= common %></td></tr>',
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

Templates.amr = [
    '<div name="test">',
      '<div><span class="common">Common</span>/<span class="gold">Gold Only</span>/<span class="system">System Only</span></div>',
      '<div><table id="all_stats" class="edm edm_table edm_stats">',
        '<tr><th>Total</th><td class="stats_value"><%= total %></td></tr>',
        '<tr><th>Common </th><td class="stats_value"><%= common %></td></tr>',
        '</table></div>',
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

Templates.stat_entry = [
    '<tr class="<%=stat_class%>"><th><%= padding %><%= name %></th><td><%= value %></td></tr>'].join("\n");

Templates.successStatus = [
    '<p id="parse-status">Showing <%= numResults %> of <%= readings %> analyses.</p>',
    '<div id="text-input"><%= input %></div>'
].join("\n");

function EDM(parentElement, edm) {
  var self = { };
  var predicates = [];
  var stats = edm.stats;
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
    ); // end parentElement
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

function capitalizeFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
}

function number_and_percentage(value, total) {
  if (value != total) {
    var v = Number.parseInt(value),
        t = Number.parseInt(total);
    var precentage = ((v / t) * 100).toFixed(2) + '%';
    return "(" + precentage + ") " + value;
  }
  return value;
}

function update_sub_types($stat, $html, prefix, error_types, stats_class) {
  $html.append(
    $(Templates.stat_entry({name: capitalizeFirstLetter(prefix),
        padding: "&nbsp;&nbsp;", stat_class: stats_class,
        value: $stat[prefix]["count"]})));
  for (var i = 0; i < error_types.length; i++) {
    var error_str = error_types[i];
    if (error_str in $stat[prefix]) {
      $html.append(
        $(Templates.stat_entry({name: capitalizeFirstLetter(error_str),
            padding: "&nbsp;&nbsp;&nbsp;&nbsp;", stat_class: stats_class,
            value: $stat[prefix][error_str]["count"]})));
        var subtotal = $stat[prefix][error_str]["count"];
        $.each($stat[prefix][error_str], function(name, value) {
          if (value instanceof Object) {
            $html.append(
              $(Templates.stat_entry({name: name,
                  padding: "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;", stat_class: "",
                  value: number_and_percentage($stat[prefix][error_str][name]["count"], subtotal)})));
            $.each(value, function(k, v) {
              if (k !== "count") {
                $html.append(
                  $(Templates.stat_entry({name: k,
                      padding: "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;", stat_class: "",
                      value: v})));
              }
            });
          }
        });
    }
  }
}
function update_type_stats(type_str, $stat, $edm) {
    var stats_class = type_str + "_row";
    var $html = $($edm).find("#all_stats");

    if ("format disagreement" in $stat) {
      $.each($stat["format disagreement"], function(name, value) {
        $html.append(
          $(Templates.stat_entry({name: name,
              padding: "&nbsp;&nbsp;", stat_class: "", value: value})));
      });
    }
    var stats_type = ["not well formed", "not connected", "unknown"];
    for (var i = 0; i < stats_type.length; i++) {
      var stats_type_str = stats_type[i];
      if (stats_type_str in $stat) {
        $html.append(
          $(Templates.stat_entry({name: stats_type_str + " count",
              padding: "&nbsp;&nbsp;", stat_class: "",
              value: $stat[stats_type_str]})));
      }
    }

    if ("predicate error" in $stat) {
      var prefix = "predicate error";
      var error_types = ["not in gold", "not in system", "not in erg"];
      update_sub_types($stat, $html, prefix, error_types, stats_class);
    }
    if ("argument error" in $stat) {
      var prefix = "argument error";
      var error_types = ["incorrect", "extra", "duplicated"];
      update_sub_types($stat, $html, prefix, error_types, stats_class);
    }
}

function update_stats($stat, $edm, is_summary) {
  var $html = $($edm).find("#all_stats");
  if (is_summary && $stat.summary !== undefined) {
    $html.append(
      $(Templates.stat_entry({name: "Summary",
          padding: "", stat_class: "",
          value: $stat.summary["count"]})));
    var summary_array = ["matched", "has predicate error", "has predicate incorrect arg",
        "has predicate error extra arg", "has predicate error extra arg and incorrect arg"];
    for (var i = 0; i < summary_array.length; i++) {
      var summary_str = summary_array[i];
      if (summary_str in $stat.summary) {
        $html.append(
          $(Templates.stat_entry({name: summary_str,
              padding: "&nbsp;&nbsp;", stat_class: "",
              value: number_and_percentage($stat.summary[summary_str], $stat.summary["count"]) })));
      }
    }
  }
  var type_array = ["gold", "system"];
  for (var i = 0; i < type_array.length; i++) {
    var type_str = type_array[i];
    $html.append(
      $(Templates.stat_entry({name: capitalizeFirstLetter(type_array[i]) + " Only",
          padding: "", stat_class: "",
          value: $stat.shared[type_str]})));
    var type_stat = $stat[type_str + "_stats"];
    if (type_stat !== undefined) {
      update_type_stats(type_str, type_stat, $edm);
    }
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
        var $edm = $(Templates.edm({total: $stat.shared.total,
                                    common: $stat.shared.common})).appendTo($inner);
        update_stats($stat, $edm, false);
    }
    if (self.data.amr) {
        var $stat = self.data.amr;
        console.log($stat);
        var $view = $(Templates.amr({total: $stat.shared.total,
                                    common: $stat.shared.common})).appendTo($inner);
        self.edm = EDM($view, self.data.amr);
        update_stats($stat, $view, false);
    }
    if (self.data.summary) {
      var $stat = self.data.summary;
      var $view = $(Templates.amr({total: $stat.shared.total,
                                  common: $stat.shared.common,
                                  predicate: $stat.shared.predicate,
                                  predicate_arg: $stat.predicate_arg})).appendTo($inner);
      update_stats($stat, $view, true);
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


var MAX = 1799;
function updateLinks(index, edm_diff_only) {
  var previous = parseInt(index) - 1;
  var next = parseInt(index) + 1;
  var current = index;
  if (previous === -1) {
      previous = MAX;
  }
  if (next > MAX) {
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
