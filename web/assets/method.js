// Method page: fill the test-results table and the drift section from summary.json.
(function () {
  "use strict";

  var el = function (tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };
  var mean = function (xs) { return xs.reduce(function (a, b) { return a + b; }, 0) / xs.length; };

  function withCi(value, ci) {
    var td = el("td", null, value + " ");
    td.appendChild(el("span", "ci", ci));
    return td;
  }

  function row(label, cells, cls, tag) {
    var tr = el("tr", cls);
    var th = el("th", null, label);
    th.scope = "row";
    if (tag) th.appendChild(el("span", "tag", tag));
    tr.appendChild(th);
    cells.forEach(function (c) { tr.appendChild(typeof c === "string" ? el("td", null, c) : c); });
    return tr;
  }

  function render(summary) {
    Common.fillCommon(summary);
    var pct = Common.pct;
    var body = document.getElementById("results");
    var names = Object.keys(summary.projects);

    names.forEach(function (name) {
      var p = summary.projects[name];
      var c = p.coverage["90"];
      var mae = withCi(p.mae.toFixed(2), p.mae_ci[0].toFixed(2) + "–" + p.mae_ci[1].toFixed(2));
      if (p.mae > p.mae_median_baseline) mae.classList.add("worse");
      body.appendChild(row(name, [
        String(p.n_test),
        mae,
        p.mae_median_baseline.toFixed(2),
        withCi(pct(c.coverage), pct(c.ci[0]) + "–" + pct(c.ci[1])),
        c.mean_width.toFixed(1),
      ], p.drift ? "drift-row" : null, p.drift ? "drift" : null));
    });

    var ps = names.map(function (n) { return summary.projects[n]; });
    body.appendChild(row("Mean of projects", [
      String(ps.reduce(function (a, p) { return a + p.n_test; }, 0)),
      mean(ps.map(function (p) { return p.mae; })).toFixed(2),
      mean(ps.map(function (p) { return p.mae_median_baseline; })).toFixed(2),
      pct(summary.overall.coverage_90, 1),
      summary.overall.mean_width_90.toFixed(1),
    ], "total"));

    var gains = names.map(function (n) { return n + " " + Common.signedPct(summary.projects[n].gain_vs_median); });
    document.getElementById("results-note").textContent =
      "MAE and width are in story points. Against always predicting the training median, the MAE changes by " +
      gains.join(", ") + ": the issue text alone carries little signal about effort, so the interval is the main output.";

    var drift = Common.driftProject(summary);
    if (drift) {
      var d = drift.info;
      document.querySelector('[data-drift="medians"]').textContent = d.median_sp_train + "→" + d.median_sp_test;
      document.querySelector('[data-drift="name"]').textContent = drift.name;
      document.querySelector('[data-drift="body"]').textContent =
        "Intervals calibrated on older issues sat too high for new ones. At the 50% level they covered only " +
        pct(d.coverage["50"].coverage) + " of later issues instead of 50%. At 90% they still held (" +
        pct(d.coverage["90"].coverage) + "), because a wide margin reaches down to the new, smaller estimates.";
    }
  }

  Common.loadSummary().then(render).catch(Common.showLoadError);
})();
