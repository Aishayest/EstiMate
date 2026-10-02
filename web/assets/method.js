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
      body.appendChild(row(Common.teamName(name), [
        String(p.n_test),
        mae,
        p.mae_median_baseline.toFixed(2),
        withCi(pct(c.coverage), pct(c.ci[0]) + "–" + pct(c.ci[1])),
        c.mean_width.toFixed(1),
      ], p.drift ? "drift-row" : null, p.drift ? "changed" : null));
    });

    var ps = names.map(function (n) { return summary.projects[n]; });
    body.appendChild(row("All four teams", [
      String(ps.reduce(function (a, p) { return a + p.n_test; }, 0)),
      mean(ps.map(function (p) { return p.mae; })).toFixed(2),
      mean(ps.map(function (p) { return p.mae_median_baseline; })).toFixed(2),
      pct(summary.overall.coverage_90, 1),
      summary.overall.mean_width_90.toFixed(1),
    ], "total"));

    var gains = names.map(function (n) {
      var g = summary.projects[n].gain_vs_median;
      return Common.teamName(n) + " " + Math.abs(g * 100).toFixed(1) + "% " + (g >= 0 ? "smaller" : "larger");
    });
    document.getElementById("results-note").textContent =
      "All numbers are in story points. \u201cAvg. miss\u201d is how far the single guess was from the team's real " +
      "size (MAE). Compared with always guessing the team's usual size, the tool's miss is " + gains.join(", ") +
      ". The text alone says little about effort, so the useful part is the range: it caught the real size about " +
      "as often as promised.";

    var drift = Common.driftProject(summary);
    if (drift) {
      var d = drift.info;
      document.querySelector('[data-drift="medians"]').textContent = d.median_sp_train + "→" + d.median_sp_test;
      document.querySelector('[data-drift="name"]').textContent = Common.teamName(drift.name);
      document.querySelector('[data-drift="body"]').textContent =
        "Ranges set on older tasks sat too high for the new, smaller ones. The narrow 50% range caught the real size " +
        "only " + pct(d.coverage["50"].coverage) + " of the time instead of half. The safe 90% range still held (" +
        pct(d.coverage["90"].coverage) + "), because it is wide enough to reach down to the smaller sizes.";
    }
  }

  Common.loadSummary().then(render).catch(Common.showLoadError);
})();
