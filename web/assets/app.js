// Estimate page: project/level selection, in-browser inference, interval axis, badges.
(function () {
  "use strict";

  var LEVELS = [50, 80, 90];
  var SCALE_MARKS = [1, 2, 3, 5, 8, 13, 21, 40];
  var AXIS_MAXES = [13, 21, 40];
  var N_EXAMPLES_SHOWN = 3;

  var state = {
    summary: null,
    project: null,
    level: 90,
    model: null,
    exampleOffset: 0,
    loaded: null, // test example currently in the textarea, if unedited
    result: null,
  };
  var models = {};

  var $ = function (id) { return document.getElementById(id); };
  var el = function (tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };

  function loadModel(project) {
    if (!models[project]) {
      models[project] = fetch("artifacts/" + project + ".json")
        .then(function (r) {
          if (!r.ok) throw new Error(project + ".json: HTTP " + r.status);
          return r.json();
        })
        .then(Estimator.prepare);
    }
    return models[project];
  }

  // ---------- controls ----------

  function renderChips(container, items, isOn, onPick) {
    container.replaceChildren();
    items.forEach(function (item) {
      var b = el("button", "chip", item.label);
      b.type = "button";
      b.setAttribute("aria-pressed", String(isOn(item.value)));
      b.addEventListener("click", function () { onPick(item.value); });
      container.appendChild(b);
    });
  }

  function renderProjects() {
    var names = Object.keys(state.summary.projects);
    renderChips($("projects"), names.map(function (n) { return { label: n, value: n }; }),
      function (v) { return v === state.project; }, selectProject);
  }

  function renderLevels() {
    renderChips($("levels"), LEVELS.map(function (l) { return { label: l + "%", value: l }; }),
      function (v) { return v === state.level; },
      function (l) {
        state.level = l;
        renderLevels();
        if (state.result) run();
        else renderBadge();
      });
  }

  function renderExamples() {
    var box = $("examples");
    box.replaceChildren();
    if (!state.model) return;
    var list = state.model.examples;
    for (var i = 0; i < N_EXAMPLES_SHOWN && i < list.length; i++) {
      var ex = list[(state.exampleOffset + i) % list.length];
      var b = el("button", "example", "↳ " + ex.title);
      b.type = "button";
      b.title = ex.key + " — " + ex.title;
      b.setAttribute("aria-pressed", String(state.loaded !== null && state.loaded.key === ex.key));
      b.addEventListener("click", loadExample.bind(null, ex));
      box.appendChild(b);
    }
  }

  function loadExample(ex) {
    var text = ex.title + (ex.description ? "\n\n" + ex.description : "");
    $("task").value = text;
    state.loaded = { key: ex.key, sp: ex.sp, text: text };
    renderExamples();
    run();
  }

  function selectProject(project) {
    state.project = project;
    state.model = null;
    state.exampleOffset = 0;
    state.loaded = null;
    renderProjects();
    renderExamples();
    document.querySelectorAll("[data-project-name]").forEach(function (n) { n.textContent = project; });
    $("out-project").textContent = project;
    setBusy(true);
    return loadModel(project).then(function (model) {
      if (state.project !== project) return; // a newer selection won
      state.model = model;
      setBusy(false);
      renderExamples();
      if ($("task").value.trim()) run();
      else { clearOutput(); renderBadge(); }
    });
  }

  function setBusy(busy) {
    var go = $("go");
    go.disabled = busy;
    go.firstElementChild.textContent = busy ? "Loading model…" : "Estimate";
  }

  // ---------- output ----------

  function clearOutput() {
    state.result = null;
    $("point").textContent = "—";
    $("interval").textContent = "—";
    $("interval-sub").textContent = "Enter an issue and press Estimate.";
    $("level-label").textContent = String(state.level);
    renderAxis(null);
    $("truth").hidden = true;
    $("note").hidden = true;
    $("neighbours").replaceChildren(emptyRow("Estimate an issue to see the most similar training issues."));
  }

  function run() {
    var text = $("task").value.trim();
    if (!state.model) return;
    if (!text) { clearOutput(); renderBadge(); return; }
    var r = Estimator.estimate(state.model, text, state.level);
    state.result = r;

    $("level-label").textContent = String(state.level);
    $("point").textContent = String(Math.round(r.point));
    $("interval").textContent = r.lo.toFixed(1) + "–" + r.hi.toFixed(1);
    $("interval-sub").textContent = "ŷ = " + r.point.toFixed(1) + " · as whole SP: " +
      (r.loInt === r.hiInt ? r.loInt : r.loInt + "–" + r.hiInt);
    renderAxis(r);
    renderBadge();
    renderTruth(r);

    var note = $("note");
    note.hidden = r.known > 0;
    note.textContent = "None of these words are in the " + state.project + " training vocabulary, " +
      "so the estimate is the model's baseline value.";

    renderNeighbours(Estimator.nearest(state.model, r.vec, 4));
  }

  function renderTruth(r) {
    var truth = $("truth");
    var ex = state.loaded;
    if (!ex) { truth.hidden = true; return; }
    var inside = ex.sp >= r.loInt && ex.sp <= r.hiInt;
    truth.replaceChildren(
      document.createTextNode(ex.key + " was actually "),
      el("strong", null, ex.sp + " SP"),
      document.createTextNode(" — " + (inside ? "inside" : "outside") + " the " + state.level + "% interval.")
    );
    truth.hidden = false;
  }

  function renderAxis(r) {
    var axis = $("axis");
    axis.replaceChildren();
    var hi = r ? r.hi : 13;
    var max = AXIS_MAXES.find(function (m) { return m >= hi; }) || Math.ceil(hi);
    var pos = function (v) { return (Math.min(v, max) / max * 100).toFixed(2) + "%"; };
    var marks = SCALE_MARKS.filter(function (v) { return v <= max; });
    var step = max <= 13 ? 0.5 : 1;

    axis.classList.toggle("empty", !r);
    if (r) {
      var lo = el("span", "edge", r.lo.toFixed(1));
      lo.style.left = pos(r.lo);
      var hiLabel = el("span", "edge", r.hi > max ? "> " + max : r.hi.toFixed(1));
      hiLabel.style.left = pos(r.hi);
      var band = el("div", "band");
      band.style.left = pos(r.lo);
      band.style.width = ((Math.min(r.hi, max) - r.lo) / max * 100).toFixed(2) + "%";
      var point = el("div", "point");
      point.style.left = pos(r.point);
      axis.append(lo, hiLabel, band, point);
      axis.setAttribute("aria-label", state.level + "% interval from " + r.lo.toFixed(1) + " to " +
        r.hi.toFixed(1) + " story points, point estimate " + r.point.toFixed(1));
    } else {
      axis.setAttribute("aria-label", "No estimate yet");
    }
    axis.appendChild(el("div", "line"));
    for (var v = 0; v <= max + 1e-9; v += step) {
      var t = el("div", "tick");
      t.style.left = pos(v);
      var tall = marks.indexOf(v) !== -1 || (max <= 13 && v % 1 === 0) || v === 0;
      t.style.height = tall ? "12px" : "6px";
      axis.appendChild(t);
    }
    marks.forEach(function (m) {
      var label = el("span", "tick-label", String(m));
      label.style.left = pos(m);
      axis.appendChild(label);
    });
  }

  function renderBadge() {
    var badge = $("badge");
    var info = state.summary && state.project ? state.summary.projects[state.project] : null;
    if (!info) return;
    var c = info.coverage[String(state.level)];
    var cov = Common.pct(c.coverage);
    var ci = Common.pct(c.ci[0]) + "–" + Common.pct(c.ci[1]);
    if (info.drift) {
      badge.className = "badge warn";
      $("badge-title").textContent = "! Drift";
      $("badge-body").textContent = "This team's typical estimate fell from " + info.median_sp_train +
        " to " + info.median_sp_test + " SP over time, so intervals built on older issues sit too high. At 50% " +
        "they covered only " + Common.pct(info.coverage["50"].coverage) + " of later issues." +
        (state.level === 50 ? "" : " At " + state.level + "%: " + cov + " (95% CI " + ci + ").");
    } else {
      badge.className = "badge ok";
      $("badge-title").textContent = "✓ Held on test";
      $("badge-body").textContent = "On " + info.n_test + " later " + state.project + " issues the model never " +
        "saw, " + state.level + "% intervals contained the real estimate " + cov + " of the time (95% CI " + ci +
        "). Mean width " + c.mean_width.toFixed(1) + " SP.";
    }
  }

  function emptyRow(text) {
    var tr = el("tr", "empty-row");
    var td = el("td", null, text);
    td.colSpan = 4;
    tr.appendChild(td);
    return tr;
  }

  function renderNeighbours(list) {
    var body = $("neighbours");
    if (!list.length) {
      body.replaceChildren(emptyRow("No training issue shares a word with this text."));
      return;
    }
    body.replaceChildren.apply(body, list.map(function (n, i) {
      var tr = el("tr");
      tr.appendChild(el("td", "num", String(i + 1).padStart(2, "0")));
      var issue = el("td", null, n.title);
      issue.appendChild(el("span", "key", n.key));
      tr.appendChild(issue);
      tr.appendChild(el("td", "sp", String(n.sp)));
      var sim = el("td", "sim");
      var bar = el("div", "simbar");
      var track = el("div");
      var fill = el("i");
      fill.style.width = (n.similarity * 100).toFixed(0) + "%";
      track.appendChild(fill);
      bar.append(track, el("span", null, n.similarity.toFixed(2).replace(/^0/, "")));
      sim.appendChild(bar);
      tr.appendChild(sim);
      return tr;
    }));
  }

  // ---------- facts ----------

  function renderStats(summary) {
    var set = function (key, text) { document.querySelector('[data-stat="' + key + '"]').textContent = text; };
    var names = Object.keys(summary.projects);
    set("coverage", Common.pct(summary.overall.coverage_90, 1));
    set("coverage-text", "Mean test coverage at a 90% target across " + names.length +
      " projects. Mean width " + summary.overall.mean_width_90.toFixed(1) + " SP.");

    var gains = names.map(function (n) { return [n, summary.projects[n].gain_vs_median]; })
      .sort(function (a, b) { return b[1] - a[1]; });
    set("gain", "≤" + Math.round(gains[0][1] * 100) + "%");
    set("gain-text", "MAE gain over always predicting the median: " + gains.map(function (g) {
      return g[0] + " " + Common.signedPct(g[1]);
    }).join(", ") + ". Effort mostly lives outside the issue text.");

    var drift = Common.driftProject(summary);
    if (drift) {
      set("drift", drift.info.median_sp_train + "→" + drift.info.median_sp_test);
      set("drift-text", "Typical estimate in " + drift.name + ", training period → test period. " +
        "The guarantee holds only while new issues look like old ones.");
    }
  }

  // ---------- wiring ----------

  $("go").addEventListener("click", run);
  $("task").addEventListener("keydown", function (e) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); run(); }
  });
  $("task").addEventListener("input", function () {
    if (state.loaded && $("task").value !== state.loaded.text) {
      state.loaded = null;
      renderExamples();
      $("truth").hidden = true;
    }
  });
  $("shuffle").addEventListener("click", function () {
    if (!state.model) return;
    state.exampleOffset = (state.exampleOffset + N_EXAMPLES_SHOWN) % state.model.examples.length;
    renderExamples();
  });

  renderLevels();
  clearOutput();
  Common.loadSummary()
    .then(function (summary) {
      state.summary = summary;
      Common.fillCommon(summary);
      renderStats(summary);
      return selectProject(Object.keys(summary.projects)[0]);
    })
    .then(function () {
      if (state.model && state.model.examples.length) loadExample(state.model.examples[0]);
    })
    .catch(Common.showLoadError);
})();
