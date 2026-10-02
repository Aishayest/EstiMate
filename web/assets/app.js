// "Try it" page: team/level selection, in-browser estimate, plain-language answer.
(function () {
  "use strict";

  var LEVELS = [50, 80, 90];
  var LEVEL_LABELS = { 50: "50% · narrow", 80: "80%", 90: "90% · safe" };
  var OUT_OF = { 50: "1 in 2", 80: "4 in 5", 90: "9 in 10" };
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
  var points = function (n) { return n + (n === 1 ? " point" : " points"); };
  var range = function (lo, hi) { return lo === hi ? points(lo) : lo + "–" + points(hi); };

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
    renderChips($("projects"), names.map(function (n) { return { label: Common.teamName(n), value: n }; }),
      function (v) { return v === state.project; }, selectProject);
    var team = Common.TEAMS[state.project];
    var info = state.summary.projects[state.project];
    $("team-about").textContent = team.name + " builds " + team.about + ". Its typical task was sized at " +
      points(info.median_sp_train) + ".";
  }

  function renderLevels() {
    renderChips($("levels"), LEVELS.map(function (l) { return { label: LEVEL_LABELS[l], value: l }; }),
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
    $("out-project").textContent = Common.teamName(project);
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
    go.firstElementChild.textContent = busy ? "Loading…" : "Estimate";
  }

  // ---------- output ----------

  function clearOutput() {
    state.result = null;
    $("point").textContent = "—";
    $("point-unit").textContent = "points";
    $("interval").textContent = "—";
    $("interval-sub").textContent = "Paste a task and press Estimate.";
    renderAxis(null);
    $("truth").hidden = true;
    $("note").hidden = true;
    $("neighbours").replaceChildren(emptyRow("Estimate a task to see similar ones this team has already sized."));
  }

  function run() {
    var text = $("task").value.trim();
    if (!state.model) return;
    if (!text) { clearOutput(); renderBadge(); return; }
    var r = Estimator.estimate(state.model, text, state.level);
    state.result = r;

    var point = Math.max(1, Math.round(r.point));
    $("point").textContent = String(point);
    $("point-unit").textContent = point === 1 ? "point" : "points";
    $("interval").textContent = range(r.loInt, r.hiInt);
    $("interval-sub").textContent = "Built to catch the team's real size in " + OUT_OF[state.level] +
      " tasks. Exact guess: " + r.point.toFixed(1) + ".";
    renderAxis(r);
    renderBadge();
    renderTruth(r);

    var note = $("note");
    note.hidden = r.known > 0;
    note.textContent = "The tool doesn't recognise any words in this text, so it falls back to this team's " +
      "usual size. Try adding more detail.";

    renderNeighbours(Estimator.nearest(state.model, r.vec, 4));
  }

  function renderTruth(r) {
    var truth = $("truth");
    var ex = state.loaded;
    if (!ex) { truth.hidden = true; return; }
    var inside = ex.sp >= r.loInt && ex.sp <= r.hiInt;
    truth.className = "truth " + (inside ? "hit" : "miss");
    truth.replaceChildren(
      document.createTextNode((inside ? "✓ " : "✗ ") + "The team really sized " + ex.key + " at "),
      el("strong", null, points(ex.sp)),
      document.createTextNode(inside
        ? " — inside the range."
        : " — outside the range. That is expected now and then: a " + state.level + "% range should miss about " +
          { 50: "1 in 2", 80: "1 in 5", 90: "1 in 10" }[state.level] + " tasks.")
    );
    truth.hidden = false;
  }

  function renderAxis(r) {
    var axis = $("axis");
    axis.replaceChildren();
    var top = r ? r.hiInt : 13;
    var max = AXIS_MAXES.find(function (m) { return m >= top; }) || Math.ceil(top);
    var pos = function (v) { return (Math.min(v, max) / max * 100).toFixed(2) + "%"; };
    var marks = SCALE_MARKS.filter(function (v) { return v <= max; });
    var step = max <= 13 ? 0.5 : 1;

    axis.classList.toggle("empty", !r);
    if (r) {
      // The band covers the whole-point range shown in words; a single value gets a small bar.
      var lo = r.loInt === r.hiInt ? r.loInt - 0.25 : r.loInt;
      var hi = r.loInt === r.hiInt ? r.hiInt + 0.25 : r.hiInt;
      var loLabel = el("span", "edge", String(r.loInt));
      loLabel.style.left = pos(lo);
      var hiLabel = el("span", "edge", String(r.hiInt));
      hiLabel.style.left = pos(hi);
      var band = el("div", "band");
      band.style.left = pos(lo);
      band.style.width = ((Math.min(hi, max) - lo) / max * 100).toFixed(2) + "%";
      var point = el("div", "point");
      point.style.left = pos(r.point);
      axis.append(band, point);
      if (r.loInt !== r.hiInt) axis.append(loLabel, hiLabel);
      else axis.append(loLabel);
      axis.setAttribute("aria-label", "Likely range " + range(r.loInt, r.hiInt) +
        ", most likely " + points(Math.max(1, Math.round(r.point))));
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
    var name = Common.teamName(state.project);
    var c = info.coverage[String(state.level)];
    var cov = Common.pct(c.coverage);
    if (info.drift) {
      badge.className = "badge warn";
      $("badge-title").textContent = "! Be careful";
      $("badge-body").textContent = "This team started giving smaller sizes over time (typical task: " +
        info.median_sp_train + " → " + info.median_sp_test + " points). The tool learned from the older tasks, so it " +
        "tends to aim too high here. Its narrow 50% range caught the real answer only " +
        Common.pct(info.coverage["50"].coverage) + " of the time" +
        (state.level === 50 ? "." : "; the " + state.level + "% range still held: " + cov + ".");
    } else {
      badge.className = "badge ok";
      $("badge-title").textContent = "✓ Checked";
      $("badge-body").textContent = "We tested the " + state.level + "% range on " + info.n_test + " newer " + name +
        " tasks the tool had never seen. It caught the team's real size " + cov + " of the time (aim: " +
        state.level + "%).";
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
      body.replaceChildren(emptyRow("No task from this team shares a word with this text."));
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
      bar.append(track, el("span", null, Common.pct(n.similarity)));
      sim.appendChild(bar);
      tr.appendChild(sim);
      return tr;
    }));
  }

  // ---------- facts ----------

  function renderStats(summary) {
    var set = function (key, text) { document.querySelector('[data-stat="' + key + '"]').textContent = text; };
    var names = Object.keys(summary.projects);
    var nTest = names.reduce(function (a, n) { return a + summary.projects[n].n_test; }, 0);
    document.querySelectorAll("[data-n-test]").forEach(function (n) { n.textContent = nTest.toLocaleString("en-US"); });

    set("coverage", Common.pct(summary.overall.coverage_90, 1));
    set("coverage-text", "We asked for a range that catches the real answer 9 times in 10. On " +
      nTest.toLocaleString("en-US") + " newer tasks it did so " + Common.pct(summary.overall.coverage_90, 1) +
      " of the time. A typical range is about " + summary.overall.mean_width_90.toFixed(0) + " points wide.");

    var gains = names.map(function (n) { return [n, summary.projects[n].gain_vs_median]; })
      .sort(function (a, b) { return b[1] - a[1]; });
    var better = gains.filter(function (g) { return g[1] > 0; });
    var worse = gains.filter(function (g) { return g[1] < 0; });
    set("gain", "≤" + Math.round(gains[0][1] * 100) + "%");
    set("gain-text", "Guessing the exact number from text alone is hard. Compared with always guessing the " +
      "team's usual size, the tool is more accurate by " + better.map(function (g) {
        return Common.signedPct(g[1]).replace("+", "") + " on " + Common.teamName(g[0]);
      }).join(", ") + (worse.length ? ", and less accurate on " + worse.map(function (g) {
        return Common.teamName(g[0]);
      }).join(", ") : "") + ". The value is in the honest range.");

    var drift = Common.driftProject(summary);
    if (drift) {
      set("drift", drift.info.median_sp_train + "→" + drift.info.median_sp_test);
      set("drift-text", Common.teamName(drift.name) + "'s typical task shrank from " + drift.info.median_sp_train +
        " to " + points(drift.info.median_sp_test) + " over time. A range is only reliable while new tasks look " +
        "like old ones, so the tool warns you about this team.");
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
