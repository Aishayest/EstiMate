// Shared page behaviour: theme toggle, mobile menu, summary loading, formatting.
(function (root) {
  "use strict";

  var darkQuery = window.matchMedia("(prefers-color-scheme: dark)");

  // Human-readable names for the four Deep-SE projects.
  var TEAMS = {
    mesos: { name: "Apache Mesos", about: "software that runs apps across a cluster of servers" },
    springxd: { name: "Spring XD", about: "a system for data pipelines and streaming" },
    appceleratorstudio: { name: "Appcelerator Studio", about: "an editor for building mobile apps" },
    talenddataquality: { name: "Talend Data Quality", about: "a tool for checking and cleaning data" },
  };
  var teamName = function (key) { return (TEAMS[key] || { name: key }).name; };

  function currentTheme() {
    return document.documentElement.dataset.theme || (darkQuery.matches ? "dark" : "light");
  }

  function initTheme() {
    var button = document.querySelector("[data-theme-toggle]");
    if (!button) return;
    var sync = function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      button.textContent = next === "dark" ? "Dark" : "Light";
      button.setAttribute("aria-label", "Switch to " + next + " theme");
    };
    button.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("estimate-theme", next); } catch (e) { /* ignore */ }
      sync();
    });
    darkQuery.addEventListener("change", sync);
    sync();
  }

  function initMenu() {
    var button = document.querySelector("[data-menu]");
    var nav = document.getElementById("nav");
    if (!button || !nav) return;
    button.addEventListener("click", function () {
      var open = nav.classList.toggle("open");
      button.setAttribute("aria-expanded", String(open));
      button.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    });
  }

  var summaryPromise = null;
  function loadSummary() {
    if (!summaryPromise) {
      summaryPromise = fetch("artifacts/summary.json").then(function (r) {
        if (!r.ok) throw new Error("summary.json: HTTP " + r.status);
        return r.json();
      });
    }
    return summaryPromise;
  }

  var pct = function (x, digits) { return (x * 100).toFixed(digits || 0) + "%"; };
  var signedPct = function (x) {
    var v = (x * 100).toFixed(1);
    return (x > 0 ? "+" : x < 0 ? "−" : "") + v.replace("-", "") + "%";
  };

  function fillCommon(summary) {
    var n = summary.n_issues.toLocaleString("en-US");
    document.querySelectorAll("[data-n-issues]").forEach(function (el) { el.textContent = n; });
  }

  // The drifting project, by the export's data-driven flag.
  function driftProject(summary) {
    var name = Object.keys(summary.projects).find(function (k) { return summary.projects[k].drift; });
    return name ? { name: name, info: summary.projects[name] } : null;
  }

  // Shown when the page is opened from disk, where fetch() is blocked.
  function showLoadError(err) {
    var box = document.createElement("div");
    box.className = "badge warn section";
    box.setAttribute("role", "alert");
    box.innerHTML = '<div class="badge-title">! Could not load the data</div>';
    var p = document.createElement("p");
    p.textContent = "The page needs its JSON files (" + err.message + "). Serve the folder over HTTP, " +
      "e.g. `python -m http.server -d web 8000`, and open http://localhost:8000.";
    box.appendChild(p);
    document.querySelector(".hero").after(box);
  }

  initTheme();
  initMenu();

  root.Common = {
    TEAMS: TEAMS,
    teamName: teamName,
    loadSummary: loadSummary,
    fillCommon: fillCommon,
    driftProject: driftProject,
    showLoadError: showLoadError,
    pct: pct,
    signedPct: signedPct,
  };
})(window);
