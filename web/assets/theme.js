// Apply a saved theme before first paint to avoid a flash of the wrong one.
(function () {
  try {
    var saved = localStorage.getItem("estimate-theme");
    if (saved === "light" || saved === "dark") document.documentElement.dataset.theme = saved;
  } catch (e) { /* storage unavailable: follow the system setting */ }
})();
