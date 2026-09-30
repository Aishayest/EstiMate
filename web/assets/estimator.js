// Browser port of the demo model: sklearn TfidfVectorizer + Ridge + split conformal.
// Mirrors src/models.py, src/conformal.py and src/build_demo.py; parity with the
// Python predictions is checked by tests/web_parity.py.
(function (root) {
  "use strict";

  // sklearn's default token_pattern (?u)\b\w\w+\b: runs of 2+ word characters.
  var TOKEN = /[\p{L}\p{N}_]{2,}/gu;

  function prepare(artifact) {
    var vocab = new Map();
    artifact.tfidf.terms.forEach(function (t, i) { vocab.set(t, i); });
    return Object.assign({}, artifact, { vocab: vocab });
  }

  function ngrams(tokens, lo, hi) {
    var out = [];
    for (var n = lo; n <= hi; n++) {
      for (var i = 0; i + n <= tokens.length; i++) out.push(tokens.slice(i, i + n).join(" "));
    }
    return out;
  }

  // Sparse, L2-normalised TF-IDF vector as a Map(index -> weight).
  function vectorize(model, text) {
    var tokens = text.toLowerCase().match(TOKEN) || [];
    var counts = new Map();
    ngrams(tokens, model.tfidf.ngram_range[0], model.tfidf.ngram_range[1]).forEach(function (g) {
      var idx = model.vocab.get(g);
      if (idx !== undefined) counts.set(idx, (counts.get(idx) || 0) + 1);
    });
    var vec = new Map();
    var norm = 0;
    counts.forEach(function (c, idx) {
      var w = (1 + Math.log(c)) * model.tfidf.idf[idx]; // sublinear_tf
      vec.set(idx, w);
      norm += w * w;
    });
    norm = Math.sqrt(norm);
    if (norm > 0) vec.forEach(function (w, idx) { vec.set(idx, w / norm); });
    return vec;
  }

  function pointEstimate(model, vec) {
    var y = model.ridge.intercept;
    vec.forEach(function (w, idx) { y += w * model.ridge.coef[idx]; });
    if (model.ridge.log_target) y = Math.expm1(y);
    return Math.min(Math.max(y, model.clip[0]), model.clip[1]);
  }

  // numpy's round-half-to-even, so empty-interval fallbacks match Python.
  function roundHalfEven(x) {
    var r = Math.round(x);
    return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
  }

  // Inward integer snapping from src/conformal.py: same coverage for integer labels.
  function snap(lo, hi) {
    var loI = Math.max(Math.ceil(lo), 1);
    var hiI = Math.floor(hi);
    if (loI > hiI) {
      var mid = Math.max(roundHalfEven((lo + hi) / 2), 1);
      loI = mid;
      hiI = mid;
    }
    return [loI, hiI];
  }

  function estimate(model, text, level) {
    var vec = vectorize(model, text);
    var point = pointEstimate(model, vec);
    var q = model.q[String(level)];
    var lo = point - q;
    var hi = point + q;
    var snapped = snap(lo, hi);
    return {
      point: point,
      q: q,
      lo: Math.max(lo, 1),
      hi: hi,
      loInt: snapped[0],
      hiInt: snapped[1],
      known: vec.size,
      vec: vec,
    };
  }

  // Cosine similarity against train issues (both sides are L2-normalised).
  function nearest(model, vec, k) {
    var scores = [];
    model.train.vectors.forEach(function (row, i) {
      var idx = row[0];
      var val = row[1];
      var s = 0;
      for (var j = 0; j < idx.length; j++) {
        var w = vec.get(idx[j]);
        if (w !== undefined) s += w * val[j];
      }
      if (s > 0) scores.push([s, i]);
    });
    scores.sort(function (a, b) { return b[0] - a[0]; });
    return scores.slice(0, k).map(function (p) {
      return {
        key: model.train.keys[p[1]],
        title: model.train.titles[p[1]],
        sp: model.train.sp[p[1]],
        similarity: p[0],
      };
    });
  }

  var api = { prepare: prepare, estimate: estimate, nearest: nearest, snap: snap };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Estimator = api;
})(typeof self !== "undefined" ? self : this);
