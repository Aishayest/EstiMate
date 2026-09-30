// Recompute every reference case with web/assets/estimator.js and report mismatches.
const fs = require("fs");
const path = require("path");
const E = require("../web/assets/estimator.js");

const ref = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
let cases = 0, pointFail = 0, intervalFail = 0, maxDiff = 0;
for (const [project, items] of Object.entries(ref)) {
  const art = JSON.parse(fs.readFileSync(path.join(__dirname, "../web/artifacts", project + ".json"), "utf8"));
  const model = E.prepare(art);
  for (const it of items) {
    for (const level of [50, 80, 90]) {
      const r = E.estimate(model, it.text, level);
      const d = Math.abs(r.point - it.point);
      maxDiff = Math.max(maxDiff, d);
      if (d > 1e-4) pointFail++;
      const [lo, hi] = it.intervals[level];
      if (r.loInt !== lo || r.hiInt !== hi) intervalFail++;
      cases++;
    }
  }
}
console.log(JSON.stringify({ cases, pointFail, intervalFail, maxDiff }));
process.exit(pointFail || intervalFail ? 1 : 0);
