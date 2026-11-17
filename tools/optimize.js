/**
 * Sucht die t-SNE-Einstellungen, mit denen die drei Weizensorten am besten getrennt werden,
 * und speichert das beste 3D-Ergebnis in data/tsne-best.json.
 *
 * Aufruf (im Ordner aufgabe10):  node tools/optimize.js
 *
 * tSNEJS verwendet Math.random. Damit jeder Lauf reproduzierbar ist, wird Math.random hier
 * durch einen Zufallsgenerator mit festem Startwert (mulberry32) ersetzt – nur in diesem Skript,
 * nicht in der Webanwendung.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const tsnejs = require("../lib/tsne.js");

const ROOT = path.join(__dirname, "..");
const { X, labels } = require("./data.js").load(path.join(ROOT, "data/seeds.csv"));

function mulberry32(seed) {
  return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

const dist = (a, b) => Math.hypot(...a.map((v, i) => v - b[i]));

/** Gütemaße im 3D-Ergebnis: k-NN-Genauigkeit (leave one out), Silhouette, Trennungsverhältnis */
function evaluate(Y) {
  const n = Y.length, D = Y.map((a) => Y.map((b) => dist(a, b)));
  let knnOk = 0;
  for (let i = 0; i < n; i++) {
    const nn = D[i].map((d, j) => [d, j]).filter(([, j]) => j !== i).sort((a, b) => a[0] - b[0]).slice(0, 5);
    const votes = [0, 0, 0, 0];
    nn.forEach(([, j]) => votes[labels[j]]++);
    if (votes.indexOf(Math.max(...votes)) === labels[i]) knnOk++;
  }
  let sil = 0;
  for (let i = 0; i < n; i++) {
    const mean = (c) => { let s = 0, k = 0; for (let j = 0; j < n; j++) if (j !== i && labels[j] === c) { s += D[i][j]; k++; } return s / k; };
    const a = mean(labels[i]);
    const b = Math.min(...[1, 2, 3].filter((c) => c !== labels[i]).map(mean));
    sil += (b - a) / Math.max(a, b);
  }
  return { knn: knnOk / n, silhouette: sil / n };
}

function run({ perplexity, epsilon, iterations, seed }) {
  Math.random = mulberry32(seed);
  const t = new tsnejs.tSNE({ dim: 3, perplexity, epsilon });
  t.initDataRaw(X);
  let cost = 0;
  for (let k = 0; k < iterations; k++) cost = t.step();
  return { Y: t.getSolution().map((p) => p.slice()), cost };
}

const PERPLEXITIES = [5, 10, 15, 20, 30, 40, 50];
const EPSILONS = [5, 10, 20];
const SEEDS = [1, 2, 3, 4, 5];
const ITERATIONS = 1000;

const results = [];
for (const perplexity of PERPLEXITIES) for (const epsilon of EPSILONS) {
  const runs = SEEDS.map((seed) => {
    const { Y, cost } = run({ perplexity, epsilon, iterations: ITERATIONS, seed });
    return { seed, cost, Y, ...evaluate(Y) };
  });
  const avg = (k) => runs.reduce((s, r) => s + r[k], 0) / runs.length;
  const best = runs.reduce((a, b) => (b.knn > a.knn || (b.knn === a.knn && b.silhouette > a.silhouette) ? b : a));
  results.push({ perplexity, epsilon, knnMean: avg("knn"), silMean: avg("silhouette"), costMean: avg("cost"), best });
  console.log(`perplexity ${String(perplexity).padStart(2)}  epsilon ${String(epsilon).padStart(2)}  kNN Ø ${(avg("knn") * 100).toFixed(1)} %  Silhouette Ø ${avg("silhouette").toFixed(3)}  KL Ø ${avg("cost").toFixed(3)}  | bester Lauf: Seed ${best.seed}, kNN ${(best.knn * 100).toFixed(1)} %, Sil. ${best.silhouette.toFixed(3)}`);
}

// Bewertung der Einstellung: Mittel über alle Startwerte (robust), dann Silhouette
const score = (r) => r.knnMean + 0.5 * r.silMean;
results.sort((a, b) => score(b) - score(a));
const win = results[0];
console.log(`\nBeste Einstellung: Perplexity ${win.perplexity}, Lernrate ${win.epsilon} – bester Lauf Seed ${win.best.seed}`);

// Vergleich: Rohdaten ohne Standardisierung mit der besten Einstellung
const raw = require("./data.js").load(path.join(ROOT, "data/seeds.csv"), { standardize: false });
Math.random = mulberry32(win.best.seed);
const tr = new tsnejs.tSNE({ dim: 3, perplexity: win.perplexity, epsilon: win.epsilon });
tr.initDataRaw(raw.X);
for (let k = 0; k < ITERATIONS; k++) tr.step();
const rawEval = evaluate(tr.getSolution());
console.log(`Ohne Standardisierung: kNN ${(rawEval.knn * 100).toFixed(1)} %, Silhouette ${rawEval.silhouette.toFixed(3)}`);

const out = {
  description: "Bestes 3D-t-SNE-Ergebnis für den Seeds-Datensatz (UCI), erzeugt mit tools/optimize.js und tSNEJS",
  created: new Date().toISOString().slice(0, 10),
  params: { dim: 3, perplexity: win.perplexity, epsilon: win.epsilon, iterations: ITERATIONS, seed: win.best.seed, preprocessing: "z-Standardisierung je Merkmal" },
  quality: { knn5: win.best.knn, silhouette: win.best.silhouette, cost: win.best.cost, rawKnn5: rawEval.knn, rawSilhouette: rawEval.silhouette },
  grid: results.map((r) => ({ perplexity: r.perplexity, epsilon: r.epsilon, knnMean: +r.knnMean.toFixed(4), silMean: +r.silMean.toFixed(4), costMean: +r.costMean.toFixed(4), bestKnn: +r.best.knn.toFixed(4) }))
    .sort((a, b) => a.perplexity - b.perplexity || a.epsilon - b.epsilon),
  labels,
  points: require("./data.js").alignPCA(win.best.Y),          // auf Hauptachsen gedreht (Abstände unverändert)
};
fs.writeFileSync(path.join(ROOT, "data/tsne-best.json"), JSON.stringify(out));
console.log("data/tsne-best.json gespeichert.");
