/** Liest data/seeds.csv (Node) und standardisiert die sieben Merkmale (z-Werte). */
"use strict";
const fs = require("fs");

function load(file, { standardize = true } = {}) {
  const rows = fs.readFileSync(file, "utf8").trim().split(/\r?\n/).slice(1).map((l) => l.split(",").map(Number));
  const labels = rows.map((r) => r[7]);
  let X = rows.map((r) => r.slice(0, 7));
  if (standardize) {
    const m = X[0].map((_, j) => X.reduce((s, r) => s + r[j], 0) / X.length);
    const sd = X[0].map((_, j) => Math.sqrt(X.reduce((s, r) => s + (r[j] - m[j]) ** 2, 0) / X.length));
    X = X.map((r) => r.map((v, j) => (v - m[j]) / sd[j]));
  }
  return { X, labels };
}
module.exports = { load };

/**
 * Dreht ein 3D-Ergebnis auf seine Hauptachsen (PCA): längste Ausdehnung → x, zweitlängste → y.
 * Eine Drehung ändert keine Abstände – Nachbarschaften und Gütemaße bleiben gleich,
 * die Wolke liegt aber in der Startansicht breit vor der Kamera.
 */
function alignPCA(Y) {
  const n = Y.length, c = [0, 1, 2].map((d) => Y.reduce((s, p) => s + p[d], 0) / n);
  const Z = Y.map((p) => p.map((v, d) => v - c[d]));
  const A = [0, 1, 2].map((i) => [0, 1, 2].map((j) => Z.reduce((s, p) => s + p[i] * p[j], 0) / n));
  let V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 50; sweep++) {                 // Jacobi-Verfahren für symmetrische 3×3-Matrizen
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      if (Math.abs(A[p][q]) < 1e-12) continue;
      const th = 0.5 * Math.atan2(2 * A[p][q], A[q][q] - A[p][p]), cs = Math.cos(th), sn = Math.sin(th);
      for (let k = 0; k < 3; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = cs * akp - sn * akq; A[k][q] = sn * akp + cs * akq; }
      for (let k = 0; k < 3; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = cs * apk - sn * aqk; A[q][k] = sn * apk + cs * aqk; }
      for (let k = 0; k < 3; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = cs * vkp - sn * vkq; V[k][q] = sn * vkp + cs * vkq; }
    }
  }
  const order = [0, 1, 2].sort((a, b) => A[b][b] - A[a][a]);
  const axes = order.map((i) => [V[0][i], V[1][i], V[2][i]]);
  axes[2] = [axes[0][1] * axes[1][2] - axes[0][2] * axes[1][1], axes[0][2] * axes[1][0] - axes[0][0] * axes[1][2], axes[0][0] * axes[1][1] - axes[0][1] * axes[1][0]]; // rechtshändig
  return Z.map((p) => axes.map((a) => +(a[0] * p[0] + a[1] * p[1] + a[2] * p[2]).toFixed(4)));
}
module.exports.alignPCA = alignPCA;
