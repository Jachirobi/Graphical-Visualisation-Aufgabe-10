/**
 * Aufgabe 10 – Seeds-Datensatz mit t-SNE in 3D
 *
 * Beim Start wird das beste, vorab berechnete Ergebnis (data/tsne-best.json) geladen.
 * Restart (r) startet tSNEJS neu – mit Math.random, also jedes Mal anders –,
 * Step (t) rechnet einen Schritt, Step 10 (T) zehn Schritte.
 * Die drei Weizensorten sind durch Farbe UND Form unterschieden:
 * Kama = Kugel (orange), Rosa = Würfel (rosa), Canadian = Oktaeder (blau).
 */
(() => {
  "use strict";

  // Mathe
  const PI = Math.PI;
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
  const M4 = {
    perspective(fovy, aspect, near, far) {
      const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
      return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0];
    },
    lookAt(eye, center, up) {
      const z = norm(sub(eye, center)), x = norm(cross(up, z)), y = cross(z, x);
      return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1];
    },
  };

  // Klassen
  const CLASSES = [
    null,
    { name: "Kama", shape: "Kugel", color: [0.93, 0.5, 0.12], css: "#ee7f1f" },
    { name: "Rosa", shape: "Würfel", color: [0.86, 0.25, 0.6], css: "#db4099" },
    { name: "Canadian", shape: "Oktaeder", color: [0.16, 0.45, 0.85], css: "#2973d9" },
  ];
  const FEATURES = ["Fläche A", "Umfang P", "Kompaktheit C", "Kernlänge", "Kernbreite", "Asymmetrie", "Furchenlänge"];

  // Glyphen
  function mesh() { return { pos: [], nor: [], idx: [] }; }
  function addTri(m, a, b, c, n) { const i = m.pos.length / 3; m.pos.push(...a, ...b, ...c); m.nor.push(...n, ...n, ...n); m.idx.push(i, i + 1, i + 2); }
  /** Kugel aus einem zweimal unterteilten Oktaeder, glatte Normalen */
  function sphere(depth = 2) {
    let tris = [[[0, 1, 0], [0, 0, 1], [1, 0, 0]], [[0, 1, 0], [1, 0, 0], [0, 0, -1]], [[0, 1, 0], [0, 0, -1], [-1, 0, 0]], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
      [[0, -1, 0], [1, 0, 0], [0, 0, 1]], [[0, -1, 0], [0, 0, -1], [1, 0, 0]], [[0, -1, 0], [-1, 0, 0], [0, 0, -1]], [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]];
    const mid = (a, b) => norm([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2]);
    for (let d = 0; d < depth; d++) tris = tris.flatMap(([a, b, c]) => { const ab = mid(a, b), bc = mid(b, c), ca = mid(c, a); return [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]; });
    const m = mesh();
    tris.forEach(([a, b, c]) => { const i = m.pos.length / 3; m.pos.push(...a, ...b, ...c); m.nor.push(...a, ...b, ...c); m.idx.push(i, i + 1, i + 2); });
    return m;
  }
  function cube() {
    const m = mesh(), s = 0.82;
    [[0, 1, 2], [1, 2, 0], [2, 0, 1]].forEach(([x, y, z]) => [-1, 1].forEach((sg) => {
      const P = (u, v) => { const p = [0, 0, 0]; p[x] = sg * s; p[y] = u * s; p[z] = v * s; return p; };
      const n = [0, 0, 0]; n[x] = sg;
      const q = sg > 0 ? [P(-1, -1), P(1, -1), P(1, 1), P(-1, 1)] : [P(-1, -1), P(-1, 1), P(1, 1), P(1, -1)];
      addTri(m, q[0], q[1], q[2], n); addTri(m, q[0], q[2], q[3], n);
    }));
    return m;
  }
  function octahedron() {
    const m = mesh(), s = 1.3, V = [[s, 0, 0], [-s, 0, 0], [0, s, 0], [0, -s, 0], [0, 0, s], [0, 0, -s]];
    [[2, 4, 0], [2, 0, 5], [2, 5, 1], [2, 1, 4], [3, 0, 4], [3, 5, 0], [3, 1, 5], [3, 4, 1]].forEach(([a, b, c]) => {
      const n = norm(cross(sub(V[b], V[a]), sub(V[c], V[a]))); addTri(m, V[a], V[b], V[c], n);
    });
    return m;
  }
  /** Würfelkanten als Orientierungsrahmen */
  function frameBox(s) {
    const m = { pos: [], nor: [], idx: [] }, c = [];
    for (let i = 0; i < 8; i++) c.push([(i & 1 ? 1 : -1) * s, (i & 2 ? 1 : -1) * s, (i & 4 ? 1 : -1) * s]);
    c.forEach((p) => { m.pos.push(...p); m.nor.push(0, 1, 0); });
    [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]].forEach(([a, b]) => m.idx.push(a, b));
    return m;
  }

  // WebGL
  const VS = `
    attribute vec3 aPos;
    attribute vec3 aNormal;
    uniform mat4 uView, uProj;
    uniform vec3 uOffset;
    uniform float uScale;
    varying vec3 vNormal;
    varying vec3 vWorld;
    void main() {
      vec3 p = aPos * uScale + uOffset;              // Glyphe an die Datenposition schieben
      vWorld = p; vNormal = aNormal;
      gl_Position = uProj * uView * vec4(p, 1.0);
    }`;
  const FS = `
    precision mediump float;
    varying vec3 vNormal;
    varying vec3 vWorld;
    uniform vec3 uColor;
    uniform vec3 uEye;
    uniform float uLit;
    uniform float uAlpha;
    void main() {
      vec3 N = normalize(vNormal), V = normalize(uEye - vWorld);
      vec3 L = normalize(vec3(0.5, 0.9, 0.6));
      float d = abs(dot(N, L));
      float s = pow(max(dot(reflect(-L, N), V), 0.0), 30.0);
      vec3 c = mix(uColor, uColor * (0.35 + 0.75 * d) + vec3(0.35 * s), uLit);
      gl_FragColor = vec4(c, uAlpha);
    }`;

  const $ = (id) => document.getElementById(id);
  const canvas = $("gl");
  const setStatus = (t, err = false) => { $("status").textContent = t; $("status").classList.toggle("is-error", err); };
  let gl, loc, glyphs = [], frame;

  function compile(type, src) {
    const sh = gl.createShader(type); gl.shaderSource(sh, src); gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(`Shader-Fehler: ${gl.getShaderInfoLog(sh)}`);
    return sh;
  }
  function upload(m) {
    const mk = (t, d) => { const b = gl.createBuffer(); gl.bindBuffer(t, b); gl.bufferData(t, d, gl.STATIC_DRAW); return b; };
    return { pos: mk(gl.ARRAY_BUFFER, new Float32Array(m.pos)), nor: mk(gl.ARRAY_BUFFER, new Float32Array(m.nor)), idx: mk(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(m.idx)), n: m.idx.length };
  }
  function initGL() {
    gl = canvas.getContext("webgl", { antialias: true });
    if (!gl) throw new Error("Ihr Browser unterstützt kein WebGL. Bitte aktivieren Sie die Hardwarebeschleunigung.");
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(`Programm-Fehler: ${gl.getProgramInfoLog(prog)}`);
    gl.useProgram(prog);
    loc = { aPos: gl.getAttribLocation(prog, "aPos"), aNormal: gl.getAttribLocation(prog, "aNormal") };
    ["uView", "uProj", "uOffset", "uScale", "uColor", "uEye", "uLit", "uAlpha"].forEach((n) => (loc[n] = gl.getUniformLocation(prog, n)));
    gl.enable(gl.DEPTH_TEST);
    glyphs = [null, upload(sphere()), upload(cube()), upload(octahedron())];
    glyphs.frame = upload(frameBox(1.25));
  }

  // Daten
  let X = [], labels = [], raw = [], best = null;
  /** z-Standardisierung je Merkmal – dieselbe Vorverarbeitung wie im Optimierungsskript */
  function standardize(rows) {
    const m = rows[0].map((_, j) => rows.reduce((s, r) => s + r[j], 0) / rows.length);
    const sd = rows[0].map((_, j) => Math.sqrt(rows.reduce((s, r) => s + (r[j] - m[j]) ** 2, 0) / rows.length));
    return rows.map((r) => r.map((v, j) => (v - m[j]) / sd[j]));
  }
  async function loadData() {
    const [csv, json] = await Promise.all([
      fetch("data/seeds.csv").then((r) => { if (!r.ok) throw new Error(`data/seeds.csv: HTTP ${r.status}`); return r.text(); }),
      fetch("data/tsne-best.json").then((r) => { if (!r.ok) throw new Error(`data/tsne-best.json: HTTP ${r.status}`); return r.json(); }),
    ]);
    const rows = csv.trim().split(/\r?\n/).slice(1).map((l) => l.split(",").map(Number));
    if (rows.length !== 210 || rows.some((r) => r.length !== 8 || r.some(Number.isNaN))) throw new Error("data/seeds.csv hat nicht das erwartete Format (210 Zeilen, 7 Merkmale + Klasse).");
    raw = rows.map((r) => r.slice(0, 7));
    labels = rows.map((r) => r[7]);
    X = standardize(raw);
    best = json;
  }

  // Zustand
  const CAM0 = { yaw: 12, pitch: 16, dist: 5.0 };
  const cam = { ...CAM0 };
  const state = { Y: [], step: 0, cost: null, source: "best", auto: false, perplexity: 10, epsilon: 20, quality: null };
  let tsne = null;

  /** Ergebnis für die Anzeige zentrieren und skalieren (t-SNE-Koordinaten haben keine feste Einheit) */
  function displayPoints(Y) {
    const n = Y.length, c = [0, 1, 2].map((d) => Y.reduce((s, p) => s + p[d], 0) / n);
    // robuste Skalierung: 90 % der Punkte liegen innerhalb von Radius 0,9 – einzelne Ausreißer stauchen die Wolke nicht
    const rs = Y.map((p) => Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2])).sort((a, b) => a - b);
    const r = (rs[Math.floor(0.9 * (n - 1))] || 1) / 0.9;
    return Y.map((p) => [(p[0] - c[0]) / r, (p[1] - c[1]) / r, (p[2] - c[2]) / r]);
  }

  /** Gütemaße: 5-NN-Genauigkeit (leave one out) und mittlere Silhouette */
  function quality(Y) {
    const n = Y.length, D = Y.map((a) => Y.map((b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])));
    let ok = 0, sil = 0;
    for (let i = 0; i < n; i++) {
      const nn = D[i].map((d, j) => [d, j]).filter(([, j]) => j !== i).sort((a, b) => a[0] - b[0]).slice(0, 5);
      const v = [0, 0, 0, 0]; nn.forEach(([, j]) => v[labels[j]]++);
      if (v.indexOf(Math.max(...v)) === labels[i]) ok++;
      const mean = (c) => { let s = 0, k = 0; for (let j = 0; j < n; j++) if (j !== i && labels[j] === c) { s += D[i][j]; k++; } return s / k; };
      const a = mean(labels[i]), b = Math.min(...[1, 2, 3].filter((c) => c !== labels[i]).map(mean));
      sil += (b - a) / (Math.max(a, b) || 1);
    }
    return { knn: ok / n, silhouette: sil / n };
  }

  // Zeichnen
  function eyePos() {
    const y = (cam.yaw * PI) / 180, p = (cam.pitch * PI) / 180;
    return [cam.dist * Math.cos(p) * Math.sin(y), cam.dist * Math.sin(p), cam.dist * Math.cos(p) * Math.cos(y)];
  }
  function bind(o) {
    gl.bindBuffer(gl.ARRAY_BUFFER, o.pos); gl.enableVertexAttribArray(loc.aPos); gl.vertexAttribPointer(loc.aPos, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, o.nor); gl.enableVertexAttribArray(loc.aNormal); gl.vertexAttribPointer(loc.aNormal, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, o.idx);
  }
  function draw() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    gl.viewport(0, 0, w, h);
    const dark = document.documentElement.getAttribute("data-theme") === "dark";
    gl.clearColor(...(dark ? [0.1, 0.12, 0.16] : [0.96, 0.96, 0.97]), 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const eye = eyePos();
    gl.uniformMatrix4fv(loc.uProj, false, new Float32Array(M4.perspective((40 * PI) / 180, w / h, 0.1, 50)));
    gl.uniformMatrix4fv(loc.uView, false, new Float32Array(M4.lookAt(eye, [0, 0, 0], [0, 1, 0])));
    gl.uniform3fv(loc.uEye, eye);
    gl.uniform1f(loc.uAlpha, 1);

    // Rahmen
    gl.uniform1f(loc.uLit, 0); gl.uniform1f(loc.uScale, 1); gl.uniform3fv(loc.uOffset, [0, 0, 0]);
    gl.uniform3fv(loc.uColor, dark ? [0.36, 0.4, 0.48] : [0.72, 0.74, 0.78]);
    bind(glyphs.frame); gl.drawElements(gl.LINES, glyphs.frame.n, gl.UNSIGNED_SHORT, 0);

    // Datenpunkte, nach Klasse gruppiert (weniger Pufferwechsel)
    gl.uniform1f(loc.uLit, 1); gl.uniform1f(loc.uScale, 0.05);
    const P = displayPoints(state.Y);
    [1, 2, 3].forEach((c) => {
      if (!$(`show-${c}`).checked) return;
      bind(glyphs[c]); gl.uniform3fv(loc.uColor, CLASSES[c].color);
      P.forEach((p, i) => { if (labels[i] !== c) return; gl.uniform3fv(loc.uOffset, p); gl.drawElements(gl.TRIANGLES, glyphs[c].n, gl.UNSIGNED_SHORT, 0); });
    });
    const err = gl.getError();
    if (err !== gl.NO_ERROR) console.error(`WebGL-Fehler 0x${err.toString(16)}`);
  }
  const redraw = () => { if (frame) return; frame = requestAnimationFrame(() => { frame = null; draw(); }); };

  // Anzeige
  const f = (v, d = 2) => v.toLocaleString("de-DE", { minimumFractionDigits: d, maximumFractionDigits: d });
  const pct = (v) => `${f(v * 100, 1)} %`;
  function renderReadout(withQuality = true) {
    if (withQuality) state.quality = quality(state.Y);
    $("step-count").textContent = state.step.toLocaleString("de-DE");
    $("ro-source").textContent = state.source === "best"
      ? `Gespeichertes bestes Ergebnis (Perplexity ${best.params.perplexity}, Lernrate ${best.params.epsilon}, ${best.params.iterations} Schritte)`
      : `Neuer Lauf (Perplexity ${state.perplexity}, Lernrate ${state.epsilon})${state.step < 100 ? " – Phase „early exaggeration“ bis Schritt 100" : ""}`;
    $("ro-cost").textContent = state.cost === null ? "– (vor dem ersten Schritt)" : f(state.cost, 3);
    $("ro-knn").textContent = state.quality ? pct(state.quality.knn) : "–";
    $("ro-sil").textContent = state.quality ? f(state.quality.silhouette, 3) : "–";
    $("btn-auto").setAttribute("aria-pressed", String(state.auto));
    $("btn-auto").querySelector(".ico").textContent = state.auto ? "❚❚" : "▶";
    $("canvas-desc").textContent =
      `3D-Punktwolke der 210 Weizenkörner nach t-SNE, Schritt ${state.step}. Kama als orange Kugeln, Rosa als rosa Würfel, Canadian als blaue Oktaeder. ` +
      (state.quality ? `${pct(state.quality.knn)} der Körner haben überwiegend Nachbarn derselben Sorte.` : "");
  }

  // t-SNE-Aktionen
  function loadBest() {
    setAuto(false, false);
    tsne = null;
    state.Y = best.points.map((p) => p.slice());
    state.step = best.params.iterations; state.cost = best.quality.cost; state.source = "best";
    renderReadout(); redraw();
    setStatus(`Bestes gespeichertes Ergebnis geladen (Schritt ${state.step}).`);
  }
  function restart() {
    setAuto(false, false);
    state.perplexity = Number($("perplexity").value);
    state.epsilon = Number($("epsilon").value);
    tsne = new tsnejs.tSNE({ dim: 3, perplexity: state.perplexity, epsilon: state.epsilon });
    tsne.initDataRaw(X);                             // neue Zufallswerte aus Math.random
    state.Y = tsne.getSolution(); state.step = 0; state.cost = null; state.source = "run";
    renderReadout(); redraw();
    setStatus(`Restart: Schritt 0 mit neuen Zufallswerten (Perplexity ${state.perplexity}, Lernrate ${state.epsilon}).`);
  }
  function step(n, announce = true) {
    if (!tsne) { restart(); if (announce) setStatus("Neuer Lauf gestartet – das gespeicherte Ergebnis lässt sich nicht weiterrechnen."); }
    for (let k = 0; k < n; k++) state.cost = tsne.step();
    state.Y = tsne.getSolution(); state.step = tsne.iter;
    renderReadout(announce || state.step % 10 === 0); redraw();
    if (announce) setStatus(`${n === 1 ? "1 Schritt" : `${n} Schritte`} gerechnet – Step-Count ${state.step}.`);
  }
  let autoId = null;
  function setAuto(on, announce = true) {
    state.auto = on;
    if (on && autoId === null) {
      const tick = () => { step(2, false); autoId = state.auto ? requestAnimationFrame(tick) : null; };
      autoId = requestAnimationFrame(tick);
    }
    if (!on && autoId !== null) { cancelAnimationFrame(autoId); autoId = null; if (tsne) renderReadout(); }
    $("btn-auto").setAttribute("aria-pressed", String(on));
    $("btn-auto").querySelector(".ico").textContent = on ? "❚❚" : "▶";
    if (announce) setStatus(on ? "Automatisch iterieren läuft." : `Angehalten bei Schritt ${state.step}.`);
  }

  // Kamera
  const orbit = (d) => { cam.yaw += d; redraw(); };
  const tilt = (d) => { cam.pitch = clamp(cam.pitch + d, -85, 85); redraw(); };
  const zoom = (d) => { cam.dist = clamp(cam.dist + d, 2.4, 12); redraw(); };
  const resetCam = () => { Object.assign(cam, CAM0); redraw(); setStatus("Ansicht zurückgesetzt."); };

  // Bedienung
  $("btn-restart").addEventListener("click", restart);
  $("btn-step").addEventListener("click", () => step(1));
  $("btn-step10").addEventListener("click", () => step(10));
  $("btn-auto").addEventListener("click", () => { if (!tsne) restart(); setAuto(!state.auto); });
  $("btn-best").addEventListener("click", loadBest);
  $("btn-left").addEventListener("click", () => orbit(-10));
  $("btn-right").addEventListener("click", () => orbit(10));
  $("btn-up").addEventListener("click", () => tilt(10));
  $("btn-down").addEventListener("click", () => tilt(-10));
  $("btn-in").addEventListener("click", () => zoom(-0.4));
  $("btn-out").addEventListener("click", () => zoom(0.4));
  $("btn-cam").addEventListener("click", resetCam);
  [1, 2, 3].forEach((c) => $(`show-${c}`).addEventListener("change", (e) => { redraw(); setStatus(`${CLASSES[c].name} ${e.target.checked ? "eingeblendet" : "ausgeblendet"}.`); }));
  $("perplexity").addEventListener("input", (e) => { $("perplexity-out").textContent = e.target.value; });
  $("perplexity").addEventListener("change", (e) => setStatus(`Perplexity ${e.target.value} – wirkt beim nächsten Restart.`));
  $("epsilon").addEventListener("change", (e) => setStatus(`Lernrate ${e.target.value} – wirkt beim nächsten Restart.`));

  let drag = null;
  canvas.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    orbit(-(e.clientX - drag.x) * 0.4); tilt((e.clientY - drag.y) * 0.4);
    drag = { x: e.clientX, y: e.clientY };
  });
  ["pointerup", "pointercancel"].forEach((t) => canvas.addEventListener(t, () => (drag = null)));
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); zoom(Math.sign(e.deltaY) * 0.3); }, { passive: false });

  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* privater Modus */ } },
  };
  const shortcuts = $("toggle-shortcuts");
  shortcuts.checked = store.get("shortcuts") !== "off";
  shortcuts.addEventListener("change", () => {
    store.set("shortcuts", shortcuts.checked ? "on" : "off");
    setStatus(shortcuts.checked ? "Tastenkürzel eingeschaltet." : "Tastenkürzel ausgeschaltet – Bedienung über die Schaltflächen.");
  });

  document.addEventListener("keydown", (e) => {
    if (!shortcuts.checked || e.metaKey || e.ctrlKey || e.altKey || !best) return;
    const t = e.target;
    if (t.closest && t.closest("input, textarea, select")) return;
    const k = e.key;
    if (k.startsWith("Arrow") && t.closest && (t.closest("#buehne") || t === document.body)) {
      e.preventDefault();
      if (k === "ArrowLeft") orbit(-5);
      if (k === "ArrowRight") orbit(5);
      if (k === "ArrowUp") tilt(5);
      if (k === "ArrowDown") tilt(-5);
      return;
    }
    switch (k) {
      case "t": e.preventDefault(); step(1); break;          // Step
      case "T": e.preventDefault(); step(10); break;         // Step 10 (Umschalt + t)
      case "r": case "R": if (!e.repeat) { e.preventDefault(); restart(); } break;
      case "a": case "A": if (!e.repeat) { e.preventDefault(); if (!tsne) restart(); setAuto(!state.auto); } break;
      case "b": case "B": if (!e.repeat) { e.preventDefault(); loadBest(); } break;
      case "+": e.preventDefault(); zoom(-0.3); break;
      case "-": e.preventDefault(); zoom(0.3); break;
      case "c": case "C": if (!e.repeat) { e.preventDefault(); resetCam(); } break;
    }
  });

  const themeSwitch = $("theme-switch");
  function applyTheme(th) {
    document.documentElement.setAttribute("data-theme", th);
    themeSwitch.setAttribute("aria-checked", String(th === "dark"));
    if (gl) redraw();
  }
  themeSwitch.addEventListener("click", () => {
    const next = themeSwitch.getAttribute("aria-checked") === "true" ? "light" : "dark";
    applyTheme(next); store.set("theme", next);
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => { if (!store.get("theme")) applyTheme(e.matches ? "dark" : "light"); });

  // Dokumentation aus der JSON-Datei
  function renderDocs() {
    const p = best.params, q = best.quality;
    $("doc-params").innerHTML = [
      ["Ausgabedimension", `${p.dim}`], ["Perplexity", `${p.perplexity}`], ["Lernrate (epsilon)", `${p.epsilon}`],
      ["Iterationen", `${p.iterations}`], ["Startwert des Zufallsgenerators", `${p.seed} (mulberry32, nur im Optimierungsskript)`],
      ["Vorverarbeitung", p.preprocessing], ["Abstandsmaß", "euklidisch (tSNEJS initDataRaw)"],
      ["Early Exaggeration", "Faktor 4 bis Schritt 100 (fest in tSNEJS)"], ["Momentum", "0,5 bis Schritt 250, danach 0,8 (fest in tSNEJS)"],
    ].map(([k, v]) => `<tr><th scope="row">${k}</th><td>${v}</td></tr>`).join("");
    $("doc-quality").innerHTML = [
      ["5-NN-Genauigkeit in 3D", pct(q.knn5), "Anteil der Körner, deren fünf nächste Nachbarn überwiegend dieselbe Sorte haben"],
      ["Silhouette", f(q.silhouette, 3), "−1 … 1; je größer, desto kompakter und getrennter die Gruppen"],
      ["KL-Divergenz (Kosten)", f(q.cost, 3), "Wert, den tSNEJS nach dem letzten Schritt zurückgibt"],
      ["Zum Vergleich: ohne Standardisierung", `${pct(q.rawKnn5)} / ${f(q.rawSilhouette, 3)}`, "gleiche Einstellungen, Rohdaten"],
    ].map(([k, v, d]) => `<tr><th scope="row">${k}</th><td class="num">${v}</td><td>${d}</td></tr>`).join("");
    $("doc-grid").innerHTML = best.grid.map((g) => {
      const win = g.perplexity === p.perplexity && g.epsilon === p.epsilon;
      return `<tr${win ? ' class="is-current"' : ""}><td class="num">${g.perplexity}</td><td class="num">${g.epsilon}</td><td class="num">${pct(g.knnMean)}</td><td class="num">${f(g.silMean, 3)}</td><td class="num">${pct(g.bestKnn)}</td><td>${win ? '<span class="ok">✓ gewählt</span>' : ""}</td></tr>`;
    }).join("");
    // Mittelwerte der Merkmale je Sorte
    $("doc-features").innerHTML = FEATURES.map((name, j) => `<tr><th scope="row">${name}</th>${[1, 2, 3].map((c) => {
      const v = raw.filter((_, i) => labels[i] === c).map((r) => r[j]);
      return `<td class="num">${f(v.reduce((s, x) => s + x, 0) / v.length, 3)}</td>`;
    }).join("")}</tr>`).join("");
  }

  // Start
  applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light");
  (async () => {
    try {
      if (typeof tsnejs === "undefined") throw new Error("Die Bibliothek lib/tsne.js wurde nicht geladen.");
      initGL();
      await loadData();
      $("perplexity").value = best.params.perplexity; $("perplexity-out").textContent = best.params.perplexity;
      $("epsilon").value = String(best.params.epsilon);
      renderDocs();
      loadBest();
      new ResizeObserver(() => redraw()).observe(canvas);
      canvas.addEventListener("webglcontextlost", (e) => { e.preventDefault(); setAuto(false, false); setStatus("WebGL-Kontext verloren – bitte Seite neu laden.", true); });
      console.info(`Seeds: ${X.length} Körner, ${X[0].length} Merkmale; bestes t-SNE-Ergebnis geladen (Perplexity ${best.params.perplexity}).`);
    } catch (err) {
      console.error(err);
      setStatus(`${err.message}${location.protocol === "file:" ? " – Die Seite bitte über einen Webserver öffnen (z. B. python3 -m http.server)." : ""}`, true);
    }
  })();
})();
