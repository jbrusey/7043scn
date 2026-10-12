/* Lecture 6 (Function Approximation): interactive demonstrations.
   Each Demos.register(id, ...) mounts into the <section id="id"> in fa.html.
   All numbers are computed locally; random runs use fixed, disclosed seeds. */
(function () {
  'use strict';
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  const grid = (fn, a, b, n = 81) => Array.from({length: n}, (_, i) => { const x = a + (b - a) * i / (n - 1); return [x, fn(x)]; });
  // Smallest "nice" axis maximum >= x, with a tick count that gives round labels.
  const LADDER = [[2, 4], [5, 5], [10, 5], [20, 4], [50, 5], [100, 4], [200, 4], [500, 5], [1000, 4], [2000, 4], [5000, 5], [10000, 5]];
  const nice = x => LADDER.find(([m]) => m >= x) || LADDER[LADDER.length - 1];
  const num = x => (x < 0 ? `(${+x.toFixed(4)})` : String(+x.toFixed(4)));

  // Redraw on any control change. Reset puts the controls back to their defaults
  // without firing events, resets the demo's own state, then redraws once.
  function wire(root, defaults, draw, onReset) {
    root.querySelectorAll('input,select').forEach(el => el.addEventListener('input', draw));
    const reset = root.querySelector('[data-action="reset"]');
    if (reset) reset.addEventListener('click', () => {
      Object.entries(defaults).forEach(([id, value]) => {
        const el = root.querySelector('#' + id), out = el.closest('label').querySelector('output');
        el.value = String(value);
        if (out) out.textContent = el.value + (out.dataset.unit || '');
      });
      if (onReset) onReset();
      draw();
    });
    draw();
  }
  const val = (root, id) => +root.querySelector('#' + id).value;
  const sel = (root, id) => root.querySelector('#' + id).value;
  const on = (root, action, fn) => root.querySelector(`[data-action="${action}"]`).addEventListener('click', fn);
  const legend = items => `<div class="legend">${items.map(([c, t]) => `<span><i style="background:${c}"></i>${t}</span>`).join('')}</div>`;

  // Mountain Car dynamics (Sutton & Barto, Example 10.1). a is -1, 0 or +1.
  function carStep(x, v, a) {
    v = Math.max(-0.07, Math.min(0.07, v + 0.001 * a - 0.0025 * Math.cos(3 * x)));
    x += v;
    if (x < -1.2) { x = -1.2; v = 0; }
    if (x >= 0.5) return [0.5, v, true];
    return [x, v, false];
  }

  // Three fixed policies; the hill and the position-velocity plane share one x axis.
  Demos.register('fa-car-lab', (root, H) => {
    const T = 200, FLOOR = -Math.PI / 6, W = 560, ML = 70, MR = 14, SHADE = '#dce6ee';
    const policy = {right: () => 1, position: x => (x > FLOOR ? 1 : -1), momentum: (x, v) => (v < 0 ? -1 : 1)};
    root.querySelector('[data-controls]').innerHTML =
      H.select('fa-car-policy', 'Fixed policy', [{value: 'right', label: 'Always push right'}, {value: 'position', label: 'Push uphill (x only)'}, {value: 'momentum', label: 'Push with the motion'}], 'right') +
      H.range('fa-car-x0', 'Start position x₀', -0.5, -0.6, -0.4, .05) + H.range('fa-car-t', 'Time step t', 0, 0, T, 1) + H.button('Reset', 'reset');
    const X = x => ML + (x + 1.2) / 1.8 * (W - ML - MR);
    function phase(path, t, mode) {
      const h = 186, top = 8, bot = h - 46, V = v => bot - (v + 0.07) / 0.14 * (bot - top), f = z => z.toFixed(1);
      let s = `<svg class="chart" viewBox="0 0 ${W} ${h}" role="img" aria-label="Position-velocity plane: trajectory up to step t and the region where the policy pushes left"><title>Position-velocity plane</title>`;
      if (mode === 'position') s += `<rect x="${f(X(-1.2))}" y="${top}" width="${f(X(FLOOR) - X(-1.2))}" height="${bot - top}" fill="${SHADE}"/>` +
        `<text x="${f((X(-1.2) + X(FLOOR)) / 2)}" y="${top + 20}" text-anchor="middle">push left</text><text x="${f((X(FLOOR) + X(0.5)) / 2)}" y="${top + 20}" text-anchor="middle">push right</text>`;
      else if (mode === 'momentum') s += `<rect x="${f(X(-1.2))}" y="${f(V(0))}" width="${f(X(0.6) - X(-1.2))}" height="${f(bot - V(0))}" fill="${SHADE}"/>` +
        `<text x="${f(X(-0.35))}" y="${top + 20}" text-anchor="middle">push right (ẋ ≥ 0)</text><text x="${f(X(-0.35))}" y="${bot - 8}" text-anchor="middle">push left (ẋ &lt; 0)</text>`;
      else s += `<text x="${f(X(-0.35))}" y="${top + 20}" text-anchor="middle">push right everywhere</text>`;
      [-0.07, 0, 0.07].forEach(v => { s += `<path d="M${ML},${f(V(v))}H${W - MR}" stroke="#d6d6cc"/><text x="${ML - 8}" y="${f(V(v) + 6)}" text-anchor="end">${v}</text>`; });
      [-1.2, -0.9, -0.6, -0.3, 0, 0.3, 0.6].forEach(x => { s += `<text x="${f(X(x))}" y="${bot + 22}" text-anchor="middle">${x}</text>`; });
      s += `<path d="M${f(X(0.5))},${top}V${bot}" stroke="${H.colors[2]}" stroke-width="2" stroke-dasharray="6 4"/>`;
      if (t > 0) s += `<polyline points="${path.slice(0, t + 1).map(([x, v]) => `${f(X(x))},${f(V(v))}`).join(' ')}" fill="none" stroke="${H.colors[1]}" stroke-width="2"/>`;
      const [xt, vt] = path[t];
      s += `<circle cx="${f(X(xt))}" cy="${f(V(vt))}" r="5" fill="${H.colors[1]}"/>` +
           `<path d="M${ML},${top}V${bot}H${W - MR}" fill="none" stroke="#444"/>` +
           `<text x="${(W + ML - MR) / 2}" y="${h - 4}" text-anchor="middle">Position x</text>` +
           `<text x="16" y="${(top + bot) / 2}" transform="rotate(-90 16 ${(top + bot) / 2})" text-anchor="middle">Velocity ẋ</text></svg>`;
      return s;
    }
    const draw = () => {
      const mode = sel(root, 'fa-car-policy'), pi = policy[mode], t = val(root, 'fa-car-t');
      let x = val(root, 'fa-car-x0'), v = 0, done = false, goal = 0;
      const path = [[x, v]];
      for (let k = 1; k <= T; k++) {
        if (!done) { [x, v, done] = carStep(x, v, pi(x, v)); if (done) goal = k; }
        path.push([x, v]);
      }
      const [xt, vt] = path[t], over = goal && t >= goal, a = pi(xt, vt), seen = path.slice(0, t + 1).map(p => p[0]);
      const hi = Math.max(...seen), lo = Math.min(...seen);
      root.querySelector('[data-plot]').innerHTML = H.plot({
        series: [{label: 'Hill: height sin 3x', points: grid(z => Math.sin(3 * z), -1.2, 0.5, 120), color: '#7d7466'},
                 {label: 'Car at step t', points: [[xt, Math.sin(3 * xt)]], color: H.colors[1], dots: true},
                 {label: 'Goal, x = 0.5', points: [[0.5, Math.sin(1.5)]], color: H.colors[2], dots: true}],
        xDomain: [-1.2, 0.6], yDomain: [-1.2, 1.2], xTicks: 6, yTicks: 4, xLabel: '', yLabel: 'Height', title: 'Car on the hill at step t', height: 200}) +
        phase(path, t, mode) + legend([[H.colors[1], 'trajectory to step t'], [H.colors[2], 'goal, x = 0.5']]);
      root.querySelector('[data-readout]').innerHTML =
        `Step ${t}: x = ${H.fmt(xt, 3)}, ẋ = ${H.fmt(vt, 4)}, ${over ? 'episode over' : a > 0 ? 'push right (+1)' : 'push left (-1)'}.<br>` +
        (over ? `Goal reached at step <strong>${goal}</strong>.`
          : t < T ? `Highest position so far <strong>${H.fmt(hi, 3)}</strong>; the goal is x = 0.5.`
          : `No goal within ${T} steps: x stayed between ${H.fmt(lo, 3)} and <strong>${H.fmt(hi, 3)}</strong>.` +
            (mode === 'position' && lo > FLOOR ? '<br>It never entered the push-left region: same as full throttle.' : ''));
    };
    wire(root, {'fa-car-policy': 'right', 'fa-car-x0': -0.5, 'fa-car-t': 0}, draw);
  });

  // One linear update from w = 0 towards target 1 at x = 0 (alpha * delta = 0.5).
  Demos.register('fa-footprint-lab', (root, H) => {
    const xs = Array.from({length: 9}, (_, i) => -1 + i * .25), C = [-.65, 0, .65];
    root.querySelector('[data-controls]').innerHTML =
      H.select('fa-repr', 'Representation', [{value: 'table', label: 'Table (9 one-hot)'}, {value: 'bins', label: 'Aggregation (3 regions)'}, {value: 'rbf', label: 'Radial basis (3)'}], 'table') +
      H.range('fa-width', 'Radial basis width σ', .35, .1, 1, .05) + H.button('Reset', 'reset');
    const repr = root.querySelector('#fa-repr');
    // The width only matters for radial basis features, so moving it selects them (registered before the redraw).
    root.querySelector('#fa-width').addEventListener('input', () => { repr.value = 'rbf'; });
    const draw = () => {
      const mode = repr.value, sigma = val(root, 'fa-width');
      const x = s => mode === 'table' ? xs.map(c => (Math.abs(c - s) < .01 ? 1 : 0))
        : mode === 'bins' ? [0, 1, 2].map(i => (i === (s < -.33 ? 0 : s > .33 ? 2 : 1) ? 1 : 0))
        : C.map(c => Math.exp(-.5 * ((s - c) / sigma) ** 2));
      const x0 = x(0), dv = xs.map(s => .5 * dot(x(s), x0));
      root.querySelector('[data-plot]').innerHTML = H.bars({labels: xs.map(String), values: dv, min: 0, max: 1.2, ticks: 4, colors: [H.colors[0]], yLabel: 'Change in v̂(s′, w)', title: 'Change in predicted value at each state after one update at x = 0', height: 270})
        .replace('</svg>', '<text x="315" y="268" text-anchor="middle">State s′</text></svg>');
      root.querySelector('[data-readout]').innerHTML =
        `Change at x = 0: <strong>${H.fmt(dv[4], 3)}</strong>; at x = 0.25: <strong>${H.fmt(dv[5], 3)}</strong>.<br>` +
        (mode === 'table' ? '<b>x</b>(0) is one-hot, ‖<b>x</b>(0)‖² = 1: distinct states share nothing.'
          : mode === 'bins' ? '<b>x</b>(0) = (0, 1, 0), ‖<b>x</b>(0)‖² = 1: the middle region shares one weight.'
          : `<b>x</b>(0) = (${x0.map(f => H.fmt(f, 2)).join(', ')}), ‖<b>x</b>(0)‖² = ${H.fmt(dot(x0, x0), 2)}: ${dv[5] < .05 ? 'narrow features barely overlap' : 'overlap spreads the change'}.`);
    };
    wire(root, {'fa-repr': 'table', 'fa-width': .35}, draw);
  });

  // Rewards 1, 2, 4 then terminal; current estimate 2.
  Demos.register('fa-target-lab', (root, H) => {
    root.querySelector('[data-controls]').innerHTML =
      H.range('fa-gamma', 'Discount γ', .9, 0, 1, .05) + H.range('fa-next', 'Next-state estimate v̂(S<sub>t+1</sub>,&nbsp;<b>w</b>)', 3, -2, 8, .1) + H.button('Reset', 'reset');
    // gamma * next can have three decimals (0.05 x 0.1 steps): show them, so that target - 2 = error exactly.
    const f = x => (x.toFixed(3).endsWith('0') ? x.toFixed(2) : x.toFixed(3));
    const draw = () => {
      const g = val(root, 'fa-gamma'), next = val(root, 'fa-next'), mc = 1 + 2 * g + 4 * g * g, td = 1 + g * next;
      root.querySelector('[data-plot]').innerHTML = H.bars({labels: ['Current estimate', 'MC target', 'TD target'], values: [2, mc, td], min: -2, max: 10, ticks: 6, colors: [H.colors[3], H.colors[0], H.colors[1]], yLabel: 'Value', title: 'Current estimate and the two targets', height: 250});
      root.querySelector('[data-readout]').innerHTML =
        `MC: G<sub>t</sub> = 1 + ${num(g)}×2 + ${num(g * g)}×4 = <strong>${f(mc)}</strong>, error ${f(mc - 2)}<br>` +
        `TD: R<sub>t+1</sub> + γv̂(S<sub>t+1</sub>, <b>w</b>) = 1 + ${num(g)}×${num(next)} = <strong>${f(td)}</strong>, δ<sub>t</sub> = ${f(td - 2)}` +
        (g === 0 ? '<br>With γ = 0 both targets are just R<sub>t+1</sub> = 1.' : '');
    };
    wire(root, {'fa-gamma': .9, 'fa-next': 3}, draw);
  });

  // 100-state random walk (a scaled version of Sutton & Barto, Example 9.1).
  // Table and state aggregation learn from the same seeded episodes.
  Demos.register('fa-walk-lab', (root, H) => {
    const N = 100, K = 10, START = 50, G = 10, SEED = 6, INIT = 10, MAXEP = 2000;
    const group = s => Math.floor((s - 1) / (N / G));
    // exact v_pi by Gauss-Seidel; on-policy distribution mu from expected visits
    const v = new Float64Array(N + 2), eta = new Float64Array(N + 2);
    for (let it = 0; it < 5000; it++) {
      let d = 0;
      for (let s = 1; s <= N; s++) {
        let acc = 0;
        for (let j = 1; j <= K; j++) acc += (s - j < 1 ? -1 : v[s - j]) + (s + j > N ? 1 : v[s + j]);
        d = Math.max(d, Math.abs(acc / (2 * K) - v[s])); v[s] = acc / (2 * K);
      }
      if (d < 1e-12) break;
    }
    for (let it = 0; it < 5000; it++) {
      const e2 = new Float64Array(N + 2); e2[START] = 1;
      for (let s = 1; s <= N; s++) for (let j = 1; j <= K; j++) { if (s - j >= 1) e2[s - j] += eta[s] / (2 * K); if (s + j <= N) e2[s + j] += eta[s] / (2 * K); }
      let d = 0; for (let s = 1; s <= N; s++) { d = Math.max(d, Math.abs(e2[s] - eta[s])); eta[s] = e2[s]; }
      if (d < 1e-10) break;
    }
    const total = eta.reduce((a, b) => a + b, 0), mu = Array.from(eta, e => e / total);
    const rmsVE = f => Math.sqrt(mu.reduce((acc, m, s) => (s >= 1 && s <= N ? acc + m * (v[s] - f(s)) ** 2 : acc), 0));
    const best = Array.from({length: G}, (_, k) => { let a = 0, b = 0; for (let s = k * 10 + 1; s <= k * 10 + 10; s++) { a += mu[s] * v[s]; b += mu[s]; } return a / b; });
    const bestVE = rmsVE(s => best[group(s)]);
    let episodes = INIT;
    root.querySelector('[data-controls]').innerHTML =
      H.select('fa-walk-method', 'Method', [{value: 'td', label: 'Semi-gradient TD(0)'}, {value: 'mc', label: 'Gradient Monte Carlo'}], 'td') +
      H.select('fa-walk-alpha', 'Step size α', ['0.002', '0.005', '0.01', '0.02', '0.05', '0.1'].map(a => ({value: a, label: a})), '0.05') +
      H.button('+10 episodes', 'ten') + H.button('+100', 'hundred') + H.button('+1000', 'thousand') + H.button('Reset', 'reset');
    function learn(method, alpha, n) {
      const r = H.rng(SEED), wT = new Float64Array(N + 1), wA = new Float64Array(G);
      for (let e = 0; e < n; e++) {
        const S = []; let s = START, ret = 0;
        for (let k = 0; k < 100000; k++) {
          S.push(s);
          const j = 1 + Math.floor(r() * K), nxt = r() < .5 ? s - j : s + j;
          if (nxt < 1) { ret = -1; break; }
          if (nxt > N) { ret = 1; break; }
          s = nxt;
        }
        S.forEach((s, t) => {
          const last = t === S.length - 1;
          const uT = method === 'mc' || last ? ret : wT[S[t + 1]];
          const uA = method === 'mc' || last ? ret : wA[group(S[t + 1])];
          wT[s] += alpha * (uT - wT[s]);
          wA[group(s)] += alpha * (uA - wA[group(s)]);
        });
      }
      return {wT, wA};
    }
    const draw = () => {
      const {wT, wA} = learn(sel(root, 'fa-walk-method'), val(root, 'fa-walk-alpha'), episodes);
      const states = Array.from({length: N}, (_, i) => i + 1), full = episodes >= MAXEP;
      root.querySelector('[data-plot]').innerHTML = H.plot({
        series: [{label: 'True value', points: states.map(s => [s, v[s]]), color: '#000', width: 2},
                 {label: 'Table (100 weights)', points: states.map(s => [s, wT[s]]), color: H.colors[0], width: 2},
                 {label: 'Aggregation (10 weights)', points: wA.reduce((p, w, k) => p.concat([[k * 10 + .5, w], [k * 10 + 10.5, w]]), []), color: H.colors[1]}],
        xDomain: [0, 100], yDomain: [-1, 1], xTicks: 4, yTicks: 4, xLabel: 'State', yLabel: 'Value', height: 280});
      root.querySelector('[data-readout]').innerHTML = `After <strong>${episodes}</strong> episodes (seed ${SEED}), μ-weighted RMS error √VE:<br>table <strong>${H.fmt(rmsVE(s => wT[s]), 3)}</strong>, aggregation <strong>${H.fmt(rmsVE(s => wA[group(s)]), 3)}</strong> (best possible for aggregation ${H.fmt(bestVE, 3)})` +
        (full ? `<br>Limit of ${MAXEP} episodes reached; Reset returns to ${INIT}.` : '');
      ['ten', 'hundred', 'thousand'].forEach(a => { root.querySelector(`[data-action="${a}"]`).disabled = full; });
    };
    const add = n => { episodes = Math.min(MAXEP, episodes + n); draw(); };
    on(root, 'ten', () => add(10));
    on(root, 'hundred', () => add(100));
    on(root, 'thousand', () => add(1000));
    wire(root, {'fa-walk-method': 'td', 'fa-walk-alpha': '0.05'}, draw, () => { episodes = INIT; });
  });

  // Four tilings of [0, 1], tile width 1/4, offsets 0, 1/4, 1/2, 3/4 of a tile.
  // Every tile is one binary feature, numbered 1-19 across the tilings.
  Demos.register('fa-tiles-lab', (root, H) => {
    const W = .25, L = 84, S = 440, FIRST = [1, 5, 10, 15];
    const off = row => row * W / 4;
    const tile = (x, row) => Math.floor((Math.min(x, 1 - Number.EPSILON) + off(row)) / W);
    root.querySelector('[data-controls]').innerHTML = H.range('fa-tile-x', 'Position x', .49, 0, 1, .01) + H.range('fa-tile-y', 'Compare with x′', .55, 0, 1, .01) + H.button('Reset', 'reset');
    const draw = () => {
      const x = val(root, 'fa-tile-x'), y = val(root, 'fa-tile-y'), X = u => L + u * S, ax = [], ay = [];
      // Draw order: tiles, x' outlines, the two position lines, then tile numbers with a halo in the tile colour.
      let svg = '<svg class="chart" viewBox="0 0 540 250" role="img" aria-label="Four offset tilings with the active tiles for x and x′ marked"><title>Four offset tilings</title>', labels = '';
      for (let row = 0; row < 4; row++) {
        const kx = tile(x, row), ky = tile(y, row), top = 8 + row * 48;
        ax.push(FIRST[row] + kx); ay.push(FIRST[row] + ky);
        svg += `<text x="6" y="${top + 25}">Tiling ${row + 1}</text>`;
        for (let k = 0; k <= 4; k++) {
          const a = Math.max(0, k * W - off(row)), b = Math.min(1, (k + 1) * W - off(row));
          if (b <= a) continue;
          svg += `<rect x="${X(a)}" y="${top}" width="${(b - a) * S}" height="38" fill="${k === kx ? H.colors[0] : '#e2e4dc'}" fill-opacity="${k === kx ? .5 : 1}" stroke="#f0f1eb" stroke-width="3"/>`;
          if (k === ky) svg += `<rect x="${X(a) + 3}" y="${top + 3}" width="${Math.max(0, (b - a) * S - 6)}" height="32" fill="none" stroke="${H.colors[1]}" stroke-width="4"/>`;
          labels += `<text x="${X((a + b) / 2)}" y="${top + 25}" text-anchor="middle" stroke="${k === kx ? '#90acbe' : '#e2e4dc'}" stroke-width="10" stroke-linejoin="round" paint-order="stroke">${FIRST[row] + k}</text>`;
        }
      }
      svg += `<line x1="${X(x)}" x2="${X(x)}" y1="2" y2="200" stroke="#000" stroke-width="3"/>` +
             `<line x1="${X(y)}" x2="${X(y)}" y1="2" y2="200" stroke="${H.colors[1]}" stroke-width="3" stroke-dasharray="7 5"/>` + labels +
             [0, .25, .5, .75, 1].map(u => `<text x="${X(u)}" y="226" text-anchor="middle">${u}</text>`).join('') +
             `<text x="${L + S / 2}" y="248" text-anchor="middle">Position</text></svg>` +
             legend([['#000', 'x and its active tiles'], [H.colors[1], 'x′ and its active tiles (outlined)']]);
      root.querySelector('[data-plot]').innerHTML = svg;
      const shared = ax.filter(f => ay.includes(f));
      root.querySelector('[data-readout]').innerHTML = `Active features at x: ${ax.join(', ')}, so v̂(x, <b>w</b>) = ${ax.map(f => `w<sub>${f}</sub>`).join(' + ')}.<br>` +
        `x′ shares <strong>${shared.length} of 4</strong>${shared.length ? ` (${shared.join(', ')})` : ''}: an update at x moves v̂(x′) ${shared.length}/4 as far as v̂(x).`;
    };
    wire(root, {'fa-tile-x': .49, 'fa-tile-y': .55}, draw);
  });

  // Semi-gradient Sarsa with grid tile coding on Mountain Car (Sutton & Barto, §10.1).
  // Greedy (epsilon = 0), w = 0 (optimistic), reward -1 per step, gamma = 1, alpha = eta / m.
  Demos.register('fa-sarsa-lab', (root, H) => {
    const SEED = 10, CAP = 10000, INIT = 10, MAXEP = 500, NT = 8, PER = (NT + 1) * (NT + 1);
    const PW = 470, PH = 240, PL = 70, PT = 10;
    let episodes = INIT, memo = {};
    root.querySelector('[data-controls]').innerHTML =
      H.select('fa-mc-tilings', 'Features', [{value: '8', label: '8 tilings, 8 × 8 each'}, {value: '1', label: '1 tiling, 8 × 8'}], '8') +
      H.range('fa-mc-eta', 'Step size η, α = η/m', .5, .1, .5, .1) +
      H.select('fa-mc-view', 'Show', [{value: 'cost', label: 'Cost-to-go, −max q̂'}, {value: 'steps', label: 'Steps per episode'}], 'cost') +
      H.button('+1 episode', 'one') + H.button('+10', 'ten') + H.button('+100', 'hundred') + H.button('Reset', 'reset');
    const features = (m, x, v) => {
      const px = (x + 1.2) / 1.7 * NT, pv = (v + 0.07) / 0.14 * NT, out = [];
      for (let k = 0; k < m; k++) {
        const i = Math.min(NT, Math.floor(px + (k % m) / m)), j = Math.min(NT, Math.floor(pv + (3 * k % m) / m));
        out.push(k * PER + i * (NT + 1) + j);
      }
      return out;
    };
    function train(m, eta, n) {
      const key = `${m}|${eta}|${n}`;
      if (memo.key === key) return memo.res;
      const r = H.rng(SEED), d = m * PER, w = new Float64Array(3 * d), alpha = eta / m, lens = [];
      const q = (f, a) => f.reduce((s, i) => s + w[a * d + i], 0);
      const greedy = f => { const qs = [0, 1, 2].map(a => q(f, a)), top = Math.max(...qs), ties = [0, 1, 2].filter(a => qs[a] >= top - 1e-9); return ties[Math.floor(r() * ties.length)]; };
      let path = [];
      for (let e = 0; e < n; e++) {
        let x = -0.6 + 0.2 * r(), v = 0, f = features(m, x, v), a = greedy(f), t = 0;
        path = [[x, v]];
        while (t < CAP) {
          const [x2, v2, done] = carStep(x, v, a - 1);
          t++; path.push([x2, v2]);
          if (done) { const delta = -1 - q(f, a); f.forEach(i => { w[a * d + i] += alpha * delta; }); break; }
          const f2 = features(m, x2, v2), a2 = greedy(f2), delta = -1 + q(f2, a2) - q(f, a);
          f.forEach(i => { w[a * d + i] += alpha * delta; });
          x = x2; v = v2; f = f2; a = a2;
        }
        lens.push(t);
      }
      const res = {lens, path, cost: (x, v) => { const f = features(m, x, v); return -Math.max(q(f, 0), q(f, 1), q(f, 2)); }};
      memo = {key, res};
      return res;
    }
    function heatmap(res) {
      const nx = 34, nv = 24, cells = [];
      let top = 0;
      for (let i = 0; i < nx; i++) for (let j = 0; j < nv; j++) {
        const c = Math.max(0, res.cost(-1.2 + 1.7 * (i + .5) / nx, -0.07 + 0.14 * (j + .5) / nv));
        cells.push([i, j, c]); top = Math.max(top, c);
      }
      const scale = top > 0 ? nice(top)[0] : 1, X = x => PL + (x + 1.2) / 1.7 * PW, Y = v => PT + PH - (v + 0.07) / 0.14 * PH;
      let s = `<svg class="chart" viewBox="0 0 560 310" role="img" aria-label="Learned cost-to-go over position and velocity"><title>Learned cost-to-go over position and velocity</title>`;
      cells.forEach(([i, j, c]) => { s += `<rect x="${(PL + i * PW / nx).toFixed(1)}" y="${(PT + PH - (j + 1) * PH / nv).toFixed(1)}" width="${(PW / nx + .6).toFixed(1)}" height="${(PH / nv + .6).toFixed(1)}" fill="${H.colors[0]}" fill-opacity="${(.04 + .9 * c / scale).toFixed(3)}"/>`; });
      const pts = res.path, stride = Math.max(1, Math.ceil(pts.length / 800));
      if (pts.length > 1) s += `<polyline points="${pts.filter((_, k) => k % stride === 0 || k === pts.length - 1).map(([x, v]) => `${X(x).toFixed(1)},${Y(v).toFixed(1)}`).join(' ')}" fill="none" stroke="${H.colors[1]}" stroke-width="2"/>`;
      [-1.2, -0.8, -0.4, 0, 0.4].forEach(x => { s += `<text x="${X(x)}" y="${PT + PH + 22}" text-anchor="middle">${x}</text>`; });
      [-0.07, 0, 0.07].forEach(v => { s += `<text x="${PL - 8}" y="${Y(v) + 6}" text-anchor="end">${v}</text>`; });
      s += `<path d="M${PL},${PT}V${PT + PH}H${PL + PW}" fill="none" stroke="#444"/>` +
           `<text x="${PL + PW / 2}" y="304" text-anchor="middle">Position x</text>` +
           `<text x="16" y="${PT + PH / 2}" transform="rotate(-90 16 ${PT + PH / 2})" text-anchor="middle">Velocity ẋ</text></svg>`;
      return s + legend([[H.colors[0], `cost-to-go: 0 (light) to ${scale} steps (dark)`], [H.colors[1], 'last episode']]);
    }
    // Steps per episode on a log axis from 10 to 10,000 steps (the episode limit), so no episode is cut off.
    function stepsPlot(L) {
      const w = 560, h = 290, ml = 70, mr = 14, mt = 16, mb = 52, [xm, xt] = nice(Math.max(10, L.length));
      const X = k => ml + k / xm * (w - ml - mr), Y = n => h - mb - (Math.log10(n) - 1) / 3 * (h - mt - mb);
      let s = `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="Steps per episode, logarithmic axis"><title>Steps per episode, logarithmic axis</title>`;
      [10, 100, 1000, 10000].forEach(n => { s += `<path d="M${ml},${Y(n)}H${w - mr}" stroke="#d6d6cc"/><text x="${ml - 8}" y="${Y(n) + 6}" text-anchor="end">${n.toLocaleString('en-GB')}</text>`; });
      for (let i = 0; i <= xt; i++) s += `<text x="${X(i * xm / xt)}" y="${h - mb + 22}" text-anchor="middle">${i * xm / xt}</text>`;
      s += `<polyline points="${L.map((n, k) => `${X(k + 1).toFixed(1)},${Y(n).toFixed(1)}`).join(' ')}" fill="none" stroke="${H.colors[0]}" stroke-width="2"/>`;
      if (L.length === 1) s += `<circle cx="${X(1)}" cy="${Y(L[0])}" r="4" fill="${H.colors[0]}"/>`;
      s += `<path d="M${ml},${mt}V${h - mb}H${w - mr}" fill="none" stroke="#444"/>` +
           `<text x="${(w + ml - mr) / 2}" y="${h - 6}" text-anchor="middle">Episode</text>` +
           `<text x="16" y="${(h - mb + mt) / 2}" transform="rotate(-90 16 ${(h - mb + mt) / 2})" text-anchor="middle">Steps (log scale)</text></svg>`;
      return s + legend([[H.colors[0], 'steps to reach the goal in each episode']]);
    }
    const draw = () => {
      const m = val(root, 'fa-mc-tilings'), res = train(m, val(root, 'fa-mc-eta'), episodes), L = res.lens, full = episodes >= MAXEP;
      root.querySelector('[data-plot]').innerHTML = sel(root, 'fa-mc-view') === 'cost' ? heatmap(res) : stepsPlot(L);
      const last10 = L.slice(-10), mean = last10.reduce((a, b) => a + b, 0) / last10.length, capped = L.filter(n => n >= CAP).length;
      root.querySelector('[data-readout]').innerHTML =
        `Episodes <strong>${episodes}</strong> (seed ${SEED}): first ${L[0]} steps, last <strong>${L[L.length - 1]}</strong>, mean of last ${last10.length}: <strong>${H.fmt(mean, 0)}</strong>.` +
        (capped ? `<br>${capped} episode(s) stopped at the ${CAP.toLocaleString('en-GB')}-step limit.` : '') +
        (full ? `<br>Limit of ${MAXEP} episodes reached; Reset returns to ${INIT}.` : '');
      ['one', 'ten', 'hundred'].forEach(a => { root.querySelector(`[data-action="${a}"]`).disabled = full; });
    };
    const add = n => { episodes = Math.min(MAXEP, episodes + n); draw(); };
    on(root, 'one', () => add(1));
    on(root, 'ten', () => add(10));
    on(root, 'hundred', () => add(100));
    wire(root, {'fa-mc-tilings': '8', 'fa-mc-eta': .5, 'fa-mc-view': 'cost'}, draw, () => { episodes = INIT; });
  });
})();
