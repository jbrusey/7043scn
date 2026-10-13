/* Lecture 9 (Exploration and Exploitation): interactive demonstrations.
   Each Demos.register(id, ...) mounts into the <section id="id"> in explore.html.
   All numbers are computed locally; random runs use fixed, disclosed seeds.
   The bandit model is also exposed as window.ExploreModel so that numbers
   quoted in the speaker notes can be checked from the browser console. */
(function () {
  'use strict';
  const seeded = seed => window.H.rng(seed);
  const normal = rng => window.H.normal(rng);
  const ARMS = ['A', 'B', 'C'];
  const sum = xs => xs.reduce((a, b) => a + b, 0);
  const list = (xs, d = 2) => xs.map(x => Number(x).toFixed(d)).join(', ');

  // ---------- random variates and the Beta distribution ----------
  // Gamma(shape, 1) by Marsaglia and Tsang; Beta as a ratio of two gammas.
  function gammaSample(shape, rng) {
    if (shape < 1) return gammaSample(shape + 1, rng) * Math.pow(Math.max(1e-12, rng()), 1 / shape);
    const d = shape - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (let k = 0; k < 1000; k++) { // bounded; each try is accepted with probability > 0.95
      const x = normal(rng), z = 1 + c * x;
      if (z <= 0) continue;
      const v = z * z * z, u = rng();
      if (u < 1 - .0331 * x ** 4 || Math.log(Math.max(u, 1e-12)) < .5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
    return d;
  }
  const betaSample = (a, b, rng) => { const x = gammaSample(a, rng), y = gammaSample(b, rng); return x / (x + y); };
  // Exact density for integer shapes, normalised in log space.
  function betaDensity(x, a, b) {
    let logB = 0;
    for (let k = 1; k < a; k++) logB += Math.log(k);
    for (let k = 1; k < b; k++) logB += Math.log(k);
    for (let k = 1; k < a + b; k++) logB -= Math.log(k);
    return Math.exp((a - 1) * Math.log(Math.max(x, 1e-12)) + (b - 1) * Math.log(Math.max(1 - x, 1e-12)) - logB);
  }
  // P(arm i has the largest mean) for independent Beta posteriors (trapezoidal rule).
  function probBest(shapes, steps = 4000) {
    const h = 1 / steps, xs = Array.from({length: steps + 1}, (_, k) => k * h);
    const dens = shapes.map(([a, b]) => xs.map(x => betaDensity(x, a, b)));
    const cdf = dens.map(f => { let c = 0; return f.map((y, k) => (c += k ? (y + f[k - 1]) * h / 2 : 0)); });
    return shapes.map((_, i) => {
      const g = xs.map((_, k) => dens[i][k] * cdf.reduce((p, F, j) => (j === i ? p : p * F[k]), 1));
      return h * (sum(g) - (g[0] + g[steps]) / 2);
    });
  }
  // Index of the largest value; ties broken uniformly at random.
  function argmax(xs, rng) {
    const best = Math.max(...xs), ties = xs.map((v, i) => (v === best ? i : -1)).filter(i => i >= 0);
    return ties.length === 1 ? ties[0] : ties[Math.floor(rng() * ties.length)];
  }

  // ---------- Bernoulli bandit ----------
  // Each arm has its own reward stream, so every strategy run with the same
  // seed sees the same n-th reward from a given arm (paired comparisons).
  // Q is the sample mean, updated incrementally; regret adds the gap of each pull.
  function makeBandit(means, seed) {
    return {
      means: means.slice(), best: Math.max(...means), n: means.map(() => 0), s: means.map(() => 0),
      q: means.map(() => 0), t: 0, total: 0, regret: 0, history: [[0, 0]], last: null,
      rng: seeded(seed + 77777), streams: means.map((_, i) => seeded(seed + 1009 * (i + 1)))
    };
  }
  function pull(b, a) {
    const r = b.streams[a]() < b.means[a] ? 1 : 0;
    b.n[a] += 1; b.s[a] += r; b.q[a] += (r - b.q[a]) / b.n[a];
    b.t += 1; b.total += r; b.regret += b.best - b.means[a];
    b.history.push([b.t, b.regret]); b.last = {a, r};
    return r;
  }
  // UCB bonus and index after t pulls: U = sqrt(c ln t / N). UCB1 (Auer et al. 2002) has c = 2.
  const ucbBonus = (b, c = 2) => b.n.map(n => Math.sqrt(c * Math.log(b.t) / n));
  const ucbIndex = (b, c = 2) => ucbBonus(b, c).map((u, i) => b.q[i] + u);
  // Choose the next arm. Every method first pulls each arm once.
  function choose(b, method, o = {}) {
    const untried = b.n.indexOf(0);
    if (untried >= 0) return untried;
    if (method === 'ucb') return argmax(ucbIndex(b, o.c ?? 2), b.rng);
    if (method === 'ts') return argmax(b.n.map((n, i) => betaSample(1 + b.s[i], 1 + n - b.s[i], b.rng)), b.rng);
    if (method === 'epsilon' && o.eps > 0 && b.rng() < o.eps) return Math.floor(b.rng() * b.means.length);
    if (method === 'decay' && b.rng() < Math.min(1, o.c / (b.t + 1))) return Math.floor(b.rng() * b.means.length);
    return argmax(b.q, b.rng);
  }
  function run(means, seed, method, horizon, o) {
    const b = makeBandit(means, seed);
    for (let t = 0; t < horizon; t++) pull(b, choose(b, method, o));
    return b;
  }
  // Mean regret curve over `repeats` runs with seeds 9000 + 113 i, the standard
  // error of the final value across runs, and the number of runs whose most
  // pulled arm is not the best arm.
  function ensemble(means, specs, horizon, repeats) {
    return specs.map(sp => {
      const runs = Array.from({length: repeats}, (_, i) => run(means, 9000 + 113 * i, sp.method, horizon, sp));
      const points = Array.from({length: horizon + 1}, (_, t) => [t, sum(runs.map(r => r.history[t][1])) / repeats]);
      const mean = points[horizon][1], v = sum(runs.map(r => (r.regret - mean) ** 2)) / (repeats - 1);
      const wrong = runs.filter(r => r.n.indexOf(Math.max(...r.n)) !== r.means.indexOf(r.best)).length;
      return {points, mean, se: Math.sqrt(v / repeats), wrong};
    });
  }
  window.ExploreModel = {makeBandit, pull, choose, run, ensemble, ucbBonus, ucbIndex, betaSample, betaDensity, probBest};

  // ---------- chart axes with round tick labels ----------
  function niceStep(raw) {
    const p = 10 ** Math.floor(Math.log10(raw));
    return [1, 2, 2.5, 5, 10].map(k => k * p).find(s => s >= raw * (1 - 1e-9));
  }
  // Smallest round range [lo, hi] with n equal ticks that contains [a, b].
  function niceRange(a, b, n = 4) {
    let step = niceStep(Math.max(b - a, 1e-9) / n);
    for (let k = 0; k < 8; k++) {
      const lo = Math.floor(a / step + 1e-9) * step;
      if (lo + n * step >= b - 1e-9) return {lo, hi: lo + n * step};
      step = niceStep(step * 1.01);
    }
    return {lo: a, hi: b};
  }

  // ---------- control wiring (same pattern as the other decks) ----------
  // Show a slider's value in its <output> without firing another input event.
  const showValue = el => { const o = el.closest('label') && el.closest('label').querySelector('output'); if (o) o.textContent = el.value + (o.dataset.unit || ''); };
  // Every input redraws; Reset restores the defaults and redraws once.
  function wire(root, defaults, draw, onReset) {
    root.querySelectorAll('input,select').forEach(el => el.addEventListener('input', draw));
    const reset = root.querySelector('[data-action="reset"]');
    if (reset) reset.addEventListener('click', () => {
      Object.entries(defaults).forEach(([id, v]) => { const el = root.querySelector('#' + id); el.value = String(v); showValue(el); });
      if (onReset) onReset();
      draw();
    });
    draw();
  }
  const val = (root, id) => +root.querySelector('#' + id).value;
  const on = (root, action, fn) => root.querySelector(`[data-action="${action}"]`).addEventListener('click', fn);
  const btn = (root, action) => root.querySelector(`[data-action="${action}"]`);
  // A nested .demo-controls row keeps a group of buttons together on its own line.
  const row = (...items) => `<div class="demo-controls">${items.join('')}</div>`;

  // 1. Pull the arms by hand. Means (0.35, 0.50, 0.65), seed 901.
  Demos.register('ex-play', (root, H) => {
    const MEANS = [.35, .5, .65];
    let b, reveal;
    root.querySelector('[data-controls]').innerHTML = ARMS.map((a, i) => H.button('Pull ' + a, 'pull' + i)).join('') +
      H.button('Reveal true means', 'truth') + H.button('Reset', 'reset');
    const draw = () => {
      const plot = root.querySelector('[data-plot]');
      plot.innerHTML = H.bars({labels: ARMS.map((a, i) => (b.n[i] ? `${a} (${b.s[i]}/${b.n[i]})` : `${a} (untried)`)),
        values: b.q, min: 0, max: 1, yLabel: 'Success rate so far', height: 250});
      // An untried arm has no estimate: its bar is a placeholder at zero, so hide its value.
      // (H.bars writes a value label and an arm label, both centred, for every bar.)
      const centred = plot.querySelectorAll('svg text[text-anchor="middle"]');
      b.n.forEach((n, i) => { if (!n && centred[2 * i]) centred[2 * i].textContent = ''; });
      root.querySelector('[data-readout]').innerHTML = (b.t ? `${b.t} pull${b.t === 1 ? '' : 's'}, <strong>${b.total}</strong> success${b.total === 1 ? '' : 'es'}. Last: ${ARMS[b.last.a]} → ${b.last.r}.` : 'No evidence yet: choose any arm.') +
        '<br>' + (reveal ? 'True means: A 0.35, B 0.50, C 0.65.' : 'True means are hidden.');
      btn(root, 'truth').textContent = reveal ? 'Hide true means' : 'Reveal true means';
    };
    const restart = () => { b = makeBandit(MEANS, 901); reveal = false; draw(); };
    ARMS.forEach((_, i) => on(root, 'pull' + i, () => { pull(b, i); draw(); }));
    on(root, 'truth', () => { reveal = !reveal; draw(); });
    on(root, 'reset', restart);
    restart();
  });

  // 2. Regret growth: greedy, fixed epsilon and decaying epsilon_t = min(1, c/t).
  //    30 paired runs of 500 pulls, means (0.35, 0.50, 0.65), seeds 9000 + 113 i.
  Demos.register('ex-epsilon-lab', (root, H) => {
    const MEANS = [.35, .5, .65], T = 500, RUNS = 30;
    root.querySelector('[data-controls]').innerHTML = H.range('ex-eps', 'Fixed ε', .1, 0, 1, .05) + H.range('ex-decay', 'Decay constant c', 10, 1, 50, 1) + H.button('Reset', 'reset');
    const draw = () => {
      const eps = val(root, 'ex-eps'), c = val(root, 'ex-decay');
      const res = ensemble(MEANS, [{method: 'greedy'}, {method: 'epsilon', eps}, {method: 'decay', c}], T, RUNS);
      const top = niceRange(0, Math.max(10, ...res.map(r => r.mean)), 4);
      const names = ['Greedy', `Fixed ε = ${eps}`, `Decaying, c = ${c}`], short = ['greedy', 'fixed', 'decaying'];
      root.querySelector('[data-plot]').innerHTML = H.plot({series: res.map((r, i) => ({label: names[i], points: r.points})),
        xDomain: [0, T], yDomain: [0, top.hi], xTicks: 5, yTicks: 4, xLabel: 'Pulls t', yLabel: 'Mean regret (successes)', height: 270});
      root.querySelector('[data-readout]').innerHTML = 'Regret at 500 pulls (mean ± SE): ' +
        res.map((r, i) => `${short[i]} <strong>${H.fmt(r.mean, 1)}</strong> ± ${H.fmt(r.se, 1)}`).join(', ') + '.<br>' +
        'Runs (of 30) whose most-pulled arm is not C: ' + res.map((r, i) => `${short[i]} ${r.wrong}`).join(', ') + '.';
    };
    wire(root, {'ex-eps': .1, 'ex-decay': 10}, draw);
  });

  // 3. UCB run step by step: the index of each arm over time.
  //    Means (0.35, 0.50, 0.65), seed 933; the run opens at t = 30 and ends at t = 300.
  Demos.register('ex-ucb-steps', (root, H) => {
    const MEANS = [.35, .5, .65], SEED = 933, TMAX = 300, START = 30, STEPS = ['step1', 'step10', 'step100'];
    let b, idx;
    root.querySelector('[data-controls]').innerHTML = H.range('ex-ucb-c', 'Bonus constant c', 2, 0, 4, .5) +
      row(H.button('1 step', 'step1'), H.button('10 steps', 'step10'), H.button('100 steps', 'step100'), H.button('Reset', 'reset'));
    const slider = root.querySelector('#ex-ucb-c');
    // Pull up to k more times; once every arm is tried, record every index (unclipped).
    const advance = k => {
      const c = +slider.value;
      for (let i = 0; i < k && b.t < TMAX; i++) {
        pull(b, choose(b, 'ucb', {c}));
        if (b.n.every(x => x > 0)) ucbIndex(b, c).forEach((v, a) => idx[a].push([b.t, v]));
      }
    };
    // Rerun from t = 0 with the current c: same reward streams and tie-break stream.
    const replay = t => { b = makeBandit(MEANS, SEED); idx = [[], [], []]; advance(t); };
    const draw = () => {
      const c = +slider.value, u = ucbBonus(b, c), cur = ucbIndex(b, c), top = Math.max(...cur);
      const lead = ARMS.filter((_, a) => cur[a] === top);
      // the time axis grows in steps of 50 pulls so early steps stay readable
      const xmax = Math.max(50, 50 * Math.ceil(b.t / 50)), xt = xmax <= 100 ? 5 : xmax === 150 ? 6 : xmax / 50;
      // the index axis covers every recorded value, in steps of 0.5 (0.25 when c = 0)
      const peak = Math.max(1, ...idx.flat().map(p => p[1])), ystep = peak <= 1 ? .25 : .5, ymax = ystep * Math.ceil(peak / ystep - 1e-9);
      root.querySelector('[data-plot]').innerHTML = H.plot({
        series: [...idx.map((p, a) => ({label: `${ARMS[a]}: Q + U`, points: p, width: 2.5})),
                 ...MEANS.map((m, a) => ({label: '', points: [[0, m], [xmax, m]], color: H.colors[a], dashed: true, width: 1.5}))],
        xDomain: [0, xmax], yDomain: [0, ymax], xTicks: xt, yTicks: Math.round(ymax / ystep), xLabel: 'Pulls t', yLabel: 'UCB index', height: 270});
      root.querySelector('[data-table] tbody').innerHTML = ARMS.map((a, i) => {
        const cell = x => (lead.includes(a) ? `<strong>${x}</strong>` : x);
        return `<tr><td>${cell(a)}</td><td>${b.n[i]}</td><td>${H.fmt(b.q[i], 3)}</td><td>${H.fmt(u[i], 3)}</td><td>${cell(H.fmt(cur[i], 3))}</td></tr>`;
      }).join('');
      const done = b.t >= TMAX;
      STEPS.forEach(s => { btn(root, s).disabled = done; });
      root.querySelector('[data-readout]').innerHTML = `t = ${b.t}, ln t = ${H.fmt(Math.log(b.t), 3)}, c = ${c}. Regret so far ${H.fmt(b.regret, 2)}.<br>` +
        (done ? `The run ends at t = ${TMAX}: press Reset, or change c to replay it.`
              : `Next pull: <strong>${lead.join(' or ')}</strong>, the largest index${lead.length > 1 ? ' (a tie, broken at random)' : ''}.`);
    };
    const go = k => { advance(k); draw(); };
    slider.addEventListener('input', () => { replay(b.t); draw(); });
    on(root, 'step1', () => go(1));
    on(root, 'step10', () => go(10));
    on(root, 'step100', () => go(100));
    on(root, 'reset', () => { slider.value = '2'; showValue(slider); replay(START); draw(); });
    replay(START);
    draw();
  });

  // 4. Beta posterior after s successes and f failures (uniform prior).
  Demos.register('ex-posterior-lab', (root, H) => {
    root.querySelector('[data-controls]').innerHTML = H.range('ex-success', 'Successes s', 2, 0, 30, 1) + H.range('ex-failure', 'Failures f', 2, 0, 30, 1) +
      H.button('Set 20 and 20', 'many') + H.button('Reset', 'reset');
    const draw = () => {
      const s = val(root, 'ex-success'), f = val(root, 'ex-failure'), a = 1 + s, b = 1 + f;
      const mean = a / (a + b), sd = Math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)));
      const pts = Array.from({length: 201}, (_, i) => [i / 200, betaDensity(i / 200, a, b)]);
      const top = niceRange(0, Math.max(2, ...pts.map(p => p[1])), 4);
      root.querySelector('[data-plot]').innerHTML = H.plot({series: [{label: `Posterior Beta(${a}, ${b})`, points: pts}, {label: 'Prior Beta(1, 1)', points: [[0, 1], [1, 1]], color: '#888', dashed: true, width: 2}],
        xDomain: [0, 1], yDomain: [0, top.hi], xTicks: 4, yTicks: 4, xLabel: 'Success probability θ', yLabel: 'Density', height: 270});
      root.querySelector('[data-readout]').innerHTML = `Beta(1, 1) → Beta(1 + ${s}, 1 + ${f}) = Beta(${a}, ${b}).<br>` +
        `Posterior mean ${a}/${a + b} = <strong>${H.fmt(mean, 3)}</strong>, standard deviation <strong>${H.fmt(sd, 3)}</strong>.`;
      btn(root, 'many').disabled = s === 20 && f === 20;
    };
    on(root, 'many', () => {
      ['ex-success', 'ex-failure'].forEach(id => { const el = root.querySelector('#' + id); el.value = '20'; showValue(el); });
      draw();
    });
    wire(root, {'ex-success': 2, 'ex-failure': 2}, draw);
  });

  // 5. Probability matching with frozen posteriors. Seed 916.
  Demos.register('ex-thompson', (root, H) => {
    const SHAPES = [[7, 3], [2, 2], [4, 4]], exact = probBest(SHAPES);
    const curves = SHAPES.map(([a, b]) => Array.from({length: 201}, (_, i) => [i / 200, betaDensity(i / 200, a, b)]));
    let rng, count, wins, sample;
    root.querySelector('[data-controls]').innerHTML = row(H.button('Draw 1 world', 'one'), H.button('Draw 1,000 worlds', 'many'), H.button('Reset', 'reset'));
    const draw = () => {
      const dots = count ? sample.map((x, a) => ({label: '', points: [[x, betaDensity(x, ...SHAPES[a])]], color: H.colors[a], dots: true})) : [];
      root.querySelector('[data-plot]').innerHTML = H.plot({series: [...curves.map((p, a) => ({label: `${ARMS[a]}: Beta(${SHAPES[a].join(', ')})`, points: p, width: 2.5})), ...dots],
        xDomain: [0, 1], yDomain: [0, 3], xTicks: 4, yTicks: 3, xLabel: 'Mean θ', yLabel: 'Posterior density', height: 270});
      root.querySelector('[data-readout]').innerHTML = count
        ? `Draw ${count.toLocaleString('en-GB')}: θ̃ = (${list(sample, 3)}), so <strong>${ARMS[sample.indexOf(Math.max(...sample))]}</strong> is selected.<br>Dots mark this draw. Won: each arm's share of ${count === 1 ? 'this draw' : `all ${count.toLocaleString('en-GB')} draws`}.`
        : 'No draws yet.<br>Each draw samples one θ̃ per arm and selects the largest.';
      root.querySelector('[data-table]').innerHTML = '<table class="compact"><thead><tr><th>Arm</th><th>Mean</th><th>Won</th><th>Exact P(best)</th></tr></thead><tbody>' +
        ARMS.map((a, i) => `<tr><td>${a}</td><td>${H.fmt(SHAPES[i][0] / (SHAPES[i][0] + SHAPES[i][1]), 2)}</td><td>${count ? H.fmt(wins[i] / count, 3) : '—'}</td><td>${H.fmt(exact[i], 3)}</td></tr>`).join('') + '</tbody></table>';
    };
    const drawWorlds = k => {
      for (let i = 0; i < k; i++) { sample = SHAPES.map(([a, b]) => betaSample(a, b, rng)); wins[sample.indexOf(Math.max(...sample))] += 1; count += 1; }
      draw();
    };
    const restart = () => { rng = seeded(916); count = 0; wins = [0, 0, 0]; sample = [0, 0, 0]; draw(); };
    on(root, 'one', () => drawWorlds(1));
    on(root, 'many', () => drawWorlds(1000));
    on(root, 'reset', restart);
    restart();
  });

  // 6. Four strategies on two instances: 30 paired runs of 400 pulls, seeds 9000 + 113 i.
  Demos.register('ex-tournament', (root, H) => {
    const INST = {wide: [.2, .5, .8], close: [.45, .5, .55]}, T = 400, RUNS = 30;
    const NAMES = ['Greedy', 'ε-greedy (ε = 0.1)', 'UCB1', 'Thompson'];
    root.querySelector('[data-controls]').innerHTML = H.select('ex-instance', 'Bandit instance', [{value: 'wide', label: 'Separated: (0.2, 0.5, 0.8)'}, {value: 'close', label: 'Close: (0.45, 0.5, 0.55)'}], 'wide') + H.button('Reset', 'reset');
    const draw = () => {
      const res = ensemble(INST[root.querySelector('#ex-instance').value], [{method: 'greedy'}, {method: 'epsilon', eps: .1}, {method: 'ucb'}, {method: 'ts'}], T, RUNS);
      const top = niceRange(0, Math.max(10, ...res.map(r => r.mean)), 4);
      root.querySelector('[data-plot]').innerHTML = H.plot({series: res.map((r, i) => ({label: NAMES[i], points: r.points})),
        xDomain: [0, T], yDomain: [0, top.hi], xTicks: 4, yTicks: 4, xLabel: 'Pulls t', yLabel: 'Mean regret (successes)', height: 270});
      root.querySelector('[data-readout]').innerHTML = 'Regret at 400 pulls (mean ± SE): ' +
        res.map((r, i) => `${['greedy', 'ε-greedy', 'UCB1', 'Thompson'][i]} <strong>${H.fmt(r.mean, 1)}</strong> ± ${H.fmt(r.se, 1)}`).join(', ') + '.';
    };
    wire(root, {'ex-instance': 'wide'}, draw);
  });

  /* ======================== Exploring with a Learned Model ========================
     Dyna (moved here from Lecture 8) and Dyna-Q+ on two of Sutton and Barto's mazes. A block scope keeps
     these helpers apart from the bandit code above. */
  {
  const H = window.H;
  const put = (root, sel, html) => { root.querySelector(sel).innerHTML = html; };
  const int = x => Math.round(x).toLocaleString('en-GB');
  // Line chart with a legend whose swatches show solid and dashed lines (H.plot's legend draws solid blocks).
  const legend = lines => '<div class="legend">' + lines.map(l =>
    `<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><path d="M1,5H29" stroke="${l.color}" stroke-width="${l.width || 3}"${l.dashed ? ' stroke-dasharray="6 4"' : ''}/></svg>${H.esc(l.label)}</span>`).join('') + '</div>';
  const plotWithLegend = (o, lines) => H.plot(Object.assign({}, o, {series: lines.map(l => Object.assign({}, l, {label: ''}))})) + legend(lines);

  /* ---- Dyna maze (Sutton & Barto, Example 8.1): 6 x 9 grid, cells numbered row * 9 + col ---- */
  const MAZE = {rows: 6, cols: 9, start: 18, goal: 8, walls: new Set([11, 20, 29, 41, 7, 16, 25])};
  const MOVES = [[-1, 0], [1, 0], [0, -1], [0, 1]];   // up, down, left, right
  function mazeStep(s, a) {
    const r = Math.floor(s / MAZE.cols) + MOVES[a][0], c = s % MAZE.cols + MOVES[a][1], s2 = r * MAZE.cols + c;
    return r < 0 || r >= MAZE.rows || c < 0 || c >= MAZE.cols || MAZE.walls.has(s2) ? s : s2;
  }
  // Tabular Dyna-Q; acting and planning use separate seeded streams. Returns episode lengths,
  // the number of value updates and (optionally) a copy of Q after every episode.
  function dynaQ(n, seed, keepQ) {
    const EPISODES = 50, CAP = 10000, alpha = .1, gamma = .95, eps = .1;
    const act = H.rng(seed), plan = H.rng(seed + 500000);
    const Q = new Float64Array(MAZE.rows * MAZE.cols * 4), model = new Map(), seen = [];
    const maxQ = s => Math.max(Q[4 * s], Q[4 * s + 1], Q[4 * s + 2], Q[4 * s + 3]);
    // Greedy action with random tie-breaking: every action within 1e-12 of the maximum counts as tied.
    const greedy = s => {
      const best = maxQ(s), ties = [];
      for (let a = 0; a < 4; a++) if (best - Q[4 * s + a] <= 1e-12) ties.push(a);
      return ties[Math.floor(act() * ties.length)];
    };
    const update = (k, r, s2) => { Q[k] += alpha * (r + (s2 === MAZE.goal ? 0 : gamma * maxQ(s2)) - Q[k]); };
    const lens = [], snaps = [];
    let updates = 0;
    for (let e = 0; e < EPISODES; e++) {
      let s = MAZE.start, t = 0;
      while (s !== MAZE.goal && t < CAP) {
        const a = act() < eps ? Math.floor(act() * 4) : greedy(s), s2 = mazeStep(s, a), r = s2 === MAZE.goal ? 1 : 0, k = 4 * s + a;
        update(k, r, s2); updates++;
        if (!model.has(k)) seen.push(k);
        model.set(k, [r, s2]);
        for (let i = 0; i < n; i++) { const kk = seen[Math.floor(plan() * seen.length)], m = model.get(kk); update(kk, m[0], m[1]); updates++; }
        s = s2; t++;
      }
      lens.push(t);
      if (keepQ) snaps.push(Float64Array.from(Q));
    }
    return {lens, updates, snaps};
  }
  function mazeSVG(Q) {
    const C = 30, W = MAZE.cols * C, Ht = MAZE.rows * C;
    let s = `<svg class="chart" viewBox="-2 -2 ${W + 4} ${Ht + 4}" role="img" aria-label="Dyna maze with the current greedy actions"><title>Dyna maze: greedy action in each state with a positive value</title>`;
    for (let i = 0; i < MAZE.rows * MAZE.cols; i++) {
      const x = (i % MAZE.cols) * C, y = Math.floor(i / MAZE.cols) * C, wall = MAZE.walls.has(i);
      s += `<rect x="${x}" y="${y}" width="${C}" height="${C}" fill="${wall ? '#6b6359' : '#fafaf6'}" stroke="#c9c5bb"/>`;
      if (wall) continue;
      const q = Array.from(Q.slice(4 * i, 4 * i + 4)), best = Math.max(...q), valued = best > 0 && i !== MAZE.goal;
      if (valued) s += `<rect x="${x}" y="${y}" width="${C}" height="${C}" fill="${H.colors[0]}" fill-opacity="0.12"/>`;
      if (i === MAZE.start || i === MAZE.goal) { s += `<text x="${x + C / 2}" y="${y + C / 2 + 6}" text-anchor="middle" font-weight="bold">${i === MAZE.start ? 'S' : 'G'}</text>`; continue; }
      if (!valued) continue;
      // arrow for the greedy action: shaft plus triangular head
      const [dy, dx] = MOVES[q.indexOf(best)], cx = x + C / 2, cy = y + C / 2, px = -dy, py = dx;
      s += `<path d="M${cx - dx * 9},${cy - dy * 9}L${cx + dx * 4},${cy + dy * 4}" stroke="${H.colors[0]}" stroke-width="2.5"/>` +
           `<path d="M${cx + dx * 11},${cy + dy * 11}L${cx + dx * 3 + px * 5},${cy + dy * 3 + py * 5}L${cx + dx * 3 - px * 5},${cy + dy * 3 - py * 5}Z" fill="${H.colors[0]}"/>`;
    }
    return s + '</svg>';
  }

  // Dyna-Q on the Dyna maze: mean episode length over 20 seeded runs, n = 0 against the chosen n.
  Demos.register('ex-dyna-lab', (root, H) => {
    const RUNS = 20, TOP = 800, SHORT = 25, cache = new Map();
    let seedSet = 1;
    put(root, '[data-controls]', H.select('ex-dyna-n', 'Planning steps n', [0, 5, 10, 50].map(v => ({value: String(v), label: String(v)})), '5') +
      H.range('ex-dyna-k', 'Maze after episode', 2, 1, 10, 1) + H.button('New seeds', 'seeds') + H.button('Reset', 'reset'));
    const runs = n => {
      const key = n + '|' + seedSet;
      if (!cache.has(key)) {
        const mean = new Array(50).fill(0);
        let steps = 0, updates = 0, first = null;
        for (let i = 0; i < RUNS; i++) {
          const res = dynaQ(n, 1000 * seedSet + i + 1, i === 0);
          if (i === 0) first = res.snaps;
          res.lens.forEach((l, j) => { mean[j] += l / RUNS; });
          steps += res.lens.reduce((a, b) => a + b, 0) / RUNS; updates += res.updates / RUNS;
        }
        // first episode from which the mean length stays at or below SHORT steps
        let from = 51;
        for (let j = 49; j >= 0 && mean[j] <= SHORT; j--) from = j + 1;
        cache.set(key, {mean, steps, updates, first, from});
      }
      return cache.get(key);
    };
    const draw = () => {
      const n = val(root, 'ex-dyna-n'), k = val(root, 'ex-dyna-k'), base = runs(0), cur = runs(n);
      const s0 = 1000 * seedSet + 1;
      const pts = m => m.slice(1).map((v, j) => [j + 2, Math.min(TOP, v)]);
      const lines = [{label: `n = ${n}`, points: pts(cur.mean), color: H.colors[0]}];
      if (n !== 0) lines.push({label: 'n = 0 (Q-learning)', points: pts(base.mean), color: H.colors[1], dashed: true});
      put(root, '[data-plot]', plotWithLegend({xDomain: [0, 50], yDomain: [0, TOP], xTicks: 5, xLabel: 'Episode (from 2)', yLabel: 'Steps per episode', height: 270,
        title: 'Mean steps per episode against episode'}, lines));
      const clipped = cur.mean.slice(1).concat(base.mean.slice(1)).some(v => v > TOP) ? ' Values above 800 are drawn at 800.' : '';
      const short = r => r.from <= 50 ? `episode ${r.from}` : 'never (within 50 episodes)';
      const saved = base.steps - cur.steps, cost = saved > 0 ? `${H.fmt((cur.updates - base.updates) / saved, 1)} extra updates per real step saved` : 'no real steps saved';
      put(root, '[data-readout]', n === 0
        ? `Seeds ${s0}–${s0 + RUNS - 1}. Episode 1, a random walk: ${int(cur.mean[0])} steps; episode 2: <strong>${int(cur.mean[1])}</strong>.<br>` +
          `Mean ≤ ${SHORT} steps from <strong>${short(cur)}</strong>; shortest path 14.<br>50 episodes: ${int(cur.steps)} real steps, one value update each.${clipped}`
        : `Seeds ${s0}–${s0 + RUNS - 1}. Episode 1: ${int(cur.mean[0])} steps for every n. Episode 2: ${int(base.mean[1])} (n = 0) vs <strong>${int(cur.mean[1])}</strong> (n = ${n}).<br>` +
          `Mean ≤ ${SHORT} steps from ${short(base)} (n = 0) vs <strong>${short(cur)}</strong> (n = ${n}); shortest path 14.<br>` +
          `50 episodes: ${int(base.steps)} vs ${int(cur.steps)} real steps, ${int(base.updates)} vs ${int(cur.updates)} value updates: <strong>${cost}</strong>.${clipped}`);
      const Q = cur.first[k - 1];
      let valued = 0;
      for (let s = 0; s < MAZE.rows * MAZE.cols; s++) if (s !== MAZE.goal && !MAZE.walls.has(s) && Math.max(Q[4 * s], Q[4 * s + 1], Q[4 * s + 2], Q[4 * s + 3]) > 0) valued++;
      put(root, '[data-maze]', mazeSVG(Q));
      put(root, '[data-maze-note]', `Greedy actions after episode ${k}, n = ${n}, seed ${s0}: ${valued} of 46 states have a positive value.`);
    };
    on(root, 'seeds', () => { seedSet += 1; draw(); });
    wire(root, {'ex-dyna-n': '5', 'ex-dyna-k': 2}, draw, () => { seedSet = 1; });
  });

  /* ---- Shortcut maze (after Sutton & Barto, Example 8.3): same 6 x 9 numbering; the wall in row 3
     runs from column 1 to 8 for the first 3000 steps, then its right end (cell 35) opens ---- */
  const SHORTCUT = {start: 48, goal: 8, gap: 35, change: 3000, T: 6000,
                    before: new Set([28, 29, 30, 31, 32, 33, 34, 35]), after: new Set([28, 29, 30, 31, 32, 33, 34])};
  function shortcutStep(walls, s, a) {
    const r = Math.floor(s / 9) + MOVES[a][0], c = s % 9 + MOVES[a][1], s2 = r * 9 + c;
    return r < 0 || r >= 6 || c < 0 || c >= 9 || walls.has(s2) ? s : s2;
  }
  // Greedy route from S in the opened maze (first best action on ties), at most 40 steps.
  function greedyRoute(Q) {
    let s = SHORTCUT.start;
    const path = [s];
    for (let i = 0; i < 40 && s !== SHORTCUT.goal; i++) {
      let b = 0;
      for (let a = 1; a < 4; a++) if (Q[4 * s + a] > Q[4 * s + b]) b = a;
      s = shortcutStep(SHORTCUT.after, s, b); path.push(s);
    }
    return path;
  }
  // Dyna-Q (kappa = 0) or Dyna-Q+ (kappa > 0) for 6000 steps, n = 50 planning updates per step; each
  // episode restarts at S after reaching G (reward 1). Dyna-Q+ (Sutton & Barto §8.3) plans with reward
  // r + kappa * sqrt(tau), tau = steps since (s, a) was last tried for real, and enters actions never tried
  // from a visited state into the model as "stay put, reward 0", last tried at step 0.
  function dynaQPlus(kappa, seed) {
    const alpha = .1, gamma = .95, eps = .1, n = 50, plus = kappa > 0;
    const act = H.rng(seed), plan = H.rng(seed + 500000);
    const Q = new Float64Array(54 * 4), mR = new Float64Array(54 * 4), mS = new Int32Array(54 * 4).fill(-1), last = new Float64Array(54 * 4);
    const seen = [], visited = new Uint8Array(54);
    const maxQ = s => Math.max(Q[4 * s], Q[4 * s + 1], Q[4 * s + 2], Q[4 * s + 3]);
    const greedy = s => {
      const best = maxQ(s), ties = [];
      for (let a = 0; a < 4; a++) if (best - Q[4 * s + a] <= 1e-12) ties.push(a);
      return ties[Math.floor(act() * ties.length)];
    };
    const update = (k, r, s2) => { Q[k] += alpha * (r + (s2 === SHORTCUT.goal ? 0 : gamma * maxQ(s2)) - Q[k]); };
    const cum = [[0, 0]];
    let s = SHORTCUT.start, goals = 0, before = 0;
    for (let t = 1; t <= SHORTCUT.T; t++) {
      const walls = t <= SHORTCUT.change ? SHORTCUT.before : SHORTCUT.after;
      if (plus && !visited[s]) {
        visited[s] = 1;
        for (let a = 0; a < 4; a++) { const k = 4 * s + a; if (mS[k] < 0) { mS[k] = s; mR[k] = 0; last[k] = 0; seen.push(k); } }
      }
      const a = act() < eps ? Math.floor(act() * 4) : greedy(s), s2 = shortcutStep(walls, s, a), r = s2 === SHORTCUT.goal ? 1 : 0, k = 4 * s + a;
      update(k, r, s2);
      if (mS[k] < 0) seen.push(k);
      mS[k] = s2; mR[k] = r; last[k] = t;
      for (let i = 0; i < n; i++) { const kk = seen[Math.floor(plan() * seen.length)]; update(kk, mR[kk] + (plus ? kappa * Math.sqrt(t - last[kk]) : 0), mS[kk]); }
      if (s2 === SHORTCUT.goal) { goals++; s = SHORTCUT.start; } else s = s2;
      if (t === SHORTCUT.change) before = goals;
      if (t % 50 === 0) cum.push([t, goals]);
    }
    return {cum, before, after: goals - before, route: greedyRoute(Q)};
  }
  const usesShortcut = p => p[p.length - 1] === SHORTCUT.goal && p.includes(SHORTCUT.gap);
  // The opened maze with routes drawn through cell centres; the new opening is shaded.
  function shortcutSVG(routes) {
    const C = 30, cx = i => (i % 9) * C + C / 2, cy = i => Math.floor(i / 9) * C + C / 2;
    let s = `<svg class="chart" viewBox="-2 -2 274 184" role="img" aria-label="Shortcut maze after step 3000 with each agent's greedy route"><title>Shortcut maze after the change: greedy routes of Dyna-Q (dashed) and Dyna-Q+ (solid)</title>`;
    for (let i = 0; i < 54; i++) {
      const x = (i % 9) * C, y = Math.floor(i / 9) * C, wall = SHORTCUT.after.has(i), gap = i === SHORTCUT.gap;
      s += `<rect x="${x}" y="${y}" width="${C}" height="${C}" fill="${wall ? '#6b6359' : gap ? '#e3eedf' : '#fafaf6'}" stroke="${gap ? H.colors[2] : '#c9c5bb'}"${gap ? ' stroke-width="2" stroke-dasharray="4 3"' : ''}/>`;
    }
    routes.forEach(r => {
      // A route that stops short of G ends where its greedy move keeps it in place: mark that cell with a ring.
      const cells = r.path.filter((c, i) => i === 0 || c !== r.path[i - 1]), end = cells[cells.length - 1];
      if (cells.length > 1) s += `<polyline points="${cells.map(i => `${cx(i) + r.dx},${cy(i) + r.dx}`).join(' ')}" fill="none" stroke="${r.color}" stroke-width="3"${r.dashed ? ' stroke-dasharray="6 4"' : ''} stroke-linejoin="round"/>`;
      if (end !== SHORTCUT.goal) s += `<circle cx="${cx(end) + r.dx}" cy="${cy(end) + r.dx}" r="9" fill="none" stroke="${r.color}" stroke-width="3"/>`;
    });
    [[SHORTCUT.start, 'S'], [SHORTCUT.goal, 'G']].forEach(([i, t]) => { s += `<text x="${cx(i)}" y="${cy(i) + 6}" text-anchor="middle" font-weight="bold">${t}</text>`; });
    return s + '</svg>';
  }
  Object.assign(window.ExploreModel, {dynaQ, dynaQPlus, greedyRoute});

  // Dyna-Q against Dyna-Q+ on the shortcut maze: cumulative goals over 6000 steps, mean of 10 seeded runs.
  Demos.register('ex-shortcut-lab', (root, H) => {
    const RUNS = 10, cache = new Map();
    let seedSet = 1;
    put(root, '[data-controls]', H.select('ex-shortcut-kappa', 'Dyna-Q+ bonus κ', ['0.0001', '0.001', '0.01'].map(v => ({value: v, label: v})), '0.001') +
      H.button('New seeds', 'seeds') + H.button('Reset', 'reset'));
    const runs = kappa => {
      const key = kappa + '|' + seedSet;
      if (!cache.has(key)) {
        const cum = new Array(SHORTCUT.T / 50 + 1).fill(0);
        let before = 0, after = 0, found = 0, route = null;
        for (let i = 0; i < RUNS; i++) {
          const res = dynaQPlus(kappa, 1000 * seedSet + i + 1);
          res.cum.forEach(([, g], j) => { cum[j] += g / RUNS; });
          before += res.before / RUNS; after += res.after / RUNS;
          if (usesShortcut(res.route)) found++;
          if (i === 0) route = res.route;
        }
        cache.set(key, {cum: cum.map((g, j) => [50 * j, g]), before, after, found, route});
      }
      return cache.get(key);
    };
    // A greedy move that leaves the state unchanged is a move into a wall or the edge, chosen for its bonus.
    const stalled = p => p.some((c, i) => i > 0 && c === p[i - 1]);
    const describe = p => p[p.length - 1] === SHORTCUT.goal ? `${p.length - 1} steps, ${usesShortcut(p) ? 'through the shortcut' : 'round the left end'}`
      : stalled(p) ? 'stops to re-test a move into a wall or the edge (ringed): its bonus outweighs heading for G' : 'does not head for G (still exploring)';
    const draw = () => {
      const kappa = val(root, 'ex-shortcut-kappa'), q = runs(0), p = runs(kappa), s0 = 1000 * seedSet + 1;
      const top = Math.max(100, Math.ceil(Math.max(q.cum[q.cum.length - 1][1], p.cum[p.cum.length - 1][1]) / 100) * 100);
      put(root, '[data-plot]', plotWithLegend({xDomain: [0, SHORTCUT.T], yDomain: [0, top], xTicks: 6, yTicks: top / 100, xLabel: 'Time steps', yLabel: 'Cumulative goals', height: 270,
        title: 'Cumulative goals reached against time steps'}, [
        {label: `Dyna-Q+ (κ = ${kappa})`, points: p.cum, color: H.colors[0]},
        {label: 'Dyna-Q', points: q.cum, color: H.colors[1], dashed: true},
        {label: 'shortcut opens', points: [[SHORTCUT.change, 0], [SHORTCUT.change, top]], color: '#9b9488', width: 2, dashed: true}]));
      const f = x => H.fmt(x, 1);
      put(root, '[data-readout]', `Seeds ${s0}–${s0 + RUNS - 1}. Goals in steps 1–3000: Dyna-Q ${f(q.before)}, Dyna-Q+ <strong>${f(p.before)}</strong>. ` +
        `Steps 3001–6000: Dyna-Q ${f(q.after)}, Dyna-Q+ <strong>${f(p.after)}</strong>.<br>` +
        `Greedy route at step 6000 takes the shortcut in Dyna-Q ${q.found}/${RUNS} runs, Dyna-Q+ <strong>${p.found}/${RUNS}</strong> (old route 16 steps, shortcut 10).`);
      put(root, '[data-maze]', shortcutSVG([{path: q.route, color: H.colors[1], dashed: true, dx: -3}, {path: p.route, color: H.colors[0], dx: 3}]));
      put(root, '[data-maze-note]', `Greedy routes at step 6000, seed ${s0}: Dyna-Q (dashed) ${describe(q.route)}; Dyna-Q+ (solid) ${describe(p.route)}. Shaded: the new opening.`);
    };
    on(root, 'seeds', () => { seedSet += 1; draw(); });
    wire(root, {'ex-shortcut-kappa': '0.001'}, draw, () => { seedSet = 1; });
  });
  }
})();
