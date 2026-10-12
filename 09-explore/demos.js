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
})();
