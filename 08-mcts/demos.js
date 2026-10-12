/* Lecture 8 (Monte-Carlo Tree Search): interactive demonstrations.
   Each Demos.register(id, ...) mounts into the <section id="id"> in mcts.html.
   All numbers are computed locally; random runs use fixed, disclosed seeds. */
(function () {
  'use strict';
  const H = window.H;

  // Redraw on any control change; Reset restores the listed defaults.
  // Demos with internal state register their own Reset listener first.
  function wire(root, defaults, draw) {
    root.querySelectorAll('input,select').forEach(el => el.addEventListener('input', draw));
    const reset = root.querySelector('[data-action="reset"]');
    if (reset) reset.addEventListener('click', () => {
      Object.entries(defaults).forEach(([id, v]) => { root.querySelector('#' + id).value = String(v); });
      root.querySelectorAll('input,select').forEach(el => el.dispatchEvent(new Event('input', {bubbles: true})));
      draw();
    });
    draw();
  }
  const val = (root, id) => +root.querySelector('#' + id).value;
  const on = (root, action, fn) => root.querySelector(`[data-action="${action}"]`).addEventListener('click', fn);
  const put = (root, sel, html) => { root.querySelector(sel).innerHTML = html; };
  const pick = (a, b) => Math.abs(a - b) < 1e-9 ? 'either (tie)' : a > b ? 'A' : 'B';
  const int = x => Math.round(x).toLocaleString('en-GB');
  // Score axes: 0–2 in steps of 0.5, otherwise whole numbers up to the next integer.
  const axis = hi => { const m = Math.max(2, Math.ceil(hi - 1e-9)); return {max: m, ticks: m === 2 ? 4 : m}; };
  // Line chart with a legend whose swatches show solid and dashed lines (H.plot's legend draws solid blocks).
  const legend = lines => '<div class="legend">' + lines.map(l =>
    `<span><svg width="30" height="10" viewBox="0 0 30 10" aria-hidden="true"><path d="M1,5H29" stroke="${l.color}" stroke-width="${l.width || 3}"${l.dashed ? ' stroke-dasharray="6 4"' : ''}/></svg>${H.esc(l.label)}</span>`).join('') + '</div>';
  const plotWithLegend = (o, lines) => H.plot(Object.assign({}, o, {series: lines.map(l => Object.assign({}, l, {label: ''}))})) + legend(lines);

  /* ---- A small planning tree shared by the tree trace and the comparison ----
     One agent chooses at both levels; deterministic payoffs at the leaves; γ = 1. */
  const TREE = {R: {children: ['A', 'B']}, A: {children: ['A1', 'A2']}, B: {children: ['B1', 'B2']},
                A1: {reward: .2}, A2: {reward: .8}, B1: {reward: .55}, B2: {reward: .65}};
  const newSearch = seed => ({stats: Object.fromEntries(Object.keys(TREE).map(k => [k, {n: 0, w: 0, stored: k === 'R'}])),
                              rng: H.rng(seed), path: [], trail: [], picks: [], G: null, done: 0});
  // UCT score of a stored child; N(s) is the parent's own visit count (at the root, the sum of child visits).
  const uct = (S, parent, child, c) => {
    const s = S.stats[child];
    return s.n === 0 ? Infinity : s.w / s.n + c * Math.sqrt(Math.log(Math.max(1, S.stats[parent].n)) / s.n);
  };
  // 1. Selection: follow UCT through fully expanded nodes (ties go left). Records the scores compared.
  function select(S, c) {
    S.path = ['R']; S.trail = []; S.picks = []; S.G = null;
    let node = 'R';
    while (TREE[node].children && TREE[node].children.every(k => S.stats[k].stored)) {
      const ch = TREE[node].children, sc = ch.map(k => uct(S, node, k, c));
      let b = 0;
      sc.forEach((v, i) => { if (v > sc[b]) b = i; });
      S.picks.push({node, ch, sc, choice: ch[b], tie: sc.filter(v => Math.abs(v - sc[b]) < 1e-12).length > 1});
      node = ch[b];
      S.path.push(node);
    }
    return node;
  }
  // 2. Expansion: store the first untried child (none at a leaf).
  function expand(S) {
    const ch = TREE[S.path[S.path.length - 1]].children, child = ch && ch.find(k => !S.stats[k].stored);
    if (child) { S.stats[child].stored = true; S.path.push(child); }
    return child;
  }
  // 3. Evaluation: uniform random rollout to a leaf; rollout states are not stored.
  function rollout(S) {
    let node = S.path[S.path.length - 1];
    S.trail = [node];
    while (TREE[node].children) { const ch = TREE[node].children; node = ch[Math.floor(S.rng() * ch.length)]; S.trail.push(node); }
    S.G = TREE[node].reward;
  }
  // 4. Backup: add one visit and the return G to every node on the stored path.
  function backup(S) { S.path.forEach(k => { S.stats[k].n += 1; S.stats[k].w += S.G; }); S.done += 1; }
  function simulate(S, c) { select(S, c); expand(S); rollout(S); backup(S); }

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
    const greedy = s => {
      let best = -Infinity, ties = [];
      for (let a = 0; a < 4; a++) { const q = Q[4 * s + a]; if (q > best + 1e-12) { best = q; ties = [a]; } else if (Math.abs(q - best) <= 1e-12) ties.push(a); }
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

  /* ======================== Model-Based RL ======================== */

  // Silver's AB example: sample episodes from the table lookup model, evaluate by every-visit MC.
  Demos.register('mcts-ab-lab', (root, H) => {
    const SEED = 806, MAX = 200;
    let rng, n, sum, cnt, last, ptsA, ptsB;
    const clear = () => { rng = H.rng(SEED); n = 0; sum = {A: 0, B: 0}; cnt = {A: 0, B: 0}; last = ''; ptsA = []; ptsB = []; };
    put(root, '[data-controls]', H.button('Sample 1 episode', 'one') + H.button('Sample 20', 'twenty') + H.button('Reset', 'reset'));
    // One model episode: start at A or B with probability 1/2, then B ends with reward 1 (probability 0.75) or 0.
    const sample = () => {
      const fromA = rng() < .5, r = rng() < .75 ? 1 : 0;
      if (fromA) { sum.A += r; cnt.A++; }
      sum.B += r; cnt.B++; n++;
      last = fromA ? `A, 0, B, ${r}` : `B, ${r}`;
      if (cnt.A) ptsA.push([n, sum.A / cnt.A]);
      ptsB.push([n, sum.B / cnt.B]);
    };
    const draw = () => {
      const xMax = [20, 50, 100, 200].find(m => n <= m);
      put(root, '[data-plot]', plotWithLegend({xDomain: [0, xMax], yDomain: [0, 1], xTicks: xMax === 20 || xMax === 200 ? 4 : 5,
        xLabel: 'Episodes sampled from the model', yLabel: 'Value estimate', height: 270, title: 'Monte-Carlo value estimates from model samples'}, [
        {label: 'V(A) from samples', points: ptsA, color: H.colors[0]},
        {label: 'V(B) from samples', points: ptsB, color: H.colors[1]},
        {label: 'model MDP value 0.75', points: [[0, .75], [xMax, .75]], color: H.colors[2], width: 2, dashed: true},
        {label: 'real-data V(A) = 0', points: [[0, 0], [xMax, 0]], color: H.colors[3], width: 2, dashed: true}]));
      const vA = cnt.A ? `<strong>${H.fmt(sum.A / cnt.A)}</strong>` : 'undefined (no episode has started at A yet)';
      put(root, '[data-readout]', n
        ? `${n} sampled episode${n > 1 ? 's' : ''}, ${cnt.A} starting at A. Last: ${last}.<br>From samples: V(A) = ${vA}, V(B) = <strong>${H.fmt(sum.B / cnt.B)}</strong>.` +
          (n >= MAX ? ' Limit of 200 episodes reached: Reset to start again.' : '')
        : 'No model samples yet. Real data (MC): V(A) = 0, V(B) = 0.75.<br>Predict V(A), then sample episodes from the model.');
      root.querySelectorAll('[data-action="one"],[data-action="twenty"]').forEach(b => { b.disabled = n >= MAX; });
    };
    on(root, 'one', () => { if (n < MAX) sample(); draw(); });
    on(root, 'twenty', () => { for (let i = 0; i < 20 && n < MAX; i++) sample(); draw(); });
    on(root, 'reset', () => { clear(); draw(); });
    clear(); draw();
  });

  // Analytical: model-predicted value of the risky action with a ±2 SE band for n simulations.
  Demos.register('mcts-bias-lab', (root, H) => {
    put(root, '[data-controls]', H.range('mcts-bias-p', 'Model success probability', .8, .05, .95, .05) + H.range('mcts-bias-n', 'Simulations n', 10, 10, 1000, 10) + H.button('Reset', 'reset'));
    const chart = (q, e) => {
      const w = 560, h = 270, top = 34, base = h - 50, Y = v => base - v / 2 * (base - top), bw = (w - 80) / 3;
      const bars = [{label: 'Real risky', v: 1.2, c: H.colors[3]}, {label: 'Model risky', v: q, c: H.colors[0], e}, {label: 'Safe', v: 1.3, c: H.colors[2]}];
      let s = `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="Expected reward: real and model-predicted risky action against safe"><title>Real reward against model prediction with a ±2 SE band</title><text x="4" y="16">Expected reward</text>`;
      for (let j = 0; j <= 4; j++) s += `<path d="M62 ${Y(j / 2)}H${w - 6}" stroke="#d6d6cc"/><text x="54" y="${Y(j / 2) + 6}" text-anchor="end">${j / 2}</text>`;
      bars.forEach((b, i) => {
        const x = 70 + i * bw, cx = x + bw / 2, hi = b.e ? Math.min(2, b.v + b.e) : b.v;
        s += `<rect x="${x + 14}" y="${Y(b.v)}" width="${bw - 28}" height="${Math.max(1, base - Y(b.v))}" fill="${b.c}"/>`;
        if (b.e) {
          const lo = Math.max(0, b.v - b.e);
          s += `<path d="M${cx},${Y(lo)}V${Y(hi)}M${cx - 14},${Y(lo)}H${cx + 14}M${cx - 14},${Y(hi)}H${cx + 14}" stroke="#222" stroke-width="2.5" fill="none"/>`;
        }
        s += `<text x="${cx}" y="${Y(hi) - 8}" text-anchor="middle">${H.fmt(b.v)}${b.e ? ' ± ' + H.fmt(b.e) : ''}</text>` +
             `<text x="${cx}" y="${h - 24}" text-anchor="middle">${b.label}</text>`;
      });
      return s + '</svg>';
    };
    const draw = () => {
      const p = val(root, 'mcts-bias-p'), n = val(root, 'mcts-bias-n'), q = 2 * p, e = 4 * Math.sqrt(p * (1 - p) / n);
      const lo = Math.max(0, q - e), hi = Math.min(2, q + e), pref = Math.abs(q - 1.3) < 1e-9 ? 'neither (tie)' : q > 1.3 ? 'risky' : 'safe';
      put(root, '[data-plot]', chart(q, e));
      put(root, '[data-readout]', `±2 SE band at n = ${n}: ${H.fmt(lo)} to ${H.fmt(hi)}, which ${lo <= 1.3 && 1.3 <= hi ? 'contains' : 'excludes'} the safe 1.30.<br>` +
        `Model bias ${H.fmt(q - 1.2).replace('-', '−')} at every n. Model prefers <strong>${pref}</strong>; reality prefers <strong>safe</strong>.`);
    };
    wire(root, {'mcts-bias-p': .8, 'mcts-bias-n': 10}, draw);
  });

  /* ======================== Integrated Architectures ======================== */

  // Dyna-Q on the Dyna maze: mean episode length over 20 seeded runs, n = 0 against the chosen n.
  Demos.register('mcts-dyna-lab', (root, H) => {
    const RUNS = 20, TOP = 800, SHORT = 25, cache = new Map();
    let seedSet = 1;
    put(root, '[data-controls]', H.select('mcts-dyna-n', 'Planning steps n', [0, 5, 10, 50].map(v => ({value: String(v), label: String(v)})), '5') +
      H.range('mcts-dyna-k', 'Maze after episode', 2, 1, 10, 1) + H.button('New seeds', 'seeds') + H.button('Reset', 'reset'));
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
      const n = val(root, 'mcts-dyna-n'), k = val(root, 'mcts-dyna-k'), base = runs(0), cur = runs(n);
      const s0 = 1000 * seedSet + 1, seeds = `seeds ${s0}–${s0 + RUNS - 1}`;
      const pts = m => m.slice(1).map((v, j) => [j + 2, Math.min(TOP, v)]);
      const lines = [{label: `n = ${n}`, points: pts(cur.mean), color: H.colors[0]}];
      if (n !== 0) lines.push({label: 'n = 0 (Q-learning)', points: pts(base.mean), color: H.colors[1], dashed: true});
      put(root, '[data-plot]', plotWithLegend({xDomain: [0, 50], yDomain: [0, TOP], xTicks: 5, xLabel: 'Episode (from 2)', yLabel: 'Steps per episode', height: 270,
        title: 'Mean steps per episode against episode'}, lines));
      const clipped = cur.mean.slice(1).concat(base.mean.slice(1)).some(v => v > TOP) ? ' Values above 800 are drawn at 800.' : '';
      const short = r => r.from <= 50 ? `episode ${r.from}` : 'never within 50 episodes';
      const saved = base.steps - cur.steps, cost = saved > 0 ? `${H.fmt((cur.updates - base.updates) / saved, 1)} extra value updates per real step saved` : 'no real steps saved';
      put(root, '[data-readout]', n === 0
        ? `Mean of ${seeds}. Episode 1, a random walk: ${int(cur.mean[0])} steps; episode 2: <strong>${int(cur.mean[1])}</strong>.<br>` +
          `Mean stays ≤ ${SHORT} steps from <strong>${short(cur)}</strong> (shortest path 14). Real steps in 50 episodes: ${int(cur.steps)}, one value update each.${clipped}`
        : `Mean of ${seeds}. Episode 1 is the same random walk for every n (${int(cur.mean[0])} steps). Episode 2: ${int(base.mean[1])} (n = 0), <strong>${int(cur.mean[1])}</strong> (n = ${n}).<br>` +
          `Mean stays ≤ ${SHORT} steps from ${short(base)} (n = 0), <strong>${short(cur)}</strong> (n = ${n}); shortest path 14.<br>` +
          `50 episodes: ${int(base.steps)} vs ${int(cur.steps)} real steps, ${int(base.updates)} vs ${int(cur.updates)} value updates: <strong>${cost}</strong>.${clipped}`);
      const Q = cur.first[k - 1];
      let valued = 0;
      for (let s = 0; s < MAZE.rows * MAZE.cols; s++) if (s !== MAZE.goal && !MAZE.walls.has(s) && Math.max(Q[4 * s], Q[4 * s + 1], Q[4 * s + 2], Q[4 * s + 3]) > 0) valued++;
      put(root, '[data-maze]', mazeSVG(Q));
      put(root, '[data-maze-note]', `Greedy actions after episode ${k}, n = ${n}, seed ${s0}: ${valued} of 46 states have a positive value.`);
    };
    on(root, 'seeds', () => { seedSet += 1; draw(); });
    on(root, 'reset', () => { seedSet = 1; });
    wire(root, {'mcts-dyna-n': '5', 'mcts-dyna-k': 2}, draw);
  });

  /* ======================== Monte-Carlo Tree Search ======================== */

  Demos.register('mcts-uct-lab', (root, H) => {
    put(root, '[data-controls]', H.range('mcts-uct-c', 'Exploration c', 1, 0, 3, .1) + H.range('mcts-uct-nb', 'Visits to B, N(s,B)', 3, 1, 30, 1) + H.button('Reset', 'reset'));
    const draw = () => {
      const c = val(root, 'mcts-uct-c'), nb = val(root, 'mcts-uct-nb'), N = 12 + nb;
      const ba = c * Math.sqrt(Math.log(N) / 12), bb = c * Math.sqrt(Math.log(N) / nb), a = .55 + ba, b = .45 + bb, ax = axis(Math.max(a, b));
      put(root, '[data-plot]', H.bars({labels: ['Q(A)', 'Score A', 'Q(B)', 'Score B'], values: [.55, a, .45, b], min: 0, max: ax.max, ticks: ax.ticks,
        colors: [H.colors[3], H.colors[0], H.colors[3], H.colors[1]], yLabel: 'Mean and UCT score', title: 'UCT score of each root action'}));
      put(root, '[data-readout]', `N(s) = 12 + ${nb} = ${N}. A: 0.55 + ${H.fmt(ba, 3)} = ${H.fmt(a, 3)}; B: 0.45 + ${H.fmt(bb, 3)} = ${H.fmt(b, 3)}.<br>Next simulation: <strong>${pick(a, b)}</strong>.`);
    };
    wire(root, {'mcts-uct-c': 1, 'mcts-uct-nb': 3}, draw);
  });

  // Phase-by-phase MCTS on TREE, drawn as an SVG tree. The state is a function of (c, phases taken):
  // changing c replays the search from the start with the new c to the same point.
  Demos.register('mcts-tree-lab', (root, H) => {
    const SEED = 808, BUDGET = 200, NAMES = ['Select', 'Expand', 'Evaluate', 'Back up'];
    const POS = {R: [294, 40], A: [146, 150], B: [442, 150], A1: [72, 262], A2: [220, 262], B1: [368, 262], B2: [516, 262]};
    const BW = 138, BH = 72, VW = 590, VH = 300;
    const READY = 'Ready: no simulations yet. Step 1 selects from the root R.';
    let S, phase, msg, note;
    put(root, '[data-controls]', H.range('mcts-tree-c', 'Exploration c', 1, 0, 3, .1) + H.button('Step 1: Select', 'phase') +
      H.button('Run 10 simulations', 'ten') + H.button('Reset', 'reset'));
    const cIn = root.querySelector('#mcts-tree-c'), c = () => +cIn.value;
    const f3 = x => H.fmt(x, 3);
    function advance() {
      const cc = c();
      if (phase === 0) {
        const node = select(S, cc);
        const parts = S.picks.map(p => `at ${p.node}, ${p.ch.map((k, i) => `${k} ${f3(p.sc[i])}`).join(' vs ')} → ${p.choice}${p.tie ? ' (tie: left)' : ''}`);
        msg = `<strong>Select</strong>: ` + (parts.length ? `UCT ${parts.join('; ')}. ` : '') +
              (TREE[node].children ? `${node} has an untried child, so selection stops at ${node}.` : `${node} is a leaf, so selection stops there.`);
      } else if (phase === 1) {
        const child = expand(S);
        msg = child ? `<strong>Expand</strong>: store ${child} as a new node; its N stays 0 until the backup.`
                    : `<strong>Expand</strong>: ${S.path[S.path.length - 1]} is a leaf, so nothing is added.`;
      } else if (phase === 2) {
        rollout(S);
        const end = S.trail[S.trail.length - 1];
        msg = S.trail.length > 1 ? `<strong>Evaluate</strong>: uniform random rollout ${S.trail.join(' → ')} returns G = ${S.G}; ${end} is visited but not stored.`
                                 : `<strong>Evaluate</strong>: ${end} is a leaf, so G = ${S.G} without a rollout.`;
      } else {
        backup(S);
        msg = `<strong>Back up</strong>: G = ${S.G} along ${S.path.join(' → ')}; N + 1 and W + ${S.G} at each node.`;
      }
      phase = (phase + 1) % 4;
    }
    const replay = steps => { S = newSearch(SEED); phase = 0; msg = READY; for (let i = 0; i < steps; i++) advance(); };
    function tree() {
      const onPath = (a, b) => S.path.indexOf(b) > 0 && S.path[S.path.indexOf(b) - 1] === a;
      const onTrail = (a, b) => S.trail.indexOf(b) > 0 && S.trail[S.trail.indexOf(b) - 1] === a;
      let s = `<svg class="chart" viewBox="0 0 ${VW} ${VH}" role="img" aria-label="MCTS tree with visits N, total return W and mean Q"><title>Search tree: leaf payoffs A1 0.2, A2 0.8, B1 0.55, B2 0.65</title>`;
      s += `<path d="M6,14H34" stroke="${H.colors[0]}" stroke-width="4"/><text x="40" y="20">tree path</text>` +
           `<path d="M6,40H34" stroke="${H.colors[1]}" stroke-width="3" stroke-dasharray="6 4"/><text x="40" y="46">rollout</text>` +
           `<text x="${VW - 4}" y="20" text-anchor="end">simulations: ${S.done}</text>`;
      Object.entries(TREE).forEach(([p, d]) => (d.children || []).forEach(k => {
        const path = onPath(p, k), roll = !path && onTrail(p, k);
        s += `<path d="M${POS[p][0]},${POS[p][1] + BH / 2}L${POS[k][0]},${POS[k][1] - BH / 2}" stroke="${path ? H.colors[0] : roll ? H.colors[1] : '#b8b3a8'}" stroke-width="${path ? 4 : roll ? 3 : 2}"${roll ? ' stroke-dasharray="6 4"' : ''}/>`;
      }));
      Object.entries(POS).forEach(([k, [x, y]]) => {
        const st = S.stats[k], active = S.path.includes(k), leaf = S.trail.length && S.trail[S.trail.length - 1] === k;
        const box = st.stored ? `fill="${active ? H.colors[0] : '#fafaf6'}"${active ? ' fill-opacity="0.14"' : ''} stroke="${leaf ? H.colors[1] : '#51483d'}" stroke-width="${leaf ? 3 : 1.5}"`
                              : `fill="#f0f1eb" stroke="${leaf ? H.colors[1] : '#9b9488'}" stroke-width="${leaf ? 3 : 1.5}" stroke-dasharray="5 4"`;
        const dim = st.stored ? '' : ' opacity="0.6"';
        const name = `<tspan font-weight="bold">${k}</tspan>` + (TREE[k].children ? '' : ` = ${TREE[k].reward}`);
        s += `<rect x="${x - BW / 2}" y="${y - BH / 2}" width="${BW}" height="${BH}" rx="6" ${box}/>` +
             `<text x="${x}" y="${y - 14}" text-anchor="middle"${dim}>${name}</text>` +
             (st.stored ? `<text x="${x}" y="${y + 6}" text-anchor="middle">N ${st.n} · W ${H.fmt(st.w)}</text>` +
                          `<text x="${x}" y="${y + 26}" text-anchor="middle">Q ${st.n ? H.fmt(st.w / st.n) : '–'}</text>`
                        : `<text x="${x}" y="${y + 14}" text-anchor="middle"${dim}>not stored</text>`);
      });
      return s + '</svg>';
    }
    const draw = () => {
      const A = S.stats.A, B = S.stats.B, full = S.done >= BUDGET && phase === 0;
      const rec = S.done === 0 ? 'none yet' : A.n === B.n ? `A (N(A) = N(B) = ${A.n}, tie broken left)` : `${A.n > B.n ? 'A' : 'B'} (N(A) = ${A.n}, N(B) = ${B.n})`;
      put(root, '[data-tree]', tree());
      put(root, '[data-readout]', (note ? note + '<br>' : '') + msg + `<br>Commit now by most visits: <strong>${rec}</strong>.` +
        (full ? `<br>Budget of ${BUDGET} simulations reached; Reset to start again.` : ''));
      const step = root.querySelector('[data-action="phase"]');
      step.textContent = `Step ${phase + 1}: ${NAMES[phase]}`;
      step.disabled = full;
      root.querySelector('[data-action="ten"]').disabled = full;
    };
    on(root, 'phase', () => { if (!(S.done >= BUDGET && phase === 0)) { note = ''; advance(); } draw(); });
    on(root, 'ten', () => {
      const start = S.done, partial = phase !== 0, target = Math.min(BUDGET, S.done + 10);
      while (S.done < target) advance();            // at most 40 phases
      const ran = target - start;
      note = partial ? `Finished the simulation in progress and ran ${ran - 1} more (${S.done} in total).` : `Ran ${ran} simulation${ran === 1 ? '' : 's'} (${S.done} in total). Last one:`;
      draw();
    });
    cIn.addEventListener('input', () => {
      const steps = 4 * S.done + phase;
      replay(steps);
      note = steps ? `c = ${c()}: the search was replayed from the start with this c to the same point.` : `c = ${c()}.`;
      draw();
    });
    on(root, 'reset', () => { replay(0); cIn.value = cIn.dataset.default; cIn.dispatchEvent(new Event('input', {bubbles: true})); note = ''; draw(); });
    note = ''; replay(0); draw();
  });

  // Simple (flat) Monte-Carlo search against MCTS on the same tree and rollout policy.
  Demos.register('mcts-flat-lab', (root, H) => {
    const BUDGET = 200;
    let seed = 808;
    put(root, '[data-controls]', H.range('mcts-flat-c', 'MCTS exploration c', 1, 0, 3, .1) + H.button('New seed', 'seed') + H.button('Reset', 'reset'));
    const draw = () => {
      const c = val(root, 'mcts-flat-c'), S = newSearch(seed), rng = H.rng(seed);
      const flat = {A: {n: 0, w: 0}, B: {n: 0, w: 0}}, mA = [], mB = [], fA = [], fB = [];
      for (let k = 1; k <= BUDGET; k++) {
        simulate(S, c);
        if (S.stats.A.n) mA.push([k, S.stats.A.w / S.stats.A.n]);
        if (S.stats.B.n) mB.push([k, S.stats.B.w / S.stats.B.n]);
        const a = k % 2 ? 'A' : 'B', leaf = TREE[a].children[Math.floor(rng() * 2)];
        flat[a].n++; flat[a].w += TREE[leaf].reward;
        (a === 'A' ? fA : fB).push([k, flat[a].w / flat[a].n]);
      }
      put(root, '[data-plot]', plotWithLegend({xDomain: [0, BUDGET], yDomain: [0, 1], yTicks: 5, xLabel: 'Simulations', yLabel: 'Root value Q(s,a)', height: 280,
        title: 'Root action values against simulations'}, [
        {label: 'MCTS: A', points: mA, color: H.colors[0]}, {label: 'MCTS: B', points: mB, color: H.colors[1]},
        {label: 'Simple: A', points: fA, color: H.colors[0], width: 2, dashed: true}, {label: 'Simple: B', points: fB, color: H.colors[1], width: 2, dashed: true}]));
      const A = S.stats.A, B = S.stats.B, qa = flat.A.w / flat.A.n, qb = flat.B.w / flat.B.n;
      put(root, '[data-readout]', `Simple (100 rollouts each): Q(A) = ${H.fmt(qa)}, Q(B) = ${H.fmt(qb)} → <strong>${pick(qa, qb)}</strong> by mean.<br>` +
        `MCTS (c = ${c}): Q(A) = ${H.fmt(A.w / A.n)} (N = ${A.n}), Q(B) = ${H.fmt(B.w / B.n)} (N = ${B.n}) → <strong>${A.n >= B.n ? 'A' : 'B'}</strong> by visits.<br>Optimal: A, via A2 (0.8). Seed ${seed}.`);
    };
    on(root, 'seed', () => { seed += 1; draw(); });
    on(root, 'reset', () => { seed = 808; });
    wire(root, {'mcts-flat-c': 1}, draw);
  });
})();
