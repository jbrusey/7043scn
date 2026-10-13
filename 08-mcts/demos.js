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

  /* ======================== Simulation-Based Search ======================== */

  // Static figure: the planning tree from "now", with the best line of play (A then A2) highlighted.
  Demos.register('mcts-forward', (root, H) => {
    const POS = {R: [190, 24], A: [95, 112], B: [285, 112], A1: [46, 200], A2: [144, 200], B1: [238, 200], B2: [336, 200]};
    const BW = 88, BH = 40, best = ['R', 'A', 'A2'];
    const onBest = (a, b) => best.includes(a) && best.includes(b);
    let s = '<svg class="chart" viewBox="0 0 382 222" role="img" aria-label="Search tree from the current state: moves A and B, then A1 0.2, A2 0.8, B1 0.55, B2 0.65">' +
      '<title>Search tree from the current state; the best line of play is A then A2, worth 0.8</title>';
    Object.entries(TREE).forEach(([p, d]) => (d.children || []).forEach(k => {
      s += `<path d="M${POS[p][0]},${POS[p][1] + BH / 2}L${POS[k][0]},${POS[k][1] - BH / 2}" stroke="${onBest(p, k) ? H.colors[0] : '#b8b3a8'}" stroke-width="${onBest(p, k) ? 4 : 2}"/>`;
    }));
    Object.entries(POS).forEach(([k, [x, y]]) => {
      const label = k === 'R' ? 'now' : TREE[k].children ? k : `${k}: ${TREE[k].reward}`;
      s += `<rect x="${x - BW / 2}" y="${y - BH / 2}" width="${BW}" height="${BH}" rx="6" fill="${best.includes(k) ? H.colors[0] : '#fafaf6'}"${best.includes(k) ? ' fill-opacity="0.14"' : ''} stroke="#51483d" stroke-width="1.5"/>` +
           `<text x="${x}" y="${y + 6}" text-anchor="middle"${TREE[k].children ? ' font-weight="bold"' : ''}>${label}</text>`;
    });
    put(root, '[data-figure]', s + '</svg>');
  });

  /* ======================== UCB1 ======================== */

  /* Three Bernoulli arms with means hidden from the algorithm. Each arm has its own seeded reward
     stream, so the k-th pull of an arm pays the same whatever c is: c changes only which arm is pulled. */
  const BANDIT = {means: [.3, .5, .7], seed: 846};
  const newBandit = seed => ({streams: BANDIT.means.map((m, a) => { const r = H.rng(seed * 10 + a); return () => (r() < m ? 1 : 0); }),
                              n: [0, 0, 0], w: [0, 0, 0], t: 0, last: null});
  // Index of the largest value; ties go to the lowest-numbered arm.
  const argmax = v => v.reduce((b, x, i) => (x > v[b] + 1e-12 ? i : b), 0);
  // UCB1 score Q(a) + c √(ln t / N(a)); an untried arm scores Infinity, so every arm is tried once first.
  const ucbScores = (B, c) => B.n.map((n, a) => (n === 0 ? Infinity : B.w[a] / n + c * Math.sqrt(Math.log(B.t) / n)));
  function pull(B, c) {
    const sc = ucbScores(B, c), a = argmax(sc), r = B.streams[a]();
    B.last = {t: B.t + 1, arm: a, r, q: B.n.map((n, i) => (n ? B.w[i] / n : null))};
    B.n[a] += 1; B.w[a] += r; B.t += 1;
  }
  // Stacked bars: solid = mean Q(a), light = bonus, total height = score. The arm pulled next is outlined.
  function ucbChart(B, sc, next, truth) {
    const w = 560, h = 300, L = 62, top = 44, base = h - 56, bw = (w - L - 10) / 3;
    const hiF = Math.max(1, ...sc.filter(Number.isFinite)), step = hiF <= 2.5 ? .5 : 1, ymax = step * Math.ceil(hiF / step - 1e-9);
    const Y = v => base - Math.min(v, ymax) / ymax * (base - top), ink = '#51483d';
    let s = `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="UCB1 score of each arm: mean plus exploration bonus">` +
      `<title>UCB1 score of each arm: mean Q(a) (solid) plus bonus (light)</title>` +
      `<rect x="${L}" y="8" width="16" height="16" fill="${ink}"/><text x="${L + 22}" y="22">mean Q(a)</text>` +
      `<rect x="${L + 134}" y="8" width="16" height="16" fill="${ink}" fill-opacity="0.22" stroke="${ink}"/><text x="${L + 156}" y="22">+ bonus = score</text>` +
      (truth ? `<path d="M${L + 318},16H${L + 348}" stroke="#222" stroke-width="2.5" stroke-dasharray="6 4"/><text x="${L + 354}" y="22">true mean</text>` : '');
    for (let v = 0; v <= ymax + 1e-9; v += step) s += `<path d="M${L} ${Y(v)}H${w - 6}" stroke="#d6d6cc"/><text x="${L - 8}" y="${Y(v) + 6}" text-anchor="end">${+v.toFixed(1)}</text>`;
    B.n.forEach((n, a) => {
      const x = L + a * bw + 24, bwid = bw - 48, cx = x + bwid / 2, col = H.colors[a], isNext = a === next, bold = isNext ? ' font-weight="bold"' : '';
      if (n === 0) {
        s += `<rect x="${x}" y="${top}" width="${bwid}" height="${base - top}" fill="${col}" fill-opacity="0.06" stroke="${isNext ? '#222' : col}" stroke-width="${isNext ? 2.5 : 1.5}" stroke-dasharray="6 4"/>` +
             `<text x="${cx}" y="${(top + base) / 2 - 4}" text-anchor="middle"${bold}>untried</text><text x="${cx}" y="${(top + base) / 2 + 18}" text-anchor="middle"${bold}>score ∞</text>`;
      } else {
        const q = B.w[a] / n;
        s += `<rect x="${x}" y="${Y(q)}" width="${bwid}" height="${Math.max(1, base - Y(q))}" fill="${col}"/>` +
             `<rect x="${x}" y="${Y(sc[a])}" width="${bwid}" height="${Math.max(0, Y(q) - Y(sc[a]))}" fill="${col}" fill-opacity="0.22" stroke="${col}"/>`;
        if (isNext) s += `<rect x="${x - 5}" y="${Y(sc[a]) - 5}" width="${bwid + 10}" height="${base - Y(sc[a]) + 5}" fill="none" stroke="#222" stroke-width="2.5"/>`;
        s += `<text x="${cx}" y="${Y(sc[a]) - 11}" text-anchor="middle"${bold}>${H.fmt(sc[a], 3)}</text>`;
      }
      if (truth) s += `<path d="M${x - 12},${Y(BANDIT.means[a])}H${x + bwid + 12}" stroke="#222" stroke-width="2.5" stroke-dasharray="6 4"/>`;
      s += `<text x="${cx}" y="${base + 24}" text-anchor="middle"${bold}>Arm ${a + 1}${isNext ? ' (next)' : ''}</text>` +
           `<text x="${cx}" y="${base + 46}" text-anchor="middle">N(${a + 1}) = ${n}</text>`;
    });
    return s + `<path d="M${L},${top - 8}V${base}H${w - 6}" fill="none" stroke="#444"/></svg>`;
  }

  // UCB1 step by step on the three arms. Changing c replays the run from t = 0 to the same t.
  Demos.register('mcts-ucb-lab', (root, H) => {
    const MAX = 200;
    let B, truth = false;
    put(root, '[data-controls]', H.range('mcts-ucb-c', 'Exploration c', 1, 0, 3, .1) + H.button('Pull 1', 'one') + H.button('Pull 10', 'ten') +
      H.button('Show true means', 'truth') + H.button('Reset', 'reset'));
    const cIn = root.querySelector('#mcts-ucb-c'), c = () => +cIn.value, f3 = x => H.fmt(x, 3);
    const replay = t => { B = newBandit(BANDIT.seed); while (B.t < t) pull(B, c()); };
    const draw = () => {
      const sc = ucbScores(B, c()), next = argmax(sc), L = B.last, done = B.t >= MAX;
      const ties = sc.filter(v => Math.abs(v - sc[next]) < 1e-12 || v === sc[next]).length > 1;
      put(root, '[data-plot]', ucbChart(B, sc, next, truth));
      put(root, '[data-table] tbody', B.n.map((n, a) => {
        const cell = x => (a === next ? `<strong>${x}</strong>` : x), q = n ? B.w[a] / n : null;
        return `<tr><td>${cell('Arm ' + (a + 1))}</td><td>${n}</td><td>${q === null ? '–' : f3(q)}</td><td>${q === null ? '∞' : f3(sc[a] - q)}</td><td>${cell(q === null ? '∞' : f3(sc[a]))}</td></tr>`;
      }).join(''));
      let last = 'No pulls yet. Every arm is untried (score ∞), so UCB1 tries each arm once, from Arm 1.';
      if (L) {
        const best = Math.max(...L.q.map(q => (q === null ? -Infinity : q))), q = L.q[L.arm];
        const why = q === null ? 'it was untried' : q >= best - 1e-9 ? 'highest score and highest mean: <em>exploit</em>'
                  : `highest score, although its mean ${f3(q)} was not the highest: <em>explore</em>`;
        last = `Pull ${L.t}: Arm ${L.arm + 1} (${why}). It paid <strong>${L.r}</strong>.`;
      }
      const nextTxt = done ? `Budget of ${MAX} pulls reached: Reset, or move c to replay the run.`
        : `t = ${B.t}${B.t ? `, ln t = ${f3(Math.log(B.t))}` : ''}. Next pull: <strong>Arm ${next + 1}</strong>` +
          (sc[next] === Infinity ? ' (untried)' : ties ? ' (a tie: the lower-numbered arm)' : '') + '.';
      put(root, '[data-readout]', last + '<br>' + nextTxt + (truth ? '<br>True means 0.3, 0.5 and 0.7, hidden from UCB1.' : ''));
      root.querySelectorAll('[data-action="one"],[data-action="ten"]').forEach(b => { b.disabled = done; });
      root.querySelector('[data-action="truth"]').textContent = truth ? 'Hide true means' : 'Show true means';
    };
    on(root, 'one', () => { if (B.t < MAX) pull(B, c()); draw(); });
    on(root, 'ten', () => { for (let i = 0; i < 10 && B.t < MAX; i++) pull(B, c()); draw(); });
    on(root, 'truth', () => { truth = !truth; draw(); });
    cIn.addEventListener('input', () => { replay(B.t); draw(); });
    on(root, 'reset', () => { truth = false; cIn.value = cIn.dataset.default; cIn.dispatchEvent(new Event('input', {bubbles: true})); replay(0); draw(); });
    replay(0); draw();
  });

  /* ======================== Monte-Carlo Tree Search ======================== */

  // Static figure: the four phases of one simulation on a small generic tree.
  Demos.register('mcts-phases', (root, H) => {
    const PW = 225, VH = 346, P = {R: [112, 64], L: [62, 128], Rt: [162, 128], LL: [37, 192], LR: [87, 192]};
    const TITLES = ['1 Select', '2 Expand', '3 Evaluate', '4 Back up'];
    const CAPS = [['UCT picks a child', 'at each stored node'], ['store one new', 'child node'], ['roll out at random', 'to the end: return G'], ['N + 1 and W + G', 'on every path node']];
    const blue = H.colors[0], red = H.colors[1], green = H.colors[2], ink = '#51483d';
    let s = `<svg class="chart" viewBox="0 0 ${4 * PW} ${VH}" role="img" aria-label="The four phases of one MCTS simulation: select, expand, evaluate by rollout, back up">` +
      `<title>Four phases of one MCTS simulation</title>` +
      `<defs><marker id="mcts-phase-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0L10,5L0,10z" fill="${blue}"/></marker></defs>`;
    TITLES.forEach((title, i) => {
      const ox = i * PW, X = k => ox + P[k][0], Y = k => P[k][1];
      const edge = (a, b, attrs) => `<path d="M${X(a)},${Y(a) + 16}L${X(b)},${Y(b) - 16}" ${attrs}/>`;
      const node = (k, attrs) => `<circle cx="${X(k)}" cy="${Y(k)}" r="16" ${attrs}/>`;
      s += `<rect x="${ox + 5}" y="4" width="${PW - 10}" height="${VH - 8}" rx="8" fill="#fafaf6" stroke="#d6d6cc"/>` +
           `<text x="${ox + PW / 2}" y="32" text-anchor="middle" font-weight="bold">${title}</text>`;
      const sel = i === 0, stored = `fill="#fafaf6" stroke="${ink}" stroke-width="1.5"`;
      s += edge('R', 'L', sel ? `stroke="${blue}" stroke-width="4" marker-end="url(#mcts-phase-arrow)"` : `stroke="#b8b3a8" stroke-width="2"`) +
           edge('R', 'Rt', 'stroke="#b8b3a8" stroke-width="2"') + edge('L', 'LL', 'stroke="#b8b3a8" stroke-width="2"') +
           (i === 0 ? edge('L', 'LR', 'stroke="#b8b3a8" stroke-width="1.5" stroke-dasharray="5 4"') : edge('L', 'LR', 'stroke="#b8b3a8" stroke-width="2"'));
      ['R', 'L', 'Rt', 'LL'].forEach(k => { s += node(k, sel && (k === 'R' || k === 'L') ? `fill="${blue}" fill-opacity="0.18" stroke="${blue}" stroke-width="2.5"` : stored); });
      if (i === 0) s += node('LR', 'fill="#f0f1eb" stroke="#9b9488" stroke-width="1.5" stroke-dasharray="4 3"') + `<text x="${X('LR') + 22}" y="${Y('LR') + 6}">untried</text>`;
      if (i === 1) s += node('LR', `fill="${green}" fill-opacity="0.18" stroke="${green}" stroke-width="3"`) + `<text x="${X('LR') + 22}" y="${Y('LR') + 6}" font-weight="bold">new</text>`;
      if (i >= 2) s += node('LR', stored);
      if (i === 2) {
        const x = X('LR'), y = Y('LR') + 16;
        s += `<path d="M${x},${y}L${x + 12},${y + 14}L${x - 6},${y + 28}L${x + 10},${y + 42}" fill="none" stroke="${red}" stroke-width="3" stroke-dasharray="6 4"/>` +
             `<rect x="${x + 3}" y="${y + 44}" width="14" height="14" fill="${red}"/><text x="${x + 24}" y="${y + 57}" font-weight="bold">G</text>`;
      }
      if (i === 3) {
        const up = (a, b) => `<path d="M${X(b)},${Y(b) - 18}L${X(a)},${Y(a) + 20}" stroke="${blue}" stroke-width="3.5" marker-end="url(#mcts-phase-arrow)"/>`;
        s += up('L', 'LR') + up('R', 'L') +
             `<text x="${X('R') + 22}" y="${Y('R') - 8}">+1, +G</text><text x="${X('L') + 20}" y="${Y('L') + 2}">+1, +G</text><text x="${X('LR') + 22}" y="${Y('LR') + 6}">+1, +G</text>`;
      }
      s += CAPS[i].map((line, j) => `<text x="${ox + PW / 2}" y="${304 + 22 * j}" text-anchor="middle">${line}</text>`).join('');
    });
    put(root, '[data-figure]', s + '</svg>');
  });

  // Phase-by-phase MCTS on TREE, drawn as an SVG tree. The state is a function of (c, phases taken):
  // changing c replays the search from the start with the new c to the same point.
  Demos.register('mcts-tree-lab', (root, H) => {
    const SEED = 808, BUDGET = 200, NAMES = ['Select', 'Expand', 'Evaluate', 'Back up'];
    const DY = 38, POS = {R: [294, 40 + DY], A: [146, 150 + DY], B: [442, 150 + DY], A1: [72, 262 + DY], A2: [220, 262 + DY], B1: [368, 262 + DY], B2: [516, 262 + DY]};
    const BW = 138, BH = 72, VW = 590, VH = 370;
    const READY = 'Ready: no simulations yet. Step 1 selects from the root R.';
    let S, phase, msg, note, expanded;
    put(root, '[data-controls]', H.range('mcts-tree-c', 'Exploration c', 1, 0, 3, .1) + H.button('Step 1: Select', 'phase') +
      H.button('Run 1 simulation', 'one') + H.button('Run 10', 'ten') + H.button('Reset', 'reset'));
    const cIn = root.querySelector('#mcts-tree-c'), c = () => +cIn.value;
    const f3 = x => H.fmt(x, 3);
    const uctText = p => `UCT ${p.ch.map((k, i) => `${k} ${f3(p.sc[i])}`).join(' vs ')} → ${p.choice}${p.tie ? ' (tie: left)' : ''}`;
    function advance() {
      const cc = c();
      if (phase === 0) {
        const node = select(S, cc);
        msg = `<strong>Select</strong>: ` + (S.picks.length ? `${S.picks.map(p => `at ${p.node}, ${uctText(p)}`).join('; ')}. ` : '') +
              (TREE[node].children ? `${node} has an untried child, so selection stops at ${node}.` : `${node} is a leaf, so selection stops there.`);
      } else if (phase === 1) {
        expanded = expand(S);
        msg = expanded ? `<strong>Expand</strong>: store ${expanded} as a new node; its N stays 0 until the backup.`
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
    // One line for a whole simulation, built from the search state after its backup.
    const summary = () => `<strong>Simulation ${S.done}</strong>: ` + (S.picks.length ? S.picks.map(uctText).join('; ') : 'the root has an untried move') +
      `; expand ${expanded || 'nothing (leaf)'}; ` + (S.trail.length > 1 ? `rollout ${S.trail.join(' → ')}` : `${S.trail[0]} is a leaf`) +
      `, G = ${S.G}; back up along ${S.path.join(' → ')}.`;
    const replay = steps => { S = newSearch(SEED); phase = 0; msg = READY; expanded = null; for (let i = 0; i < steps; i++) advance(); };
    function tree() {
      const onPath = (a, b) => S.path.indexOf(b) > 0 && S.path[S.path.indexOf(b) - 1] === a;
      const onTrail = (a, b) => S.trail.indexOf(b) > 0 && S.trail[S.trail.indexOf(b) - 1] === a;
      const lastDone = S.done === 0 && phase === 0 ? -1 : (phase + 3) % 4;
      let s = `<svg class="chart" viewBox="0 0 ${VW} ${VH}" role="img" aria-label="MCTS tree with visits N, total return W and mean Q"><title>Search tree: leaf payoffs A1 0.2, A2 0.8, B1 0.55, B2 0.65</title>`;
      // phase strip: the phase just completed is filled
      NAMES.forEach((name, i) => {
        const x = 6 + i * 146, on = i === lastDone;
        s += `<rect x="${x}" y="4" width="140" height="30" rx="5" fill="${on ? H.colors[0] : '#fafaf6'}"${on ? ' fill-opacity="0.16"' : ''} stroke="${on ? H.colors[0] : '#b8b3a8'}" stroke-width="${on ? 2.5 : 1.2}"/>` +
             `<text x="${x + 70}" y="25" text-anchor="middle"${on ? ' font-weight="bold"' : ''}>${i + 1} ${name}</text>`;
      });
      Object.entries(TREE).forEach(([p, d]) => (d.children || []).forEach(k => {
        const path = onPath(p, k), roll = !path && onTrail(p, k);
        s += `<path d="M${POS[p][0]},${POS[p][1] + BH / 2}L${POS[k][0]},${POS[k][1] - BH / 2}" stroke="${path ? H.colors[0] : roll ? H.colors[1] : '#b8b3a8'}" stroke-width="${path ? 4 : roll ? 3 : 2}"${roll ? ' stroke-dasharray="6 4"' : ''}/>`;
      }));
      // UCT scores compared in the latest selection, written beside each edge
      S.picks.forEach(p => p.ch.forEach((k, i) => {
        const [px, py] = POS[p.node], [kx, ky] = POS[k], mx = (px + kx) / 2, my = (py + BH / 2 + ky - BH / 2) / 2, left = kx < px;
        s += `<text x="${mx + (left ? -12 : 12)}" y="${my + 2}" text-anchor="${left ? 'end' : 'start'}" font-size="15"${k === p.choice ? ' font-weight="bold"' : ''}>UCT ${f3(p.sc[i])}</text>`;
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
      s += `<path d="M6,${VH - 18}H34" stroke="${H.colors[0]}" stroke-width="4"/><text x="40" y="${VH - 12}">tree path</text>` +
           `<path d="M150,${VH - 18}H178" stroke="${H.colors[1]}" stroke-width="3" stroke-dasharray="6 4"/><text x="184" y="${VH - 12}">rollout</text>` +
           `<text x="${VW - 4}" y="${VH - 12}" text-anchor="end">simulations: ${S.done}</text>`;
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
      root.querySelectorAll('[data-action="phase"],[data-action="one"],[data-action="ten"]').forEach(b => { b.disabled = full; });
    };
    // Complete k simulations; a simulation already in progress is finished first and counts as one.
    const run = k => {
      const start = S.done, partial = phase !== 0, target = Math.min(BUDGET, S.done + k);
      while (S.done < target) advance();            // at most 40 phases
      const ran = S.done - start;
      if (k === 1) { note = partial ? 'Finished the simulation in progress.' : ''; msg = summary(); }
      else { note = partial ? `Finished the simulation in progress and ran ${ran - 1} more (${S.done} in total). Last one:` : `Ran ${ran} simulations (${S.done} in total). Last one:`; msg = summary(); }
      draw();
    };
    on(root, 'phase', () => { if (!(S.done >= BUDGET && phase === 0)) { note = ''; advance(); } draw(); });
    on(root, 'one', () => run(1));
    on(root, 'ten', () => run(10));
    cIn.addEventListener('input', () => {
      const steps = 4 * S.done + phase;
      replay(steps);
      if (steps && phase === 0) msg = summary();
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
