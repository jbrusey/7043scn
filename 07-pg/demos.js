/* Lecture 7 (Policy Gradient): interactive demonstrations.
   Each Demos.register(id, ...) mounts into the <section id="id"> in pg.html.
   All numbers are computed locally; random runs use fixed, disclosed seeds.
   The two-action bandit used throughout: r(1) = 3, r(0) = 1, p = P(action 1). */
(function () {
  'use strict';
  const sig = x => 1 / (1 + Math.exp(-x));
  const softmax = (xs, tau) => { const m = Math.max(...xs), es = xs.map(x => Math.exp((x - m) / tau)), z = es.reduce((a, b) => a + b, 0); return es.map(x => x / z); };
  // Readout number: fixed decimals, a true minus sign, and never "−0.00".
  const num = (x, n = 2) => { const s = Number(x).toFixed(n); return (+s === 0 ? (0).toFixed(n) : s).replace('-', '−'); };
  const metric = (label, value) => `<div class="metric">${label}<strong>${value}</strong></div>`;
  const val = (root, id) => +root.querySelector('#' + id).value;

  // Redraw on any control change. Reset restores every control to its
  // data-default, lets the demo restore its own state (onReset), then redraws once.
  function wire(root, draw, onReset) {
    root.querySelectorAll('input,select').forEach(el => el.addEventListener('input', draw));
    root.querySelector('[data-action="reset"]').addEventListener('click', () => {
      root.querySelectorAll('[data-default]').forEach(el => {
        el.value = el.dataset.default;
        const o = el.closest('label') && el.closest('label').querySelector('output');
        if (o) o.textContent = el.value + (o.dataset.unit || '');
      });
      if (onReset) onReset();
      draw();
    });
    draw();
  }

  Demos.register('pg-softmax-lab', (root, H) => {
    root.querySelector('[data-controls]').innerHTML =
      H.range('pg-h0', 'Preference h(s, Left)', 0, -3, 3, .1) + H.range('pg-h1', 'Preference h(s, Stay)', 1, -3, 3, .1) +
      H.range('pg-h2', 'Preference h(s, Right)', -1, -3, 3, .1) + H.range('pg-temp', 'Temperature τ', 1, .2, 3, .1) + H.button('Reset', 'reset');
    const draw = () => {
      const hs = [0, 1, 2].map(i => val(root, 'pg-h' + i)), p = softmax(hs, val(root, 'pg-temp'));
      const ent = -p.reduce((a, x) => a + (x > 0 ? x * Math.log(x) : 0), 0);
      root.querySelector('[data-plot]').innerHTML = H.bars({labels: ['Left', 'Stay', 'Right'], values: p, min: 0, max: 1, title: 'Action probabilities at one state', yLabel: 'π(a | s)'});
      root.querySelector('[data-readout]').innerHTML = `π(· | s) = (${p.map(x => num(x, 3)).join(', ')}) for Left, Stay, Right.<br>Entropy ${num(ent, 3)} nats (uniform: ln 3 ≈ 1.099).`;
    };
    wire(root, draw);
  });

  Demos.register('pg-update-lab', (root, H) => {
    root.querySelector('[data-controls]').innerHTML =
      H.range('pg-theta', 'Initial θ', 0, -3, 3, .1) +
      H.select('pg-action', 'Observed action a', [{value: '1', label: 'Action 1 (reward 3)'}, {value: '0', label: 'Action 0 (reward 1)'}], '1') +
      H.range('pg-baseline', 'Baseline b', 0, 0, 4, .1) + H.range('pg-alpha', 'Step size α', .5, 0, 2, .1) + H.button('Reset', 'reset');
    const draw = () => {
      const theta = val(root, 'pg-theta'), a = val(root, 'pg-action'), b = val(root, 'pg-baseline'), alpha = val(root, 'pg-alpha');
      const p = sig(theta), r = a ? 3 : 1, score = a - p, g = (r - b) * score, next = theta + alpha * g, p2 = sig(next);
      root.querySelector('[data-plot]').innerHTML = H.bars({labels: ['Before: p', 'After: p′'], values: [p, p2], min: 0, max: 1, colors: [H.colors[3], H.colors[0]], title: 'P(action 1) before and after one update', yLabel: 'P(action 1)', height: 230});
      root.querySelector('[data-readout]').innerHTML =
        `a − p = ${num(score, 3)}, r − b = ${num(r - b, 2)}, ĝ = <strong>${num(g, 3)}</strong>, θ′ = θ + αĝ = ${num(next, 3)}<br>` +
        `P(action 1): ${num(p, 3)} → <strong>${num(p2, 3)}</strong>; J = 1 + 2p: ${num(1 + 2 * p, 3)} → <strong>${num(1 + 2 * p2, 3)}</strong>`;
    };
    wire(root, draw);
  });

  // REINFORCE on the two-action bandit: eight seeded runs, optional baseline.
  // Rewards r(1) = 3 + c and r(0) = 1 + c, plus N(0, 1) noise; c is an offset.
  Demos.register('pg-curves-lab', (root, H) => {
    const RUNS = 8, EPISODES = 200, NOISE = 1, RATE = .1;
    let seedSet = 1;
    root.querySelector('[data-controls]').innerHTML =
      H.select('pg-c-base', 'Baseline b', [{value: 'none', label: 'None (b = 0)'}, {value: 'avg', label: 'Running average of r'}], 'none') +
      H.range('pg-c-offset', 'Reward offset c', 10, 0, 10, 1) + H.range('pg-c-alpha', 'Step size α', .05, .01, .2, .01) +
      H.button('New seeds', 'seeds') + H.button('Reset', 'reset');
    function run(seed, alpha, offset, useBaseline) {
      const rng = H.rng(seed);
      let theta = 0, b = 0;
      const ps = [[0, .5]];
      for (let e = 1; e <= EPISODES; e++) {
        const p = sig(theta), a = rng() < p ? 1 : 0, r = (a ? 3 : 1) + offset + NOISE * H.normal(rng);
        theta += alpha * (r - (useBaseline ? b : 0)) * (a - p);
        if (useBaseline) b += RATE * (r - b);   // updated after use, so b never depends on the current action
        ps.push([e, sig(theta)]);
      }
      return ps;
    }
    const draw = () => {
      const alpha = val(root, 'pg-c-alpha'), offset = val(root, 'pg-c-offset'), useB = root.querySelector('#pg-c-base').value === 'avg';
      const first = 1000 * seedSet + 1;
      const runs = Array.from({length: RUNS}, (_, i) => run(first + i, alpha, offset, useB));
      const finals = runs.map(r => r[EPISODES][1]), mean = finals.reduce((a, b) => a + b, 0) / RUNS;
      const sd = Math.sqrt(finals.reduce((a, x) => a + (x - mean) ** 2, 0) / (RUNS - 1)), wrong = finals.filter(x => x < .5).length;
      const meanCurve = runs[0].map((_, k) => [k, runs.reduce((a, r) => a + r[k][1], 0) / RUNS]);
      root.querySelector('[data-plot]').innerHTML = H.plot({
        series: [...runs.map((r, i) => ({label: i ? '' : `${RUNS} seeded runs`, points: r, color: H.colors[0], width: 1.5, opacity: .6})), {label: `Mean of ${RUNS} runs`, points: meanCurve, color: '#000', width: 3}],
        xDomain: [0, EPISODES], yDomain: [0, 1], xLabel: 'Episode', yLabel: 'P(action 1)', title: 'P(action 1) against episode for eight seeded runs', height: 290});
      root.querySelector('[data-readout]').innerHTML =
        `After ${EPISODES} episodes, P(action 1): mean <strong>${num(mean, 2)}</strong>, sd across runs <strong>${num(sd, 2)}</strong>; runs preferring the worse action (p &lt; 0.5): <strong>${wrong}</strong> of ${RUNS}.<br>Seeds ${first}–${first + RUNS - 1}.`;
    };
    root.querySelector('[data-action="seeds"]').addEventListener('click', () => { seedSet += 1; draw(); });
    wire(root, draw, () => { seedSet = 1; });
  });

  // One actor-critic transition at t = 0 (so γ^t = 1); action 1 observed.
  Demos.register('pg-td-lab', (root, H) => {
    const P = .4, ALPHA = .5, BETA = .2, THETA = Math.log(P / (1 - P));
    root.querySelector('[data-controls]').innerHTML =
      H.range('pg-td-r', 'Reward R<sub>1</sub>', 1, -3, 3, .1) + H.range('pg-td-v', 'Critic v̂(s)', 2, -2, 6, .1) +
      H.range('pg-td-next', 'Critic v̂(s′)', 3, -2, 6, .1) + H.range('pg-td-gamma', 'Discount γ', .9, 0, 1, .05) +
      H.select('pg-td-done', 'Next state s′', [{value: '0', label: 'Non-terminal'}, {value: '1', label: 'Terminal'}], '0') + H.button('Reset', 'reset');
    const draw = () => {
      const r = val(root, 'pg-td-r'), v = val(root, 'pg-td-v'), n = val(root, 'pg-td-next'), gamma = val(root, 'pg-td-gamma');
      const done = root.querySelector('#pg-td-done').value === '1', target = r + (done ? 0 : gamma * n), delta = target - v;
      const step = ALPHA * delta * (1 - P), nextP = sig(THETA + step);
      ['pg-td-next', 'pg-td-gamma'].forEach(id => { root.querySelector('#' + id).disabled = done; });
      root.querySelector('[data-readout]').innerHTML =
        `<div class="metric-row">${metric('TD target', num(target, 2))}${metric('TD error δ', num(delta, 2))}${metric('Actor step αδ(1 − p)', num(step, 3))}</div>` +
        (done ? 'Terminal s′: target = R<sub>1</sub>; v̂(s′) and γ are unused.' : `Target = R<sub>1</sub> + γ v̂(s′) = ${num(r, 2)} ${gamma * n < 0 ? '−' : '+'} ${num(Math.abs(gamma * n), 2)}.`) +
        `<br>Critic: v̂(s) ${num(v, 2)} → ${num(v + BETA * delta, 2)}. Actor: P(action 1) ${num(P, 3)} → <strong>${num(nextP, 3)}</strong>.`;
      root.querySelector('[data-plot]').innerHTML = H.bars({labels: ['Before', 'After'], values: [P, nextP], min: 0, max: 1, colors: [H.colors[3], H.colors[0]], title: 'P(action 1) before and after the actor step', yLabel: 'P(action 1)', height: 220});
    };
    wire(root, draw);
  });

  // Exact expected gradients for the bandit: current policy p, data from p_old.
  Demos.register('pg-offpolicy-lab', (root, H) => {
    root.querySelector('[data-controls]').innerHTML = H.range('pg-current-p', 'Current policy p', .8, .1, .9, .05) + H.range('pg-old-p', 'Data policy p<sub>old</sub>', .2, .1, .9, .05) + H.button('Reset', 'reset');
    const draw = () => {
      const p = val(root, 'pg-current-p'), q = val(root, 'pg-old-p'), w1 = p / q, w0 = (1 - p) / (1 - q);
      const exact = p * 3 * (1 - p) + (1 - p) * 1 * (0 - p);              // a ~ p
      const raw = q * 3 * (1 - p) + (1 - q) * 1 * (0 - p);                // a ~ p_old, scored at p
      const weighted = q * w1 * 3 * (1 - p) + (1 - q) * w0 * 1 * (0 - p); // a ~ p_old, importance-weighted
      root.querySelector('[data-plot]').innerHTML = H.bars({labels: ['True', 'Old data, raw', 'Old data, weighted'], values: [exact, raw, weighted], min: -1, max: 3, title: 'Expected gradient: true, raw old data, weighted old data', yLabel: 'Expected gradient dJ/dθ'});
      root.querySelector('[data-readout]').innerHTML = `True ${num(exact, 3)}; raw <strong>${num(raw, 3)}</strong>; weighted ${num(weighted, 3)}.<br>Weights: action 1 × ${num(w1, 2)}, action 0 × ${num(w0, 2)}.`;
    };
    wire(root, draw);
  });

  Demos.register('pg-step-lab', (root, H) => {
    root.querySelector('[data-controls]').innerHTML = H.range('pg-step-alpha', 'Step size α', .25, 0, 1.5, .05) + H.button('Reset', 'reset');
    const J = x => -((x - 2) ** 2 + 1);
    const pts = Array.from({length: 161}, (_, i) => { const x = -2 + i * .05; return [x, J(x)]; });
    const draw = () => {
      const a = val(root, 'pg-step-alpha'), m = 4 * a;
      root.querySelector('[data-plot]').innerHTML = H.plot({series: [{label: 'Exact J(μ)', points: pts}, {label: 'Start μ = 0', points: [[0, J(0)]], color: H.colors[2], dots: true}, {label: 'After one step μ′', points: [[m, J(m)]], color: H.colors[1], dots: true}], xDomain: [-2, 6], yDomain: [-20, 0], xLabel: 'Policy mean μ', yLabel: 'Expected reward J'});
      const verdict = a === 0 ? 'No update.' : a === .5 ? 'Lands on the optimum μ = 2.' : a < 1 ? 'Expected reward improves.' : a === 1 ? 'Same J, on the other side of the optimum.' : 'The exact-gradient step makes J worse.';
      root.querySelector('[data-readout]').innerHTML = `Gradient at μ = 0 is +4, so μ′ = 4α = <strong>${num(m, 2)}</strong>.<br>J: −5.00 → <strong>${num(J(m), 2)}</strong>. ${verdict}`;
    };
    wire(root, draw);
  });
})();
