/* Lecture 10 (Case Study: RL in Car Climate Control): interactive demonstrations.
   Each Demos.register(id, ...) mounts into the <section id="id"> in climate.html.
   The cabin model below is a disclosed teaching toy (one temperature node, an
   electric heater, no cooling). It is NOT the validated cabin model of Brusey et
   al. (2018) and its results are not the paper's results. All numbers are
   computed locally; random runs use fixed, disclosed seeds. */
(function () {
  'use strict';
  const H = window.H;

  // ---- toy single-zone cabin --------------------------------------------------
  const DT = 60;                       // decision step: 1 minute, in s
  const C_KJ = 90, UA = 50;            // capacitance (kJ/K) and heat-loss conductance (W/K)
  const LEVELS = [0, 1, 2, 3, 4, 5];   // heater power, kW (electrical = thermal)
  const LO = 20, HI = 24;              // comfort band on cabin air temperature, °C
  const TRIP = 30;                     // trip length, minutes
  const KWH_PER_KM = 0.15;             // assumed driving consumption (150 Wh/km)
  const W_DEF = 5;                     // default energy weight w, kW
  const NA = LEVELS.length;

  // Exact solution of C dT/dt = UA (Tamb - T) + P + Qsol over one step with
  // constant inputs (zero-order hold): T' = Tinf + (T - Tinf) exp(-DT UA / C).
  function step(T, P, Tamb, o) {
    const ua = (o && o.UA) || UA, c = ((o && o.C) || C_KJ) * 1000, q = (o && o.Qsol) || 0;
    const Tinf = Tamb + (P + q) * 1000 / ua;
    return Tinf + (T - Tinf) * Math.exp(-DT * ua / c);
  }
  const comfy = T => T >= LO && T <= HI;
  // R_{t+1} = -c_{t+1} - P_t / w: one per uncomfortable minute, energy weighted by w (kW)
  const reward = (Tn, P, w) => -(comfy(Tn) ? 0 : 1) - P / w;

  // Run one trip; policy(T, t, prevP, scenario) returns heater power in kW.
  function rollout(policy, sc, w, n) {
    n = n || TRIP;
    let T = sc.T0, prev = 0, comfort = 0, kwh = 0, ret = 0;
    const temps = [[0, T]];
    for (let t = 0; t < n; t++) {
      const P = policy(T, t, prev, sc), Tn = step(T, P, sc.Tamb);
      comfort += comfy(Tn) ? 1 : 0; kwh += P * DT / 3600; ret += reward(Tn, P, w);
      T = Tn; prev = P; temps.push([t + 1, T]);
    }
    return {temps, comfort, kwh, ret, n};
  }

  // Baselines: all act through the same six heater levels.
  const bangbang = (T, t, prev) => T < 21 ? 5 : T > 23 ? 0 : prev;
  const proportional = kp => T => H.clamp(Math.round(kp * (22 - T)), 0, 5);

  // Planner (oracle): finite-horizon dynamic programming with the exact model and
  // known ambient, on a 0.1 °C grid from -20 to 50 °C with linear interpolation.
  const GMIN = -20, GSTEP = 0.1, GN = 701, GMAX = GMIN + (GN - 1) * GSTEP;
  const transCache = new Map(), planCache = new Map();
  function transitions(Tamb) {
    if (transCache.has(Tamb)) return transCache.get(Tamb);
    const idx = new Int32Array(GN * NA), frac = new Float64Array(GN * NA), cost = new Float64Array(GN * NA);
    for (let i = 0; i < GN; i++) for (let a = 0; a < NA; a++) {
      const Tn = step(GMIN + i * GSTEP, LEVELS[a], Tamb), x = (H.clamp(Tn, GMIN, GMAX) - GMIN) / GSTEP, j = Math.min(GN - 2, Math.floor(x));
      idx[i * NA + a] = j; frac[i * NA + a] = x - j; cost[i * NA + a] = comfy(Tn) ? 0 : 1;
    }
    const out = {idx, frac, cost};
    transCache.set(Tamb, out);
    return out;
  }
  const interp = (V, T) => {
    const x = (H.clamp(T, GMIN, GMAX) - GMIN) / GSTEP, j = Math.min(GN - 2, Math.floor(x)), f = x - j;
    return V[j] * (1 - f) + V[j + 1] * f;
  };
  function plan(Tamb, w, n) {
    n = n || TRIP;
    const key = Tamb + '|' + w + '|' + n;
    if (planCache.has(key)) return planCache.get(key);
    const tr = transitions(Tamb), Vs = new Array(n + 1);
    Vs[n] = new Float64Array(GN);
    for (let t = n - 1; t >= 0; t--) {
      const Vn = Vs[t + 1], V = new Float64Array(GN);
      for (let i = 0; i < GN; i++) {
        let best = -Infinity;
        for (let a = 0; a < NA; a++) {
          const k = i * NA + a, j = tr.idx[k], f = tr.frac[k];
          const v = -tr.cost[k] - LEVELS[a] / w + Vn[j] * (1 - f) + Vn[j + 1] * f;
          if (v > best) best = v;
        }
        V[i] = best;
      }
      Vs[t] = V;
    }
    // act greedily with a one-step lookahead from the exact (off-grid) temperature;
    // ties go to the lower power
    const policy = (T, t) => {
      const Vn = Vs[Math.min(t + 1, n)];
      let best = -Infinity, bp = 0;
      LEVELS.forEach(P => { const Tn = step(T, P, Tamb), v = reward(Tn, P, w) + interp(Vn, Tn); if (v > best + 1e-9) { best = v; bp = P; } });
      return bp;
    };
    if (planCache.size > 300) planCache.clear();
    const out = {policy};
    planCache.set(key, out);
    return out;
  }

  // Test set (seed 2018): 200 trips, integer ambient -10..10 °C, start = ambient + 0..10 °C.
  function scenarios(seed, count) {
    const r = H.rng(seed), out = [];
    for (let i = 0; i < count; i++) { const Tamb = Math.floor(r() * 21) - 10; out.push({Tamb, T0: Tamb + Math.floor(r() * 11)}); }
    return out;
  }
  const TESTS = scenarios(2018, 200);
  function evaluate(policyFor, w) {
    let c = 0, e = 0, ret = 0, steps = 0;
    TESTS.forEach(sc => { const x = rollout(policyFor(sc), sc, w); c += x.comfort; e += x.kwh; ret += x.ret; steps += x.n; });
    return {comfort: 100 * c / steps, kw: e * 3600 / (steps * DT), ret: ret / TESTS.length};
  }

  // Tabular Sarsa(lambda) with accumulating traces on a discretised observation:
  // cabin temperature in bins of `bin` °C over [-12, 36) (clipped at the ends),
  // optionally the ambient band (five 4 °C bands over [-10, 10]). Q starts at 0,
  // which is optimistic because every reward is negative.
  // Training trips (seeded): ambient ~ U[-10, 10], start = ambient + U[0, 10].
  // Traces are cleared at the start of every trip. The list of non-zero traces is
  // kept on the agent, so training in chunks gives exactly the same tables as
  // training in one call.
  const TMIN = -12, TMAX = 36;
  const SARSA = {alpha: 0.1, gamma: 0.95, eps: 0.1};
  function makeAgent(bin, useAmb, lambda, seed) {
    const nb = Math.round((TMAX - TMIN) / bin), size = nb * (useAmb ? 5 : 1) * NA;
    return {bin, useAmb, lambda, nb, Q: new Float64Array(size), E: new Float64Array(size), touched: [], rng: H.rng(seed), trips: 0};
  }
  const sIndex = (ag, T, Tamb) => (ag.useAmb ? H.clamp(Math.floor((Tamb + 10) / 4), 0, 4) * ag.nb : 0) + H.clamp(Math.floor((T - TMIN) / ag.bin), 0, ag.nb - 1);
  function greedy(ag, s) { let b = 0, bv = -Infinity; for (let a = 0; a < NA; a++) { const v = ag.Q[s * NA + a]; if (v > bv + 1e-12) { bv = v; b = a; } } return b; }
  function train(ag, trips, w) {
    const r = ag.rng, Q = ag.Q, E = ag.E, touched = ag.touched;
    const choose = s => r() < SARSA.eps ? Math.floor(r() * NA) : greedy(ag, s);
    for (let ep = 0; ep < trips; ep++) {
      const Tamb = -10 + 20 * r();
      let T = Tamb + 10 * r(), s = sIndex(ag, T, Tamb), a = choose(s);
      touched.forEach(k => { E[k] = 0; }); touched.length = 0;   // traces start at zero each trip
      for (let t = 0; t < TRIP; t++) {
        const P = LEVELS[a], Tn = step(T, P, Tamb), rew = reward(Tn, P, w);
        const s2 = sIndex(ag, Tn, Tamb), a2 = choose(s2), k0 = s * NA + a;
        // the trip end is a time limit, not a terminal state: always bootstrap
        const delta = rew + SARSA.gamma * Q[s2 * NA + a2] - Q[k0];
        E[k0] += 1;
        if (touched.indexOf(k0) < 0) touched.push(k0);
        touched.forEach(k => { Q[k] += SARSA.alpha * delta * E[k]; E[k] *= SARSA.gamma * ag.lambda; });
        T = Tn; s = s2; a = a2;
      }
      ag.trips++;
    }
  }
  const agentPolicy = ag => (T, t, prev, sc) => LEVELS[greedy(ag, sIndex(ag, T, sc.Tamb))];

  // Two-node air + interior-mass illustration (separate from the single-zone toy):
  // air 20 kJ/K, interior mass 70 kJ/K, air-mass 80 W/K, air-ambient 50 W/K,
  // ambient -5 °C, heater into the air; explicit Euler with 0.1 s steps, output every 10 s.
  function twoNode(Ta, Tm, P, minutes) {
    const Ca = 20000, Cm = 70000, h = 80, ua = 50, Tamb = -5, dt = 0.1, out = [[0, Ta, Tm]];
    for (let k = 1; k <= minutes * 600; k++) {
      const dTa = (P * 1000 + h * (Tm - Ta) + ua * (Tamb - Ta)) / Ca, dTm = h * (Ta - Tm) / Cm;
      Ta += dt * dTa; Tm += dt * dTm;
      if (k % 100 === 0) out.push([k / 600, Ta, Tm]);
    }
    return out;
  }

  window.CabinModel = {step, reward, rollout, bangbang, proportional, plan, scenarios, TESTS, evaluate, makeAgent, train, agentPolicy, twoNode, LEVELS, TRIP, KWH_PER_KM};

  // ---- shared UI helpers (as in the pilot deck) ---------------------------------
  // Redraw on any control change; Reset restores the listed defaults.
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
  const metric = (label, value) => `<div class="metric">${label}<strong>${value}</strong></div>`;
  const signed = (x, n) => (x < -0.5 * Math.pow(10, -n) ? '−' : '') + H.fmt(Math.abs(x), n);
  const temp = x => signed(x, 1) + ' °C';
  const band = [{y0: LO, y1: HI}];
  const GREY = '#8a8a80';

  // Reference results of the baselines on the test set (w = 5), computed once.
  let refCache = null;
  const refs = () => refCache || (refCache = {
    bang: evaluate(() => bangbang, W_DEF), prop: evaluate(() => proportional(1), W_DEF), plan: evaluate(sc => plan(sc.Tamb, W_DEF).policy, W_DEF)});

  // 1. Heat balance from a cold soak, constant heater power, plotted for an hour.
  Demos.register('cc-balance-lab', (root, H) => {
    const MIN = 60;
    root.querySelector('[data-controls]').innerHTML =
      H.range('cc-bal-p', 'Heater power P', 3, 0, 5, 1, ' kW') + H.range('cc-bal-amb', 'Ambient T<sub>amb</sub>', -5, -15, 15, 1, ' °C') +
      H.range('cc-bal-sol', 'Solar gain Q<sub>sol</sub>', 0, 0, 1, 0.1, ' kW') + H.range('cc-bal-c', 'Capacitance C', 90, 45, 180, 15, ' kJ/K') + H.button('Reset', 'reset');
    const draw = () => {
      const P = val(root, 'cc-bal-p'), Tamb = val(root, 'cc-bal-amb'), q = val(root, 'cc-bal-sol'), C = val(root, 'cc-bal-c');
      const pts = [[0, Tamb]];
      let T = Tamb;
      for (let t = 1; t <= MIN; t++) { T = step(T, P, Tamb, {C, Qsol: q}); pts.push([t, T]); }
      const Tinf = Tamb + (P + q) * 1000 / UA, tau = C * 1000 / UA / 60;
      // the cabin rises monotonically from ambient towards Tinf, so the axis only has to contain Tinf
      const top = Math.max(40, 20 * Math.ceil((Tinf + 2) / 20));
      root.querySelector('[data-plot]').innerHTML = H.plot({
        series: [{label: 'Cabin temperature', points: pts}, {label: 'T∞ (steady state)', points: [[0, Tinf], [MIN, Tinf]], color: H.colors[1], dashed: true, width: 2},
          {label: 'Ambient', points: [[0, Tamb], [MIN, Tamb]], color: GREY, dashed: true, width: 2}],
        bands: band, xDomain: [0, MIN], yDomain: [-20, top], xTicks: 6, yTicks: top > 40 ? (top + 20) / 20 : 6, xLabel: 'Minutes from cold soak', yLabel: 'Temperature (°C)', height: 280});
      let reach = '<strong>Never reaches 20 °C</strong> (T<sub>∞</sub> ≤ 20 °C)';
      if (Tinf > LO) {
        const t20 = tau * Math.log((Tinf - Tamb) / (Tinf - LO));
        reach = `Reaches 20 °C after <strong>${H.fmt(t20, 1)} min</strong>${t20 > MIN ? ' (beyond the plotted hour)' : ''}`;
      }
      const hold = Math.max(0, UA * (LO - Tamb) / 1000 - q);
      root.querySelector('[data-readout]').innerHTML = `T<sub>∞</sub> = <strong>${temp(Tinf)}</strong>, τ = C/UA = <strong>${H.fmt(tau, 0)} min</strong>. ${reach}.<br>30 min at ${P} kW use ${H.fmt(P / 2, 2)} kWh; holding 20 °C needs ${H.fmt(hold, 2)} kW.`;
    };
    wire(root, {'cc-bal-p': 3, 'cc-bal-amb': -5, 'cc-bal-sol': 0, 'cc-bal-c': 90}, draw);
  });

  // 2. Hidden variable: same air temperature, different interior-mass temperature.
  Demos.register('cc-hidden-lab', (root, H) => {
    root.querySelector('[data-controls]').innerHTML =
      H.range('cc-hid-mass', 'Cabin B interior mass', -5, -10, 25, 1, ' °C') + H.range('cc-hid-p', 'Heater power (both cabins)', 2, 0, 3, 0.5, ' kW') + H.button('Reset', 'reset');
    const draw = () => {
      const m = val(root, 'cc-hid-mass'), P = val(root, 'cc-hid-p'), A = twoNode(20, 20, P, 10), B = twoNode(20, m, P, 10);
      root.querySelector('[data-plot]').innerHTML = H.plot({
        series: [{label: 'A: air (interior 20 °C)', points: A.map(p => [p[0], p[1]])}, {label: `B: air (interior ${signed(m, 0)} °C)`, points: B.map(p => [p[0], p[1]])},
          {label: 'B: ET, still air', points: B.map(p => [p[0], (p[1] + p[2]) / 2]), color: H.colors[3], dashed: true, width: 2.5}],
        xDomain: [0, 10], yDomain: [-10, 40], xTicks: 5, yTicks: 5, xLabel: 'Minutes', yLabel: 'Temperature (°C)', height: 280});
      // outputs every 10 s: index 0 is now, 6 is 1 minute, 60 is 10 minutes
      const row = (name, f) => `<tr><td>${name}</td>${[0, 6, 60].map(i => `<td>${temp(f(i))}</td>`).join('')}</tr>`;
      root.querySelector('[data-readout]').innerHTML = `<table class="compact"><thead><tr><th></th><th>Now</th><th>After 1 min</th><th>After 10 min</th></tr></thead><tbody>` +
        row('A: air', i => A[i][1]) + row('B: air', i => B[i][1]) + row('B: ET = ½(air + interior)', i => (B[i][1] + B[i][2]) / 2) + '</tbody></table>';
    };
    wire(root, {'cc-hid-mass': -5, 'cc-hid-p': 2}, draw);
  });

  // 3. Energy weight w: the planner's optimal trip from a cold soak.
  Demos.register('cc-tradeoff-lab', (root, H) => {
    const W_MAX = 20;
    root.querySelector('[data-controls]').innerHTML =
      H.range('cc-tr-w', 'Energy weight w', W_DEF, 1, W_MAX, 0.5, ' kW') + H.range('cc-tr-amb', 'Ambient', -5, -15, 10, 1, ' °C') +
      H.range('cc-tr-n', 'Trip length', TRIP, 10, 60, 5, ' min') + H.button('Reset', 'reset');
    const draw = () => {
      const w = val(root, 'cc-tr-w'), Tamb = val(root, 'cc-tr-amb'), n = val(root, 'cc-tr-n'), sc = {Tamb, T0: Tamb};
      const x = rollout(plan(Tamb, w, n).policy, sc, w, n);
      root.querySelector('[data-plot]').innerHTML = H.plot({
        series: [{label: 'Cabin under the planner', points: x.temps}, {label: 'Ambient', points: [[0, Tamb], [n, Tamb]], color: GREY, dashed: true, width: 2}],
        bands: band, xDomain: [0, n], yDomain: [-20, 30], xTicks: n <= 30 || n % 10 ? n / 5 : n / 10, yTicks: 5, xLabel: 'Minute of trip', yLabel: 'Cabin temperature (°C)', height: 280});
      // Why the planner heats or not: the unheated cabin (at ambient, at most 10 °C) is never
      // comfortable, so heating wins exactly when its N comfortable minutes exceed its energy
      // cost sum(P_t)/w. When the planner does not heat at this w, the heated trip shown in the
      // text is the one it chooses at the largest w on the slider.
      const heat = x.kwh > 0 ? x : rollout(plan(Tamb, W_MAX, n).policy, sc, W_MAX, n);
      const kwMin = Math.round(heat.kwh * 60), N = heat.comfort, cost = kwMin / w;   // sum of P_t over 1-minute steps, kW·min
      const cmp = cost < N ? '&lt;' : cost > N ? '&gt;' : '=';
      const why = heat.kwh === 0
        ? `Even at w = ${W_MAX} kW the planner does not heat: on a ${n}-minute trip the warm-up leaves too few comfortable minutes.`
        : x.kwh > 0
          ? `Heating buys <strong>${N} comfortable minutes</strong> for an energy cost ΣP<sub>t</sub>/w = ${kwMin}/${w} = ${H.fmt(cost, 1)} ${cmp} ${N}, so it pays (for any <strong>w &gt; ${H.fmt(kwMin / N, 2)} kW</strong>).`
          : `Heating would buy <strong>${N} comfortable minutes</strong> for an energy cost ΣP<sub>t</sub>/w = ${kwMin}/${w} = ${H.fmt(cost, 1)} ${cmp} ${N}, so it does not pay (it would for <strong>w &gt; ${H.fmt(kwMin / N, 2)} kW</strong>).`;
      root.querySelector('[data-readout]').innerHTML = `<div class="metric-row">${metric('Comfortable', `${x.comfort} / ${n} min`)}${metric('Heater energy', H.fmt(x.kwh, 2) + ' kWh')}${metric('Range cost', H.fmt(x.kwh / KWH_PER_KM, 1) + ' km')}${metric('Return', signed(x.ret, 1))}</div>${why}`;
    };
    wire(root, {'cc-tr-w': W_DEF, 'cc-tr-amb': -5, 'cc-tr-n': TRIP}, draw);
  });

  // 4. Same trip, three controllers.
  Demos.register('cc-compare-lab', (root, H) => {
    root.querySelector('[data-controls]').innerHTML =
      H.range('cc-cmp-amb', 'Ambient', -5, -15, 10, 1, ' °C') + H.range('cc-cmp-start', 'Start above ambient', 0, 0, 15, 1, ' °C') +
      H.range('cc-cmp-kp', 'Proportional gain K<sub>p</sub>', 1, 0.5, 3, 0.25, ' kW/°C') + H.button('Reset', 'reset');
    const draw = () => {
      const Tamb = val(root, 'cc-cmp-amb'), sc = {Tamb, T0: Tamb + val(root, 'cc-cmp-start')}, kp = val(root, 'cc-cmp-kp');
      const runs = [['Bang-bang', bangbang, H.colors[3]], ['Proportional', proportional(kp), H.colors[1]], ['Planner', plan(Tamb, W_DEF).policy, H.colors[2]]]
        .map(([name, pol, color]) => ({name, color, x: rollout(pol, sc, W_DEF)}));
      root.querySelector('[data-plot]').innerHTML = H.plot({
        series: runs.map(r => ({label: r.name, points: r.x.temps, color: r.color, width: 2.5})),
        bands: band, xDomain: [0, TRIP], yDomain: [-20, 30], xTicks: 6, yTicks: 5, xLabel: 'Minute of trip', yLabel: 'Cabin temperature (°C)', height: 280});
      root.querySelector('[data-readout]').innerHTML = `<table class="compact"><thead><tr><th></th><th>Comfort</th><th>Energy</th><th>Return</th></tr></thead><tbody>` +
        runs.map(r => `<tr><td>${r.name}</td><td>${r.x.comfort} min</td><td>${H.fmt(r.x.kwh, 2)} kWh</td><td>${signed(r.x.ret, 1)}</td></tr>`).join('') + '</tbody></table>' +
        (runs[2].x.kwh === 0 ? 'At w = 5 kW, not heating gives the planner its best return.' : '');
    };
    wire(root, {'cc-cmp-amb': -5, 'cc-cmp-start': 0, 'cc-cmp-kp': 1}, draw);
  });

  // 5. Sarsa(lambda) learning curves: lambda = 0.9 and lambda = 0 side by side, three
  //    seeded agents each (seeds 1-3), cabin + ambient state, 1 °C bins. The demo opens
  //    after the first 500 training trips; Reset returns to that view.
  Demos.register('cc-train-lab', (root, H) => {
    const LAMS = [0.9, 0], SEEDS = [1, 2, 3], COLS = [H.colors[0], H.colors[4]], CAP = 6000, EVERY = 250, START = 500;
    let runs = null, hist = [];
    const trips = () => runs[0][0].trips;
    const record = () => hist.push({trips: trips(), e: runs.map(row => row.map(ag => evaluate(() => agentPolicy(ag), W_DEF)))});
    const trainBy = n => {
      const stop = Math.min(CAP, trips() + n);
      while (trips() < stop) { const k = Math.min(EVERY, stop - trips()); runs.forEach(row => row.forEach(ag => train(ag, k, W_DEF))); record(); }
    };
    const init = () => { runs = LAMS.map(l => SEEDS.map(s => makeAgent(1, true, l, s))); hist = []; record(); trainBy(START); };
    root.querySelector('[data-controls]').innerHTML = H.button('Train 500 trips', 'train500') + H.button('Train 2000 trips', 'train2000') + H.button('Reset', 'reset');
    const draw = () => {
      const R = refs(), last = hist[hist.length - 1], mean = (e, k) => e.reduce((a, x) => a + x[k], 0) / e.length;
      const seedLines = [], meanLines = [];
      LAMS.forEach((l, i) => {
        SEEDS.forEach((s, j) => seedLines.push({label: '', points: hist.map(h => [h.trips, h.e[i][j].comfort]), color: COLS[i], width: 1.2, opacity: 0.45}));
        meanLines.push({label: `λ = ${l} (mean of 3 seeds)`, points: hist.map(h => [h.trips, mean(h.e[i], 'comfort')]), color: COLS[i], width: 3.5});
      });
      root.querySelector('[data-plot]').innerHTML = H.plot({
        series: [...seedLines, ...meanLines,
          {label: 'Proportional', points: [[0, R.prop.comfort], [CAP, R.prop.comfort]], color: H.colors[1], dashed: true, width: 2},
          {label: 'Bang-bang', points: [[0, R.bang.comfort], [CAP, R.bang.comfort]], color: H.colors[3], dashed: true, width: 2}],
        xDomain: [0, CAP], yDomain: [0, 100], xTicks: 6, yTicks: 5, xLabel: 'Training trips per seed', yLabel: 'Test time in comfort (%)', height: 280});
      const res = i => `λ = ${LAMS[i]}: <strong>${H.fmt(mean(last.e[i], 'comfort'), 1)}%</strong> comfort, ${H.fmt(mean(last.e[i], 'kw'), 2)} kW`;
      root.querySelector('[data-readout]').innerHTML = `After <strong>${last.trips}</strong> training trips per seed: ${res(0)}; ${res(1)}.<br>` +
        (last.trips >= CAP ? 'Training budget of 6000 trips reached: press Reset to start again.' : 'Greedy policies on the 200 test trips, mean of seeds 1–3.');
      root.querySelectorAll('[data-action^="train"]').forEach(b => { b.disabled = last.trips >= CAP; });
    };
    root.querySelector('[data-action="reset"]').addEventListener('click', () => { init(); draw(); });
    root.querySelector('[data-action="train500"]').addEventListener('click', () => { trainBy(500); draw(); });
    root.querySelector('[data-action="train2000"]').addEventListener('click', () => { trainBy(2000); draw(); });
    init(); draw();
  });

  // 6. Fair evaluation on the 200 unseen test trips: comfort and power, five seeds (1-5).
  Demos.register('cc-eval-lab', (root, H) => {
    const SEEDS = [1, 2, 3, 4, 5], WS = [2, 2.5, 3, 3.5, 4, 5], evals = {};
    let front = null;
    root.querySelector('[data-controls]').innerHTML =
      H.select('cc-ev-budget', 'Training budget per seed', [1000, 4000, 16000].map(b => ({value: String(b), label: b + ' trips'})), '4000') +
      H.range('cc-ev-kp', 'Proportional gain K<sub>p</sub>', 1, 0.75, 3, 0.25, ' kW/°C') + H.button('Reset', 'reset');
    // each budget trains fresh agents, so results do not depend on viewing order
    const learned = budget => evals[budget] || (evals[budget] = SEEDS.map(s => { const ag = makeAgent(1, true, 0.9, s); train(ag, budget, W_DEF); return evaluate(() => agentPolicy(ag), W_DEF); }));
    const draw = () => {
      const budget = val(root, 'cc-ev-budget'), kp = val(root, 'cc-ev-kp'), R = refs(), L = learned(budget), P = evaluate(() => proportional(kp), W_DEF);
      front = front || WS.map(w => evaluate(sc => plan(sc.Tamb, w).policy, w));
      const mean = k => L.reduce((a, e) => a + e[k], 0) / L.length, sd = k => Math.sqrt(L.reduce((a, e) => a + (e[k] - mean(k)) ** 2, 0) / (L.length - 1));
      root.querySelector('[data-plot]').innerHTML = H.plot({
        series: [{label: 'Planner, w = 2 to 5', points: front.map(e => [e.comfort, e.kw]), color: H.colors[2], width: 2},
          {label: 'Bang-bang', points: [[R.bang.comfort, R.bang.kw]], color: H.colors[3], dots: true},
          {label: 'Proportional', points: [[P.comfort, P.kw]], color: H.colors[1], dots: true},
          {label: 'Learned (5 seeds)', points: L.map(e => [e.comfort, e.kw]), color: H.colors[0], dots: true}],
        xDomain: [40, 90], yDomain: [0.5, 2], xTicks: 5, yTicks: 3, xLabel: 'Time in comfort (%)', yLabel: 'Mean heater power (kW)', height: 300});
      const row = (name, c, kw, ret) => `<tr><td>${name}</td><td>${c}</td><td>${kw}</td><td>${ret}</td></tr>`;
      root.querySelector('[data-readout]').innerHTML = `<table class="compact"><thead><tr><th></th><th>Comfort %</th><th>kW</th><th>Return</th></tr></thead><tbody>` +
        row('Bang-bang', H.fmt(R.bang.comfort, 1), H.fmt(R.bang.kw, 2), signed(R.bang.ret, 1)) +
        row('Proportional', H.fmt(P.comfort, 1), H.fmt(P.kw, 2), signed(P.ret, 1)) +
        row('Learned', `${H.fmt(mean('comfort'), 1)} ± ${H.fmt(sd('comfort'), 1)}`, H.fmt(mean('kw'), 2), signed(mean('ret'), 1)) +
        row('Planner, w = 5', H.fmt(R.plan.comfort, 1), H.fmt(R.plan.kw, 2), signed(R.plan.ret, 1)) + '</tbody></table>';
    };
    wire(root, {'cc-ev-budget': '4000', 'cc-ev-kp': 1}, draw);
  });
})();
