/* Interactive helpers for the 7043SCN Reveal decks.
   Load order in each deck: reveal.js and plugins, interactive.js, demos.js,
   then Reveal.initialize(...). Each demonstration registers itself with
   Demos.register('section-id', (root, H) => { ... }) and is mounted once,
   before Reveal lays out the slides. No network requests are made here. */
(function () {
  'use strict';

  const colors = ['#2f6690', '#b23a2a', '#3a7d44', '#7a4e9a', '#b8860b'];
  const esc = s => String(s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));

  const H = window.H = {
    colors, esc,
    fmt: (x, n = 2) => Number(x).toFixed(n),
    clamp: (x, a, b) => Math.min(b, Math.max(a, x)),
    $: (r, s) => r.querySelector(s),
    bind: (r, s, e, f) => r.querySelector(s).addEventListener(e, f),
    math: s => `<div class="equation">${s}</div>`,
    // deterministic pseudorandom generator (mulberry32)
    rng(seed) {
      let a = seed >>> 0;
      return () => {
        a += 0x6D2B79F5;
        let t = a;
        t = Math.imul(t ^ t >>> 15, t | 1);
        t ^= t + Math.imul(t ^ t >>> 7, t | 61);
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
      };
    },
    normal(r) { return Math.sqrt(-2 * Math.log(Math.max(1e-10, r()))) * Math.cos(2 * Math.PI * r()); },

    range: (id, label, v, min, max, step = 1, unit = '') =>
      `<label class="range-label" for="${id}">${label} <output for="${id}" data-unit="${esc(unit)}">${v}${unit}</output>` +
      `<input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${v}" data-default="${v}"></label>`,
    select: (id, label, options, value) =>
      `<label class="select-label" for="${id}">${label}<select id="${id}" data-default="${esc(value)}">` +
      options.map(o => `<option value="${esc(o.value)}"${String(o.value) === String(value) ? ' selected' : ''}>${esc(o.label)}</option>`).join('') +
      `</select></label>`,
    button: (label, action, kind = '') => `<button type="button" data-action="${action}" class="${kind}">${label}</button>`,
    quiz: (question, options, correct, explanation) =>
      `<div class="quiz" data-correct="${correct}"><p class="question">${question}</p><div class="quiz-options">` +
      options.map((o, i) => `<button type="button" data-answer="${i}">${o}</button>`).join('') +
      `</div><p class="quiz-feedback" aria-live="polite"></p><template>${explanation}</template></div>`,

    // Typeset TeX inserted after page load (e.g. a quiz explanation).
    typeset(el) {
      const mj = window.MathJax;
      if (mj && mj.typesetPromise) {
        const run = () => mj.typesetPromise([el]).then(() => window.Reveal && Reveal.isReady && Reveal.isReady() && Reveal.layout()).catch(() => {});
        return mj.startup && mj.startup.promise ? mj.startup.promise.then(run) : run();
      }
      return Promise.resolve();
    },

    // Line/dot chart as an SVG string. Font sizes are in viewBox units.
    plot(o) {
      const w = o.width || 560, h = o.height || 300, m = {l: 70, r: 14, t: 16, b: 52};
      const xd = o.xDomain || [0, 1], yd = o.yDomain || [0, 1];
      const X = x => m.l + (x - xd[0]) / (xd[1] - xd[0] || 1) * (w - m.l - m.r);
      const Y = y => h - m.b - (y - yd[0]) / (yd[1] - yd[0] || 1) * (h - m.t - m.b);
      const num = x => Math.abs(x) >= 1000 ? x.toFixed(0) : String(+x.toFixed(2));
      const title = esc(o.title || `${o.yLabel || 'y'} against ${o.xLabel || 'x'}`);
      let s = `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${title}"><title>${title}</title>`;
      (o.bands || []).forEach(b => { s += `<rect x="${m.l}" y="${Y(b.y1)}" width="${w - m.l - m.r}" height="${Math.max(0, Y(b.y0) - Y(b.y1))}" fill="${b.color || '#e3eedf'}"/>`; });
      const nx = o.xTicks || 4, ny = o.yTicks || 4;
      for (let i = 0; i <= ny; i++) {
        const y = yd[0] + i * (yd[1] - yd[0]) / ny;
        s += `<path d="M${m.l},${Y(y)}H${w - m.r}" stroke="#d6d6cc"/><text x="${m.l - 8}" y="${Y(y) + 6}" text-anchor="end">${num(y)}</text>`;
      }
      for (let i = 0; i <= nx; i++) {
        const x = xd[0] + i * (xd[1] - xd[0]) / nx;
        s += `<text x="${X(x)}" y="${h - m.b + 22}" text-anchor="middle">${num(x)}</text>`;
      }
      s += `<path d="M${m.l},${m.t}V${h - m.b}H${w - m.r}" fill="none" stroke="#444"/>` +
           `<text x="${(w + m.l - m.r) / 2}" y="${h - 6}" text-anchor="middle">${esc(o.xLabel || '')}</text>` +
           `<text x="16" y="${(h - m.b + m.t) / 2}" transform="rotate(-90 16 ${(h - m.b + m.t) / 2})" text-anchor="middle">${esc(o.yLabel || '')}</text>`;
      (o.series || []).forEach((a, i) => {
        const pts = a.points.filter(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));
        const c = a.color || colors[i % colors.length];
        if (!a.dots && pts.length > 1) s += `<polyline points="${pts.map(p => `${X(p[0])},${Y(p[1])}`).join(' ')}" fill="none" stroke="${c}" stroke-width="${a.width || 3}"${a.opacity ? ` stroke-opacity="${a.opacity}"` : ''}${a.dashed ? ' stroke-dasharray="7 5"' : ''}/>`;
        else pts.forEach(p => { s += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="5" fill="${c}"/>`; });
      });
      s += '</svg>';
      if ((o.series || []).some(x => x.label)) s += `<div class="legend">${o.series.map((a, i) => a.label ? `<span><i style="background:${a.color || colors[i % colors.length]}"></i>${esc(a.label)}</span>` : '').join('')}</div>`;
      return s;
    },

    // Bar chart as an SVG string.
    bars(o) {
      const w = o.width || 560, h = o.height || 270, top = 30, base = h - 50;
      const lo = o.min ?? 0, hi = o.max ?? Math.max(1, ...o.values);
      const Y = v => base - (v - lo) / (hi - lo || 1) * (base - top);
      const bw = (w - 80) / o.values.length, cs = o.colors || colors;
      let s = `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(o.title || o.yLabel || 'Bar chart')}"><title>${esc(o.title || o.yLabel || 'Values')}</title><text x="4" y="16">${esc(o.yLabel || '')}</text>`;
      const nt = o.ticks || 4;
      for (let j = 0; j <= nt; j++) { const v = lo + (hi - lo) * j / nt; s += `<path d="M62 ${Y(v)}H${w - 6}" stroke="#d6d6cc"/><text x="54" y="${Y(v) + 6}" text-anchor="end">${+v.toFixed(2)}</text>`; }
      o.values.forEach((v, i) => {
        const x = 70 + i * bw, y = Y(H.clamp(v, lo, hi)), z = Y(H.clamp(0, lo, hi));
        s += `<rect x="${x + 6}" y="${Math.min(y, z)}" width="${Math.max(2, bw - 14)}" height="${Math.max(1, Math.abs(y - z))}" fill="${cs[i % cs.length]}"/>` +
             `<text x="${x + bw / 2}" y="${(v < 0 ? z : y) - 7}" text-anchor="middle">${+Number(v).toFixed(3)}</text>` +
             `<text x="${x + bw / 2}" y="${h - 24}" text-anchor="middle">${esc(o.labels[i])}</text>`;
      });
      return s + '</svg>';
    }
  };

  // Registry of slide demonstrations, keyed by the id of their <section>.
  const registry = [];
  window.Demos = {
    register(id, mount) { registry.push([id, mount]); },
    mountAll() {
      registry.forEach(([id, mount]) => {
        const root = document.getElementById(id);
        if (!root) { console.error(`Demos: no slide with id "${id}"`); return; }
        try { mount(root, H); } catch (err) { console.error(`Demos: mounting "${id}" failed`, err); }
      });
      activateQuizzes(document);
      document.querySelectorAll('.reveal .slides .demo-controls, .reveal .slides .quiz, .reveal .slides input, .reveal .slides select')
        .forEach(el => el.setAttribute('data-prevent-swipe', ''));
    }
  };

  function activateQuizzes(scope) {
    scope.querySelectorAll('.quiz').forEach(q => {
      if (q.dataset.ready) return;
      q.dataset.ready = '1';
      q.querySelectorAll('[data-answer]').forEach(b => b.addEventListener('click', () => {
        const right = +b.dataset.answer === +q.dataset.correct;
        q.querySelectorAll('[data-answer]').forEach(c => { c.classList.remove('correct', 'incorrect'); c.setAttribute('aria-pressed', String(c === b)); });
        b.classList.add(right ? 'correct' : 'incorrect');
        const fb = q.querySelector('.quiz-feedback');
        fb.innerHTML = right ? 'Correct. ' + q.querySelector('template').innerHTML : 'Not quite. Think about the mechanism, then try again.';
        if (right) H.typeset(fb);
      }));
    });
  }
  H.activateQuizzes = activateQuizzes;

  // Range sliders show their value in the adjacent <output>.
  document.addEventListener('input', e => {
    if (e.target.type !== 'range') return;
    const o = e.target.closest('label') && e.target.closest('label').querySelector('output');
    if (o) o.textContent = e.target.value + (o.dataset.unit || '');
  });

  // After a mouse/touch interaction, hand the arrow keys back to Reveal.
  // Keyboard users keep focus on the control.
  let pointerUsed = false;
  document.addEventListener('pointerdown', () => { pointerUsed = true; }, true);
  document.addEventListener('keydown', () => { pointerUsed = false; }, true);
  document.addEventListener('change', e => {
    if (pointerUsed && e.target.closest && e.target.closest('.reveal .slides') && /INPUT|SELECT/.test(e.target.tagName)) e.target.blur();
  }, true);
  document.addEventListener('pointerup', e => {
    const t = e.target;
    if (t && t.closest && t.closest('.reveal .slides') && (t.tagName === 'BUTTON' || t.type === 'range')) setTimeout(() => t.blur(), 0);
  }, true);

  // Reveal ignores keys only for INPUT/TEXTAREA. Also leave a focused select
  // alone, and let Space/Enter press a focused button without changing slide.
  window.DeckKeyboard = function (event) {
    const t = document.activeElement;
    if (!t || !t.closest || !t.closest('.reveal .slides')) return true;
    if (t.tagName === 'SELECT') return false;
    if (t.tagName === 'BUTTON' && (event.keyCode === 32 || event.keyCode === 13)) return false;
    return true;
  };

  // Re-centre the current slide once MathJax has finished typesetting.
  window.addEventListener('load', () => {
    const mj = window.MathJax;
    if (mj && mj.startup && mj.startup.promise) mj.startup.promise.then(() => { if (window.Reveal && Reveal.isReady && Reveal.isReady()) Reveal.layout(); });
  });
})();
