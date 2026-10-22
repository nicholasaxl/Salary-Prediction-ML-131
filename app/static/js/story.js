// Story: a sticky scatter of predicted vs actual salary that morphs through the six stages of the study,
// beside one article per stage. Every number shown is the cross-validated figure from /api/story.
import { api } from './api.js';
import { el, tween, ticker, onView, renderReport, prefersReduced, usd, kfmt, clamp, lerp, ease } from './util.js';

const MAX = 460000;
const TICKS = [0, 100000, 200000, 300000, 400000];
const DUR = 1100;
const STAGGER = 250;
const TAU = Math.PI * 2;
const FAM = { RF: 'Random forest', LGBM: 'Boosting' };
const MOBILE = '(max-width: 900px)';

const r2f = (v) => v.toFixed(2);
const r2f3 = (v) => v.toFixed(3);

// ---------------------------------------------------------------- data prep
function prep(st) {
  const N = st.yRaw.length;
  const X = new Float64Array(N);
  for (let i = 0; i < N; i++) X[i] = st.yRaw[i] * 100;
  const dedupOf = new Int32Array(N).fill(-1);
  st.keep.forEach((r, j) => { dedupOf[r] = j; });
  const keepIdx = [], dupIdx = [];
  for (let i = 0; i < N; i++) (dedupOf[i] >= 0 ? keepIdx : dupIdx).push(i);
  const order = [...Array(N).keys()].sort((a, b) => X[a] - X[b]);
  const delay = new Float32Array(N);
  order.forEach((i, r) => { delay[i] = (r / Math.max(1, N - 1)) * STAGGER; });
  return { N, X, keepIdx: Int32Array.from(keepIdx), dupIdx: Int32Array.from(dupIdx), delay };
}

const decDollars = (d) => ({ x: d.x.map((v) => v * 100), y: d.y.map((v) => v * 100) });

/** What the chart should show for a view {stage, family, fit}. NaN y = "stay where you are" (hidden duplicates). */
function makeTarget(st, P, v) {
  const y = new Float64Array(P.N);
  let dec, m, who;
  if (v.stage === 0) {
    const p = v.fit ? st.insample.pred : st.pred[v.family][0];
    for (let i = 0; i < P.N; i++) y[i] = p[i] * 100;
    dec = v.fit ? st.decileLineY : st.deciles[v.family][0];
    m = v.fit ? { r2: st.insample.r2, rmse: st.insample.rmse } : st.metrics[v.family][0];
    who = v.fit ? 'Random forest, training fit' : FAM[v.family];
  } else {
    const best = v.stage === 5;
    const fam = best ? 'BEST' : v.family;
    const k = best ? 0 : v.stage;
    const p = st.pred[fam][k];
    y.fill(NaN);
    for (let j = 0; j < p.length; j++) y[st.keep[j]] = p[j] * 100;
    dec = st.deciles[fam][k];
    m = st.metrics[fam][k];
    who = best ? st.best.name : FAM[v.family];
  }
  return { y, dupA: v.stage === 0 ? 1 : 0, dec: decDollars(dec), m, who };
}

// ---------------------------------------------------------------- mount
export async function mount(host) {
  const [st, models] = await Promise.all([api.story(), api.models().catch(() => null)]);
  const P = prep(st);
  const stages = st.stages;
  const state = { stage: 0, family: 'RF', fit: false };
  const mq = window.matchMedia(MOBILE);

  // ---------- chart card
  const steps = stages.map((s, i) => el('button', {
    class: 'st-step', type: 'button', 'data-i': i,
    'aria-label': `Stage ${i}: ${s.name}`,
    onclick: () => goTo(i),
  }, el('span', { class: 'st-step-dot', 'aria-hidden': 'true' }), el('span', { class: 'st-step-lab', 'aria-hidden': 'true' }, s.short)));
  const stepper = el('nav', { class: 'st-stepper', 'aria-label': 'Study stages (use left and right arrow keys)' }, steps);
  stepper.addEventListener('keydown', (e) => {
    const k = e.key;
    let to = null;
    if (k === 'ArrowRight' || k === 'ArrowDown') to = Math.min(stages.length - 1, state.stage + 1);
    else if (k === 'ArrowLeft' || k === 'ArrowUp') to = Math.max(0, state.stage - 1);
    else if (k === 'Home') to = 0;
    else if (k === 'End') to = stages.length - 1;
    if (to == null) return;
    e.preventDefault();
    steps[to].focus();
    goTo(to);
  });

  const famBtns = Object.entries(FAM).map(([key, name]) => el('button', {
    class: 'st-seg-btn', type: 'button', 'data-fam': key, 'aria-pressed': String(key === state.family),
    onclick: () => setFamily(key),
  }, name));
  const famGroup = el('div', { class: 'st-seg', role: 'group', 'aria-label': 'Model family' }, famBtns);
  const bestTag = el('span', { class: 'st-best-tag', hidden: true }, 'Stage 5 uses the single best model: ', el('b', {}, st.best.name));

  const hintId = 'st-fit-hint';
  const makeSwitch = (label) => {
    const b = el('button', {
      class: 'st-switch', type: 'button', role: 'switch', 'aria-checked': 'false', 'aria-describedby': hintId,
      onclick: () => setFit(!state.fit),
    }, el('span', { class: 'st-switch-track', 'aria-hidden': 'true' }), el('span', {}, label), el('span', { class: 'st-switch-state', 'aria-hidden': 'true' }, 'off'));
    return b;
  };
  const fitSwitch = makeSwitch('Show training fit (what the R report plotted)');
  const hint = el('p', { class: 'st-hint', id: hintId },
    'The R report scored its forest on the very rows it had trained on, so the cloud hugged the diagonal; switch it off to see the same forest scored on rows it never saw.');
  const ctrls = el('div', { class: 'st-ctrls' }, famGroup, bestTag, fitSwitch);

  const canvas = el('canvas', { class: 'st-canvas', role: 'img', 'aria-label': '' }, 'Scatter of predicted against actual salary.');
  const mK = el('p', { class: 'st-mbox-k' });
  const mR2L = el('span', { class: 'st-mbox-l' }, 'CV R²');
  const mR2 = el('b', { class: 'st-mbox-v num' });
  const mRmL = el('span', { class: 'st-mbox-l' }, 'CV RMSE');
  const mRm = el('b', { class: 'st-mbox-v num' });
  const mWarn = el('p', { class: 'st-mbox-warn', hidden: true }, 'training fit, not trustworthy');
  const mbox = el('div', { class: 'st-mbox', 'aria-hidden': 'true' }, mK,
    el('div', { class: 'st-mbox-row is-main' }, mR2L, mR2), el('div', { class: 'st-mbox-row' }, mRmL, mRm), mWarn);
  const plot = el('div', { class: 'st-plot' }, canvas, mbox);

  const card = el('div', { class: 'card st-card' }, el('div', { class: 'st-head' }, stepper, ctrls, hint), plot);
  const stick = el('div', { class: 'st-stick' }, card);

  const noisePct = Math.round((await api.report().then((r) => r.meta.noise).catch(() => 0.3)) * 100);
  const legend = el('ul', { class: 'st-legend', 'aria-label': 'Chart legend' },
    el('li', {}, el('i', { class: 'st-sw st-sw--pt', 'aria-hidden': 'true' }), 'one person'),
    el('li', {}, el('i', { class: 'st-sw', 'aria-hidden': 'true' }), 'average prediction within deciles of actual salary'),
    el('li', {}, el('i', { class: 'st-sw st-sw--ghost', 'aria-hidden': 'true' }), 'previous stage'),
    el('li', {}, el('i', { class: 'st-sw st-sw--diag', 'aria-hidden': 'true' }), 'perfect prediction'));
  const honest = el('p', { class: 'st-honest' },
    'Held-out R² only moves from about ', el('b', { class: 'num' }, '0.39'), ' to ', el('b', { class: 'num' }, '0.41'),
    `: people with identical profiles differ by about ${noisePct}% in salary, so no model can collapse the cloud onto the diagonal.`);
  const note = el('div', { class: 'st-note' }, legend, honest);
  const left = el('div', { class: 'st-left' }, stick, note);

  // ---------- articles
  const M = st.metrics;
  const arts = [];
  const artUI = [];
  const stepsCol = el('div', { class: 'st-steps' });
  const callSwitch = makeSwitch('Show training fit');

  for (let i = 0; i < stages.length; i++) {
    const s = stages[i];
    const eyebrow = el('p', { class: 'st-eyebrow' }, el('span', {}, `Stage ${i}`));
    if (s.leaky) eyebrow.append(el('span', { class: 'st-badge' }, 'leaky test set'));
    if (i === 1) eyebrow.append(el('span', {}, '· Correction'));
    if (i === 5) eyebrow.append(el('span', { class: 'st-badge st-badge--best' }, 'best of ten'));
    const fam = el('p', { class: 'st-fam' });
    const rmV = el('b', { class: 'st-metric-v num' });
    const r2V = el('b', { class: 'st-metric-v num' });
    const rmC = el('span', { class: 'st-chip' });
    const r2C = el('span', { class: 'st-chip' });
    const metrics = el('div', { class: 'st-metrics' },
      el('div', { class: 'st-metric' }, el('span', { class: 'st-metric-l' }, 'CV RMSE'), rmV, rmC),
      el('div', { class: 'st-metric' }, el('span', { class: 'st-metric-l' }, 'CV R²'), r2V, r2C));
    const body = el('div', { class: 'st-art-body' });
    const art = el('article', { class: 'st-art', id: `story-stage-${i}`, 'data-i': i, 'aria-labelledby': `st-h-${i}` },
      eyebrow, el('h3', { id: `st-h-${i}` }, s.name), fam, metrics);
    if (i === 0) {
      art.append(el('div', { class: 'st-callout' },
        el('p', {}, 'The R report plotted its forest against the rows it had trained on, which is why its chart hugged the diagonal. Flip the switch to see that picture, then flip it back to see the honest one.'),
        callSwitch));
    }
    if (i === 1) {
      art.append(el('p', { class: 'st-delta-note' }, el('b', {}, 'Correction, not a regression. '),
        `${(P.dupIdx.length).toLocaleString('en-US')} exact duplicate rows fade out of the chart. With them gone no test row has a twin in training, so the scores are expected to get worse.`));
    }
    art.append(body);
    arts.push(art);
    artUI.push({ fam, rmV, r2V, rmC, r2C });
    stepsCol.append(art);
  }

  const live = el('p', { class: 'st-sr', 'aria-live': 'polite', 'aria-atomic': 'true' });
  const grid = el('div', { class: 'st-grid' }, left, stepsCol);
  const glance = models ? buildGlance(models, stages, st) : null;
  const root = el('div', { class: 'st' }, grid, glance, live);
  host.append(root);

  // ---------- canvas state
  const ctx = canvas.getContext('2d');
  let W = 0, H = 0, dpr = 1;
  let pad = { l: 56, r: 14, t: 12, b: 40 };
  const PX = new Float32Array(P.N);
  const cur = new Float64Array(P.N), from = new Float64Array(P.N), to = new Float64Array(P.N);
  let dupA = 1, dupFrom = 1, dupTo = 1;
  let decCur = null, decFrom = null, decTo = null, ghost = null, ghostA = 0;
  let anim = null;
  let col = {};

  const readColors = () => {
    const cs = getComputedStyle(document.documentElement);
    const g = (n) => cs.getPropertyValue(n).trim();
    col = {
      point: g('--canvas-point'), grid: g('--canvas-grid'), faint: g('--fg-faint'), dim: g('--fg-dim'),
      bad: g('--bad'), gold: g('--gold'), bg: g('--bg-raise'), mono: g('--font-mono') || 'monospace',
    };
  };

  const sx = (v) => pad.l + (v / MAX) * (W - pad.l - pad.r);
  const sy = (v) => H - pad.b - (v / MAX) * (H - pad.t - pad.b);

  function layout() {
    const r = plot.getBoundingClientRect();
    const w = Math.max(1, Math.round(r.width)), h = Math.max(1, Math.round(r.height));
    dpr = Math.min(3, window.devicePixelRatio || 1);
    W = w; H = h;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    const small = w < 520;
    pad = small ? { l: 40, r: 10, t: 8, b: 30 } : { l: 56, r: 14, t: 12, b: 40 };
    for (let i = 0; i < P.N; i++) PX[i] = sx(P.X[i]);
    mbox.style.left = (pad.l + (small ? 6 : 10)) + 'px';
    mbox.style.top = (pad.t + (small ? 2 : 6)) + 'px';
  }

  function polyline(d) {
    ctx.beginPath();
    for (let k = 0; k < d.x.length; k++) {
      const px = sx(d.x[k]), py = sy(d.y[k]);
      if (k) ctx.lineTo(px, py); else ctx.moveTo(px, py);
    }
  }

  function draw() {
    if (!W) return;
    const c = ctx;
    const small = W < 520;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, W, H);
    const x0 = sx(0), x1 = sx(MAX), y0 = sy(0), y1 = sy(MAX);

    // grid
    c.lineWidth = 1;
    c.strokeStyle = col.grid;
    c.beginPath();
    for (const v of TICKS) {
      const gx = Math.round(sx(v)) + 0.5, gy = Math.round(sy(v)) + 0.5;
      c.moveTo(gx, y1); c.lineTo(gx, y0);
      c.moveTo(x0, gy); c.lineTo(x1, gy);
    }
    c.stroke();

    // tick labels and axis titles
    const fs = small ? 10 : 11;
    c.fillStyle = col.faint;
    c.font = `500 ${fs}px ${col.mono}`;
    c.textAlign = 'center'; c.textBaseline = 'top';
    for (const v of TICKS) c.fillText(kfmt(v), sx(v), y0 + 5);
    c.textAlign = 'right'; c.textBaseline = 'middle';
    for (const v of TICKS) c.fillText(kfmt(v), x0 - 6, sy(v));
    c.fillStyle = col.dim;
    c.font = `600 ${fs}px ${col.mono}`;
    c.textAlign = 'right'; c.textBaseline = 'bottom';
    c.fillText('ACTUAL SALARY →', x1, y0 - 4);
    c.textAlign = 'left'; c.textBaseline = 'top';
    {
      const lab = small ? '↑ PREDICTED' : '↑ PREDICTED SALARY';
      let lx = Math.max(x0 + 6, mbox.offsetLeft + mbox.offsetWidth + 10), ly = y1 + 4;
      if (lx + c.measureText(lab).width > x1 - 30) { lx = x0 + 6; ly = mbox.offsetTop + mbox.offsetHeight + 6; }
      c.fillText(lab, lx, ly);
    }

    // perfect-prediction diagonal
    c.save();
    c.beginPath(); c.rect(x0, y1, x1 - x0, y0 - y1); c.clip();
    c.strokeStyle = col.bad; c.lineWidth = 1.5; c.setLineDash([6, 5]);
    c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1); c.stroke();
    c.setLineDash([]);

    // points: one path for the unique rows, one for the (fading) duplicates
    const rad = small ? 1.7 : 2.1;
    c.fillStyle = col.point;
    c.beginPath();
    const K = P.keepIdx;
    for (let n = 0; n < K.length; n++) {
      const i = K[n], px = PX[i], py = sy(cur[i]);
      c.moveTo(px + rad, py); c.arc(px, py, rad, 0, TAU);
    }
    c.fill();
    if (dupA > 0.01) {
      c.globalAlpha = dupA;
      c.beginPath();
      const D = P.dupIdx;
      for (let n = 0; n < D.length; n++) {
        const i = D[n], px = PX[i], py = sy(cur[i]);
        c.moveTo(px + rad, py); c.arc(px, py, rad, 0, TAU);
      }
      c.fill();
      c.globalAlpha = 1;
    }

    // ghost of the previous decile line
    if (ghost && ghostA > 0.01) {
      c.globalAlpha = ghostA * 0.9;
      c.strokeStyle = col.faint; c.lineWidth = 1.75; c.setLineDash([5, 4]);
      polyline(ghost); c.stroke();
      c.setLineDash([]); c.globalAlpha = 1;
    }
    // current decile line
    if (decCur) {
      c.strokeStyle = col.gold; c.lineWidth = small ? 2.25 : 2.75; c.lineJoin = 'round'; c.lineCap = 'round';
      polyline(decCur); c.stroke();
      c.fillStyle = col.gold; c.strokeStyle = col.bg; c.lineWidth = 1.5;
      c.beginPath();
      const dr = small ? 3 : 3.6;
      for (let k = 0; k < decCur.x.length; k++) {
        const px = sx(decCur.x[k]), py = sy(decCur.y[k]);
        c.moveTo(px + dr, py); c.arc(px, py, dr, 0, TAU);
      }
      c.fill(); c.stroke();
    }
    c.restore();
  }

  // ---------- morph
  const sameLine = (a, b) => a && b && a.x.every((v, k) => Math.abs(v - b.x[k]) < 1 && Math.abs(a.y[k] - b.y[k]) < 1);

  function morph(tg, instant = false) {
    if (anim) anim.cancel();
    from.set(cur);
    for (let i = 0; i < P.N; i++) to[i] = Number.isNaN(tg.y[i]) ? cur[i] : tg.y[i];
    dupFrom = dupA; dupTo = tg.dupA;
    const dFrom = decCur ? { x: decCur.x.slice(), y: decCur.y.slice() } : tg.dec;
    decFrom = dFrom; decTo = tg.dec;
    if (!sameLine(dFrom, tg.dec)) ghost = { x: dFrom.x.slice(), y: dFrom.y.slice() };
    const gFrom = ghostA;
    const span = DUR - STAGGER;
    const frame = (t) => {
      const ms = t * DUR;
      for (let i = 0; i < P.N; i++) {
        const a = from[i], b = to[i];
        if (a === b) { cur[i] = b; continue; }
        const lt = clamp((ms - P.delay[i]) / span, 0, 1);
        cur[i] = a + (b - a) * ease.inOut(lt);
      }
      const g = ease.inOut(t);
      dupA = lerp(dupFrom, dupTo, g);
      decCur = { x: decTo.x.map((v, k) => lerp(decFrom.x[k], v, g)), y: decTo.y.map((v, k) => lerp(decFrom.y[k], v, g)) };
      ghostA = ghost ? Math.max(gFrom, g) : 0;
      draw();
    };
    if (instant) { frame(1); anim = null; return; }
    anim = tween(DUR, frame, (t) => t);
  }

  // ---------- metric box, articles, labels
  const viewTarget = () => makeTarget(st, P, state);
  const famName = () => (state.stage === 5 ? st.best.name : FAM[state.family]);

  function updateBox(tg, animate = true) {
    const fit = state.stage === 0 && state.fit;
    mK.textContent = `Stage ${state.stage} · ${tg.who}`;
    mR2L.textContent = fit ? 'Training R²' : 'CV R²';
    mRmL.textContent = fit ? 'Training RMSE' : 'CV RMSE';
    mWarn.hidden = !fit;
    mbox.classList.toggle('is-fit', fit);
    setNum(mR2, tg.m.r2, r2f, animate);
    setNum(mRm, tg.m.rmse, usd, animate);
    canvas.setAttribute('aria-label', fit
      ? `Stage 0, Random forest training fit, not trustworthy: R² ${r2f(tg.m.r2)}, RMSE ${usd(tg.m.rmse)}. Predicted against actual salary for ${P.N.toLocaleString('en-US')} rows.`
      : `Stage ${state.stage}, ${tg.who}: R² ${r2f(tg.m.r2)}, RMSE ${usd(tg.m.rmse)}. Cross-validated predicted against actual salary for ${(state.stage === 0 ? P.N : P.keepIdx.length).toLocaleString('en-US')} rows.`);
  }

  function setNum(node, v, fmt, animate) {
    if (animate && !prefersReduced()) ticker(node, v, { format: fmt, duration: 900 });
    else { if (node._tick) node._tick.cancel(); node.dataset.value = String(v); node.textContent = fmt(v); }
  }

  function chip(node, d, { lowerBetter, fmt, note }) {
    node.className = 'st-chip';
    node.removeAttribute('title');
    if (d == null) { node.classList.add('st-chip--none'); node.textContent = 'no earlier stage'; return; }
    const better = lowerBetter ? d < 0 : d > 0;
    const zero = Math.abs(d) < (lowerBetter ? 0.5 : 0.0005);
    const sign = d > 0 ? '+' : d < 0 ? '−' : '±';
    const arrow = zero ? '•' : d > 0 ? '▲' : '▼';
    const txt = `${arrow} ${sign}${fmt(Math.abs(d))}`;
    if (note) { node.classList.add('st-chip--note'); node.textContent = `${txt} · correction`; node.title = 'Correction, not a regression'; return; }
    if (zero) { node.textContent = `${txt} · no change`; return; }
    node.classList.add(better ? 'st-chip--good' : 'st-chip--bad');
    node.textContent = `${txt} · ${better ? 'better' : 'worse'}`;
  }

  function updateArticles(animate) {
    const f = state.family;
    for (let i = 0; i < arts.length; i++) {
      const u = artUI[i];
      const best = i === 5;
      const m = best ? M.BEST[0] : M[f][i];
      const prev = best ? M[f][4] : i ? M[f][i - 1] : null;
      u.fam.textContent = best
        ? `${st.best.name}, the lowest-error of ten model families · deltas versus tuned ${FAM[f].toLowerCase()}`
        : i === 0
          ? `${FAM[f]} · cross-validated on all ${P.N.toLocaleString('en-US')} rows, duplicates included`
          : `${FAM[f]} · cross-validated on ${P.keepIdx.length.toLocaleString('en-US')} unique rows${i > 1 ? ` · deltas versus stage ${i - 1}` : ''}`;
      setNum(u.rmV, m.rmse, usd, animate);
      setNum(u.r2V, m.r2, r2f3, animate);
      // deltas from the unrounded scoreboard when available, so they match the written report to the dollar
      const exact = (fam, s, fallback) => (models && models.scoreboard.find((r) => r.family === fam && r.stage === s)) || fallback;
      const mx = best ? exact('BEST', 5, m) : exact(f, i, m);
      const px = prev && (best ? exact(f, 4, prev) : exact(f, i - 1, prev));
      chip(u.rmC, px ? mx.rmse - px.rmse : null, { lowerBetter: true, fmt: usd, note: i === 1 });
      chip(u.r2C, px ? mx.r2 - px.r2 : null, { lowerBetter: false, fmt: (v) => v.toFixed(3), note: i === 1 });
    }
  }

  function syncControls() {
    const s = state.stage;
    steps.forEach((b, i) => {
      if (i === s) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current');
      b.classList.toggle('is-past', i < s);
    });
    arts.forEach((a, i) => a.classList.toggle('is-current', i === s));
    famBtns.forEach((b) => { b.disabled = s === 5; b.setAttribute('aria-pressed', String(b.dataset.fam === state.family)); });
    famGroup.hidden = s === 5;
    bestTag.hidden = s !== 5;
    for (const sw of [fitSwitch, callSwitch]) {
      sw.disabled = s !== 0;
      sw.setAttribute('aria-checked', String(state.fit));
      sw.querySelector('.st-switch-state').textContent = state.fit ? 'on' : 'off';
      sw.title = s === 0 ? '' : 'Available at stage 0';
    }
    hint.hidden = s !== 0;
    if (glance) glance.querySelectorAll('tbody tr').forEach((tr) => tr.classList.toggle('is-current', Number(tr.dataset.stage) === s));
  }

  function announce(extra = '') {
    const tg = currentTarget;
    const fit = state.stage === 0 && state.fit;
    live.textContent = `Stage ${state.stage} of ${stages.length - 1}, ${stages[state.stage].name}. ${tg.who}: ${fit ? 'training-fit' : 'cross-validated'} R² ${r2f(tg.m.r2)}, RMSE ${usd(tg.m.rmse)}.${extra}`;
  }

  let currentTarget = null;
  function apply({ animate = true, say = '' } = {}) {
    currentTarget = viewTarget();
    syncControls();
    updateBox(currentTarget, animate);
    morph(currentTarget, !animate);
    announce(say);
  }

  function setStage(i) {
    if (i === state.stage) return;
    state.stage = i;
    if (i !== 0) state.fit = false;
    apply();
  }
  function setFamily(f) {
    if (f === state.family || state.stage === 5) return;
    state.family = f;
    updateArticles(true);
    apply({ say: '' });
  }
  function setFit(on) {
    if (state.stage !== 0) return;
    state.fit = on;
    apply({ say: on ? ' Showing the training fit, which is not trustworthy.' : ' Showing the honest held-out predictions.' });
  }

  // ---------- scrolling: an observer around the viewport middle picks the stage; the stepper scrolls to it
  let lock = false, lockT = 0, io = null;
  const navH = () => parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--nav-h')) || 56;
  const band = () => (mq.matches ? [0.6, 0.36] : [0.45, 0.5]);

  function pickFromScroll() {
    const [topF] = band();
    const yLine = window.innerHeight * (topF + 0.025);
    for (let i = 0; i < arts.length; i++) {
      const r = arts[i].getBoundingClientRect();
      if (r.top <= yLine && r.bottom >= yLine) { setStage(i); return; }
    }
  }

  function observe() {
    if (io) io.disconnect();
    if (!('IntersectionObserver' in window)) return;
    const [a, b] = band();
    io = new IntersectionObserver((entries) => {
      if (lock) return;
      for (const e of entries) if (e.isIntersecting) setStage(Number(e.target.dataset.i));
    }, { rootMargin: `-${a * 100}% 0px -${b * 100}% 0px` });
    arts.forEach((a2) => io.observe(a2));
  }

  function unlock() { lock = false; clearTimeout(lockT); pickFromScroll(); }

  function goTo(i) {
    const off = mq.matches ? navH() + stick.offsetHeight + 8 : navH() + 16;
    const top = arts[i].getBoundingClientRect().top + window.scrollY - off;
    lock = true;
    clearTimeout(lockT);
    setStage(i);
    const smooth = !prefersReduced();
    window.scrollTo({ top, behavior: smooth ? 'smooth' : 'auto' });
    if (!smooth) { lockT = setTimeout(unlock, 60); return; }
    lockT = setTimeout(unlock, 1400);
    if ('onscrollend' in window) window.addEventListener('scrollend', () => { clearTimeout(lockT); lockT = setTimeout(unlock, 50); }, { once: true });
  }

  // ---------- theme, resize
  const redrawTheme = () => { readColors(); draw(); };
  window.addEventListener('themechange', () => requestAnimationFrame(redrawTheme));
  const lightMq = window.matchMedia('(prefers-color-scheme: light)');
  (lightMq.addEventListener ? lightMq.addEventListener('change', redrawTheme) : lightMq.addListener(redrawTheme));
  const onMq = () => { observe(); };
  (mq.addEventListener ? mq.addEventListener('change', onMq) : mq.addListener(onMq));

  const relayout = () => { layout(); draw(); };
  if ('ResizeObserver' in window) new ResizeObserver(() => relayout()).observe(plot);
  else window.addEventListener('resize', relayout);
  // devicePixelRatio changes (zoom, moving between screens) do not always resize the box
  const watchDpr = () => {
    const m = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
    const h = () => { relayout(); watchDpr(); };
    if (m.addEventListener) m.addEventListener('change', h, { once: true });
  };
  watchDpr();

  // ---------- first paint: complete at rest
  readColors();
  layout();
  currentTarget = viewTarget();
  for (let i = 0; i < P.N; i++) cur[i] = currentTarget.y[i];
  decCur = currentTarget.dec;
  dupA = 1;
  syncControls();
  updateBox(currentTarget, false);
  updateArticles(false);
  draw();
  announce();
  live.textContent = '';
  observe();
  // count the box up once it is first seen (its final values are already in place before that)
  onView(plot, () => { if (!prefersReduced()) { const m = currentTarget.m; ticker(mR2, m.r2, { from: 0, format: r2f, duration: 1100 }); ticker(mRm, m.rmse, { from: 0, format: usd, duration: 1100 }); } }, { threshold: 0.4 });

  // the written report under every stage
  await Promise.all(arts.map((a, i) => renderReport(a.lastElementChild, 'stage' + i, { open: true })));
  pickFromScroll();
}

// ---------------------------------------------------------------- all stages at a glance
function buildGlance(models, stages, st) {
  const rows = models.scoreboard || [];
  const at = (fam, s) => rows.find((r) => r.family === fam && r.stage === s);
  const maxR2 = 0.5;
  let d = 0;
  const r2Cell = (r, best) => {
    if (!r) return el('td', {}, '–');
    const bar = el('span', { class: 'st-bar', 'aria-hidden': 'true' }, el('i', { style: `--v:${clamp(r.r2 / maxR2, 0, 1).toFixed(3)};--d:${d++}` }));
    return el('td', { class: 'num' }, el('span', { class: 'st-r2' }, bar, el('span', {}, r.r2.toFixed(3))));
  };
  const body = el('tbody');
  for (const s of stages) {
    if (s.id < 5) {
      const rf = at('RF', s.id), lg = at('LGBM', s.id);
      body.append(el('tr', { 'data-stage': s.id, class: s.leaky ? 'is-leaky' : '' },
        el('th', { scope: 'row' }, `${s.id} ${s.short}`, el('small', {}, s.leaky ? 'leaky test set' : s.name)),
        el('td', { class: 'num' }, rf ? usd(rf.rmse) : '–'), r2Cell(rf),
        el('td', { class: 'num' }, lg ? usd(lg.rmse) : '–'), r2Cell(lg)));
    } else {
      const b = at('BEST', 5) || { rmse: st.metrics.BEST[0].rmse, r2: st.metrics.BEST[0].r2 };
      const bar = el('span', { class: 'st-bar', 'aria-hidden': 'true' }, el('i', { style: `--v:${clamp(b.r2 / maxR2, 0, 1).toFixed(3)};--d:${d++}` }));
      body.append(el('tr', { 'data-stage': 5, class: 'is-best' },
        el('th', { scope: 'row' }, `5 ${s.short}`, el('small', {}, `Best model: ${st.best.name}`)),
        el('td', { colspan: 4, class: 'num' }, el('span', { class: 'st-r2' },
          el('span', {}, `RMSE ${usd(b.rmse)}`), bar, el('span', {}, `R² ${b.r2.toFixed(3)}`), el('span', {}, '★ best overall')))));
    }
  }
  const table = el('table', { class: 'st-table' },
    el('caption', { class: 'st-sr' }, 'Cross-validated error and R² for every stage, random forest and boosting, plus the best model.'),
    el('thead', {},
      el('tr', {}, el('th', { scope: 'col', rowspan: 2 }, 'Stage'), el('th', { scope: 'colgroup', colspan: 2 }, 'Random forest'), el('th', { scope: 'colgroup', colspan: 2 }, 'Boosting')),
      el('tr', {}, el('th', { scope: 'col' }, 'RMSE'), el('th', { scope: 'col' }, 'R²'), el('th', { scope: 'col' }, 'RMSE'), el('th', { scope: 'col' }, 'R²'))),
    body);
  const wrap = el('div', { class: 'st-glance' },
    el('h3', {}, 'All stages at a glance'),
    el('p', {}, 'Cross-validated on held-out folds. Lower RMSE and higher R² are better; stage 0 is flattered by duplicate rows leaking into its test folds.'),
    el('div', { class: 'card st-tablewrap' }, table));
  if (!prefersReduced() && 'IntersectionObserver' in window) {
    wrap.classList.add('is-pre');
    const grow = () => { wrap.classList.add('is-grow'); requestAnimationFrame(() => wrap.classList.remove('is-pre')); };
    onView(wrap, grow, { threshold: 0.2 });
    setTimeout(() => { if (wrap.classList.contains('is-pre')) grow(); }, 6000);
  }
  return wrap;
}
