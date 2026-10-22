// Model arena: animated leaderboard, decision-rule stepper, verdict cards and the quantile-model exception.
// Everything is drawn from /api/models; every sentence is computed from the table, nothing is hard-coded.
import { el, ticker, onView, renderReport, prefersReduced, usd, kfmt } from './util.js';
import { api } from './api.js';

const nameOf = (key) => (key.startsWith('tuned_') ? 'tuned ' : '') + key.replace(/^(search|tuned)_/, '');
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const pct = (v, d = 1) => (v * 100).toFixed(d) + '%';
const usd2 = (n) => '$' + n.toFixed(2);
const secs =(s) => (s < 10 ? s.toFixed(1) : String(Math.round(s))) + ' s';
const list = (a) => (a.length < 2 ? a.join('') : a.slice(0, -1).join(', ') + ' and ' + a[a.length - 1]);
const pos = (v, a, b) => Math.min(100, Math.max(0, ((v - a) / (b - a)) * 100)) + '%';

export async function mount(host) {
  let data;
  try { data = await api.models(); } catch (e) {
    host.append(el('p', { class: 'err' }, 'The model results could not load: ' + e.message));
    return;
  }

  const rows = data.search;
  const byKey = new Map(rows.map((r) => [r.key, r]));
  const dec = data.decision;
  const baseline = rows.find((r) => r.model === 'Median baseline');
  const contenders = rows.filter((r) => r !== baseline);
  const fams = [...new Set(contenders.map((r) => r.model))];
  const defRow = (f) => byKey.get('search_' + f);
  const tunRow = (f) => byKey.get('tuned_' + f) || null;
  const rowFor = (f, m) => (m === 'tuned' && tunRow(f)) || defRow(f);
  const noTuned = fams.filter((f) => !tunRow(f));

  const winner = byKey.get(dec.accuracy_winner);
  const pars = byKey.get(dec.parsimonious_pick);
  const prod = byKey.get(dec.product_pick);
  const tiedRows = rows.filter((r) => r.tied).sort((a, b) => a.rmse - b.rmse);
  const nTied = dec.n_tied ?? tiedRows.length;
  const folds = (dec.n_repeats || 3) * 5;

  // Fixed axis for both modes so a toggle never rescales the bars.
  const maxEnd = Math.max(...contenders.map((r) => r.gap_to_best + r.gap_se));
  const AX = Math.ceil(maxEnd / 500) * 500;
  const TICK = AX <= 2000 ? 500 : 1000;
  const ticks = []; for (let v = 0; v <= AX; v += TICK) ticks.push(v);
  const pctOf = (v) => (v / AX) * 100 + '%';

  const root = el('div', { class: 'ar-wrap' });
  root.style.setProperty('--ar-step', String(TICK / AX));
  host.append(root);

  // ======================================================== 1. leaderboard
  let mode = 'default';
  let hl = new Map(); // key -> flag label (set by the stepper)

  const btnD = el('button', { type: 'button', class: 'ar-seg-btn', 'aria-pressed': 'true', onclick: () => setMode('default') }, 'Default');
  const btnT = el('button', { type: 'button', class: 'ar-seg-btn', 'aria-pressed': 'false', onclick: () => setMode('tuned') }, 'Tuned');
  const seg = el('div', { class: 'ar-seg', role: 'group', 'aria-label': 'Model settings shown on the leaderboard' }, btnD, btnT);
  const status = el('p', { class: 'visually-hidden', role: 'status' });
  const board = el('ol', { class: 'ar-board ar-pre ar-intro', role: 'list', 'aria-label': 'Model families ranked by cross-validated error, lowest first' });

  const head = el('div', { class: 'ar-head', 'aria-hidden': 'true' },
    el('span', { class: 'ar-h-name' }, 'Model'),
    el('div', { class: 'ar-axis' }, ticks.map((v) => el('span', { class: 'ar-tick' + (v === 0 ? ' is-zero' : ''), style: `left:${pctOf(v)}` }, v === 0 ? '$0' : '$' + v / 1000 + 'k'))),
    el('span', { class: 'ar-h-val' }, 'RMSE'), el('span', { class: 'ar-h-r2' }, 'R²'), el('span', { class: 'ar-h-sec' }, 'CV time'));

  const rowObj = {};
  for (const f of fams) {
    const rank = el('span', { class: 'ar-rank num' });
    const mark = el('sup', { class: 'ar-mark', hidden: true, title: 'Shown at default settings' }, '*');
    const pill = el('span', { class: 'ar-pill', hidden: true }, 'tuned');
    const flag = el('span', { class: 'ar-flag', hidden: true });
    const name = el('div', { class: 'ar-name' }, rank, el('span', { class: 'ar-nm' }, f, mark), pill, flag);
    const bar = el('i', { class: 'ar-bar' });
    const whisk = el('i', { class: 'ar-whisk' });
    const track = el('div', { class: 'ar-track', 'aria-hidden': 'true' }, bar, whisk);
    const rm = el('b', { class: 'num ar-rmse' });
    const gp = el('small', { class: 'num ar-gap' });
    const val = el('div', { class: 'ar-val' }, rm, gp);
    const r2 = el('span', { class: 'num ar-r2' });
    const sc = el('span', { class: 'num ar-sec' });
    const meta = el('div', { class: 'ar-meta' },
      el('span', {}, el('span', { class: 'ar-lab' }, 'R² '), r2), el('span', {}, el('span', { class: 'ar-lab' }, 'CV time '), sc));
    const li = el('li', { class: 'ar-row' }, name, track, val, meta);
    rowObj[f] = { f, li, rank, mark, pill, flag, bar, whisk, rm, gp, r2, sc };
  }

  function fill(o, r, rank, tick) {
    o.li.dataset.key = r.key;
    o.li.style.setProperty('--i', rank - 1);
    o.rank.textContent = rank;
    o.pill.hidden = !r.tuned;
    const fallback = mode === 'tuned' && !tunRow(o.f);
    o.mark.hidden = !fallback;
    o.li.classList.toggle('is-tied', !!r.tied);
    o.li.classList.toggle('is-best', r.gap_to_best === 0);
    const g = r.gap_to_best, se = r.gap_se, lo = Math.max(0, g - se);
    o.bar.style.width = pctOf(g);
    o.whisk.hidden = !(se > 0);
    o.whisk.style.left = pctOf(lo);
    o.whisk.style.width = pctOf(g + se - lo);
    o.whisk.classList.toggle('is-clip', g - se < 0);
    if (tick && o.rm.dataset.value && Math.round(+o.rm.dataset.value) !== Math.round(r.rmse)) ticker(o.rm, r.rmse, { duration: 650 });
    else { if (o.rm._tick) o.rm._tick.cancel(); o.rm.textContent = usd(r.rmse); o.rm.dataset.value = String(r.rmse); }
    o.gp.textContent = g === 0 ? 'best' : `+${usd(g)} ±${usd(se)}`;
    o.r2.textContent = r.r2.toFixed(3);
    o.sc.textContent = secs(r.seconds);
    o.li.setAttribute('aria-label', `Rank ${rank}: ${nameOf(r.key)}${fallback ? ' (default settings, no tuned run)' : ''}. RMSE ${usd(r.rmse)}. ` +
      (g === 0 ? 'Best configuration. ' : `${usd(g)} behind the best, standard error ${usd(se)}. `) +
      `R squared ${r.r2.toFixed(3)}. Cross-validation time ${secs(r.seconds)}.` + (r.tied ? ' Statistically tied with the best.' : ''));
  }

  function flip(fn) {
    const els = fams.map((f) => rowObj[f].li);
    const first = els.map((e) => e.getBoundingClientRect().top);
    fn();
    els.forEach((e, i) => {
      const dy = first[i] - e.getBoundingClientRect().top;
      if (Math.abs(dy) > 1 && e.animate) e.animate([{ transform: `translateY(${dy}px)` }, { transform: 'none' }], { duration: 520, easing: 'cubic-bezier(0.2,0.7,0.2,1)' });
    });
  }

  const hint = el('p', { class: 'ar-hint', hidden: true });
  const hintBtn = el('button', { type: 'button', class: 'ar-link', onclick: () => setMode(mode === 'default' ? 'tuned' : 'default') });
  const hintWrap = el('div', { class: 'ar-hintwrap', hidden: true }, hint, hintBtn);
  const footFallback = el('p', { class: 'faint ar-note', hidden: true });
  const flagged = new Set();

  function applyHL() {
    const visible = new Set();
    for (const f of fams) {
      const o = rowObj[f], k = o.li.dataset.key, lab = hl.get(k);
      visible.add(k);
      o.li.classList.toggle('is-hl', !!lab);
      o.flag.hidden = !lab;
      if (lab) o.flag.textContent = lab;
      if (lab && !flagged.has(k)) { o.li.classList.remove('ar-pulse'); void o.li.offsetWidth; o.li.classList.add('ar-pulse'); }
      if (lab) flagged.add(k); else flagged.delete(k);
    }
    const hidden = [...hl.keys()].filter((k) => !visible.has(k));
    hintWrap.hidden = !hidden.length;
    if (hidden.length) {
      const other = mode === 'default' ? 'Tuned' : 'Default';
      hint.textContent = `${hl.size - hidden.length} of the ${hl.size} highlighted configurations show in this view. ${cap(list(hidden.map(nameOf)))} ${hidden.length > 1 ? 'show' : 'shows'} under ${other}.`;
      hintBtn.textContent = `Show ${other}`;
    }
  }

  function paint(animate) {
    const sel = fams.map((f) => ({ f, r: rowFor(f, mode) })).sort((a, b) => a.r.rmse - b.r.rmse);
    const apply = () => sel.forEach(({ f, r }, i) => { board.append(rowObj[f].li); fill(rowObj[f], r, i + 1, animate); });
    if (animate && !prefersReduced()) flip(apply); else apply();
    applyHL();
    footFallback.hidden = mode !== 'tuned' || !noTuned.length;
    status.textContent = `Showing ${mode} settings, ranked by error. Best: ${nameOf(sel[0].r.key)}.`;
  }

  function setMode(m) {
    if (m === mode) { applyHL(); return; }
    mode = m;
    btnD.setAttribute('aria-pressed', String(m === 'default'));
    btnT.setAttribute('aria-pressed', String(m === 'tuned'));
    paint(true);
  }

  const others = noTuned.filter((f) => f !== 'CatBoost');
  footFallback.textContent = '* Shown at default settings, because no tuned result exists: ' +
    [noTuned.includes('CatBoost') ? 'CatBoost was evaluated at default settings only' : null,
      others.length ? `${list(others)} ${others.length > 1 ? 'have' : 'has'} no tuned run` : null].filter(Boolean).join('; ') + '.';

  const legend = el('ul', { class: 'ar-legend', role: 'list' },
    el('li', {}, el('i', { class: 'ar-key ar-key-bar' }), 'Bar: how much worse than the best configuration (RMSE gap)'),
    el('li', {}, el('i', { class: 'ar-key ar-key-whisk' }), '± corrected standard error of that gap'),
    el('li', {}, el('i', { class: 'ar-key ar-key-tied' }), 'Statistically tied with the best: the data cannot separate them'));

  const baseNote = baseline ? el('p', { class: 'faint ar-note' },
    `Not drawn: the median baseline (always guess the typical salary) has an RMSE of ${usd(baseline.rmse)} and R² ${baseline.r2.toFixed(2)}. Its gap of ${usd(baseline.gap_to_best)} would run ${(baseline.gap_to_best / AX).toFixed(1)} times past the end of this axis.`) : null;

  const boardCard = el('div', { class: 'card ar-card', id: 'ar-leaderboard' },
    el('div', { class: 'ar-card-head' },
      el('div', {}, el('h3', {}, 'Leaderboard'),
        el('p', { class: 'muted ar-sub' }, `Cross-validated error, ${folds} held-out folds. Bars start at zero and show the gap to the best configuration in dollars; shorter is better.`)),
      seg),
    status, head, board, hintWrap, legend, footFallback, baseNote);
  root.append(boardCard);

  paint(false);

  const introDone = () => { board.classList.remove('ar-pre'); setTimeout(() => board.classList.remove('ar-intro'), 2000); };
  if (prefersReduced()) { board.classList.remove('ar-pre', 'ar-intro'); } else {
    onView(board, () => requestAnimationFrame(introDone), { threshold: 0.2 });
    setTimeout(introDone, 9000); // never leave bars at zero
  }

  // ======================================================== 2. arena report
  const repArena = el('div', { class: 'ar-report' });
  root.append(repArena);
  const pArena = renderReport(repArena, 'arena');

  // ======================================================== 3. decision stepper
  const span = Math.max(...tiedRows.map((r) => r.gap_to_best));
  const maxCx = Math.max(...contenders.map((r) => r.complexity));
  const minCx = Math.min(...tiedRows.map((r) => r.complexity));
  const simplest = tiedRows.filter((r) => r.complexity === minCx);
  const capable = tiedRows.filter((r) => r.interval_capable);
  const incapable = tiedRows.filter((r) => !r.interval_capable);
  const sameAsPars = prod && pars && prod.key === pars.key;

  const steps = [
    { title: 'Accuracy winner', map: new Map([[winner.key, 'winner']]), rows: [winner],
      text: `Rank every configuration by mean error over ${folds} held-out folds. ${cap(nameOf(winner.key))} comes first at ${usd(winner.rmse)} RMSE.` },
    { title: 'Tied with the winner', map: new Map(tiedRows.map((r) => [r.key, 'tied'])), rows: tiedRows,
      text: `A configuration whose gap to the winner is smaller than its own standard error cannot be told apart from it. ${nTied} of ${rows.length} configurations qualify, all within ${usd(span)} of the winner.` },
    { title: 'Simplest among the tied', map: new Map([[pars.key, 'simplest']]), rows: [pars],
      text: `${cap(String(simplest.length))} of the ${nTied} tied share the lowest complexity score among them (${minCx} on a scale up to ${maxCx}). The tie-break then prefers default settings, then lower error, which gives ${nameOf(pars.key)}, only ${usd(pars.gap_to_best)} behind the winner.` },
    { title: 'For a range: simplest tied model with quantile output', map: new Map([[prod.key, 'range pick']]), rows: [prod],
      text: `A salary range needs a model that predicts quantiles directly. ${capable.length} of the tied do (${list(capable.map((r) => nameOf(r.key)))})${incapable.length ? `; ${list(incapable.map((r) => nameOf(r.key)))} ${incapable.length > 1 ? 'do' : 'does'} not` : ''}. The same tie-break picks ${nameOf(prod.key)}${sameAsPars ? ', the same model as step 3' : ''}.` },
  ];

  let cur = 0;
  const stepN = el('p', { class: 'ar-step-n' });
  const stepT = el('h4', { class: 'ar-step-title' });
  const stepX = el('p', { class: 'ar-step-text' });
  const chips = el('ul', { class: 'ar-chips', role: 'list' });
  const stepBody = el('div', { class: 'ar-step', 'aria-live': 'polite' }, stepN, stepT, stepX, chips);
  const prev = el('button', { type: 'button', class: 'btn ar-nav-btn', onclick: () => go(cur - 1) }, '← Previous');
  const next = el('button', { type: 'button', class: 'btn primary ar-nav-btn', onclick: () => go(cur + 1) }, 'Next →');
  const dots = steps.map((s, i) => el('button', { type: 'button', class: 'ar-dot', 'aria-label': `Step ${i + 1}: ${s.title}`, onclick: () => go(i) }, el('span', { class: 'visually-hidden' }, String(i + 1))));
  const showBtn = el('button', { type: 'button', class: 'ar-link', onclick: () => boardCard.scrollIntoView({ behavior: prefersReduced() ? 'auto' : 'smooth', block: 'start' }) }, 'See it on the leaderboard');

  function go(i) {
    cur = Math.max(0, Math.min(steps.length - 1, i));
    const s = steps[cur];
    stepN.textContent = `Step ${cur + 1} of ${steps.length}`;
    stepT.textContent = s.title;
    stepX.textContent = s.text;
    chips.textContent = '';
    s.rows.forEach((r) => chips.append(el('li', { class: 'ar-chip' }, el('span', {}, cap(nameOf(r.key))), el('b', { class: 'num' }, usd(r.rmse)))));
    stepBody.classList.remove('ar-swap'); void stepBody.offsetWidth; stepBody.classList.add('ar-swap');
    dots.forEach((d, k) => { d.classList.toggle('is-on', k === cur); d.classList.toggle('is-past', k < cur); if (k === cur) d.setAttribute('aria-current', 'step'); else d.removeAttribute('aria-current'); });
    prev.disabled = cur === 0; next.disabled = cur === steps.length - 1;
    // highlight on the leaderboard; make sure the picked row is the one on screen
    hl = s.map;
    let want = mode;
    if (hl.size === 1) {
      const k = [...hl.keys()][0], f = byKey.get(k).model;
      if (k.startsWith('tuned_')) want = 'tuned'; else if (mode === 'tuned' && tunRow(f)) want = 'default';
    }
    if (want !== mode) setMode(want); else applyHL();
  }

  const stepCard = el('div', { class: 'card ar-card ar-stepper' },
    el('div', { class: 'ar-card-head' }, el('div', {}, el('h3', {}, 'The decision rule, step by step'),
      el('p', { class: 'muted ar-sub' }, 'Fixed before the results were known. Each step narrows the field, and highlights the rows it keeps on the leaderboard.')), showBtn),
    stepBody,
    el('div', { class: 'ar-step-nav' }, prev, el('div', { class: 'ar-dots', role: 'group', 'aria-label': 'Decision steps' }, dots), next));
  root.append(stepCard);

  // ======================================================== 4. verdict cards and notes
  function vcard(label, r, reason) {
    const rm = el('b', { class: 'num ar-v-rmse' }, usd(r.rmse)); rm.dataset.value = String(r.rmse);
    const c = el('article', { class: 'card ar-vcard' },
      el('p', { class: 'ar-v-label' }, label),
      el('h4', { class: 'ar-v-name' }, cap(nameOf(r.key))),
      el('p', { class: 'ar-v-nums' }, rm, el('span', { class: 'faint' }, ' RMSE'), el('span', { class: 'ar-v-gap num' }, r.gap_to_best === 0 ? 'the best' : `+${usd(r.gap_to_best)} vs best`)),
      el('p', { class: 'muted ar-v-reason' }, reason));
    if (!prefersReduced()) onView(c, () => ticker(rm, r.rmse, { from: 0, duration: 900 }), { threshold: 0.4 });
    return c;
  }
  const verdicts = el('div', { class: 'ar-verdicts' },
    vcard('Accuracy winner', winner, `Lowest error of all ${rows.length} configurations, but it blends several models and takes ${secs(winner.seconds)} to cross-validate against ${secs(pars.seconds)} for ${nameOf(pars.key)}.`),
    vcard('Parsimonious pick', pars, `Statistically tied with the winner and the simplest of the ${nTied} tied, for ${usd(pars.gap_to_best)} more error.`),
    vcard('Product pick', prod, `Simplest tied model with native quantile output, so it can give a 10-90% range${sameAsPars ? ' as well as a point estimate' : ''}.`));

  // near-tie: Ridge default
  const ridge = byKey.get('search_Ridge');
  let nearCard = null;
  if (ridge) {
    const miss = ridge.gap_to_best - ridge.gap_se;
    const mx = Math.max(ridge.gap_to_best, ridge.gap_se);
    const mrow = (label, v, cls) => el('div', { class: 'ar-mrow' }, el('span', { class: 'ar-mlab' }, label),
      el('div', { class: 'ar-mtrack', 'aria-hidden': 'true' }, el('i', { class: 'ar-mbar ' + cls, style: `width:${(v / mx) * 100}%` })), el('b', { class: 'num' }, usd2(v)));
    const meter = el('div', { class: 'ar-meter ar-pre' }, mrow('Ridge gap to best', ridge.gap_to_best, ''), mrow('Tie cutoff (its standard error)', ridge.gap_se, 'is-cut'));
    onView(meter, () => requestAnimationFrame(() => meter.classList.remove('ar-pre')), { threshold: 0.5 });
    setTimeout(() => meter.classList.remove('ar-pre'), 9000);
    if (prefersReduced()) meter.classList.remove('ar-pre');
    nearCard = el('div', { class: 'card ar-note-card' },
      el('p', { class: 'ar-v-label' }, 'Near-tie worth knowing'),
      el('h4', { class: 'ar-n-title' }, 'Ridge regression misses the cutoff by a hair'),
      el('p', { class: 'ar-n-text' },
        `Ridge (default settings) is ${usd2(ridge.gap_to_best)} behind the best, which is ${pct(ridge.gap_to_best / winner.rmse, 2)} of the error. To count as tied, that gap must be smaller than its standard error of ${usd2(ridge.gap_se)}. ` +
        (miss > 0 ? `It misses by ${usd2(miss)}: a hair, but a miss by the rule. ` : `It sits ${usd2(-miss)} inside the cutoff. `) +
        `It is also the most interpretable model here, a linear model whose coefficients can be read directly, and it runs in ${secs(ridge.seconds)}. If explaining every prediction matters more than the last ${usd(ridge.gap_to_best)} of error, it is a defensible choice.`),
      meter);
  }

  // LightGBM exception + quantile chart
  const iv = data.intervals || [];
  const app = iv.find((r) => r.model === 'LightGBM');
  let exCard = null;
  if (app && iv.length) {
    const lowP = Math.min(...iv.map((r) => r.pinball)), highP = Math.max(...iv.map((r) => r.pinball));
    const A0 = 0.70, A1 = 0.85, TARGET = 0.80;
    const WMAX = Math.ceil(Math.max(...iv.map((r) => r.median_width)) / 20000) * 20000;
    const wTicks = [0, 1, 2, 3].map((i) => (WMAX / 3) * i);
    const qrow = (r, track, value) => el('div', { class: 'ar-qrow' + (r.model === 'LightGBM' ? ' is-app' : '') }, el('span', { class: 'ar-qname' }, r.model), track, value);
    const covRows = iv.map((r) => qrow(r,
      el('div', { class: 'ar-qtrack', 'aria-hidden': 'true' }, el('i', { class: 'ar-qtarget', style: `left:${pos(TARGET, A0, A1)}` }), el('i', { class: 'ar-qdot', style: `left:${pos(r.coverage, A0, A1)}` })),
      el('b', { class: 'num ar-qval' }, pct(r.coverage))));
    const widRows = iv.map((r) => qrow(r,
      el('div', { class: 'ar-qtrack', 'aria-hidden': 'true' }, el('i', { class: 'ar-qbar', style: `width:${(r.median_width / WMAX) * 100}%` })),
      el('b', { class: 'num ar-qval' }, usd(r.median_width))));
    const axisRow = (labels) => el('div', { class: 'ar-qrow ar-qaxis', 'aria-hidden': 'true' }, el('span'),
      el('div', { class: 'ar-qtrack' }, labels.map(([p, t]) => el('span', { class: 'ar-tick', style: `left:${p}` }, t))), el('span'));
    const covAxis = axisRow([0.70, 0.75, 0.80, 0.85].map((v) => [pos(v, A0, A1), Math.round(v * 100) + '%']));
    const widAxis = axisRow(wTicks.map((v) => [(v / WMAX) * 100 + '%', v === 0 ? '$0' : kfmt(v)]));
    const charts = el('div', { class: 'ar-qcharts ar-pre' },
      el('figure', { class: 'ar-qfig' },
        el('figcaption', {}, el('b', {}, 'Coverage of the 10-90% range'), el('span', { class: 'faint' }, ' Share of held-out salaries that fell inside the range. Axis runs from 70% to 85%, not from zero; the dashed line is the 80% target.')),
        el('div', { class: 'ar-qrows' }, covRows, covAxis)),
      el('figure', { class: 'ar-qfig' },
        el('figcaption', {}, el('b', {}, 'Median width of the range'), el('span', { class: 'faint' }, ` Narrower is sharper. Axis starts at zero and runs to ${usd(WMAX)}.`)),
        el('div', { class: 'ar-qrows' }, widRows, widAxis)));
    onView(charts, () => requestAnimationFrame(() => charts.classList.remove('ar-pre')), { threshold: 0.3 });
    setTimeout(() => charts.classList.remove('ar-pre'), 9000);
    if (prefersReduced()) charts.classList.remove('ar-pre');
    exCard = el('div', { class: 'card ar-note-card' },
      el('p', { class: 'ar-v-label' }, 'A deliberate exception'),
      el('h4', { class: 'ar-n-title' }, 'The web app uses LightGBM quantile models: a deliberate exception to the rule'),
      el('p', { class: 'ar-n-text' },
        `The rule's product pick is ${nameOf(prod.key)}, but the estimator serves LightGBM. On the 10-90% range it has the lowest pinball loss of the ${iv.length} quantile models (${usd(app.pinball)}), its range covered ${pct(app.coverage)} of held-out salaries against the 80% target, and its median width is ${usd(app.median_width)}. ` +
        `The ${iv.length} models are within ${pct(highP / lowP - 1)} of each other on pinball loss, and the rule never scored interval quality, so the exception is stated rather than hidden.`),
      charts);
  }

  root.append(verdicts);
  if (nearCard) root.append(nearCard);
  if (exCard) root.append(exCard);

  // ======================================================== 5. decision report
  const repDec = el('div', { class: 'ar-report' });
  root.append(repDec);
  const pDec = renderReport(repDec, 'decision');

  go(0);
  await Promise.all([pArena, pDec]);
}
