// Estimator: describe a role, get a range, the pay-band ribbon, observed peers and what moved the number.
import { el, usd, usdK, ticker, renderReport, clamp } from './util.js';
import { api } from './api.js';
import { ribbon, ribbonLegend } from './ribbon.js';

const FIELDS = [
  ['job_title', 'Job title'],
  ['experience_level', 'Experience level'],
  ['company_location', 'Company location'],
  ['company_size', 'Company size'],
  ['employment_type', 'Employment type'],
  ['remote_ratio', 'Work setup'],
  ['work_year', 'Year'],
];

const DEFAULTS = { job_title: 'Data Scientist', experience_level: 'SE', company_location: 'US', company_size: 'M', employment_type: 'FT', remote_ratio: '0' };

const PRESETS = [
  { name: 'Entry analyst in India', set: { job_title: 'Data Analyst', experience_level: 'EN', company_location: 'IN' } },
  { name: 'Senior ML engineer, US', set: { job_title: 'Machine Learning Engineer', experience_level: 'SE', company_location: 'US' } },
  { name: 'A rare job title', set: { job_title: '3D Computer Vision Researcher', experience_level: 'SE', company_location: 'US' } },
];

const signed = (v) => (v < 0 ? '−' : '+') + usd(Math.abs(v));
const scaleMax = (high) => Math.max(300000, Math.ceil((high * 1.08) / 50000) * 50000);

export async function mount(host) {
  const optRes = await api.options();
  const OPT = optRes.options;
  const selects = {};
  const items = {
    ...OPT,
    work_year: [...OPT.work_year].sort((a, b) => b - a).map((y) => ({ value: y, label: String(y) })),
  };

  // ---- form
  const fieldEl = ([key, label]) => {
    const sel = el('select', { id: 'es-' + key, name: key });
    for (const o of items[key]) sel.append(el('option', { value: o.value }, o.n != null ? `${o.label} (${o.n.toLocaleString('en-US')})` : o.label));
    const want = key === 'work_year' ? String(Math.max(...OPT.work_year)) : DEFAULTS[key];
    sel.value = [...sel.options].some((o) => o.value === want) ? want : sel.options[0].value;
    sel.addEventListener('change', schedule);
    selects[key] = sel;
    return el('div', { class: 'es-field' }, el('label', { for: sel.id }, label), sel);
  };

  const presetBtns = PRESETS.filter((p) => Object.entries(p.set).every(([k, v]) => items[k].some((o) => o.value === v))).map((p) =>
    el('button', {
      type: 'button', class: 'es-chip',
      onclick: () => { for (const [k, v] of Object.entries(p.set)) selects[k].value = v; schedule(); },
    }, p.name));

  const form = el('form', { class: 'card es-form', 'aria-label': 'Role details', onsubmit: (e) => e.preventDefault() },
    el('h3', { class: 'es-h' }, 'Describe the role'),
    FIELDS.map(fieldEl),
    presetBtns.length ? el('div', { class: 'es-presets' }, el('span', { class: 'faint' }, 'Or try'), presetBtns) : null);

  // ---- result
  const loEl = el('span', { class: 'es-lo num' }, '…');
  const hiEl = el('span', { class: 'es-hi num' }, '…');
  const midEl = el('b', { class: 'num es-mid' }, '…');
  const srEl = el('p', { class: 'visually-hidden', role: 'status', 'aria-live': 'polite' });
  const errEl = el('div', { class: 'es-error', role: 'alert', hidden: true });
  const ribHost = el('div', { class: 'es-ribbon' });
  const legendPeers = ribbonLegend(true);
  const legendPlain = ribbonLegend(false);
  legendPlain.hidden = true;
  const peersEl = el('p', { class: 'es-peers' });
  const warnEl = el('div', { class: 'es-warn', hidden: true });
  const baseEl = el('p', { class: 'es-base muted' });
  const factorList = el('ul', { class: 'es-factors', 'aria-label': 'Factors that moved this estimate' });

  const main = el('div', { class: 'es-main' },
    el('p', { class: 'es-kicker' }, 'Estimated salary, 80% range'),
    el('p', { class: 'es-range', 'aria-hidden': 'true' }, loEl, el('span', { class: 'es-to' }, ' to '), hiEl),
    el('p', { class: 'es-unit faint' }, 'per year, USD'),
    ribHost, legendPeers, legendPlain,
    el('p', { class: 'es-likely' }, 'Most likely about ', midEl, ' a year.'),
    peersEl);

  const why = el('div', { class: 'es-why' },
    el('h3', { class: 'es-h' }, 'What moves this estimate'),
    baseEl, factorList);

  const result = el('div', { class: 'card es-result', 'aria-busy': 'true' },
    el('div', { class: 'es-bar', 'aria-hidden': 'true' }),
    errEl, warnEl, main, srEl, why);

  const layout = el('div', { class: 'es' }, form, result);
  host.append(layout);

  // ---- behaviour
  let timer = 0, seq = 0, rb = null;
  const prevFactor = new Map();

  function body() {
    const b = {};
    for (const [k] of FIELDS) b[k] = selects[k].value;
    b.remote_ratio = Number(b.remote_ratio);
    b.work_year = Number(b.work_year);
    return b;
  }

  function schedule() {
    clearTimeout(timer);
    result.classList.add('is-loading');
    timer = setTimeout(run, 80);
  }

  async function run() {
    const mine = ++seq;
    result.classList.add('is-loading');
    try {
      const r = await api.predict(body());
      if (mine !== seq) return;
      errEl.hidden = true;
      result.classList.remove('is-stale');
      show(r);
    } catch (e) {
      if (mine !== seq) return;
      errEl.textContent = 'Could not estimate this profile: ' + e.message;
      errEl.hidden = false;
      result.classList.add('is-stale');
    }
    if (mine === seq) { result.classList.remove('is-loading'); result.setAttribute('aria-busy', 'false'); }
  }

  function show(r) {
    const first = !rb;
    const hasPeers = !!(r.peers && r.peers.n > 0);
    const dur = first ? 1100 : 600;
    ticker(loEl, r.low, { duration: dur, format: usdK });
    ticker(hiEl, r.high, { duration: dur, format: usdK });
    ticker(midEl, r.mid, { duration: dur, format: usdK });

    const ropts = { low: r.low, mid: r.mid, high: r.high, peers: hasPeers ? r.peers : null, max: scaleMax(r.high) };
    if (!rb) rb = ribbon(ribHost, ropts); else rb.update(ropts);
    legendPeers.hidden = !hasPeers;
    legendPlain.hidden = hasPeers;

    if (hasPeers) {
      const p = r.peers;
      peersEl.textContent = `${p.n.toLocaleString('en-US')} people in the data with the ${p.match} reported a median of ${usd(p.median)}. The middle half earned ${usd(p.p25)} to ${usd(p.p75)}.`;
    } else peersEl.textContent = 'No one in the data matches closely enough to compare against.';

    warnEl.replaceChildren();
    if (r.warnings && r.warnings.length) {
      warnEl.append(el('b', { class: 'es-warn-tag' }, 'Take care'), ...r.warnings.map((w) => el('p', {}, w)));
      warnEl.hidden = false;
    } else warnEl.hidden = true;

    baseEl.textContent = `Starting from a typical salary of ${usd(r.baseline)}, each factor below adds or subtracts. The factors sum to the most likely value.`;
    drawFactors(r.factors);

    srEl.textContent = `Estimated salary ${usdK(r.low)} to ${usdK(r.high)} a year, most likely about ${usdK(r.mid)}.` +
      (r.warnings && r.warnings.length ? ' Warning: ' + r.warnings.join(' ') : '');
  }

  function drawFactors(factors) {
    const top = Math.max(...factors.map((f) => Math.abs(f.effect)), 8000);
    const pending = [];
    const rows = factors.map((f) => {
      const prev = prevFactor.get(f.feature) || { s: 0, v: 0 };
      const s = clamp((f.effect / top) * 50, -50, 50);
      const bar = el('b', { class: 'es-fbar ' + (f.effect < 0 ? 'neg' : 'pos') });
      const val = el('span', { class: 'es-fval num ' + (f.effect < 0 ? 'neg' : 'pos') }, signed(prev.v));
      const place = (sv) => { bar.style.left = (sv >= 0 ? 50 : 50 + sv) + '%'; bar.style.width = Math.abs(sv) + '%'; };
      place(prev.s);
      pending.push(() => { place(s); ticker(val, f.effect, { from: prev.v, format: signed, duration: 700 }); });
      prevFactor.set(f.feature, { s, v: f.effect });
      return el('li', { class: 'es-factor' },
        el('span', { class: 'es-fname' }, f.label, ' ', el('span', { class: 'es-fcat' }, f.value)),
        el('span', { class: 'es-ftrack', role: 'img', 'aria-label': `${f.label} ${f.value}: ${signed(f.effect)}` }, bar),
        val);
    });
    factorList.replaceChildren(...rows);
    void factorList.offsetWidth; // commit the starting widths so the bars grow from where they were
    pending.forEach((fn) => fn());
  }

  run();
  await renderReport(host, 'estimator');
}
