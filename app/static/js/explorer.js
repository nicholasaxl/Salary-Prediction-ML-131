// Explorer: group and filter the salary rows; histogram plus one range bar per group.
import { el, usd, kfmt, renderReport, clamp } from './util.js';
import { api } from './api.js';

const GROUPS = [
  ['job_title', 'Job title'], ['experience_level', 'Experience level'], ['company_location', 'Company location'],
  ['company_size', 'Company size'], ['employment_type', 'Employment type'], ['work_year', 'Year'], ['remote_ratio', 'Work setup'],
];
const FILTERS = [
  ['job_title', 'Job title'], ['experience_level', 'Experience'], ['company_location', 'Location'],
  ['company_size', 'Company size'], ['employment_type', 'Employment type'], ['remote_ratio', 'Work setup'], ['work_year', 'Year'],
];
const GROUP_LABEL = Object.fromEntries(GROUPS);

function niceStep(max) {
  for (const s of [10000, 20000, 25000, 50000, 100000]) if (max / s <= 6) return s;
  return 100000;
}

export async function mount(host) {
  const OPT = (await api.options()).options;
  const items = { ...OPT, work_year: [...OPT.work_year].sort((a, b) => b - a).map((y) => ({ value: y, label: String(y) })) };
  const state = { group_by: 'job_title', min_n: 10 };
  const sel = {};

  // ---- filter bar
  const mkSelect = (id, options, any) => {
    const s = el('select', { id });
    if (any) s.append(el('option', { value: '' }, any));
    for (const o of options) s.append(el('option', { value: o.value }, o.n != null ? `${o.label} (${o.n.toLocaleString('en-US')})` : o.label));
    return s;
  };
  const field = (label, control, cls = '') => el('div', { class: 'ex-field ' + cls }, el('label', { for: control.id }, label), control);

  const groupSel = mkSelect('ex-group', GROUPS.map(([value, label]) => ({ value, label })));
  groupSel.value = state.group_by;
  groupSel.addEventListener('change', () => { state.group_by = groupSel.value; refresh(); });

  const filterFields = FILTERS.map(([key, label]) => {
    const s = mkSelect('ex-f-' + key, items[key], 'All');
    s.addEventListener('change', () => { s.classList.toggle('is-set', s.value !== ''); refresh(); });
    sel[key] = s;
    return field(label, s);
  });

  const minOut = el('output', { class: 'num', for: 'ex-min' }, String(state.min_n));
  const minRange = el('input', { id: 'ex-min', type: 'range', min: '1', max: '100', value: String(state.min_n), 'aria-valuetext': state.min_n + ' rows' });
  let minTimer = 0;
  minRange.addEventListener('input', () => {
    minOut.textContent = minRange.value;
    minRange.setAttribute('aria-valuetext', minRange.value + ' rows');
    clearTimeout(minTimer);
    minTimer = setTimeout(() => { state.min_n = Number(minRange.value); refresh(); }, 120);
  });
  const clearBtn = el('button', { type: 'button', class: 'ex-clear', onclick: clearFilters }, 'Clear filters');

  const bar = el('form', { class: 'card ex-bar', 'aria-label': 'Group and filter', onsubmit: (e) => e.preventDefault() },
    field('Group by', groupSel, 'ex-group'),
    el('div', { class: 'ex-filters' }, filterFields),
    el('div', { class: 'ex-lower' },
      el('div', { class: 'ex-field ex-min' }, el('label', { for: 'ex-min' }, 'Minimum rows per group: ', minOut), minRange),
      clearBtn));

  // ---- output
  const title = el('h3', { class: 'ex-title' }, 'Salary by job title');
  const summary = el('p', { class: 'ex-summary muted', 'aria-live': 'polite' }, 'Loading…');
  const hist = el('div', { class: 'ex-hist', role: 'img', 'aria-label': 'Salary distribution' });
  const histMed = el('i', { class: 'ex-hmed' }, el('span', { class: 'num' }));
  const histAxis = el('div', { class: 'ex-haxis num', 'aria-hidden': 'true' }, el('span'), el('span'), el('span'));
  const histWrap = el('div', { class: 'ex-histwrap' }, hist, histMed, histAxis);
  const axisEl = el('div', { class: 'ex-axis', 'aria-hidden': 'true' });
  const rowsEl = el('ul', { class: 'ex-rows', 'aria-label': 'Groups' });
  const empty = el('p', { class: 'ex-empty muted', hidden: true });
  const legend = el('div', { class: 'ex-legend' },
    el('span', {}, el('i', { class: 'ex-key iqr' }), 'middle half (25th to 75th percentile)'),
    el('span', {}, el('i', { class: 'ex-key med' }), 'median'),
    el('span', { class: 'faint' }, 'Select a row to filter to it'));

  const out = el('div', { class: 'card ex-out' },
    el('div', { class: 'ex-head' }, title),
    summary, histWrap, legend, axisEl, rowsEl, empty);

  host.append(el('div', { class: 'ex' }, bar, out));

  // ---- behaviour
  let seq = 0;
  const prev = new Map();
  let bins = [];

  function activeFilters() {
    const f = {};
    for (const [k] of FILTERS) if (sel[k].value !== '') f[k] = sel[k].value;
    return f;
  }
  function clearFilters() {
    for (const [k] of FILTERS) { sel[k].value = ''; sel[k].classList.remove('is-set'); }
    refresh();
  }

  async function refresh() {
    const mine = ++seq;
    const filters = activeFilters();
    clearBtn.disabled = Object.keys(filters).length === 0;
    out.classList.add('is-loading');
    title.textContent = 'Salary by ' + GROUP_LABEL[state.group_by].toLowerCase();
    let r;
    try {
      r = await api.explore({ group_by: state.group_by, min_n: state.min_n, ...filters });
    } catch (e) { r = { detail: e.message }; }
    if (mine !== seq) return;
    out.classList.remove('is-loading');
    if (r.detail) { showEmpty('Could not load the data: ' + (typeof r.detail === 'string' ? r.detail : 'invalid request')); return; }
    draw(r);
  }

  function showEmpty(msg) {
    summary.textContent = '';
    empty.textContent = msg;
    empty.hidden = false;
    rowsEl.replaceChildren(); axisEl.replaceChildren();
    prev.clear();
    drawHist(null);
  }

  function draw(r) {
    if (!r.rows) { showEmpty('No rows match these filters. Clear a filter to see data again.'); summary.textContent = '0 rows match.'; return; }
    const o = r.overall;
    summary.textContent = `${r.rows.toLocaleString('en-US')} rows, median ${usd(o.median)} (middle half ${usd(o.p25)} to ${usd(o.p75)})`
      + (r.hidden_groups ? `. ${r.hidden_groups} small group${r.hidden_groups > 1 ? 's' : ''} hidden below ${state.min_n} rows.` : '.');
    drawHist(r);
    if (!r.groups.length) {
      empty.textContent = `Every group has fewer than ${state.min_n} rows. Lower the minimum with the slider.`;
      empty.hidden = false;
      rowsEl.replaceChildren(); axisEl.replaceChildren(); prev.clear();
      return;
    }
    empty.hidden = true;
    drawRows(r.groups);
  }

  function drawHist(r) {
    if (!r) { hist.replaceChildren(); bins = []; histMed.hidden = true; return; }
    const { counts, edges } = r.histogram;
    if (bins.length !== counts.length) {
      hist.replaceChildren(...counts.map(() => el('i', { class: 'ex-bin' })));
      bins = [...hist.children];
    }
    const top = Math.max(...counts, 1);
    counts.forEach((c, i) => {
      bins[i].style.height = Math.max(c > 0 ? 3 : 0, (c / top) * 100) + '%';
      bins[i].title = `${kfmt(edges[i])} to ${kfmt(edges[i + 1])}: ${c.toLocaleString('en-US')} rows`;
    });
    hist.setAttribute('aria-label', `Salary distribution of ${r.rows.toLocaleString('en-US')} rows, from ${kfmt(edges[0])} to ${kfmt(edges[edges.length - 1])}, median ${usd(r.overall.median)}`);
    const [a, m, b] = histAxis.children;
    a.textContent = kfmt(edges[0]); b.textContent = kfmt(edges[edges.length - 1]);
    m.textContent = kfmt((edges[0] + edges[edges.length - 1]) / 2);
    histMed.hidden = false;
    histMed.style.left = (clamp((r.overall.median - edges[0]) / (edges[edges.length - 1] - edges[0]), 0, 1) * 100).toFixed(2) + '%';
    histMed.firstChild.textContent = 'median ' + kfmt(r.overall.median);
  }

  function drawRows(groups) {
    const hi = Math.max(...groups.map((g) => g.p75)) * 1.05;
    const step = niceStep(hi);
    const pct = (v) => ((v / hi) * 100).toFixed(2);
    const hadFocus = rowsEl.contains(document.activeElement);

    axisEl.replaceChildren(el('span'), el('span', { class: 'ex-axis-scale num' },
      Array.from({ length: Math.floor(hi / step) + 1 }, (_, i) => el('i', { style: `left:${pct(i * step)}%` }, i === 0 ? '$0' : kfmt(i * step)))),
      el('span'), el('span'));
    rowsEl.style.setProperty('--tw', ((step / hi) * 100).toFixed(3) + '%');

    const pending = [];
    const seen = new Set();
    const rows = groups.map((g) => {
      const key = String(g.key);
      seen.add(key);
      const p = prev.get(key) || { l: Number(pct(g.p25)) + (Number(pct(g.p75)) - Number(pct(g.p25))) / 2, w: 0, m: Number(pct(g.median)) };
      const now = { l: Number(pct(g.p25)), w: Number(pct(g.p75 - g.p25)), m: Number(pct(g.median)) };
      const iqr = el('i', { class: 'ex-iqr' });
      const med = el('i', { class: 'ex-med' });
      const place = (s) => { iqr.style.left = s.l + '%'; iqr.style.width = s.w + '%'; med.style.left = s.m + '%'; };
      place(p);
      pending.push(() => place(now));
      prev.set(key, now);
      const desc = `${g.label}: median ${usd(g.median)}, middle half ${usd(g.p25)} to ${usd(g.p75)}, ${g.n.toLocaleString('en-US')} rows. Select to filter to this group.`;
      return el('li', {},
        el('button', {
          type: 'button', class: 'ex-row', 'data-key': key, 'aria-label': desc,
          title: `Range ${usd(g.min)} to ${usd(g.max)}, mean ${usd(g.mean)}`,
          onclick: () => pick(key),
        },
        el('span', { class: 'ex-name' }, g.label),
        el('span', { class: 'ex-track', 'aria-hidden': 'true' }, iqr, med),
        el('span', { class: 'ex-med-v num' }, kfmt(g.median)),
        el('span', { class: 'ex-n num' }, g.n.toLocaleString('en-US'))));
    });
    for (const k of [...prev.keys()]) if (!seen.has(k)) prev.delete(k);
    rowsEl.replaceChildren(...rows);
    void rowsEl.offsetWidth; // commit the starting positions so bars slide from where they were
    pending.forEach((fn) => fn());
    if (hadFocus) rowsEl.querySelector('button')?.focus({ preventScroll: true });
  }

  function pick(key) {
    const s = sel[state.group_by];
    if (!s) return;
    s.value = key;
    s.classList.toggle('is-set', s.value !== '');
    refresh();
  }

  clearBtn.disabled = true;
  refresh();
  await renderReport(host, 'explorer');
}
