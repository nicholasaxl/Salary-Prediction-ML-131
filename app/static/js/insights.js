// Insights: pay premiums with intervals, pay-band heatmap, model reliability, and the decisions they support.
import { el, usd, kfmt, tween, onView, renderReport, clamp, ease } from './util.js';
import { api } from './api.js';

const CAP = 140; // axis is symmetric, +/- this many percent
const GROUP_ORDER = ['Level', 'Location', 'Role', 'Company', 'Setup', 'Time trend'];
const LEVELS = ['Entry', 'Mid', 'Senior', 'Executive'];
const MIN_CELL = 10;
const THIN = 30;
const TARGET = 0.8;

const MINUS = '−';
const sgn = (v) => { const r = Math.round(v); return r === 0 ? '0' : (r < 0 ? MINUS : '+') + Math.abs(r); };
const plain = (v) => { const r = Math.round(v); return r < 0 ? MINUS + Math.abs(r) : String(Math.abs(r) === 0 ? 0 : r); };
const premText = (pct, lo, hi) => `${sgn(pct)}% (${plain(lo)}, ${plain(hi)})`;
const axisPos = (v) => (50 + (clamp(v, -CAP, CAP) / CAP) * 50);

// the same failure-safe "play once when seen" used by the premium and reliability charts
function playOnce(node, fn) {
  let done = false;
  const go = () => { if (!done) { done = true; fn(); } };
  onView(node, go, { threshold: 0.2 });
  setTimeout(go, 3500);
}

export async function mount(host) {
  const [d, rep] = await Promise.all([api.insights(), api.report().catch(() => null)]);
  host.append(premiumCard(d), heatCard(d), reliabilityCard(d));
  const decisions = rep && rep.sections.insights && rep.sections.insights.decisions;
  if (decisions && decisions.length) host.append(decisionBlock(decisions));
  await renderReport(host, 'insights');
}

// ------------------------------------------------------------------ (a) premiums
function parsePremium(p) {
  if (/^Per year/i.test(p.factor)) return { ...p, group: 'Time trend', name: 'Each extra year', ref: '' };
  const [group, rest] = p.factor.split(': ');
  const [name, ref] = (rest || '').split(' vs ');
  return { ...p, group, name: name || rest || p.factor, ref: ref ? 'vs ' + ref : '', refName: ref };
}

function premiumCard(d) {
  const prem = d.premiums.map(parsePremium);
  const refs = {};
  for (const p of prem) if (p.refName && !refs[p.group]) refs[p.group] = p.refName;
  const refSentence = GROUP_ORDER.filter((g) => refs[g]).map((g) => `${refs[g]} for ${g === 'Company' ? 'company size' : g.toLowerCase()}`).join(', ');

  const rows = []; // { pct, lo, hi, node, val }
  const body = el('div', { class: 'in-prem in-pre' });
  const axis = el('div', { class: 'in-paxis', 'aria-hidden': 'true' },
    el('span', {}),
    el('span', { class: 'in-paxis-scale' },
      [-100, 0, 100].map((v) => el('i', { style: `left:${axisPos(v)}%` }, v === 0 ? '0' : (v < 0 ? MINUS : '+') + Math.abs(v) + '%'))),
    el('span', {}));
  body.append(axis);

  for (const g of GROUP_ORDER) {
    const items = prem.filter((p) => p.group === g).sort((a, b) => a.pct - b.pct);
    if (!items.length) continue;
    const ref = refs[g];
    body.append(el('div', { class: 'in-pgroup' },
      el('h4', {}, g), ref ? el('span', { class: 'faint' }, 'compared with ' + ref) : el('span', { class: 'faint' }, 'change per year, ' + d.years[0] + ' to ' + d.years[1])));
    const list = el('ul', { class: 'in-plist', 'aria-label': g + ' premiums' });
    items.forEach((p, i) => {
      const tone = !p.significant ? 'is-null' : p.pct > 0 ? 'is-pos' : 'is-neg';
      const lo = axisPos(p.lo), hi = axisPos(p.hi);
      const val = el('span', { class: 'in-pval num' }, premText(p.pct, p.lo, p.hi));
      const label = `${g}: ${p.name} ${p.ref || ''}. ${sgn(p.pct)} percent, 90% interval ${plain(p.lo)} to ${plain(p.hi)}. ` +
        (p.significant ? (p.pct > 0 ? 'Reliably higher.' : 'Reliably lower.') : 'Interval crosses zero, no reliable difference.');
      list.append(el('li', { class: 'in-prow ' + tone, style: `--x:${axisPos(p.pct).toFixed(2)}%;--l:${lo.toFixed(2)}%;--w:${(hi - lo).toFixed(2)}%;--i:${i}` },
        el('span', { class: 'in-pname', title: `${p.n.toLocaleString('en-US')} rows` },
          el('b', {}, p.name), el('span', { class: 'faint' }, `${p.n.toLocaleString('en-US')} rows`)),
        el('span', { class: 'in-pci', role: 'img', 'aria-label': label }, el('i', { class: 'in-pline' }), el('i', { class: 'in-pdot' })),
        val));
      rows.push({ p, val });
    });
    body.append(list);
  }

  const legend = el('div', { class: 'in-legend' },
    el('span', {}, el('i', { class: 'in-key is-pos' }), 'reliably higher'),
    el('span', {}, el('i', { class: 'in-key is-neg' }), 'reliably lower'),
    el('span', {}, el('i', { class: 'in-key is-null' }), 'no reliable difference (interval crosses zero)'));

  const card = el('div', { class: 'card in-card', id: 'in-premiums' },
    el('h3', { class: 'in-h' }, 'What moves salary'),
    el('p', { class: 'in-sub muted' },
      `Each row is the pay difference against a reference group (${refSentence}), with the other factors held constant. `
      + `Based on ${d.rows.toLocaleString('en-US')} salaries from ${d.years[0]} to ${d.years[1]}, median ${usd(d.median_salary)}. Lines show a 90% interval; the axis stops at ±${CAP}%.`),
    legend, body);

  playOnce(card, () => {
    body.classList.remove('in-pre');
    tween(1100, (t) => rows.forEach(({ p, val }) => { val.textContent = premText(p.pct * t, p.lo * t, p.hi * t); }), ease.out);
  });
  return card;
}

// ------------------------------------------------------------------ (b) heatmap
function heatCard(d) {
  const markets = ['US', 'Outside US'];
  const families = [...new Set(d.bands.map((b) => b.family))].sort();
  const vmax = Math.max(...d.bands.map((b) => b.median));
  const find = (f, l, m) => d.bands.find((b) => b.family === f && b.level === l && b.market === m);
  let market = 'US';

  const readout = el('p', { class: 'in-readout muted', 'aria-live': 'polite' }, 'Hover or focus a cell to see the middle-50% band.');
  const tbody = el('tbody');
  const table = el('table', { class: 'in-heat' },
    el('caption', { class: 'visually-hidden' }, 'Median salary in thousands of USD by role family and level'),
    el('thead', {}, el('tr', {}, el('th', { scope: 'col' }, el('span', { class: 'visually-hidden' }, 'Role family')), LEVELS.map((l) => el('th', { scope: 'col' }, l)))),
    tbody);

  const say = (txt) => { readout.textContent = txt; };
  function cell(f, l) {
    const b = find(f, l, market);
    if (!b || b.n < MIN_CELL) {
      return el('td', { class: 'in-na' }, el('span', { 'aria-label': `${f}, ${l}, ${market}: not enough data, fewer than ${MIN_CELL} people` }, '–'));
    }
    const band = `${usd(b.p25)} to ${usd(b.p75)}`;
    const text = `${f}, ${l}, ${market}: middle 50% ${band}, median ${usd(b.median)}, ${b.n} people`;
    const c = el('span', {
      class: 'in-cell', tabindex: '0', title: `Middle 50%: ${band}`, 'aria-label': text,
      style: `--shade:${(6 + 48 * (b.median / vmax)).toFixed(1)}%`,
    }, el('b', { class: 'num' }, kfmt(b.median)), el('i', { class: 'num' }, 'n ' + b.n));
    c.addEventListener('mouseenter', () => say(text));
    c.addEventListener('focus', () => say(text));
    return el('td', {}, c);
  }
  function draw(animate) {
    tbody.replaceChildren(...families.map((f, i) => el('tr', { class: animate ? 'in-rowin' : '', style: `--i:${i}` },
      el('th', { scope: 'row' }, f), LEVELS.map((l) => cell(f, l)))));
    say('Hover or focus a cell to see the middle-50% band.');
  }
  draw(false);

  const btns = markets.map((m) => el('button', {
    type: 'button', class: 'in-seg', 'aria-pressed': String(m === market),
    onclick: () => {
      if (m === market) return;
      market = m;
      btns.forEach((b, i) => b.setAttribute('aria-pressed', String(markets[i] === market)));
      draw(true);
    },
  }, m));

  return el('div', { class: 'card in-card', id: 'in-bands' },
    el('div', { class: 'in-headrow' },
      el('div', {}, el('h3', { class: 'in-h' }, 'Median pay by role and level'),
        el('p', { class: 'in-sub muted' }, `Thousands of USD, with the number of people in each cell. A dash means fewer than ${MIN_CELL} people, too few to quote.`)),
      el('div', { class: 'in-seggroup', role: 'group', 'aria-label': 'Market' }, btns)),
    el('div', { class: 'in-scroll' }, table),
    readout);
}

// ------------------------------------------------------------------ (c) reliability
function reliabilityCard(d) {
  const segs = d.reliability.filter((r) => r.group !== 'All');
  const all = d.reliability.find((r) => r.group === 'All');
  const weak = segs.filter((r) => r.coverage < 0.76 && r.n >= THIN).sort((a, b) => a.coverage - b.coverage);
  const pct = (v) => Math.round(v * 100) + '%';
  const list = (arr) => (arr.length <= 1 ? arr.join('') : arr.slice(0, -1).join(', ') + ' and ' + arr[arr.length - 1]);

  const sentence = weak.length
    ? `Coverage falls furthest below the ${pct(TARGET)} target for ${list(weak.map((r) => `${r.segment} (${pct(r.coverage)})`))}. In those groups real salaries land outside the range more often than they should, so treat the range there as optimistic.`
    : `Every group with ${THIN} or more people lands close to the ${pct(TARGET)} target.`;

  const bars = [];
  const body = el('div', { class: 'in-rel in-pre' });
  body.append(el('div', { class: 'in-raxis', 'aria-hidden': 'true' },
    el('span', {}),
    el('span', { class: 'in-raxis-scale' },
      el('i', { style: 'left:0%' }, '0%'), el('i', { class: 'is-target', style: `left:${TARGET * 100}%` }, pct(TARGET) + ' target'), el('i', { style: 'left:100%' }, '100%')),
    el('span', {}), el('span', {})));
  ['Location', 'Level', 'Role'].forEach((g) => {
    const rows = segs.filter((r) => r.group === g).sort((a, b) => b.n - a.n);
    if (!rows.length) return;
    body.append(el('h4', { class: 'in-rgroup' }, g));
    const ul = el('ul', { class: 'in-rlist', 'aria-label': g + ' coverage' });
    for (const r of rows) {
      const thin = r.n < THIN, off = r.coverage < 0.76 && !thin;
      ul.append(el('li', { class: 'in-rrow' + (thin ? ' is-thin' : '') + (off ? ' is-off' : ''), style: `--c:${(r.coverage * 100).toFixed(1)}%` },
        el('span', { class: 'in-rname' }, r.segment),
        el('span', { class: 'in-rtrack', role: 'img', 'aria-label': `${r.segment}: ${pct(r.coverage)} of real salaries inside the range, ${r.n} people${thin ? ', too few to be reliable' : ''}` },
          el('i', { class: 'in-rfill' }), el('i', { class: 'in-rline' })),
        el('span', { class: 'in-rpct num' }, pct(r.coverage)),
        el('span', { class: 'in-rn num' }, 'n ' + r.n.toLocaleString('en-US'))));
    }
    body.append(ul);
  });

  const legend = el('div', { class: 'in-legend' },
    el('span', {}, el('i', { class: 'in-key is-blue' }), 'groups with 30+ people'),
    el('span', {}, el('i', { class: 'in-key is-gold' }), 'fewer than 30 people (shaky)'),
    el('span', {}, el('i', { class: 'in-key is-line' }), pct(TARGET) + ' target'));

  const card = el('div', { class: 'card in-card', id: 'in-reliability' },
    el('h3', { class: 'in-h' }, 'Where the model can be trusted'),
    el('p', { class: 'in-sub muted' }, `Share of real salaries that landed inside the model's 80% range, tested on rows it had not seen. Near ${pct(TARGET)} is well calibrated.`),
    legend, body,
    el('p', { class: 'in-finding' }, sentence),
    all ? el('p', { class: 'in-sub muted' }, `Across all ${all.n.toLocaleString('en-US')} rows the range caught ${(all.coverage * 100).toFixed(1)}% of salaries and is a median ${usd(all.median_width)} wide, so use the range, not a single number.`) : null);

  playOnce(card, () => body.classList.remove('in-pre'));
  return card;
}

// ------------------------------------------------------------------ (d) decisions
function decisionBlock(decisions) {
  const split = (c) => {
    const m = /^(High|Medium|Low)\b[:,]?\s*(.*)$/i.exec(c || '');
    return m ? { level: m[1].toLowerCase(), rest: m[2] } : { level: '', rest: c || '' };
  };
  return el('div', { class: 'in-decisions', id: 'in-decisions' },
    el('h3', { class: 'in-h' }, 'Decisions this supports'),
    el('p', { class: 'in-sub muted' }, 'Each answer carries a note on how much weight the data can bear.'),
    el('ul', { class: 'in-dgrid' }, decisions.map((x) => {
      const c = split(x.c);
      return el('li', { class: 'card in-dcard' },
        el('h4', {}, x.q), el('p', {}, x.a),
        el('p', { class: 'in-conf' },
          c.level ? el('b', { class: 'in-conf-tag', 'data-level': c.level }, c.level + ' confidence') : null,
          c.rest ? el('span', {}, c.rest) : null));
    })));
}
