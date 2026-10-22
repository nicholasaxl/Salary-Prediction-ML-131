// Page shell: theme, nav scroll-spy, hero, section loader. Each section is a module exporting mount(el).
import { api } from './api.js';
import { el, usd, usdK, ticker, reveal, renderReport, prefersReduced } from './util.js';
import { ribbon, ribbonLegend } from './ribbon.js';

const $ = (s, r = document) => r.querySelector(s);

// ---- theme (system by default; the toggle remembers a choice when storage is available)
function initTheme() {
  const root = document.documentElement;
  try { const t = localStorage.getItem('payband-theme'); if (t) root.dataset.theme = t; } catch (_) { /* storage blocked */ }
  const btn = $('#theme');
  const effective = () => root.dataset.theme || (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  btn.addEventListener('click', () => {
    const next = effective() === 'dark' ? 'light' : 'dark';
    root.dataset.theme = next;
    try { localStorage.setItem('payband-theme', next); } catch (_) { /* ignore */ }
    window.dispatchEvent(new CustomEvent('themechange', { detail: next }));
  });
}

// ---- nav: exactly one link is current, chosen by scroll position
function initNav() {
  const links = [...document.querySelectorAll('.nav-links a')];
  const sections = links.map((a) => [a, document.getElementById(a.getAttribute('href').slice(1))]).filter(([, s]) => s);
  let queued = false;
  const update = () => {
    queued = false;
    const line = window.innerHeight * 0.4;           // a section is current once its top passes this line
    let current = null;
    for (const [a, sec] of sections) if (sec.getBoundingClientRect().top <= line) current = a;
    if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4) current = sections.length ? sections[sections.length - 1][0] : current;
    links.forEach((a) => (a === current ? a.setAttribute('aria-current', 'true') : a.removeAttribute('aria-current')));
  };
  const schedule = () => { if (!queued) { queued = true; requestAnimationFrame(update); } };
  window.addEventListener('scroll', schedule, { passive: true });
  window.addEventListener('resize', schedule);
  update();
}

// ---- hero: a live example estimate drawn as the ribbon
const EXAMPLE = { job_title: 'Data Scientist', experience_level: 'SE', employment_type: 'FT', company_location: 'US', company_size: 'M', work_year: 2023, remote_ratio: 0 };

async function initHero() {
  const [rep] = await Promise.all([api.report().catch(() => null)]);
  if (rep) {
    const m = rep.meta;
    ticker($('#h-rows'), m.rows, { format: (v) => Math.round(v).toLocaleString('en-US') });
    ticker($('#h-r2'), m.r2Best, { format: (v) => v.toFixed(2), duration: 1100 });
    ticker($('#h-noise'), m.noise * 100, { format: (v) => '±' + Math.round(v) + '%', duration: 1100 });
  }
  const host = $('#hero-ribbon');
  try {
    const r = await api.predict(EXAMPLE);
    const rb = ribbon(host, { low: r.low, mid: r.mid, high: r.high, peers: r.peers && r.peers.n ? r.peers : null, max: 260000 });
    host.append(ribbonLegend(!!(r.peers && r.peers.n)));
    $('#hero-range').innerHTML = '';
    const lo = el('span', { class: 'num' }), hi = el('span', { class: 'num' });
    $('#hero-range').append(lo, ' – ', hi);
    ticker(lo, r.low, { duration: 1200, format: usdK }); ticker(hi, r.high, { duration: 1200, format: usdK });
    ticker($('#hero-mid'), r.mid, { duration: 1200, format: usdK });
    $('#hero-peers').textContent = r.peers && r.peers.n ? `${r.peers.n} observed people with the ${r.peers.match} reported a median of ${usd(r.peers.median)}.` : '';
  } catch (e) {
    host.append(el('p', { class: 'err' }, 'The live estimate is unavailable: ' + e.message));
  }
}

// ---- sections
const SECTIONS = [
  ['story', './story.js'],
  ['arena', './arena.js'],
  ['estimate', './estimator.js'],
  ['insights', './insights.js'],
  ['explore', './explorer.js'],
];

async function loadSections() {
  await Promise.all(SECTIONS.map(async ([id, mod]) => {
    const mount = $(`#${id}-mount`);
    try {
      const m = await import(mod);
      mount.textContent = '';
      await m.mount(mount);
    } catch (e) {
      console.error(id, e);
      mount.textContent = '';
      mount.append(el('p', { class: id === 'story' ? 'err' : 'mount-loading' }, `This section could not load (${e.message}).`));
    }
  }));
}

async function initFooter() {
  await renderReport($('#limits-mount'), 'limits', { open: true });
  const rep = await api.report().catch(() => null);
  if (rep) $('#repo-link').href = rep.meta.repo;
}

initTheme();
initNav();
initHero();
loadSections().then(() => { reveal(document.querySelectorAll('.sec-head, .reveal-me')); });
initFooter();
