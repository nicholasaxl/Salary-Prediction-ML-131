// Shared helpers: formatting, motion, reveal-on-view and the written-report renderer.
import { api } from './api.js';

export const prefersReduced = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
export const usd = (n) => '$' + Math.round(n).toLocaleString('en-US');
/** Model outputs are good to about a thousand dollars at best, so show them that way. */
export const usdK = (n) => '$' + (Math.round(n / 1000) * 1000).toLocaleString('en-US');
export const kfmt = (n) => '$' + Math.round(n / 1000) + 'k';
export const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
export const lerp = (a, b, t) => a + (b - a) * t;
export const ease = {
  out: (t) => 1 - Math.pow(1 - t, 3),
  inOut: (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
};

/** Tiny hyperscript: el('div', {class:'x', onclick}, 'text', childNode). */
export function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
    else if (k === 'class') n.className = v;
    else if (k === 'html') n.innerHTML = v;
    else n.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) if (kid != null) n.append(kid.nodeType ? kid : document.createTextNode(kid));
  return n;
}

/** Run onFrame(t) with t in 0..1 over `duration` ms. Returns { cancel(), done: Promise }. Jumps to the end when motion is reduced. */
export function tween(duration, onFrame, easing = ease.inOut) {
  let raf = 0, cancelled = false;
  const done = new Promise((resolve) => {
    if (prefersReduced() || duration <= 0) { onFrame(1); resolve(); return; }
    const t0 = performance.now();
    const step = (now) => {
      if (cancelled) return resolve();
      const t = clamp((now - t0) / duration, 0, 1);
      onFrame(easing(t));
      if (t < 1) raf = requestAnimationFrame(step); else resolve();
    };
    raf = requestAnimationFrame(step);
  });
  return { done, cancel() { cancelled = true; cancelAnimationFrame(raf); } };
}

/** Count a number in an element's text: ticker(el, 51366, {format: usd}). */
export function ticker(node, to, { from, format = usd, duration = 900 } = {}) {
  const start = from ?? parseFloat((node.dataset.value ?? '0')) ?? 0;
  node.dataset.value = String(to);
  if (node._tick) node._tick.cancel();
  node._tick = tween(duration, (t) => { node.textContent = format(lerp(Number.isFinite(start) ? start : 0, to, t)); }, ease.out);
  return node._tick.done;
}

/** Call cb once when el scrolls into view (immediately if IntersectionObserver is missing). */
export function onView(node, cb, { threshold = 0.25, once = true } = {}) {
  if (!('IntersectionObserver' in window)) { cb(); return; }
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) { cb(e); if (once) io.disconnect(); }
  }, { threshold });
  io.observe(node);
}

/** Fade/slide elements in as they enter the viewport. Always completes: a timer reveals anything the observer missed. */
export function reveal(nodes) {
  const list = [...nodes];
  if (prefersReduced() || !('IntersectionObserver' in window)) return;
  list.forEach((n) => n.classList.add('will-reveal'));
  const show = (n) => n.classList.add('is-in');
  list.forEach((n, i) => onView(n, () => show(n), { threshold: 0.12 }));
  setTimeout(() => list.forEach(show), 2500);
}

// ---- written reports --------------------------------------------------------
const NUM_RE = /(\$[\d,]+(?:\.\d+)?|[+-]?\d+(?:\.\d+)?%|R² [\d.]+)/g;
const rich = (text) => esc(text).replace(NUM_RE, '<span class="num">$1</span>');

/** Render the written report for a section key (stage0..stage5, arena, decision, insights, ...) into `host`. */
export async function renderReport(host, key, { open = true } = {}) {
  let rep;
  try { rep = (await api.report()).sections[key]; } catch (e) { host.append(el('p', { class: 'err' }, 'Report unavailable: ' + e.message)); return null; }
  if (!rep) return null;
  const row = (label, body, cls = '') => (body && (!Array.isArray(body) || body.length)
    ? el('div', { class: 'rrow ' + cls }, el('h4', {}, label),
        Array.isArray(body) ? el('ul', {}, body.map((b) => el('li', { html: rich(b) }))) : el('p', { html: rich(body) }))
    : null);
  const d = el('details', { class: 'report', ...(open ? { open: true } : {}) },
    el('summary', {}, el('span', { class: 'report-tag' }, 'Report'), el('span', {}, rep.title)),
    el('div', { class: 'report-body' }, row('What we did', rep.did), row('What happened', rep.happened), row('Why', rep.why), row('Takeaway', rep.takeaway, 'takeaway')));
  host.append(d);
  return rep;
}
