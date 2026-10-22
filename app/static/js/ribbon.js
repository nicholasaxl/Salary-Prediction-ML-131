// The pay-band ribbon: the signature graphic. A P10-P90 band with a median tick and an optional observed-peer bar.
import { el, usd, kfmt, ticker, prefersReduced } from './util.js';

/**
 * ribbon(host, { low, mid, high, peers: {p25, p75}|null, max })
 * Returns { update(opts) }: calling update() with new numbers slides the band to its new place (CSS transitions).
 * `max` fixes the right edge of the scale so ribbons in the same view stay comparable.
 */
export function ribbon(host, opts) {
  const band = el('div', { class: 'rb-band' });
  const mid = el('div', { class: 'rb-mid', title: 'Median' });
  const peer = el('div', { class: 'rb-peer', title: 'Observed peers, middle 50%' });
  const tLo = el('span', { class: 'rb-tick' });
  const tHi = el('span', { class: 'rb-tick' });
  const root = el('div', { class: 'rb', role: 'img' }, el('div', { class: 'rb-track' }), band, peer, mid, tLo, tHi);
  host.append(root);

  const pos = (v, max) => (Math.min(Math.max(v / max, 0), 1) * 100).toFixed(2) + '%';
  function update(o, first = false) {
    const max = o.max || Math.max(o.high, o.peers ? o.peers.p75 : 0) * 1.12;
    root.setAttribute('aria-label', `Estimated range ${usd(o.low)} to ${usd(o.high)}, most likely ${usd(o.mid)}`);
    band.style.left = pos(o.low, max);
    band.style.width = (((o.high - o.low) / max) * 100).toFixed(2) + '%';
    mid.style.left = pos(o.mid, max);
    tLo.style.left = pos(o.low, max); tHi.style.left = pos(o.high, max);
    if (first) { tLo.textContent = kfmt(o.low); tHi.textContent = kfmt(o.high); }
    else { ticker(tLo, o.low, { format: kfmt, duration: 500 }); ticker(tHi, o.high, { format: kfmt, duration: 500 }); }
    if (first) { tLo.dataset.value = o.low; tHi.dataset.value = o.high; }
    if (o.peers && o.peers.p75) {
      peer.hidden = false;
      peer.style.left = pos(o.peers.p25, max);
      peer.style.width = (((o.peers.p75 - o.peers.p25) / max) * 100).toFixed(2) + '%';
    } else peer.hidden = true;
  }
  if (!prefersReduced()) root.classList.add('rb-intro');
  update(opts, true);
  return { update: (o) => update(o, false), root };
}

export function ribbonLegend(withPeers = true) {
  return el('div', { class: 'ribbon-legend' },
    el('span', { html: '<i style="background:var(--band);border:1px solid var(--blue)"></i>model 10–90% range' }),
    el('span', { html: '<i style="background:var(--gold);width:4px;height:12px"></i>most likely' }),
    withPeers ? el('span', { html: '<i style="background:var(--gold);opacity:.8;height:5px"></i>observed peers, middle 50%' }) : null);
}
