// Thin cached fetch layer. Every section module uses these, never fetch() directly.
const cache = new Map();

async function getJSON(url) {
  if (!cache.has(url)) {
    cache.set(url, fetch(url).then(async (r) => {
      if (!r.ok) {
        let detail = r.statusText;
        try { detail = (await r.json()).detail || detail; } catch (_) { /* keep statusText */ }
        cache.delete(url);
        throw new Error(detail);
      }
      return r.json();
    }));
  }
  return cache.get(url);
}

export const api = {
  story: () => getJSON('/api/story'),        // points, per-stage predictions, decile lines, metrics
  models: () => getJSON('/api/models'),      // search table, decision, interval comparison
  report: () => getJSON('/api/report'),      // { meta, sections: { stage0..stage5, arena, decision, insights, estimator, explorer, limits } }
  insights: () => getJSON('/api/insights'),  // premiums, bands, reliability
  options: () => getJSON('/api/options'),    // category lists for the estimator form
  explore: (params) => fetch('/api/explore?' + new URLSearchParams(params)).then((r) => r.json()),
  predict: async (body) => {
    const r = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error((await r.json()).detail || 'Could not estimate');
    return r.json();
  },
};
