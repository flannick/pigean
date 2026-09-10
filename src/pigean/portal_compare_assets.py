"""Single-page UI for the PIGEAN **Comparer** (served at /compare by `pigean.portal_server`).

Two runs (A, B) are chosen on a landing card (trait -> model -> run for each, with the shared
fuzzy typeahead); the results view then shows a top-N correlation summary, A-vs-B (or
difference-vs-mean) scatters for gene scores and gene-set effects, a run-parameter diff, and
paginated ranking tables with lookup. Everything is computed server-side by
`pigean.portal_compare`; shared CSS / JS helpers come from `portal_assets_common`.
"""

from __future__ import annotations

import html

from .portal_assets_common import render_document

CSS = r"""
.landing.compare { width:min(1040px, 96vw); }
.ab { display:grid; grid-template-columns:1fr auto 1fr; gap:18px; align-items:start; }
.ab .col { border:1px solid var(--line); border-radius:12px; padding:14px 16px; }
.ab .col h3 { margin:0 0 10px; font-size:14px; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; }
.ab .swap { align-self:center; }
.tag { display:inline-block; border-radius:6px; padding:1px 7px; font-size:11px; font-weight:700; margin-right:6px; }
.tag.a { background:#dbeafe; color:#1d4ed8; } .tag.b { background:#fde68a; color:#92400e; }
.vs { color:var(--muted); font-weight:600; margin:0 6px; }
.crumb.stacked { display:grid; grid-template-columns:auto 1fr; gap:4px 10px; align-items:baseline; }
.crumb.stacked strong { font-size:19px; }
.crumb.stacked .muted { margin-left:8px; font-size:12px; }
.sumtable td.num, .sumtable th.num { text-align:right; }
.status-both { color:var(--accent); } .status-a_only { color:#1d4ed8; } .status-b_only { color:#92400e; }
.seg { display:inline-flex; border:1px solid var(--line); border-radius:9px; overflow:hidden; }
.seg button { border:0; border-radius:0; font-weight:500; padding:6px 10px; }
.seg button.active { background:var(--accent-soft); color:var(--accent); font-weight:700; }
.grid.compare { grid-template-columns:1fr 1fr; }
.tabs { display:flex; gap:6px; margin:0 0 12px; border-bottom:1px solid var(--line); }
.tabs button { border:0; border-bottom:2px solid transparent; border-radius:0; background:none; font-weight:600; color:var(--muted); padding:6px 12px; }
.tabs button.active { color:var(--accent); border-bottom-color:var(--accent); }
.tabs button:hover { background:none; color:var(--ink); }
.sumtable { width:auto; min-width:60%; }
.sumtable th, .sumtable td { padding:6px 14px; }
"""

SCRIPT = r"""
const state = { runs:[], a:null, b:null, topN:100, geneMetric:'combined', gsMetric:'beta', view:'ab', geneRows:[], gsRows:[], params:null, summary:null, sumTab:'genes' };
const RUN_KEYS = ['a', 'b'];
const runById = id => state.runs.find(r => r.run_id === id);
const runLabel = r => r ? `${r.phenotype && r.phenotype.name ? r.phenotype.name : (r.trait || r.run_id)} · ${r.model}${r.seed && r.seed !== 'main' ? ' · ' + r.seed : ''}` : '';
function hashState() { const h = new URLSearchParams(location.hash.replace(/^#/, '')); return { a: h.get('a') || '', b: h.get('b') || '' }; }

// ---------- landing: two independent trait -> model -> run selector columns
function availableTraits(k) { const model = $(`${k}_model`).value; return uniq(state.runs.filter(r => !model || r.model === model).map(r => r.trait)).sort(); }
function traitItems(k) { return q => { const traits = availableTraits(k); const ranked = q ? traits.map(t => [t, traitScore(t, q.toLowerCase())]).filter(x => x[1] > 0).sort((x, y) => y[1] - x[1]).map(x => x[0]) : traits; return ranked.map(t => { const p = phenoOf(t); return { value: t, label: t, sub: p && p.name && p.name !== t ? p.name : '' }; }); }; }
function matchingRuns(k) {
  const model = $(`${k}_model`).value, q = $(`${k}_trait`).value.trim().toLowerCase();
  const pool = state.runs.filter(r => !model || r.model === model);
  if (!q) return pool;
  const exact = pool.filter(r => (r.trait || '').toLowerCase() === q); if (exact.length) return exact;
  return pool.map(r => [r, Math.max(traitScore(r.trait || '', q), fuzzyScore(q, r.run_id) * 0.5)]).filter(x => x[1] > 0).sort((x, y) => y[1] - x[1]).map(x => x[0]);
}
function populateRuns(k) {
  const runs = matchingRuns(k);
  $(`${k}_run`).innerHTML = runs.map(r => `<option value="${esc(r.run_id)}">${esc(r.trait ? `${r.trait} · ${r.model}${r.seed && r.seed !== 'main' ? ' · ' + r.seed : ''}` : (r.title || r.run_id))}</option>`).join('');
  if (!runs.some(r => r.run_id === state[k])) state[k] = runs.length ? runs[0].run_id : null;
  if (state[k]) $(`${k}_run`).value = state[k];
  $('compare').disabled = !(state.a && state.b && state.a !== state.b);
  $('landing-note').textContent = state.a && state.b && state.a === state.b ? 'Run A and Run B are the same run — pick two different runs.' : '';
}
async function loadRuns() {
  const body = await api('/api/runs');
  state.runs = body.runs;
  if (!state.runs.length) { $('landing-note').innerHTML = '<span class="warn">No runs in this database.</span>'; return; }
  const models = uniq(state.runs.map(r => r.model));
  RUN_KEYS.forEach(k => { $(`${k}_model`).innerHTML = '<option value="">any model</option>' + models.map(m => `<option value="${esc(m)}">${esc(m)}</option>`).join(''); populateRuns(k); });
  const h = hashState();
  if (h.a && h.b && runById(h.a) && runById(h.b) && h.a !== h.b) { state.a = h.a; state.b = h.b; $('a_run').value = h.a; $('b_run').value = h.b; await openResults(); }
  else showLanding();
}
function showLanding() { $('view-results').hidden = true; $('view-landing').hidden = false; setHash({}); setTimeout(() => $('a_trait').focus(), 50); }
async function openResults() {
  if (!state.a || !state.b || state.a === state.b) return;
  $('view-landing').hidden = true; $('view-results').hidden = false;
  setHash({ a: state.a, b: state.b });
  const ra = runById(state.a), rb = runById(state.b);
  $('crumb').innerHTML = `<span class="tag a">A</span><div><strong>${esc(runLabel(ra))}</strong><span class="muted">${esc(ra.run_id)}</span></div>` +
    `<span class="tag b">B</span><div><strong>${esc(runLabel(rb))}</strong><span class="muted">${esc(rb.run_id)}</span></div>`;
  await Promise.all([loadSummary(), loadGenes(), loadGeneSets(), loadParams()]);
}
function swapRuns() { [state.a, state.b] = [state.b, state.a]; if (!$('view-results').hidden) openResults(); else { $('a_run').value = state.a || ''; $('b_run').value = state.b || ''; } }

// ---------- summary (top-N correlations)
async function loadSummary() {
  state.summary = await api('/api/compare/summary', { a: state.a, b: state.b, top_n: state.topN });
  renderSummary();
}
function renderSummary() {
  const d = state.summary; if (!d) return;
  const kind = state.sumTab, part = d[kind], c = part.counts, noun = kind === 'genes' ? 'genes' : 'gene sets';
  const ranked = kind === 'genes' ? [d.ranked.a_genes, d.ranked.b_genes] : [d.ranked.a_gene_sets, d.ranked.b_gene_sets];
  const row = m => `<tr><td><b>${esc(m.metric)}</b></td><td class="num">${m.n.toLocaleString()}</td><td class="num">${m.pearson === null ? '' : m.pearson.toFixed(3)}</td><td class="num">${m.spearman === null ? '' : m.spearman.toFixed(3)}</td><td class="num">${m.overlap === null || m.overlap === undefined ? '—' : m.overlap.toLocaleString()}</td><td class="num">${m.jaccard === null || m.jaccard === undefined ? '—' : m.jaccard.toFixed(2)}</td><td class="num">${m.n_common.toLocaleString()}</td><td class="num">${m.rank_pearson_all === null ? '' : m.rank_pearson_all.toFixed(3)}</td></tr>`;
  $('summary').innerHTML = `
    <div class="tabs"><button data-tab="genes" class="${kind==='genes'?'active':''}">Genes</button><button data-tab="gene_sets" class="${kind==='gene_sets'?'active':''}">Gene sets</button></div>
    <div class="counts">
      <div class="stat"><strong>${c.both.toLocaleString()}</strong><span class="muted">${noun} in both</span></div>
      <div class="stat"><strong>${c.a_only.toLocaleString()}</strong><span class="muted">A only</span></div>
      <div class="stat"><strong>${c.b_only.toLocaleString()}</strong><span class="muted">B only</span></div>
      <span class="muted">ranked over ${(ranked[0] || 0).toLocaleString()} (A) / ${(ranked[1] || 0).toLocaleString()} (B) ${noun} in the full PIGEAN output</span>
    </div>
    ${d.ranks_available ? '' : '<div class="warn">This database predates rank storage — rebuild it with the current pigean.portal to get ranks and top-N summaries.</div>'}
    <table class="sumtable"><thead><tr><th>score</th><th class="num">n in top-${state.topN || 'all'}</th><th class="num">Pearson</th><th class="num">Spearman</th><th class="num">top-${state.topN || 'all'} overlap</th><th class="num">Jaccard</th><th class="num">n in both</th><th class="num">rank r (all)</th></tr></thead><tbody>
      ${part.metrics.map(row).join('')}
    </tbody></table>
    <div class="muted" style="margin-top:6px">Top-N = union of A's and B's top N by that score's rank; Pearson / Spearman are on the members of that union present in both runs. Only rows that passed each build's thresholds are stored, so "A only" / "B only" means the other run fell below threshold.</div>`;
  $('summary').querySelectorAll('.tabs button').forEach(b => b.onclick = () => { state.sumTab = b.dataset.tab; renderSummary(); });
}

// ---------- scatters + ranking tables
function scatter(el, rows, metric, keyName, hoverName) {
  const both = rows.filter(r => r.status === 'both');
  const ab = state.view === 'ab';
  const x = both.map(r => ab ? r[`a_${metric}`] : (r[`a_${metric}`] + r[`b_${metric}`]) / 2);
  const y = both.map(r => ab ? r[`b_${metric}`] : r[`delta_${metric}`]);
  const cd = both.map(r => [r[`a_${metric}`], r[`b_${metric}`], r[`a_rank_${metric}`], r[`b_rank_${metric}`], r[`delta_rank_${metric}`]]);
  const traces = [{ type: 'scattergl', mode: 'markers', x, y, text: both.map(r => r[keyName]), customdata: cd,
    marker: { size: 6, opacity: 0.7, color: both.map(r => Math.abs(r[`delta_rank_${metric}`] ?? 0)), colorscale: 'Viridis', reversescale: true, colorbar: { title: '|Δrank|', thickness: 12 } },
    hovertemplate: `<b>%{text}</b><br>A ${metric} %{customdata[0]:.3f} (rank %{customdata[2]})<br>B ${metric} %{customdata[1]:.3f} (rank %{customdata[3]})<br>Δrank %{customdata[4]}<extra></extra>` }];
  let layout;
  if (ab) {
    const lo = Math.min(...x, ...y), hi = Math.max(...x, ...y);
    traces.push({ type: 'scatter', mode: 'lines', x: [lo, hi], y: [lo, hi], line: { color: '#9ca3af', dash: 'dot', width: 1 }, hoverinfo: 'skip' });
    layout = { xaxis: { title: `A: ${metric}` }, yaxis: { title: `B: ${metric}` } };
  } else {
    traces.push({ type: 'scatter', mode: 'lines', x: [Math.min(...x), Math.max(...x)], y: [0, 0], line: { color: '#9ca3af', dash: 'dot', width: 1 }, hoverinfo: 'skip' });
    layout = { xaxis: { title: `mean of A and B: ${metric}` }, yaxis: { title: `B − A: ${metric}` } };
  }
  Plotly.react(el, traces, Object.assign({ margin: { l: 55, r: 10, t: 10, b: 45 }, height: 420, hovermode: 'closest', showlegend: false }, layout), { responsive: true, displaylogo: false });
  $(el).on('plotly_click', ev => { const p = ev.points && ev.points[0]; if (p && p.text) showLookup(hoverName, p.text); });
}
const statusHtml = s => `<span class="status-${s}">${s === 'both' ? 'both' : (s === 'a_only' ? 'A only' : 'B only')}</span>`;
const rank = v => v === null || v === undefined ? '' : v.toLocaleString();
const delta = v => v === null || v === undefined ? '' : (v > 0 ? '+' : '') + (Number.isInteger(v) ? v.toLocaleString() : fmt(v));

async function loadGenes() {
  const metric = state.geneMetric;
  const d = await api('/api/compare/genes', { a: state.a, b: state.b, metric, search: $('gene_search').value.trim(), sort: $('gene_sort').value, status: $('gene_status').value, limit: 20000 });
  state.geneRows = d.rows;
  scatter('gene-scatter', d.rows, metric, 'gene', 'gene');
  $('gene-count').textContent = `${d.total.toLocaleString()} genes (${d.counts.both.toLocaleString()} in both); scatter shows genes present in both`;
  pagedTable('gene-table', `<tr><th>Gene</th><th class="num">rank A</th><th class="num">rank B</th><th class="num">Δrank</th><th class="num">A ${metric}</th><th class="num">B ${metric}</th><th class="num">Δ${metric}</th><th>status</th></tr>`, d.rows,
    r => `<tr class="row" data-id="${esc(r.gene)}"><td>${esc(r.gene)}</td><td class="num">${rank(r[`a_rank_${metric}`])}</td><td class="num">${rank(r[`b_rank_${metric}`])}</td><td class="num">${delta(r[`delta_rank_${metric}`])}</td><td class="num">${fmt(r[`a_${metric}`])}</td><td class="num">${fmt(r[`b_${metric}`])}</td><td class="num">${delta(r[`delta_${metric}`])}</td><td>${statusHtml(r.status)}</td></tr>`,
    tr => tr.onclick = () => showLookup('gene', tr.dataset.id));
}
async function loadGeneSets() {
  const metric = state.gsMetric;
  const d = await api('/api/compare/gene_sets', { a: state.a, b: state.b, metric, search: $('gs_search').value.trim(), sort: $('gs_sort').value, status: $('gs_status').value, limit: 20000 });
  state.gsRows = d.rows;
  scatter('gs-scatter', d.rows, metric, 'gene_set', 'gene_set');
  $('gs-count').textContent = `${d.total.toLocaleString()} gene sets (${d.counts.both.toLocaleString()} in both); scatter shows gene sets present in both`;
  pagedTable('gs-table', `<tr><th>Gene set</th><th>Library</th><th class="num">rank A</th><th class="num">rank B</th><th class="num">Δrank</th><th class="num">A ${metric}</th><th class="num">B ${metric}</th><th class="num">Δ${metric}</th><th>status</th></tr>`, d.rows,
    r => `<tr class="row" data-id="${esc(r.gene_set)}"><td title="${esc(r.gene_set)}">${esc(r.gene_set)}</td><td class="wrap">${esc(r.label)}</td><td class="num">${rank(r[`a_rank_${metric}`])}</td><td class="num">${rank(r[`b_rank_${metric}`])}</td><td class="num">${delta(r[`delta_rank_${metric}`])}</td><td class="num">${fmt(r[`a_${metric}`])}</td><td class="num">${fmt(r[`b_${metric}`])}</td><td class="num">${delta(r[`delta_${metric}`])}</td><td>${statusHtml(r.status)}</td></tr>`,
    tr => tr.onclick = () => showLookup('gene_set', tr.dataset.id));
}

// ---------- lookup modal (one gene / gene set in both runs)
async function showLookup(kind, id) {
  let d;
  try { d = await api('/api/compare/lookup', { a: state.a, b: state.b, kind, id }); } catch (err) { openModal(id, `<span class="warn">${esc(err.message)}</span>`); return; }
  const metrics = kind === 'gene' ? ['combined', 'prior', 'log_bf', 'huge_score'] : ['beta', 'beta_uncorrected'];
  const ra = runById(state.a), rb = runById(state.b);
  openModal(`${kind === 'gene' ? 'Gene' : 'Gene set'}: ${id}`,
    `<div class="muted" style="margin-bottom:8px">${statusHtml(d.status)}${d.label ? ' · library ' + esc(d.label) : ''}</div>
     <table><thead><tr><th>metric</th><th class="num"><span class="tag a">A</span>${esc(runLabel(ra))}</th><th class="num"><span class="tag b">B</span>${esc(runLabel(rb))}</th><th class="num">Δ (B − A)</th><th class="num">rank A</th><th class="num">rank B</th><th class="num">Δrank</th></tr></thead><tbody>
     ${metrics.map(m => `<tr><td>${m}</td><td class="num">${fmt(d[`a_${m}`])}</td><td class="num">${fmt(d[`b_${m}`])}</td><td class="num">${delta(d[`delta_${m}`])}</td><td class="num">${rank(d[`a_rank_${m}`])}</td><td class="num">${rank(d[`b_rank_${m}`])}</td><td class="num">${delta(d[`delta_rank_${m}`])}</td></tr>`).join('')}
     </tbody></table>`);
}

// ---------- run parameters diff
async function loadParams() {
  state.params = await api('/api/compare/params', { a: state.a, b: state.b });
  renderParams();
}
function renderParams() {
  const p = state.params; if (!p) return;
  const q = $('param_search').value.trim(), only = $('param_only_diff').checked;
  let rows = p.rows.filter(r => !only || r.differs);
  if (q) rows = rows.map(r => [r, fuzzyScore(q, r.parameter)]).filter(x => x[1] > 0).sort((x, y) => y[1] - x[1]).map(x => x[0]);
  $('param-count').textContent = `${p.n_differ} of ${p.n_params} parameters differ${!p.a_has_params || !p.b_has_params ? ' (a run has no stored params — build with params=)' : ''}; showing ${rows.length}`;
  pagedTable('param-table', `<tr><th>Parameter</th><th>Ver.</th><th><span class="tag a">A</span></th><th><span class="tag b">B</span></th></tr>`, rows,
    r => `<tr class="${r.differs ? 'selected' : ''}"><td>${esc(r.parameter)}</td><td>${esc(r.version)}</td><td style="white-space:normal;word-break:break-all">${esc(r.a_value ?? '')}</td><td style="white-space:normal;word-break:break-all">${esc(r.b_value ?? '')}</td></tr>`, null, 40);
}

// ---------- wiring
let t1, t2;
RUN_KEYS.forEach(k => {
  attachTypeahead($(`${k}_trait`), traitItems(k), () => populateRuns(k));
  $(`${k}_trait`).addEventListener('input', () => populateRuns(k));
  $(`${k}_model`).onchange = () => populateRuns(k);
  $(`${k}_run`).onchange = e => { state[k] = e.target.value; populateRuns(k); };
});
$('compare').onclick = openResults;
$('swap-landing').onclick = swapRuns; $('swap').onclick = swapRuns;
$('back').onclick = showLanding;
$('top_n').onchange = () => { state.topN = +$('top_n').value; loadSummary(); };
$('gene_metric').onchange = () => { state.geneMetric = $('gene_metric').value; loadGenes(); };
$('gs_metric').onchange = () => { state.gsMetric = $('gs_metric').value; loadGeneSets(); };
document.querySelectorAll('#view-toggle button').forEach(b => b.onclick = () => { state.view = b.dataset.view; document.querySelectorAll('#view-toggle button').forEach(x => x.classList.toggle('active', x === b)); scatter('gene-scatter', state.geneRows, state.geneMetric, 'gene', 'gene'); scatter('gs-scatter', state.gsRows, state.gsMetric, 'gene_set', 'gene_set'); });
['gene_search','gene_sort','gene_status'].forEach(id => $(id).oninput = $(id).onchange = () => { clearTimeout(t1); t1 = setTimeout(loadGenes, 300); });
['gs_search','gs_sort','gs_status'].forEach(id => $(id).oninput = $(id).onchange = () => { clearTimeout(t2); t2 = setTimeout(loadGeneSets, 300); });
$('param_search').oninput = renderParams; $('param_only_diff').onchange = renderParams;
$('modal-close').onclick = () => $('modal').close();
$('modal').onclick = e => { if (e.target === $('modal')) $('modal').close(); };
window.addEventListener('hashchange', () => { const h = hashState(); if (!(h.a && h.b) && !$('view-results').hidden) showLanding(); });
loadRuns().catch(err => { $('landing-note').innerHTML = `<span class="warn">${esc(err.message)}</span>`; $('view-landing').hidden = false; });
"""

BODY = r"""
<div id="view-landing" hidden>
  <div class="landing compare">
    <h1>{title}</h1>
    <p class="lede">Compare two PIGEAN runs: gene and gene-set scores, rankings, and run parameters.{api_note} <a href="./" style="color:var(--accent)">Explorer →</a></p>
    <div class="ab">
      <div class="col"><h3><span class="tag a">A</span>Run A</h3>
        <div class="field ta-wrap"><label for="a_trait">Trait</label><input id="a_trait" placeholder="Search traits by name or id" autocomplete="off"></div>
        <div class="field"><label for="a_model">Model</label><select id="a_model"></select></div>
        <div class="field"><label for="a_run">Run</label><select id="a_run"></select></div>
      </div>
      <div class="swap"><button id="swap-landing" type="button" title="swap A and B">⇄</button></div>
      <div class="col"><h3><span class="tag b">B</span>Run B</h3>
        <div class="field ta-wrap"><label for="b_trait">Trait</label><input id="b_trait" placeholder="Search traits by name or id" autocomplete="off"></div>
        <div class="field"><label for="b_model">Model</label><select id="b_model"></select></div>
        <div class="field"><label for="b_run">Run</label><select id="b_run"></select></div>
      </div>
    </div>
    <div class="actions"><button id="compare" class="primary" type="button" disabled>Compare</button><span id="landing-note" class="hint"></span></div>
  </div>
</div>
<div id="view-results" class="shell" hidden>
  <div class="bar">
    <button id="back" type="button" title="back to search">◀ Search</button>
    <div class="crumb stacked" id="crumb"></div>
    <div class="right"><button id="swap" type="button" title="swap A and B">⇄ swap</button></div>
  </div>
  <section class="panel">
    <div class="controls"><h2 style="margin:0">Agreement</h2><div style="margin-left:auto"><label for="top_n">top N</label><select id="top_n"><option value="50">50</option><option value="100" selected>100</option><option value="250">250</option><option value="500">500</option><option value="1000">1000</option><option value="0">all common</option></select></div></div>
    <div id="summary"><p class="muted">Loading…</p></div>
  </section>
  <div class="controls" style="margin:4px 0 8px"><div><label>scatter view</label><div class="seg" id="view-toggle"><button type="button" data-view="ab" class="active" title="x = A, y = B">A vs B</button><button type="button" data-view="diff" title="x = mean, y = B − A">difference vs mean</button></div></div></div>
  <div class="grid compare">
    <section class="panel">
      <div class="controls"><h2 style="margin:0">Genes</h2><div style="margin-left:auto"><label for="gene_metric">score</label><select id="gene_metric"><option value="combined">combined</option><option value="prior">prior</option><option value="log_bf">log_bf</option><option value="huge_score">huge_score</option></select></div></div>
      <div id="gene-scatter"></div>
      <div class="muted" id="gene-count"></div>
      <div class="controls" style="margin-top:8px">
        <div style="flex:1"><label for="gene_search">lookup gene</label><input id="gene_search" placeholder="e.g. TCF7L2" style="width:100%" autocomplete="off"></div>
        <div><label for="gene_sort">sort</label><select id="gene_sort"><option value="abs_delta_rank">|Δrank|</option><option value="abs_delta">|Δscore|</option><option value="delta_rank">Δrank</option><option value="delta">Δscore</option><option value="a_rank">rank A</option><option value="b_rank">rank B</option><option value="id">name</option></select></div>
        <div><label for="gene_status">show</label><select id="gene_status"><option value="">all</option><option value="both">in both</option><option value="a_only">A only</option><option value="b_only">B only</option></select></div>
      </div>
      <div id="gene-table"></div>
    </section>
    <section class="panel">
      <div class="controls"><h2 style="margin:0">Gene sets</h2><div style="margin-left:auto"><label for="gs_metric">effect</label><select id="gs_metric"><option value="beta">beta</option><option value="beta_uncorrected">beta_uncorrected</option></select></div></div>
      <div id="gs-scatter"></div>
      <div class="muted" id="gs-count"></div>
      <div class="controls" style="margin-top:8px">
        <div style="flex:1"><label for="gs_search">lookup gene set (id or library)</label><input id="gs_search" placeholder="e.g. insulin" style="width:100%" autocomplete="off"></div>
        <div><label for="gs_sort">sort</label><select id="gs_sort"><option value="abs_delta_rank">|Δrank|</option><option value="abs_delta">|Δeffect|</option><option value="delta_rank">Δrank</option><option value="delta">Δeffect</option><option value="a_rank">rank A</option><option value="b_rank">rank B</option><option value="id">name</option></select></div>
        <div><label for="gs_status">show</label><select id="gs_status"><option value="">all</option><option value="both">in both</option><option value="a_only">A only</option><option value="b_only">B only</option></select></div>
      </div>
      <div id="gs-table"></div>
    </section>
    <section class="panel" style="grid-column:1 / -1">
      <div class="controls"><h2 style="margin:0">Run parameters</h2><div style="flex:1;margin-left:16px"><label for="param_search">search parameter (fuzzy)</label><input id="param_search" placeholder="e.g. burn in, seed, chains" style="width:100%" autocomplete="off"></div><div><label><input type="checkbox" id="param_only_diff" checked style="width:auto"> only differences</label></div></div>
      <div class="muted" id="param-count"></div>
      <div id="param-table"></div>
    </section>
  </div>
</div>
<dialog id="modal" class="modal"><div class="modal-head"><h2 id="modal-title"></h2><button id="modal-close" type="button" title="close">✕</button></div><div class="modal-body" id="modal-body"></div></dialog>
"""


def render_compare_html(*, title: str, plotly_src: str, api_base: str = "") -> str:
    """Complete HTML document for the Comparer UI (see `render_portal_html` for the arguments)."""
    api_note = f" API: <code>{html.escape(api_base)}</code>" if api_base else ""
    body = BODY.replace("{title}", html.escape(title)).replace("{api_note}", api_note)
    return render_document(title=title, plotly_src=plotly_src, api_base=api_base, css=CSS, body=body, script=SCRIPT)
