"""Single-page UI for the PIGEAN results portal (served by `pigean.portal_server`).

Three panels: a gene scatter (log_bf vs prior, coloured by combined), a ranked gene-set
table, and a detail panel showing the selected gene set's gene loadings (or the selected
gene's gene-set memberships). All data comes from the JSON API; no build step.
"""

from __future__ import annotations

import html
import json

CSS = r"""
:root { --ink:#1f2933; --muted:#65758b; --line:#d9e1ea; --soft:#f4f7f8; --accent:#0f766e; --accent-soft:#dff4f1; --warn:#9a5b1f; }
* { box-sizing: border-box; }
body { margin:0; color:var(--ink); font-family: ui-sans-serif,-apple-system,"Segoe UI",sans-serif; background:#f7f9fa; font-size:14px; }
.shell { max-width:1600px; margin:0 auto; padding:20px 24px 48px; }
header { display:flex; flex-wrap:wrap; gap:16px; align-items:end; margin-bottom:16px; }
h1 { margin:0; font-size:26px; letter-spacing:-0.02em; }
h2 { margin:0 0 10px; font-size:17px; }
.muted { color:var(--muted); font-size:12px; }
.grid { display:grid; grid-template-columns: 1.1fr 0.9fr; gap:16px; }
.panel { background:#fff; border:1px solid var(--line); border-radius:14px; padding:16px; min-width:0; }
.panel.full { grid-column: 1 / -1; }
label { display:block; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; margin-bottom:4px; }
select, input { border:1px solid var(--line); border-radius:9px; padding:7px 9px; background:#fff; color:var(--ink); font-size:13px; }
input[type=number] { width:9ch; }
.controls { display:flex; flex-wrap:wrap; gap:12px; align-items:end; margin-bottom:10px; }
.controls > div { min-width:0; }
button { border:1px solid var(--line); background:#fff; border-radius:9px; padding:6px 10px; cursor:pointer; font-weight:600; }
button.active, button:hover { background:var(--accent-soft); border-color:var(--accent); }
table { width:100%; border-collapse:collapse; font-size:12.5px; }
th, td { padding:5px 7px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }
th { position:sticky; top:0; background:#fff; cursor:pointer; user-select:none; }
th.num, td.num { text-align:right; font-variant-numeric: tabular-nums; }
tr.row { cursor:pointer; }
tr.row:hover, tr.selected { background:var(--accent-soft); }
.scroll { max-height:480px; overflow:auto; border:1px solid var(--line); border-radius:9px; }
.stats { display:flex; gap:12px; flex-wrap:wrap; margin:6px 0 10px; }
.stat { background:var(--soft); border:1px solid var(--line); border-radius:9px; padding:6px 10px; }
.stat strong { font-size:16px; display:block; }
.warn { color:var(--warn); }
code { background:var(--soft); padding:1px 5px; border-radius:5px; }
#detail-title { word-break:break-all; }
.kv { display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:4px 12px; font-size:12px; margin-bottom:10px; }
.kv span { color:var(--muted); }
"""

SCRIPT = r"""
const state = { run:null, runs:[], genes:[], geneSets:[], selectedGeneSet:null, selectedGene:null, gsSort:{col:'beta',desc:true} };
const $ = id => document.getElementById(id);
const fmt = v => (v === null || v === undefined) ? '' : (Math.abs(v) >= 1000 ? v.toFixed(0) : (Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(2) : (+v).toFixed(3)));
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const API_BASE = (window.PIGEAN_PORTAL_API_BASE || '').replace(/\/+$/, '');
async function api(path, params) {
  const url = new URL(API_BASE + path, location.href);
  Object.entries(params || {}).forEach(([k,v]) => { if (v !== '' && v !== null && v !== undefined) url.searchParams.set(k, v); });
  let res;
  try { res = await fetch(url); } catch (err) { throw new Error(`cannot reach portal server at ${API_BASE || location.origin}: ${err.message}`); }
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || res.statusText);
  return body;
}
function num(id) { const v = $(id).value; return v === '' ? '' : Number(v); }

async function loadRuns() {
  const body = await api('/api/runs');
  state.runs = body.runs;
  const sel = $('run');
  sel.innerHTML = state.runs.map(r => `<option value="${esc(r.run_id)}">${esc(r.title || r.run_id)}</option>`).join('');
  if (!state.runs.length) { $('run-summary').innerHTML = '<span class="warn">No runs in this database.</span>'; return; }
  state.run = state.runs[0].run_id;
  sel.value = state.run;
  await refreshRun();
}
function runSummary() {
  const r = state.runs.find(x => x.run_id === state.run); if (!r) return;
  const f = r.filters || {};
  const filt = ['genes','gene_sets','loadings'].map(k => (f[k] && f[k].length) ? `${k}: ${f[k].join(` ${f.mode === 'all' ? 'AND' : 'OR'} `)}` : null).filter(Boolean).join(' · ') || 'no build-time filters';
  $('run-summary').innerHTML = `
    <div class="stats">
      <div class="stat"><strong>${r.n_genes.toLocaleString()}</strong><span class="muted">genes (of ${r.n_genes_input.toLocaleString()})</span></div>
      <div class="stat"><strong>${r.n_gene_sets.toLocaleString()}</strong><span class="muted">gene sets (of ${r.n_gene_sets_input.toLocaleString()})</span></div>
      <div class="stat"><strong>${r.n_loadings.toLocaleString()}</strong><span class="muted">loadings (of ${r.n_loadings_input.toLocaleString()})</span></div>
    </div>
    <div class="muted">Build filters (${esc(f.mode || 'any')}): ${esc(filt)} · built ${esc(r.built_at)}</div>
    ${(r.warnings||[]).map(w => `<div class="warn">${esc(w)}</div>`).join('')}`;
}
async function refreshRun() {
  state.selectedGeneSet = null; state.selectedGene = null;
  runSummary();
  await Promise.all([loadGenes(), loadGeneSets()]);
  $('detail').innerHTML = '<p class="muted">Click a gene set row for its gene loadings, or a gene point / row for its gene sets.</p>';
  $('detail-title').textContent = 'Detail';
}

async function loadGenes() {
  const body = await api('/api/genes', { run: state.run, min_prior: num('min_prior'), min_log_bf: num('min_log_bf'), min_combined: num('min_combined'), search: $('gene_search').value, sort: 'combined', limit: 20000 });
  state.genes = body.genes;
  drawScatter();
  renderGeneTable();
}
function drawScatter() {
  const g = state.genes;
  const trace = {
    type: 'scattergl', mode: 'markers', x: g.map(r => r.log_bf), y: g.map(r => r.prior), text: g.map(r => r.gene),
    customdata: g.map(r => [r.combined, r.huge_score]),
    hovertemplate: '<b>%{text}</b><br>log_bf %{x:.3f}<br>prior %{y:.3f}<br>combined %{customdata[0]:.3f}<br>huge %{customdata[1]:.3f}<extra></extra>',
    marker: { size: 6, opacity: 0.75, color: g.map(r => r.combined), colorscale: 'Viridis', colorbar: { title: 'combined', thickness: 12 } },
  };
  const layout = { margin: { l: 55, r: 10, t: 10, b: 45 }, xaxis: { title: 'log_bf (direct)', zeroline: true }, yaxis: { title: 'prior (indirect)', zeroline: true }, height: 460, hovermode: 'closest' };
  Plotly.react('scatter', [trace], layout, { responsive: true, displaylogo: false });
  $('scatter').on('plotly_click', ev => { const p = ev.points && ev.points[0]; if (p) showGene(p.text); });
  $('scatter-count').textContent = `${g.length.toLocaleString()} genes shown`;
}
function renderGeneTable() {
  const rows = state.genes.slice(0, 300);
  $('gene-table').innerHTML = `<thead><tr><th>Gene</th><th class="num">combined</th><th class="num">log_bf</th><th class="num">prior</th><th class="num">huge</th><th class="num">N</th></tr></thead><tbody>` +
    rows.map(r => `<tr class="row" data-gene="${esc(r.gene)}"><td>${esc(r.gene)}</td><td class="num">${fmt(r.combined)}</td><td class="num">${fmt(r.log_bf)}</td><td class="num">${fmt(r.prior)}</td><td class="num">${fmt(r.huge_score)}</td><td class="num">${fmt(r.n)}</td></tr>`).join('') + '</tbody>';
  $('gene-table').querySelectorAll('tr.row').forEach(tr => tr.onclick = () => showGene(tr.dataset.gene));
}

async function loadGeneSets() {
  const body = await api('/api/gene_sets', { run: state.run, min_beta: num('min_beta'), min_beta_uncorrected: num('min_beta_uncorrected'), search: $('gs_search').value, sort: state.gsSort.col, limit: 2000 });
  state.geneSets = body.gene_sets;
  renderGeneSetTable();
}
function renderGeneSetTable() {
  const cols = [['gene_set','Gene set',false],['label','Label',false],['n','N',true],['beta','beta',true],['beta_uncorrected','beta_unc',true],['p_orig','P',true]];
  let rows = state.geneSets.slice();
  const s = state.gsSort;
  rows.sort((a,b) => { const x=a[s.col], y=b[s.col]; if (x===y) return 0; if (x===null||x===undefined) return 1; if (y===null||y===undefined) return -1; return (x<y?-1:1) * (s.desc?-1:1); });
  $('gs-table').innerHTML = '<thead><tr>' + cols.map(([c,t,n]) => `<th class="${n?'num':''}" data-col="${c}">${t}${s.col===c?(s.desc?' ▼':' ▲'):''}</th>`).join('') + '</tr></thead><tbody>' +
    rows.map((r,i) => `<tr class="row ${r.gene_set===state.selectedGeneSet?'selected':''}" data-gs="${esc(r.gene_set)}"><td>${i+1}. ${esc(r.gene_set)}</td><td>${esc(r.label)}</td><td class="num">${fmt(r.n)}</td><td class="num">${fmt(r.beta)}</td><td class="num">${fmt(r.beta_uncorrected)}</td><td class="num">${fmt(r.p_orig)}</td></tr>`).join('') + '</tbody>';
  $('gs-table').querySelectorAll('th').forEach(th => th.onclick = () => { const c = th.dataset.col; if (s.col===c) s.desc=!s.desc; else { s.col=c; s.desc = c!=='gene_set' && c!=='label' && c!=='p_orig'; } renderGeneSetTable(); });
  $('gs-table').querySelectorAll('tr.row').forEach(tr => tr.onclick = () => showGeneSet(tr.dataset.gs));
  $('gs-count').textContent = `${rows.length.toLocaleString()} gene sets shown`;
}

function kv(obj, keys) { return '<div class="kv">' + keys.filter(k => obj[k] !== undefined && obj[k] !== null && obj[k] !== '').map(k => `<div><span>${esc(k)}</span> ${esc(typeof obj[k]==='number'?fmt(obj[k]):obj[k])}</div>`).join('') + '</div>'; }

async function showGeneSet(id) {
  state.selectedGeneSet = id; state.selectedGene = null; renderGeneSetTable();
  const d = await api('/api/gene_set', { run: state.run, id, limit: 1000 });
  $('detail-title').textContent = `Gene set: ${id}`;
  const L = d.loadings;
  $('detail').innerHTML = kv(d, ['label','n','beta','beta_uncorrected','p_orig','z_orig']) +
    `<div class="muted">${L.length.toLocaleString()} of ${d.n_loadings.toLocaleString()} gene loadings (sorted by weight, then combined)</div>` +
    `<div id="loading-plot" style="height:${Math.min(600, 80 + 16 * Math.min(L.length, 40))}px"></div>` +
    `<div class="scroll"><table id="loading-table"><thead><tr><th>Gene</th><th class="num">weight</th><th class="num">beta</th><th class="num">combined</th><th class="num">log_bf</th><th class="num">prior</th><th class="num">huge</th></tr></thead><tbody>` +
    L.map(r => `<tr class="row" data-gene="${esc(r.gene)}"><td>${esc(r.gene)}</td><td class="num">${fmt(r.weight)}</td><td class="num">${fmt(r.beta)}</td><td class="num">${fmt(r.combined)}</td><td class="num">${fmt(r.log_bf)}</td><td class="num">${fmt(r.prior)}</td><td class="num">${fmt(r.huge_score)}</td></tr>`).join('') + '</tbody></table></div>';
  $('detail').querySelectorAll('tr.row').forEach(tr => tr.onclick = () => showGene(tr.dataset.gene));
  const top = L.slice(0, 40).reverse();
  if (top.length) Plotly.react('loading-plot', [{ type: 'bar', orientation: 'h', y: top.map(r => r.gene), x: top.map(r => r.combined ?? 0), marker: { color: top.map(r => r.weight ?? 0), colorscale: 'Teal', colorbar: { title: 'weight', thickness: 10 } }, hovertemplate: '%{y}: combined %{x:.3f}<extra></extra>' }],
    { margin: { l: 110, r: 10, t: 10, b: 35 }, xaxis: { title: 'combined (gene score)' }, yaxis: { automargin: true, tickfont: { size: 10 } } }, { responsive: true, displaylogo: false });
  highlightGenes(new Set(L.map(r => r.gene)));
}
async function showGene(gene) {
  state.selectedGene = gene;
  const d = await api('/api/gene', { run: state.run, id: gene, limit: 1000 });
  $('detail-title').textContent = `Gene: ${gene}`;
  $('detail').innerHTML = kv(d, ['combined','log_bf','prior','huge_score','n','chrom','start','end']) +
    `<div class="muted">${d.gene_sets.length.toLocaleString()} gene-set loadings for this gene (sorted by beta)</div>` +
    `<div class="scroll"><table><thead><tr><th>Gene set</th><th>Label</th><th class="num">beta</th><th class="num">weight</th><th class="num">beta_unc</th></tr></thead><tbody>` +
    d.gene_sets.map(r => `<tr class="row" data-gs="${esc(r.gene_set)}"><td>${esc(r.gene_set)}</td><td>${esc(r.label)}</td><td class="num">${fmt(r.beta)}</td><td class="num">${fmt(r.weight)}</td><td class="num">${fmt(r.beta_uncorrected)}</td></tr>`).join('') + '</tbody></table></div>' +
    `<details style="margin-top:10px"><summary class="muted">all columns</summary>${kv(d.extra, Object.keys(d.extra))}</details>`;
  $('detail').querySelectorAll('tr.row').forEach(tr => tr.onclick = () => showGeneSet(tr.dataset.gs));
  highlightGenes(new Set([gene]));
}
function highlightGenes(set) {
  const g = state.genes; if (!g.length) return;
  Plotly.restyle('scatter', { 'marker.line': [{ width: g.map(r => set.has(r.gene) ? 2 : 0), color: '#c2410c' }], 'marker.size': [g.map(r => set.has(r.gene) ? 10 : 6)] });
}

let t1, t2;
$('run').onchange = e => { state.run = e.target.value; refreshRun(); };
['min_prior','min_log_bf','min_combined','gene_search'].forEach(id => $(id).oninput = () => { clearTimeout(t1); t1 = setTimeout(loadGenes, 350); });
['min_beta','min_beta_uncorrected','gs_search'].forEach(id => $(id).oninput = () => { clearTimeout(t2); t2 = setTimeout(loadGeneSets, 350); });
loadRuns().catch(err => { $('run-summary').innerHTML = `<span class="warn">${esc(err.message)}</span>`; });
"""

BODY = r"""
<div class="shell">
  <header>
    <div><h1>{title}</h1><div class="muted">Gene scatter, ranked gene sets, and gene loadings from thresholded PIGEAN outputs.{api_note}</div></div>
    <div style="margin-left:auto;min-width:260px"><label for="run">Run</label><select id="run"></select></div>
  </header>
  <div id="run-summary" class="muted"></div>
  <div class="grid">
    <section class="panel">
      <h2>Genes: direct (log_bf) vs indirect (prior)</h2>
      <div class="controls">
        <div><label>min prior</label><input id="min_prior" type="number" step="0.1"></div>
        <div><label>min log_bf</label><input id="min_log_bf" type="number" step="0.1"></div>
        <div><label>min combined</label><input id="min_combined" type="number" step="0.1"></div>
        <div style="flex:1"><label>gene search</label><input id="gene_search" placeholder="e.g. TCF7L2" style="width:100%"></div>
      </div>
      <div id="scatter"></div>
      <div class="muted" id="scatter-count"></div>
      <div class="scroll" style="max-height:260px;margin-top:8px"><table id="gene-table"></table></div>
    </section>
    <section class="panel">
      <h2>Top gene sets</h2>
      <div class="controls">
        <div><label>min beta</label><input id="min_beta" type="number" step="0.01"></div>
        <div><label>min beta_unc</label><input id="min_beta_uncorrected" type="number" step="0.01"></div>
        <div style="flex:1"><label>search (id or label)</label><input id="gs_search" placeholder="e.g. insulin" style="width:100%"></div>
      </div>
      <div class="scroll" style="max-height:760px"><table id="gs-table"></table></div>
      <div class="muted" id="gs-count"></div>
    </section>
    <section class="panel full">
      <h2 id="detail-title">Detail</h2>
      <div id="detail"><p class="muted">Loading…</p></div>
    </section>
  </div>
</div>
"""


def render_portal_html(*, title: str, plotly_src: str, api_base: str = "") -> str:
    """
    Complete HTML document for the portal UI.

    Args:
        title: Page title.
        plotly_src: URL (or data: URI) of plotly.min.js.
        api_base: Absolute URL of a `pigean.portal serve` instance for a static deployment
            (e.g. a bucket-hosted page calling `http://localhost:8765`); empty means same origin.
    """
    api_note = f" API: <code>{html.escape(api_base)}</code>" if api_base else ""
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{html.escape(title)}</title>\n<style>{CSS}</style>\n"
        f"<script src=\"{html.escape(plotly_src, quote=True)}\"></script>\n</head>\n<body>\n"
        + BODY.replace("{title}", html.escape(title)).replace("{api_note}", api_note)
        + f"<script>window.PIGEAN_PORTAL_API_BASE = {json.dumps(api_base)};</script>\n"
        + f"<script>{SCRIPT}</script>\n</body>\n</html>\n"
    )
