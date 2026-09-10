"""Single-page UI for the PIGEAN results portal (served by `pigean.portal_server`).

Layout: model -> trait -> run selectors plus a gene search in the header; a gene scatter
(log_bf vs prior, coloured by combined) with a sortable gene table; a ranked gene-set
table; and a collapsible right-hand detail sheet that opens with a gene set's gene
loadings or a gene's gene-set memberships. Genes belonging to the selected gene set are
drawn as a magenta highlight layer on top of the scatter. All data comes from the JSON
API; no build step.
"""

from __future__ import annotations

import html
import json

CSS = r"""
:root { --ink:#1f2933; --muted:#65758b; --line:#d9e1ea; --soft:#f4f7f8; --accent:#0f766e; --accent-soft:#dff4f1; --warn:#9a5b1f; --hl:#d61ac7; --hl-soft:#fbe3f8; }
* { box-sizing: border-box; }
body { margin:0; color:var(--ink); font-family: ui-sans-serif,-apple-system,"Segoe UI",sans-serif; background:#f7f9fa; font-size:14px; }
.shell { max-width:1600px; margin:0 auto; padding:20px 24px 48px; transition: padding-right .2s; }
body.sheet-open .shell { padding-right: 520px; }
header { display:flex; flex-wrap:wrap; gap:14px; align-items:end; margin-bottom:12px; }
header .selectors { display:flex; flex-wrap:wrap; gap:12px; margin-left:auto; align-items:end; }
h1 { margin:0; font-size:26px; letter-spacing:-0.02em; }
h2 { margin:0 0 10px; font-size:17px; }
h3 { margin:14px 0 6px; font-size:14px; }
.muted { color:var(--muted); font-size:12px; }
.grid { display:grid; grid-template-columns: 1.1fr 0.9fr; gap:16px; }
.panel { background:#fff; border:1px solid var(--line); border-radius:14px; padding:16px; min-width:0; }
label { display:block; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; margin-bottom:4px; }
select, input { border:1px solid var(--line); border-radius:9px; padding:7px 9px; background:#fff; color:var(--ink); font-size:13px; }
input[type=number] { width:9ch; }
.controls { display:flex; flex-wrap:wrap; gap:12px; align-items:end; margin-bottom:10px; }
button { border:1px solid var(--line); background:#fff; border-radius:9px; padding:6px 10px; cursor:pointer; font-weight:600; color:var(--ink); }
button:hover { background:var(--accent-soft); border-color:var(--accent); }
.seg { display:inline-flex; border:1px solid var(--line); border-radius:9px; overflow:hidden; }
.seg button { border:0; border-radius:0; font-weight:500; padding:6px 10px; }
.seg button.active { background:var(--hl-soft); color:var(--hl); font-weight:700; }
table { width:100%; border-collapse:collapse; font-size:12.5px; }
th, td { padding:5px 7px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }
th { position:sticky; top:0; background:#fff; cursor:pointer; user-select:none; }
th.num, td.num { text-align:right; font-variant-numeric: tabular-nums; }
tr.row { cursor:pointer; }
tr.row:hover { background:var(--accent-soft); }
tr.selected { background:var(--hl-soft); }
.scroll { max-height:480px; overflow:auto; border:1px solid var(--line); border-radius:9px; }
.stats { display:flex; gap:12px; flex-wrap:wrap; align-items:center; margin:6px 0 12px; }
.stat { background:var(--soft); border:1px solid var(--line); border-radius:9px; padding:6px 10px; }
.stat strong { font-size:16px; display:block; }
.warn { color:var(--warn); }
code { background:var(--soft); padding:1px 5px; border-radius:5px; font-size:12px; }
details.build summary { cursor:pointer; color:var(--muted); font-size:12px; user-select:none; }
details.build div { margin:6px 0 0 14px; font-size:12px; line-height:1.6; }
.kv { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:4px 12px; font-size:12px; margin-bottom:10px; }
.kv span { color:var(--muted); display:block; font-size:11px; }
.chip { display:inline-block; background:var(--hl-soft); color:var(--hl); border-radius:999px; padding:1px 8px; font-size:11px; font-weight:600; margin-left:6px; }
/* detail sheet */
#sheet { position:fixed; top:0; right:0; height:100vh; width:500px; max-width:95vw; background:#fff; border-left:1px solid var(--line); box-shadow:-12px 0 30px rgba(31,41,51,.08); transform:translateX(105%); transition:transform .2s; overflow:auto; padding:18px 20px 40px; z-index:20; }
body.sheet-open #sheet { transform:none; }
#sheet .sheet-head { display:flex; align-items:start; gap:10px; margin-bottom:8px; }
#sheet h2 { flex:1; font-size:16px; word-break:break-all; margin:0; }
#sheet .kind { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; }
#sheet .scroll { max-height:none; }
#loading-plot { margin-bottom:8px; }
"""

SCRIPT = r"""
const state = { runs:[], run:null, genes:[], geneSets:[], selectedGeneSet:null, selectedGene:null,
  geneSort:{col:'combined',desc:true}, gsSort:{col:'beta',desc:true}, highlighted:new Set(), highlightMode:'context' };
const HL = '#d61ac7';
const $ = id => document.getElementById(id);
const fmt = v => (v === null || v === undefined) ? '' : (Math.abs(v) >= 1000 ? (+v).toFixed(0) : (Math.abs(v) < 0.01 && v !== 0 ? (+v).toExponential(2) : (+v).toFixed(3)));
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
const num = id => { const v = $(id).value; return v === '' ? '' : Number(v); };
const sortRows = (rows, s) => rows.slice().sort((a,b) => { const x=a[s.col], y=b[s.col]; if (x===y) return 0; if (x===null||x===undefined) return 1; if (y===null||y===undefined) return -1; return (x<y?-1:1) * (s.desc?-1:1); });
const sortMark = (s, c) => s.col===c ? (s.desc?' ▼':' ▲') : '';
function bindSort(table, s, textCols, rerender) {
  table.querySelectorAll('th[data-col]').forEach(th => th.onclick = () => { const c = th.dataset.col; if (s.col===c) s.desc=!s.desc; else { s.col=c; s.desc = !textCols.includes(c); } rerender(); });
}

// ---------- run selection: model -> trait -> run
const uniq = a => [...new Set(a.filter(x => x !== null && x !== undefined && x !== ''))];
async function loadRuns() {
  const body = await api('/api/runs');
  state.runs = body.runs;
  if (!state.runs.length) { $('run-summary').innerHTML = '<span class="warn">No runs in this database.</span>'; return; }
  const models = uniq(state.runs.map(r => r.model));
  $('model').innerHTML = '<option value="">all models</option>' + models.map(m => `<option value="${esc(m)}">${esc(m)}</option>`).join('');
  populateTraits();
  populateRuns();
  await refreshRun();
}
function populateTraits() {
  const model = $('model').value;
  const traits = uniq(state.runs.filter(r => !model || r.model === model).map(r => r.trait)).sort();
  $('trait-list').innerHTML = traits.map(t => `<option value="${esc(t)}">`).join('');
  $('trait').placeholder = traits.length ? `search ${traits.length} traits` : 'no trait labels';
}
function matchingRuns() {
  const model = $('model').value, trait = $('trait').value.trim().toLowerCase();
  return state.runs.filter(r => (!model || r.model === model) && (!trait || (r.trait || '').toLowerCase().includes(trait) || r.run_id.toLowerCase().includes(trait)));
}
function populateRuns() {
  const runs = matchingRuns();
  const label = r => r.seed ? `${r.trait || r.run_id} · ${r.model} · seed ${r.seed}` : (r.title || r.run_id);
  $('run').innerHTML = runs.map(r => `<option value="${esc(r.run_id)}">${esc(label(r))}</option>`).join('');
  $('run-count').textContent = `${runs.length} of ${state.runs.length} runs`;
  if (!runs.some(r => r.run_id === state.run)) state.run = runs.length ? runs[0].run_id : null;
  if (state.run) $('run').value = state.run;
}
function runSummary() {
  const r = state.runs.find(x => x.run_id === state.run);
  if (!r) { $('run-summary').innerHTML = '<span class="warn">No run matches the current selection.</span>'; return; }
  const f = r.filters || {};
  const filt = ['genes','gene_sets','loadings'].map(k => (f[k] && f[k].length) ? `<b>${k}</b>: ${esc(f[k].join(` ${f.mode === 'all' ? 'AND' : 'OR'} `))}` : null).filter(Boolean).join(' &nbsp;·&nbsp; ') || 'no build-time filters';
  const paths = ['gene_stats_path','gene_set_stats_path','gene_gene_set_stats_path'].filter(k => r[k]).map(k => `<code>${esc(r[k])}</code>`).join('<br>');
  $('run-summary').innerHTML = `
    <div class="stats">
      <div class="stat"><strong>${esc(r.title || r.run_id)}</strong><span class="muted">${esc(r.run_id)}</span></div>
      <div class="stat"><strong>${r.n_genes.toLocaleString()}</strong><span class="muted">genes (of ${r.n_genes_input.toLocaleString()})</span></div>
      <div class="stat"><strong>${r.n_gene_sets.toLocaleString()}</strong><span class="muted">gene sets (of ${r.n_gene_sets_input.toLocaleString()})</span></div>
      <div class="stat"><strong>${r.n_loadings.toLocaleString()}</strong><span class="muted">loadings (of ${r.n_loadings_input.toLocaleString()})</span></div>
      <details class="build"><summary>Build details</summary><div>
        Filters (${esc(f.mode || 'any')}): ${filt}<br>
        Built ${esc(r.built_at)}<br>${paths}
        ${(r.warnings||[]).map(w => `<div class="warn">${esc(w)}</div>`).join('')}
      </div></details>
    </div>`;
}
async function refreshRun() {
  state.selectedGeneSet = null; state.selectedGene = null; state.highlighted = new Set();
  closeSheet();
  runSummary();
  if (!state.run) { state.genes = []; state.geneSets = []; drawScatter(); renderGeneTable(); renderGeneSetTable(); return; }
  await Promise.all([loadGenes(), loadGeneSets()]);
}

// ---------- genes: scatter + table
async function loadGenes() {
  const body = await api('/api/genes', { run: state.run, min_prior: num('min_prior'), min_log_bf: num('min_log_bf'), min_combined: num('min_combined'), sort: 'combined', limit: 20000 });
  state.genes = body.genes;
  $('gene-list').innerHTML = state.genes.slice(0, 3000).map(g => `<option value="${esc(g.gene)}">`).join('');
  drawScatter();
  renderGeneTable();
}
function drawScatter() {
  const g = state.genes, hl = state.highlighted, solo = state.highlightMode === 'solo' && hl.size > 0;
  // context: everything drawn (dimmed) with the highlighted genes on top; solo: only the highlighted genes
  const base = solo ? [] : g.filter(r => !hl.has(r.gene)), top = g.filter(r => hl.has(r.gene));
  const hover = '<b>%{text}</b><br>log_bf %{x:.3f}<br>prior %{y:.3f}<br>combined %{customdata[0]:.3f}<br>huge %{customdata[1]:.3f}<extra></extra>';
  const traces = [{
    name: 'genes', type: 'scattergl', mode: 'markers', x: base.map(r => r.log_bf), y: base.map(r => r.prior), text: base.map(r => r.gene),
    customdata: base.map(r => [r.combined, r.huge_score]), hovertemplate: hover,
    marker: { size: 6, opacity: hl.size ? 0.35 : 0.75, color: base.map(r => r.combined), colorscale: 'Viridis', colorbar: { title: 'combined', thickness: 12 } },
  }];
  if (top.length) traces.push({
    name: 'highlighted', type: 'scattergl', mode: 'markers', x: top.map(r => r.log_bf), y: top.map(r => r.prior), text: top.map(r => r.gene),
    customdata: top.map(r => [r.combined, r.huge_score]), hovertemplate: hover,
    marker: { size: 11, symbol: 'diamond', color: HL, opacity: 0.95, line: { width: 1.5, color: '#fff' } },
  });
  const layout = { margin: { l: 55, r: 10, t: 10, b: 45 }, xaxis: { title: 'log_bf (direct)', zeroline: true }, yaxis: { title: 'prior (indirect)', zeroline: true }, height: 460, hovermode: 'closest', showlegend: false };
  Plotly.react('scatter', traces, layout, { responsive: true, displaylogo: false });
  $('scatter').on('plotly_click', ev => { const p = ev.points && ev.points[0]; if (p) showGene(p.text); });
  $('scatter-count').innerHTML = (solo ? `${top.length} highlighted genes shown (solo)` : `${g.length.toLocaleString()} genes shown`) + (hl.size && !solo ? ` <span class="chip">${top.length} highlighted</span>` : '');
  $('hl-mode').hidden = !hl.size;
}
function renderGeneTable() {
  const s = state.geneSort;
  const cols = [['gene','Gene',false],['combined','combined',true],['log_bf','log_bf',true],['prior','prior',true],['huge_score','huge',true],['n','N',true]];
  const rows = sortRows(state.genes, s).slice(0, 300);
  $('gene-table').innerHTML = '<thead><tr>' + cols.map(([c,t,n]) => `<th class="${n?'num':''}" data-col="${c}">${t}${sortMark(s,c)}</th>`).join('') + '</tr></thead><tbody>' +
    rows.map(r => `<tr class="row ${r.gene===state.selectedGene?'selected':''}" data-gene="${esc(r.gene)}"><td>${esc(r.gene)}${state.highlighted.has(r.gene)?'<span class="chip">in set</span>':''}</td><td class="num">${fmt(r.combined)}</td><td class="num">${fmt(r.log_bf)}</td><td class="num">${fmt(r.prior)}</td><td class="num">${fmt(r.huge_score)}</td><td class="num">${fmt(r.n)}</td></tr>`).join('') + '</tbody>';
  bindSort($('gene-table'), s, ['gene'], renderGeneTable);
  $('gene-table').querySelectorAll('tr.row').forEach(tr => tr.onclick = () => showGene(tr.dataset.gene));
  $('gene-table-count').textContent = `top ${rows.length} of ${state.genes.length.toLocaleString()} by ${s.col}`;
}

// ---------- gene sets
async function loadGeneSets() {
  const body = await api('/api/gene_sets', { run: state.run, min_beta: num('min_beta'), min_beta_uncorrected: num('min_beta_uncorrected'), search: $('gs_search').value, sort: 'beta', limit: 2000 });
  state.geneSets = body.gene_sets;
  renderGeneSetTable();
}
function renderGeneSetTable() {
  const s = state.gsSort;
  const cols = [['gene_set','Gene set',false],['label','Label',false],['n','N',true],['beta','beta',true],['beta_uncorrected','beta_unc',true],['p_orig','P',true]];
  const rows = sortRows(state.geneSets, s);
  $('gs-table').innerHTML = '<thead><tr>' + cols.map(([c,t,n]) => `<th class="${n?'num':''}" data-col="${c}">${t}${sortMark(s,c)}</th>`).join('') + '</tr></thead><tbody>' +
    rows.map((r,i) => `<tr class="row ${r.gene_set===state.selectedGeneSet?'selected':''}" data-gs="${esc(r.gene_set)}"><td>${i+1}. ${esc(r.gene_set)}</td><td>${esc(r.label)}</td><td class="num">${fmt(r.n)}</td><td class="num">${fmt(r.beta)}</td><td class="num">${fmt(r.beta_uncorrected)}</td><td class="num">${fmt(r.p_orig)}</td></tr>`).join('') + '</tbody>';
  bindSort($('gs-table'), s, ['gene_set','label','p_orig'], renderGeneSetTable);
  $('gs-table').querySelectorAll('tr.row').forEach(tr => tr.onclick = () => showGeneSet(tr.dataset.gs));
  $('gs-count').textContent = `${rows.length.toLocaleString()} gene sets shown`;
}

// ---------- detail sheet
function kv(obj, keys) { return '<div class="kv">' + keys.filter(k => obj[k] !== undefined && obj[k] !== null && obj[k] !== '').map(k => `<div><span>${esc(k)}</span>${esc(typeof obj[k]==='number'?fmt(obj[k]):obj[k])}</div>`).join('') + '</div>'; }
function openSheet(kind, title, bodyHtml) {
  $('sheet-kind').textContent = kind; $('sheet-title').textContent = title; $('sheet-body').innerHTML = bodyHtml;
  document.body.classList.add('sheet-open');
  window.dispatchEvent(new Event('resize'));
}
function closeSheet() { document.body.classList.remove('sheet-open'); window.dispatchEvent(new Event('resize')); }
function setHighlight(genes) { state.highlighted = new Set(genes); drawScatter(); renderGeneTable(); }

async function showGeneSet(id) {
  const d = await api('/api/gene_set', { run: state.run, id, limit: 1000 });
  state.selectedGeneSet = id; state.selectedGene = null;
  const L = d.loadings;
  openSheet('Gene set', id,
    kv(d, ['label','n','beta','beta_uncorrected','p_orig','z_orig']) +
    `<h3>Gene loadings <span class="muted">${L.length.toLocaleString()} of ${d.n_loadings.toLocaleString()}, by weight then combined</span></h3>` +
    `<div id="loading-plot" style="height:${Math.min(560, 60 + 15 * Math.min(L.length, 35))}px"></div>` +
    `<div class="scroll"><table id="loading-table"><thead><tr><th>Gene</th><th class="num">weight</th><th class="num">beta</th><th class="num">combined</th><th class="num">log_bf</th><th class="num">prior</th></tr></thead><tbody>` +
    L.map(r => `<tr class="row" data-gene="${esc(r.gene)}"><td>${esc(r.gene)}</td><td class="num">${fmt(r.weight)}</td><td class="num">${fmt(r.beta)}</td><td class="num">${fmt(r.combined)}</td><td class="num">${fmt(r.log_bf)}</td><td class="num">${fmt(r.prior)}</td></tr>`).join('') + '</tbody></table></div>' +
    `<details style="margin-top:10px"><summary class="muted">all columns</summary>${kv(d.extra, Object.keys(d.extra))}</details>`);
  $('sheet-body').querySelectorAll('tr.row').forEach(tr => tr.onclick = () => showGene(tr.dataset.gene));
  const top = L.slice(0, 35).reverse();
  if (top.length) Plotly.react('loading-plot', [{ type: 'bar', orientation: 'h', y: top.map(r => r.gene), x: top.map(r => r.combined ?? 0), marker: { color: HL, opacity: top.map(r => 0.35 + 0.65 * (r.weight ?? 0)) }, hovertemplate: '%{y}: combined %{x:.3f}<extra></extra>' }],
    { margin: { l: 100, r: 10, t: 4, b: 30 }, xaxis: { title: 'combined (gene score)' }, yaxis: { automargin: true, tickfont: { size: 10 } } }, { responsive: true, displaylogo: false });
  setHighlight(L.map(r => r.gene));
  renderGeneSetTable();
}
async function showGene(gene) {
  let d;
  try { d = await api('/api/gene', { run: state.run, id: gene, limit: 1000 }); }
  catch (err) { openSheet('Gene', gene, `<p class="warn">${esc(err.message)} — it may not have passed the build thresholds for this run.</p>`); return; }
  state.selectedGene = gene;
  openSheet('Gene', gene,
    kv(d, ['combined','log_bf','prior','huge_score','n','chrom','start','end']) +
    `<h3>Gene sets <span class="muted">${d.gene_sets.length.toLocaleString()} loadings, by beta</span></h3>` +
    `<div class="scroll"><table><thead><tr><th>Gene set</th><th>Label</th><th class="num">beta</th><th class="num">weight</th><th class="num">beta_unc</th></tr></thead><tbody>` +
    d.gene_sets.map(r => `<tr class="row" data-gs="${esc(r.gene_set)}"><td>${esc(r.gene_set)}</td><td>${esc(r.label)}</td><td class="num">${fmt(r.beta)}</td><td class="num">${fmt(r.weight)}</td><td class="num">${fmt(r.beta_uncorrected)}</td></tr>`).join('') + '</tbody></table></div>' +
    `<details style="margin-top:10px"><summary class="muted">all columns</summary>${kv(d.extra, Object.keys(d.extra))}</details>`);
  $('sheet-body').querySelectorAll('tr.row').forEach(tr => tr.onclick = () => showGeneSet(tr.dataset.gs));
  setHighlight([gene]);
}

// ---------- wiring
let t1, t2;
$('model').onchange = () => { populateTraits(); populateRuns(); refreshRun(); };
$('trait').oninput = () => { populateRuns(); clearTimeout(t1); t1 = setTimeout(refreshRun, 250); };
$('run').onchange = e => { state.run = e.target.value; refreshRun(); };
['min_prior','min_log_bf','min_combined'].forEach(id => $(id).oninput = () => { clearTimeout(t1); t1 = setTimeout(loadGenes, 350); });
['min_beta','min_beta_uncorrected','gs_search'].forEach(id => $(id).oninput = () => { clearTimeout(t2); t2 = setTimeout(loadGeneSets, 350); });
$('gene_search').onchange = () => { const v = $('gene_search').value.trim(); if (v) showGene(v.toUpperCase() === v ? v : (state.genes.find(g => g.gene.toLowerCase() === v.toLowerCase()) || {gene: v}).gene); };
$('sheet-close').onclick = closeSheet;
$('hl-mode').querySelectorAll('button').forEach(b => b.onclick = () => { state.highlightMode = b.dataset.mode; $('hl-mode').querySelectorAll('button').forEach(x => x.classList.toggle('active', x === b)); drawScatter(); });
$('clear-highlight').onclick = () => { state.selectedGeneSet = null; state.selectedGene = null; setHighlight([]); renderGeneSetTable(); };
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeSheet(); });
loadRuns().catch(err => { $('run-summary').innerHTML = `<span class="warn">${esc(err.message)}</span>`; });
"""

BODY = r"""
<div class="shell">
  <header>
    <div><h1>{title}</h1><div class="muted">Gene scatter, ranked gene sets, and gene loadings from thresholded PIGEAN outputs.{api_note}</div></div>
    <div class="selectors">
      <div><label for="model">Model</label><select id="model"></select></div>
      <div><label for="trait">Trait</label><input id="trait" list="trait-list" placeholder="search traits" style="width:16ch"><datalist id="trait-list"></datalist></div>
      <div><label for="run">Run <span id="run-count" class="muted"></span></label><select id="run" style="min-width:24ch"></select></div>
      <div><label for="gene_search">Gene</label><input id="gene_search" list="gene-list" placeholder="e.g. TCF7L2 ⏎" style="width:14ch"><datalist id="gene-list"></datalist></div>
    </div>
  </header>
  <div id="run-summary" class="muted"></div>
  <div class="grid">
    <section class="panel">
      <h2>Genes: direct (log_bf) vs indirect (prior)</h2>
      <div class="controls">
        <div><label>min prior</label><input id="min_prior" type="number" step="0.1"></div>
        <div><label>min log_bf</label><input id="min_log_bf" type="number" step="0.1"></div>
        <div><label>min combined</label><input id="min_combined" type="number" step="0.1"></div>
        <div style="margin-left:auto;display:flex;gap:8px;align-items:end">
          <div id="hl-mode" hidden><label>Highlight</label><div class="seg"><button type="button" data-mode="context" class="active" title="highlighted genes on top of all genes">Context</button><button type="button" data-mode="solo" title="only the highlighted genes">Solo</button></div></div>
          <button id="clear-highlight" type="button">Clear highlight</button>
        </div>
      </div>
      <div id="scatter"></div>
      <div class="muted" id="scatter-count"></div>
      <div class="muted" id="gene-table-count" style="margin-top:8px"></div>
      <div class="scroll" style="max-height:300px;margin-top:4px"><table id="gene-table"></table></div>
    </section>
    <section class="panel">
      <h2>Top gene sets</h2>
      <div class="controls">
        <div><label>min beta</label><input id="min_beta" type="number" step="0.01"></div>
        <div><label>min beta_unc</label><input id="min_beta_uncorrected" type="number" step="0.01"></div>
        <div style="flex:1"><label>search (id or label)</label><input id="gs_search" placeholder="e.g. insulin" style="width:100%"></div>
      </div>
      <div class="scroll" style="max-height:820px"><table id="gs-table"></table></div>
      <div class="muted" id="gs-count"></div>
    </section>
  </div>
</div>
<aside id="sheet" aria-label="detail">
  <div class="sheet-head"><div style="flex:1"><div class="kind" id="sheet-kind"></div><h2 id="sheet-title"></h2></div><button id="sheet-close" type="button" title="close (Esc)">✕</button></div>
  <div id="sheet-body"></div>
</aside>
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
