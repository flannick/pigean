"""Single-page UI for the PIGEAN results portal (served by `pigean.portal_server`).

Layout: model -> trait -> run selectors (traits searchable by legacy id, portal name or
portal id when the build included the portal phenotype file) plus a gene search in the header; a gene scatter
(log_bf vs prior, coloured by combined) with a sortable gene table; a ranked gene-set
table; and a collapsible right-hand detail sheet that opens with a gene set's gene
loadings or a gene's gene-set memberships. Genes belonging to the selected gene set are
drawn on top of the scatter with a white ring while the rest of the genes fade. All data comes from the JSON
API; no build step.
"""

from __future__ import annotations

import html
import json

CSS = r"""
:root { --ink:#1f2933; --muted:#65758b; --line:#d9e1ea; --soft:#f4f7f8; --accent:#0f766e; --accent-soft:#dff4f1; --warn:#9a5b1f; --hl:#d61ac7; --hl-soft:#fbe3f8; }
* { box-sizing: border-box; }
body { margin:0; color:var(--ink); font-family: ui-sans-serif,-apple-system,"Segoe UI",sans-serif; background:#f7f9fa; font-size:14px; }
.shell { max-width:1600px; margin:0 auto; padding:20px 24px 48px; }
header { display:flex; flex-wrap:wrap; gap:14px; align-items:end; margin-bottom:12px; }
header .selectors { display:flex; flex-wrap:wrap; gap:12px; margin-left:auto; align-items:end; }
h1 { margin:0; font-size:26px; letter-spacing:-0.02em; }
h2 { margin:0 0 10px; font-size:17px; }
h3 { margin:14px 0 6px; font-size:14px; }
.muted { color:var(--muted); font-size:12px; }
.grid { display:grid; grid-template-columns: 1.1fr 0.9fr; gap:16px; }
@media (max-width: 1100px) { .grid { grid-template-columns: 1fr; } }
.panel { background:#fff; border:1px solid var(--line); border-radius:14px; padding:16px; min-width:0; }
label { display:block; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; margin-bottom:4px; }
select, input { border:1px solid var(--line); border-radius:9px; padding:7px 9px; background:#fff; color:var(--ink); font-size:13px; }
input[type=number] { width:9ch; }
.controls { display:flex; flex-wrap:wrap; gap:12px; align-items:end; margin-bottom:10px; }
button { border:1px solid var(--line); background:#fff; border-radius:9px; padding:6px 10px; cursor:pointer; font-weight:600; color:var(--ink); }
button:hover { background:var(--accent-soft); border-color:var(--accent); }
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
/* landing */
#view-landing { min-height:80vh; display:flex; align-items:center; justify-content:center; }
#view-landing[hidden], #view-results[hidden] { display:none !important; }
dialog.modal { border:1px solid var(--line); border-radius:14px; padding:0; width:min(900px, 94vw); max-height:88vh; box-shadow:0 24px 60px rgba(31,41,51,.18); }
dialog.modal::backdrop { background:rgba(31,41,51,.28); }
dialog.modal .modal-head { display:flex; align-items:center; gap:10px; padding:14px 18px; border-bottom:1px solid var(--line); position:sticky; top:0; background:#fff; }
dialog.modal .modal-head h2 { flex:1; margin:0; font-size:16px; }
dialog.modal .modal-body { padding:14px 18px 18px; overflow:auto; max-height:calc(88vh - 60px); font-size:12.5px; line-height:1.5; }
.linkbtn { border:0; background:none; color:var(--accent); padding:0; font-weight:600; cursor:pointer; font-size:12px; }
.linkbtn:hover { background:none; text-decoration:underline; }
.maps { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:2px 0 12px; }
.maps a { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:2px 9px; font-size:11.5px; color:var(--ink); text-decoration:none; background:#fff; }
.maps a:hover { border-color:var(--accent); color:var(--accent); }
.maps a .onto { color:var(--muted); font-weight:600; margin-right:3px; }
.landing { width:min(720px, 94vw); background:#fff; border:1px solid var(--line); border-radius:18px; padding:32px 36px 28px; box-shadow:0 18px 45px rgba(31,41,51,.08); }
.landing h1 { font-size:34px; margin-bottom:4px; }
.landing .lede { color:var(--muted); margin:0 0 22px; }
.landing .field { margin-bottom:14px; }
.landing input.big { width:100%; font-size:20px; padding:14px 16px; border-radius:12px; }
.landing .row { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.landing select, .landing input { width:100%; }
.landing .actions { display:flex; gap:12px; align-items:center; margin-top:18px; }
.landing button.primary { background:var(--accent); color:#fff; border-color:var(--accent); font-size:15px; padding:10px 18px; }
.landing button.primary:hover { background:#0b5f59; }
.landing .hint { color:var(--muted); font-size:12px; }
.trait-hits { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; max-height:120px; overflow:auto; }
.trait-hits button { font-weight:500; font-size:12px; padding:4px 9px; }
/* results header */
.bar { display:flex; flex-wrap:wrap; gap:14px; align-items:center; margin-bottom:12px; }
.bar .crumb { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }
.bar .crumb strong { font-size:22px; letter-spacing:-0.02em; }
.bar .crumb .muted { font-size:13px; }
.bar .right { margin-left:auto; display:flex; gap:10px; align-items:end; }
.counts { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:0 0 14px; }
.counts .stat { padding:5px 10px; }
.counts .stat strong { font-size:14px; display:inline; margin-right:4px; }
.counts details.build { display:inline-block; }
/* tables: sticky first column, truncated ids with tooltips, wrapping labels */
.scroll th:first-child, .scroll td:first-child { position:sticky; left:0; background:#fff; z-index:2; max-width:30ch; overflow:hidden; text-overflow:ellipsis; }
.scroll th:first-child { z-index:3; }
.scroll tr.selected td:first-child { background:var(--hl-soft); }
.scroll tr.row:hover td:first-child { background:var(--accent-soft); }
td.wrap { white-space:normal; min-width:18ch; max-width:34ch; line-height:1.25; }
.scroll { overscroll-behavior-x: contain; }
/* detail sheet */
#sheet { position:fixed; top:0; right:0; height:100vh; width:760px; max-width:92vw; background:#fff; border-left:1px solid var(--line); box-shadow:-12px 0 30px rgba(31,41,51,.08); transform:translateX(105%); transition:transform .2s; overflow:auto; padding:18px 20px 40px; z-index:20; }
body.sheet-open #sheet { transform:none; }
#sheet .sheet-head { display:flex; align-items:start; gap:10px; margin-bottom:8px; }
#sheet h2 { flex:1; font-size:18px; word-break:break-all; margin:0; }
#sheet-backdrop { position:fixed; inset:0; background:rgba(31,41,51,.18); z-index:19; display:none; }
body.sheet-open #sheet-backdrop { display:block; }
#sheet .kind { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; }
#sheet .scroll { max-height:none; }
#loading-plot { margin-bottom:8px; }
.tabs { display:flex; gap:6px; margin:10px 0 12px; border-bottom:1px solid var(--line); }
.tabs button { border:0; border-bottom:2px solid transparent; border-radius:0; background:none; font-weight:600; color:var(--muted); padding:6px 10px; }
.tabs button.active { color:var(--accent); border-bottom-color:var(--accent); }
.tabs button:hover { background:none; color:var(--ink); }
.across-controls { display:flex; gap:10px; align-items:end; flex-wrap:wrap; margin-bottom:8px; }
"""

SCRIPT = r"""
const state = { runs:[], run:null, genes:[], geneSets:[], selectedGeneSet:null, selectedGene:null,
  geneSort:{col:'combined',desc:true}, gsSort:{col:'beta',desc:true}, highlighted:new Set() };
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

// ---------- run selection: landing (trait -> model -> run -> optional gene) and results bar
const uniq = a => [...new Set(a.filter(x => x !== null && x !== undefined && x !== ''))];
// Fuzzy score: exact > prefix > substring > in-order subsequence (with a small gap penalty); 0 = no match.
function fuzzyScore(query, text) {
  if (!query) return 1; if (!text) return 0;
  const q = query.toLowerCase(), t = text.toLowerCase();
  if (t === q) return 1000; if (t.startsWith(q)) return 800 - t.length * 0.01;
  const idx = t.indexOf(q); if (idx >= 0) return 600 - idx * 0.5 - t.length * 0.01;
  let ti = 0, gaps = 0, first = -1;
  for (const ch of q) { const j = t.indexOf(ch, ti); if (j < 0) return 0; if (first < 0) first = j; gaps += j - ti; ti = j + 1; }
  return Math.max(1, 300 - gaps * 4 - first - t.length * 0.01);
}
const traitScore = (t, q) => { const p = phenoOf(t); return Math.max(fuzzyScore(q, t), p ? fuzzyScore(q, p.name || '') * 0.98 : 0, p && (p.portal_id || '').toLowerCase() === q ? 1000 : 0); };
const phenoOf = t => { const r = state.runs.find(x => x.trait === t && x.phenotype); return r ? r.phenotype : null; };
const traitLabel = t => { const p = phenoOf(t); return p && p.name && p.name !== t ? `${p.name} (${t})` : t; };
function hashState() { const h = new URLSearchParams(location.hash.replace(/^#/, '')); return { run: h.get('run') || '', gene: h.get('gene') || '', gs: h.get('gs') || '' }; }
function setHash(obj) { const h = new URLSearchParams(); Object.entries(obj).forEach(([k,v]) => { if (v) h.set(k, v); }); const next = '#' + h.toString(); if (location.hash !== next) history.replaceState(null, '', next); }
async function loadRuns() {
  const body = await api('/api/runs');
  state.runs = body.runs;
  if (!state.runs.length) { $('landing-note').innerHTML = '<span class="warn">No runs in this database.</span>'; return; }
  const models = uniq(state.runs.map(r => r.model));
  $('model').innerHTML = '<option value="">any model</option>' + models.map(m => `<option value="${esc(m)}">${esc(m)}</option>`).join('');
  populateTraits(); populateRuns();
  const h = hashState();
  if (h.run && state.runs.some(r => r.run_id === h.run)) { state.run = h.run; $('run').value = h.run; await openResults(); if (h.gene) showGene(h.gene); else if (h.gs) showGeneSet(h.gs); }
  else showLanding();
}
function populateTraits() {
  const model = $('model').value;
  const traits = uniq(state.runs.filter(r => !model || r.model === model).map(r => r.trait)).sort();
  $('trait-list').innerHTML = traits.map(t => `<option value="${esc(t)}">${esc(traitLabel(t))}</option>`).join('');
  const q = $('trait').value.trim().toLowerCase();
  const hits = q ? traits.map(t => [t, traitScore(t, q)]).filter(x => x[1] > 0).sort((a,b) => b[1] - a[1]).map(x => x[0]) : traits;
  $('trait-hits').innerHTML = hits.slice(0, 60).map(t => `<button type="button" data-trait="${esc(t)}" class="${$('trait').value === t ? 'active' : ''}">${esc(traitLabel(t))}</button>`).join('') + (hits.length > 60 ? `<span class="hint">… ${hits.length - 60} more</span>` : '');
  $('trait-hits').querySelectorAll('button').forEach(b => b.onclick = () => { $('trait').value = b.dataset.trait; populateTraits(); populateRuns(); });
}
function matchingRuns() {
  const model = $('model').value, q = $('trait').value.trim().toLowerCase();
  const pool = state.runs.filter(r => !model || r.model === model);
  if (!q) return pool;
  const exact = pool.filter(r => (r.trait || '').toLowerCase() === q);
  if (exact.length) return exact;
  return pool.map(r => [r, Math.max(traitScore(r.trait || '', q), fuzzyScore(q, r.run_id) * 0.5)]).filter(x => x[1] > 0).sort((a,b) => b[1] - a[1]).map(x => x[0]);
}
function bestGene(query) {
  const q = query.trim(); if (!q) return '';
  const ranked = state.genes.map(g => [g.gene, fuzzyScore(q, g.gene)]).filter(x => x[1] > 0).sort((a,b) => b[1] - a[1]);
  return ranked.length ? ranked[0][0] : q.toUpperCase();
}
function populateRuns() {
  const runs = matchingRuns();
  const label = r => r.trait ? `${r.trait} · ${r.model}${r.seed && r.seed !== 'main' ? ' · ' + r.seed : ''}` : (r.title || r.run_id);
  $('run').innerHTML = runs.map(r => `<option value="${esc(r.run_id)}">${esc(label(r))}</option>`).join('');
  $('run-count').textContent = runs.length === state.runs.length ? `${runs.length} runs` : `${runs.length} of ${state.runs.length} runs`;
  if (!runs.some(r => r.run_id === state.run)) state.run = runs.length ? runs[0].run_id : null;
  if (state.run) $('run').value = state.run;
  $('open').disabled = !state.run;
}
function showLanding() { $('view-results').hidden = true; $('view-landing').hidden = false; closeSheet(); setHash({}); setTimeout(() => $('trait').focus(), 50); }
async function openResults() {
  if (!state.run) return;
  $('view-landing').hidden = true; $('view-results').hidden = false;
  setHash({ run: state.run });
  await refreshRun();
}
function runSummary() {
  const r = state.runs.find(x => x.run_id === state.run);
  if (!r) { $('crumb').innerHTML = '<span class="warn">No run selected.</span>'; $('run-summary').innerHTML = ''; return; }
  const ph = r.phenotype, f = r.filters || {};
  $('crumb').innerHTML = `<strong>${esc(ph && ph.name ? ph.name : (r.trait || r.title || r.run_id))}</strong>` +
    `<span class="muted">${esc(r.trait || '')}${ph && ph.portal_id ? ' · ' + esc(ph.portal_id) : ''}${ph && ph.trait_group ? ' · ' + esc(ph.trait_group) : ''}</span>` +
    `<span class="muted">model <b>${esc(r.model_title || r.model || '')}</b>${r.seed && r.seed !== 'main' ? ' · run <b>' + esc(r.seed) + '</b>' : ''}</span>`;
  const filt = ['genes','gene_sets','loadings'].map(k => (f[k] && f[k].length) ? `<b>${k}</b>: ${esc(f[k].join(` ${f.mode === 'all' ? 'AND' : 'OR'} `))}` : null).filter(Boolean).join(' &nbsp;·&nbsp; ') || 'no build-time filters';
  const paths = ['gene_stats_path','gene_set_stats_path','gene_gene_set_stats_path','params_path'].filter(k => r[k]).map(k => `<code>${esc(r[k])}</code>`).join('<br>');
  const maps = ph && ph.mappings ? ph.mappings : [];
  const conf = m => m.confidence === null || m.confidence === undefined ? '' : (+m.confidence).toFixed(2);
  const mapChips = maps.length ? `<div class="maps">${maps.map(m => `<a href="${esc(ontologyUrl(m.target_id))}" target="_blank" rel="noopener" title="${esc(m.target_label || '')} · ${esc((m.predicate || '').replace('skos:', ''))}${conf(m) ? ' · confidence ' + conf(m) : ''}"><span class="onto">${esc(m.target_ontology || m.target_id.split(':')[0])}</span>${esc(m.target_label || m.target_id)}</a>`).join('')}
      <button class="linkbtn" id="open-mappings" type="button">all ${maps.length} mappings</button></div>` : '';
  $('run-summary').innerHTML = mapChips + `<div class="counts">
      <div class="stat"><strong>${r.n_genes.toLocaleString()}</strong><span class="muted">genes of ${r.n_genes_input.toLocaleString()}</span></div>
      <div class="stat"><strong>${r.n_gene_sets.toLocaleString()}</strong><span class="muted">gene sets of ${r.n_gene_sets_input.toLocaleString()}</span></div>
      <div class="stat"><strong>${r.n_loadings.toLocaleString()}</strong><span class="muted">loadings of ${r.n_loadings_input.toLocaleString()}</span></div>
      <button class="linkbtn" id="open-build" type="button">Build details</button>
      ${r.params_path ? '<button class="linkbtn" id="open-params" type="button">Run parameters</button>' : ''}
    </div>`;
  $('open-build').onclick = () => openModal('Build details', `Run id <code>${esc(r.run_id)}</code><br>Model <b>${esc(r.model_title || r.model || '')}</b> · run <b>${esc(r.seed || '')}</b> · trait <b>${esc(r.trait || '')}</b><br>Filters (${esc(f.mode || 'any')}): ${filt}<br>Built ${esc(r.built_at)}<br>${paths}${(r.warnings||[]).map(w => `<div class="warn">${esc(w)}</div>`).join('')}`);
  const pb = $('open-params');
  if (pb) pb.onclick = async () => { openModal('Run parameters', '<span class="muted">loading…</span>'); const b = await api('/api/run_params', { run: r.run_id });
    $('modal-body').innerHTML = `<div class="muted" style="margin-bottom:8px"><code>${esc(r.params_path)}</code> · ${b.params.length} parameters</div><table><thead><tr><th>Parameter</th><th>Ver.</th><th>Value</th></tr></thead><tbody>${b.params.map(p => `<tr><td>${esc(p.parameter)}</td><td>${esc(p.version)}</td><td style="white-space:normal;word-break:break-all">${esc(p.value)}</td></tr>`).join('')}</tbody></table>`; };
  const mb = $('open-mappings');
  if (mb) mb.onclick = () => openModal(`Phenotype mappings: ${ph.name || r.trait}`, `<div class="muted" style="margin-bottom:8px">${esc(r.trait)} · ${esc(ph.portal_id || '')} · ${esc(ph.trait_group || '')}${ph.trait_type ? ' · ' + esc(ph.trait_type) : ''}${ph.is_dichotomous === '1' || ph.is_dichotomous === 'true' ? ' · dichotomous' : ''}${ph.legacy_trait_group ? ' · legacy group ' + esc(ph.legacy_trait_group) : ''}${ph.description && ph.description !== ph.name ? '<br>' + esc(ph.description) : ''}</div>
      <table><thead><tr><th>Ontology</th><th>ID</th><th>Label</th><th>Predicate</th><th class="num">Conf.</th><th>Justification</th><th>Source</th></tr></thead><tbody>
      ${maps.map(m => `<tr><td>${esc(m.target_ontology)}</td><td><a href="${esc(ontologyUrl(m.target_id))}" target="_blank" rel="noopener">${esc(m.target_id)}</a></td><td class="wrap">${esc(m.target_label)}</td><td>${esc((m.predicate || '').replace('skos:', ''))}</td><td class="num">${conf(m)}</td><td class="muted">${esc(m.justification || '')}</td><td class="muted">${esc(m.source || '')}</td></tr>`).join('')}</tbody></table>`);
}
// Resolvable link for an ontology CURIE (MESH:D003924, MONDO:0005148, EFO:0000275, ...).
function ontologyUrl(curie) {
  const [prefix, local] = (curie || '').split(':');
  if (!local) return 'https://bioregistry.io/' + encodeURIComponent(curie || '');
  const p = prefix.toUpperCase();
  if (p === 'MESH') return `https://meshb.nlm.nih.gov/record/ui?ui=${encodeURIComponent(local)}`;
  if (p === 'ORPHANET' || p === 'ORPHA') return `https://www.orpha.net/en/disease/detail/${encodeURIComponent(local)}`;
  if (p === 'ICD10CM' || p === 'ICD10') return `https://icd.who.int/browse10/2019/en#/${encodeURIComponent(local)}`;
  if (['EFO','MONDO','DOID','HP','OBA','CMO','CHEBI','NCIT','UBERON','GO','PATO','SNOMED','SNOMEDCT'].includes(p)) return `https://www.ebi.ac.uk/ols4/search?q=${encodeURIComponent(curie)}`;
  return 'https://bioregistry.io/' + encodeURIComponent(curie);
}
function openModal(title, bodyHtml) { $('modal-title').textContent = title; $('modal-body').innerHTML = bodyHtml; const d = $('modal'); if (!d.open) d.showModal(); }
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
  fillGeneList('');
  drawScatter();
  renderGeneTable();
}
function fillGeneList(q) {
  const ranked = q ? state.genes.map(g => [g.gene, fuzzyScore(q, g.gene)]).filter(x => x[1] > 0).sort((a,b) => b[1] - a[1]).slice(0, 30).map(x => x[0]) : state.genes.slice(0, 200).map(g => g.gene);
  $('gene-list').innerHTML = ranked.map(g => `<option value="${esc(g)}">`).join('');
}
function drawScatter() {
  const g = state.genes, hl = state.highlighted;
  // Highlighted genes keep the same combined colour scale but are drawn on top with a white ring;
  // everything else fades so the set's genes stand out in context.
  const base = g.filter(r => !hl.has(r.gene)), top = g.filter(r => hl.has(r.gene));
  const hover = '<b>%{text}</b><br>log_bf %{x:.3f}<br>prior %{y:.3f}<br>combined %{customdata[0]:.3f}<br>huge %{customdata[1]:.3f}<extra></extra>';
  const vals = g.map(r => r.combined).filter(v => v !== null && v !== undefined);
  const cmin = vals.length ? Math.min(...vals) : 0, cmax = vals.length ? Math.max(...vals) : 1;
  const mk = rows => ({ type: 'scattergl', mode: 'markers', x: rows.map(r => r.log_bf), y: rows.map(r => r.prior), text: rows.map(r => r.gene), customdata: rows.map(r => [r.combined, r.huge_score]), hovertemplate: hover });
  const traces = [Object.assign(mk(base), { name: 'genes', marker: { size: 6, opacity: hl.size ? 0.12 : 0.75, color: base.map(r => r.combined), colorscale: 'Viridis', cmin, cmax, colorbar: { title: 'combined', thickness: 12 } } })];
  if (top.length) traces.push(Object.assign(mk(top), { name: 'in gene set', marker: { size: 10, opacity: 1, color: top.map(r => r.combined), colorscale: 'Viridis', cmin, cmax, showscale: false, line: { width: 2, color: '#fff' } } }));
  const layout = { margin: { l: 55, r: 10, t: 10, b: 45 }, xaxis: { title: 'log_bf (direct)', zeroline: true }, yaxis: { title: 'prior (indirect)', zeroline: true }, height: 460, hovermode: 'closest', showlegend: false, plot_bgcolor: hl.size ? '#eef1f3' : '#fff' };
  Plotly.react('scatter', traces, layout, { responsive: true, displaylogo: false });
  $('scatter').on('plotly_click', ev => { const p = ev.points && ev.points[0]; if (p) showGene(p.text); });
  $('scatter-count').innerHTML = `${g.length.toLocaleString()} genes shown` + (hl.size ? ` <span class="chip">${top.length} highlighted</span>` : '');
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
    rows.map((r,i) => `<tr class="row ${r.gene_set===state.selectedGeneSet?'selected':''}" data-gs="${esc(r.gene_set)}"><td title="${esc(r.gene_set)}">${i+1}. ${esc(r.gene_set)}</td><td class="wrap">${esc(r.label)}</td><td class="num">${fmt(r.n)}</td><td class="num">${fmt(r.beta)}</td><td class="num">${fmt(r.beta_uncorrected)}</td><td class="num">${fmt(r.p_orig)}</td></tr>`).join('') + '</tbody>';
  bindSort($('gs-table'), s, ['gene_set','label','p_orig'], () => { $('gs_sort').value = s.col; renderGeneSetTable(); });
  $('gs-table').querySelectorAll('tr.row').forEach(tr => tr.onclick = () => showGeneSet(tr.dataset.gs));
  $('gs-count').textContent = `${rows.length.toLocaleString()} gene sets shown`;
}

// ---------- detail sheet
function kv(obj, keys) { return '<div class="kv">' + keys.filter(k => obj[k] !== undefined && obj[k] !== null && obj[k] !== '').map(k => `<div><span>${esc(k)}</span>${esc(typeof obj[k]==='number'?fmt(obj[k]):obj[k])}</div>`).join('') + '</div>'; }
function openSheet(kind, title, bodyHtml, across) {
  $('sheet-kind').textContent = kind; $('sheet-title').textContent = title;
  const runLabel = (state.runs.find(r => r.run_id === state.run) || {}).trait || 'this run';
  $('sheet-body').innerHTML = across
    ? `<div class="tabs"><button class="active" data-tab="this">Details for ${esc(runLabel)}</button><button data-tab="across">Across traits</button></div><div id="tab-this">${bodyHtml}</div><div id="tab-across" hidden></div>`
    : bodyHtml;
  if (across) $('sheet-body').querySelectorAll('.tabs button').forEach(b => b.onclick = () => {
    $('sheet-body').querySelectorAll('.tabs button').forEach(x => x.classList.toggle('active', x === b));
    $('tab-this').hidden = b.dataset.tab !== 'this'; $('tab-across').hidden = b.dataset.tab !== 'across';
    if (b.dataset.tab === 'across' && !$('tab-across').dataset.loaded) { $('tab-across').dataset.loaded = '1'; across(); }
    window.dispatchEvent(new Event('resize'));
  });
  document.body.classList.add('sheet-open');
  window.dispatchEvent(new Event('resize'));
}

// ---------- across-traits view (vertical trait plot, one point per run, grouped by trait group)
const GROUP_COLORS = ['#1f77b4','#2ca02c','#9467bd','#d62728','#ff7f0e','#8c564b','#e377c2','#17becf','#bcbd22','#7f7f7f','#0f766e','#c2410c'];
async function renderAcross(kind, id, metrics) {
  const el = $('tab-across');
  const models = uniq(state.runs.map(r => r.model));
  const current = (state.runs.find(r => r.run_id === state.run) || {}).model || '';
  el.innerHTML = `<div class="across-controls">
      <div><label>metric</label><select id="across-metric">${metrics.map(m => `<option value="${m}">${m}</option>`).join('')}</select></div>
      <div><label>model</label><select id="across-model"><option value="">all models</option>${models.map(m => `<option value="${esc(m)}" ${m===current?'selected':''}>${esc(m)}</option>`).join('')}</select></div>
      <div class="muted" id="across-note" style="flex:1"></div>
    </div><div id="across-plot"></div><div class="scroll" style="max-height:320px"><table id="across-table"></table></div>`;
  const draw = async () => {
    const metric = $('across-metric').value, model = $('across-model').value;
    const body = await api(kind === 'gene' ? '/api/gene_across' : '/api/gene_set_across', { id, model });
    const rows = body.rows.filter(r => r[metric] !== null && r[metric] !== undefined);
    $('across-note').textContent = `${rows.length} run${rows.length===1?'':'s'} where ${id} passed the build thresholds`;
    if (!rows.length) { Plotly.purge('across-plot'); $('across-table').innerHTML = ''; return; }
    rows.sort((a,b) => (a.trait_group||'~').localeCompare(b.trait_group||'~') || (b[metric] - a[metric]));
    const label = r => `${r.phenotype_name || r.trait}${models.length > 1 && !model ? ' · ' + r.model : ''}${r.seed && r.seed !== 'main' ? ' · ' + r.seed : ''}`;
    const y = rows.map(label), groups = uniq(rows.map(r => r.trait_group || 'other'));
    const traces = groups.map((gname, i) => { const sel = rows.filter(r => (r.trait_group || 'other') === gname); return {
      type: 'scatter', mode: 'markers', name: gname, orientation: 'h', y: sel.map(label), x: sel.map(r => r[metric]),
      marker: { size: 9, color: GROUP_COLORS[i % GROUP_COLORS.length], line: { width: 1, color: '#fff' } },
      customdata: sel.map(r => [r.trait, r.portal_id || '', r.run_id]),
      hovertemplate: `<b>%{y}</b><br>${metric} %{x:.3f}<br>%{customdata[0]} %{customdata[1]}<br>%{customdata[2]}<extra>${esc(gname)}</extra>` }; });
    const h = Math.max(220, 40 + 18 * rows.length);
    Plotly.react('across-plot', traces, { height: Math.min(h, 900), margin: { l: 10, r: 10, t: 10, b: 40 }, xaxis: { title: metric, zeroline: true },
      yaxis: { automargin: true, categoryorder: 'array', categoryarray: y.slice().reverse(), tickfont: { size: 10 } }, legend: { orientation: 'h', y: -0.12, font: { size: 10 } }, hovermode: 'closest' }, { responsive: true, displaylogo: false });
    $('across-plot').on('plotly_click', ev => { const p = ev.points && ev.points[0]; if (p && p.customdata) { state.run = p.customdata[2]; $('model').value = ''; $('trait').value = ''; populateTraits(); populateRuns(); $('run').value = state.run; openResults().then(() => kind === 'gene' ? showGene(id) : showGeneSet(id)); } });
    const cols = kind === 'gene' ? ['combined','log_bf','prior','huge_score'] : ['beta','beta_uncorrected','n'];
    $('across-table').innerHTML = `<thead><tr><th>Trait</th><th>Model</th><th>Run</th><th>Group</th>${cols.map(c => `<th class="num">${c}</th>`).join('')}</tr></thead><tbody>` +
      rows.map(r => `<tr><td title="${esc(r.trait)} ${esc(r.portal_id||'')}">${esc(r.phenotype_name || r.trait)}</td><td>${esc(r.model)}</td><td>${esc(r.seed)}</td><td>${esc(r.trait_group||'')}</td>${cols.map(c => `<td class="num">${fmt(r[c])}</td>`).join('')}</tr>`).join('') + '</tbody>';
  };
  $('across-metric').onchange = draw; $('across-model').onchange = draw;
  await draw();
}
function closeSheet() { document.body.classList.remove('sheet-open'); if (state.run && !$('view-results').hidden) setHash({ run: state.run }); window.dispatchEvent(new Event('resize')); }
function setHighlight(genes) { state.highlighted = new Set(genes); drawScatter(); renderGeneTable(); }

async function showGeneSet(id) {
  const d = await api('/api/gene_set', { run: state.run, id, limit: 1000 });
  state.selectedGeneSet = id; state.selectedGene = null; setHash({ run: state.run, gs: id });
  const L = d.loadings;
  openSheet('Gene set', id,
    kv(d, ['label','n','beta','beta_uncorrected','p_orig','z_orig']) +
    `<h3>Gene loadings <span class="muted">${L.length.toLocaleString()} of ${d.n_loadings.toLocaleString()}, by weight then combined</span></h3>` +
    `<div id="loading-plot" style="height:${Math.min(700, 60 + 16 * Math.min(L.length, 40))}px"></div>` +
    `<div class="scroll"><table id="loading-table"><thead><tr><th>Gene</th><th class="num">weight</th><th class="num">beta</th><th class="num">combined</th><th class="num">log_bf</th><th class="num">prior</th></tr></thead><tbody>` +
    L.map(r => `<tr class="row" data-gene="${esc(r.gene)}"><td>${esc(r.gene)}</td><td class="num">${fmt(r.weight)}</td><td class="num">${fmt(r.beta)}</td><td class="num">${fmt(r.combined)}</td><td class="num">${fmt(r.log_bf)}</td><td class="num">${fmt(r.prior)}</td></tr>`).join('') + '</tbody></table></div>' +
    `<details style="margin-top:10px"><summary class="muted">all columns</summary>${kv(d.extra, Object.keys(d.extra))}</details>`,
    () => renderAcross('gene_set', id, ['beta', 'beta_uncorrected']));
  $('sheet-body').querySelectorAll('tr.row').forEach(tr => tr.onclick = () => showGene(tr.dataset.gene));
  const top = L.slice(0, 40).reverse();
  if (top.length) Plotly.react('loading-plot', [{ type: 'bar', orientation: 'h', y: top.map(r => r.gene), x: top.map(r => r.combined ?? 0), marker: { color: top.map(r => r.combined ?? 0), colorscale: 'Viridis', opacity: top.map(r => 0.45 + 0.55 * (r.weight ?? 0)) }, hovertemplate: '%{y}: combined %{x:.3f}, weight %{customdata:.2f}<extra></extra>', customdata: top.map(r => r.weight ?? 0) }],
    { margin: { l: 100, r: 10, t: 4, b: 30 }, xaxis: { title: 'combined (gene score)' }, yaxis: { automargin: true, tickfont: { size: 10 } } }, { responsive: true, displaylogo: false });
  setHighlight(L.map(r => r.gene));
  renderGeneSetTable();
}
async function showGene(gene) {
  let d;
  try { d = await api('/api/gene', { run: state.run, id: gene, limit: 1000 }); }
  catch (err) { openSheet('Gene', gene, `<p class="warn">${esc(err.message)} — it may not have passed the build thresholds for this run.</p>`, () => renderAcross('gene', gene, ['combined', 'log_bf', 'prior', 'huge_score'])); return; }
  state.selectedGene = gene; setHash({ run: state.run, gene });
  openSheet('Gene', gene,
    kv(d, ['combined','log_bf','prior','huge_score','n','chrom','start','end']) +
    `<h3>Gene sets <span class="muted">${d.gene_sets.length.toLocaleString()} loadings, by beta</span></h3>` +
    `<div class="scroll"><table><thead><tr><th>Gene set</th><th>Label</th><th class="num">beta</th><th class="num">weight</th><th class="num">beta_unc</th></tr></thead><tbody>` +
    d.gene_sets.map(r => `<tr class="row" data-gs="${esc(r.gene_set)}"><td title="${esc(r.gene_set)}">${esc(r.gene_set)}</td><td class="wrap">${esc(r.label)}</td><td class="num">${fmt(r.beta)}</td><td class="num">${fmt(r.weight)}</td><td class="num">${fmt(r.beta_uncorrected)}</td></tr>`).join('') + '</tbody></table></div>' +
    `<details style="margin-top:10px"><summary class="muted">all columns</summary>${kv(d.extra, Object.keys(d.extra))}</details>`,
    () => renderAcross('gene', gene, ['combined', 'log_bf', 'prior', 'huge_score']));
  $('sheet-body').querySelectorAll('tr.row').forEach(tr => tr.onclick = () => showGeneSet(tr.dataset.gs));
  setHighlight([gene]);
}

// ---------- wiring
let t1, t2;
$('model').onchange = () => { populateTraits(); populateRuns(); };
$('trait').oninput = () => { populateTraits(); populateRuns(); };
$('trait').onkeydown = e => { if (e.key === 'Enter') { populateRuns(); if (state.run) openWithGene(); } };
$('run').onchange = e => { state.run = e.target.value; };
$('open').onclick = openWithGene;
async function openWithGene() { await openResults(); const g = $('gene_landing').value.trim(); if (g) { const best = bestGene(g); $('gene_search').value = best; showGene(best); } }
$('back').onclick = showLanding;
['min_prior','min_log_bf','min_combined'].forEach(id => $(id).oninput = () => { clearTimeout(t1); t1 = setTimeout(loadGenes, 350); });
['min_beta','min_beta_uncorrected','gs_search'].forEach(id => $(id).oninput = () => { clearTimeout(t2); t2 = setTimeout(loadGeneSets, 350); });
$('gs_sort').onchange = () => { const c = $('gs_sort').value; state.gsSort = { col: c, desc: !['gene_set','label','p_orig'].includes(c) }; renderGeneSetTable(); };
$('gene_search').oninput = () => fillGeneList($('gene_search').value.trim());
$('gene_search').onchange = () => { const v = $('gene_search').value.trim(); if (v) showGene(bestGene(v)); };
$('modal-close').onclick = () => $('modal').close();
$('modal').onclick = e => { if (e.target === $('modal')) $('modal').close(); };
$('sheet-close').onclick = closeSheet;
$('sheet-backdrop').onclick = closeSheet;
$('clear-highlight').onclick = () => { state.selectedGeneSet = null; state.selectedGene = null; setHighlight([]); renderGeneSetTable(); };
document.addEventListener('keydown', e => { if (e.key === 'Escape' && !$('modal').open) closeSheet(); });
window.addEventListener('hashchange', () => { const h = hashState(); if (!h.run && !$('view-results').hidden) showLanding(); });
loadRuns().catch(err => { $('landing-note').innerHTML = `<span class="warn">${esc(err.message)}</span>`; $('view-landing').hidden = false; });
"""

BODY = r"""
<div id="view-landing" hidden>
  <div class="landing">
    <h1>{title}</h1>
    <p class="lede">Browse thresholded PIGEAN results: genes, gene sets and their loadings, across traits and models.{api_note}</p>
    <div class="field"><label for="trait">Trait</label><input id="trait" class="big" list="trait-list" placeholder="type a trait name, legacy id or portal id…" autocomplete="off"><datalist id="trait-list"></datalist><div id="trait-hits" class="trait-hits"></div></div>
    <div class="row">
      <div class="field"><label for="model">Model</label><select id="model"></select></div>
      <div class="field"><label for="run">Run <span id="run-count" class="muted"></span></label><select id="run"></select></div>
    </div>
    <div class="field"><label for="gene_landing">Gene <span class="muted">(optional — opens that gene's card)</span></label><input id="gene_landing" placeholder="e.g. TCF7L2" style="width:100%"></div>
    <div class="actions"><button id="open" class="primary" type="button" disabled>Open results</button><span id="landing-note" class="hint">Enter in the trait box also opens the first matching run.</span></div>
  </div>
</div>
<div id="view-results" class="shell" hidden>
  <div class="bar">
    <button id="back" type="button" title="back to search">◀ Search</button>
    <div class="crumb" id="crumb"></div>
    <div class="right"><div><label for="gene_search">Gene</label><input id="gene_search" list="gene-list" placeholder="e.g. TCF7L2 ⏎" style="width:14ch"><datalist id="gene-list"></datalist></div></div>
  </div>
  <div id="run-summary"></div>
  <div class="grid">
    <section class="panel">
      <h2>Genes: direct (log_bf) vs indirect (prior)</h2>
      <div class="controls">
        <div><label>min prior</label><input id="min_prior" type="number" step="0.1"></div>
        <div><label>min log_bf</label><input id="min_log_bf" type="number" step="0.1"></div>
        <div><label>min combined</label><input id="min_combined" type="number" step="0.1"></div>
        <div style="margin-left:auto"><button id="clear-highlight" type="button">Clear highlight</button></div>
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
        <div><label for="gs_sort">sort by</label><select id="gs_sort"><option value="beta">beta</option><option value="beta_uncorrected">beta_uncorrected</option><option value="n">N</option><option value="p_orig">P (asc)</option><option value="gene_set">name</option></select></div>
        <div style="flex:1"><label>search (id or label)</label><input id="gs_search" placeholder="e.g. insulin" style="width:100%"></div>
      </div>
      <div class="scroll" style="max-height:820px"><table id="gs-table"></table></div>
      <div class="muted" id="gs-count"></div>
    </section>
  </div>
</div>
<dialog id="modal" class="modal"><div class="modal-head"><h2 id="modal-title"></h2><button id="modal-close" type="button" title="close">✕</button></div><div class="modal-body" id="modal-body"></div></dialog>
<div id="sheet-backdrop"></div>
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
