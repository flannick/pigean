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

from .portal_assets_common import render_document

CSS = ""  # the shared stylesheet lives in portal_assets_common

SCRIPT = r"""
const state = { runs:[], run:null, genes:[], geneSets:[], selectedGeneSet:null, selectedGene:null,
  geneSort:{col:'combined',desc:true}, gsSort:{col:'beta',desc:true}, highlighted:new Set() };
// ---------- run selection: landing (trait -> model -> run -> optional gene) and results bar
function hashState() { const h = new URLSearchParams(location.hash.replace(/^#/, '')); return { run: h.get('run') || '', gene: h.get('gene') || '', gs: h.get('gs') || '' }; }
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
function availableTraits() {
  const model = $('model').value;
  return uniq(state.runs.filter(r => !model || r.model === model).map(r => r.trait)).sort();
}
function traitItems(q) {
  const traits = availableTraits();
  const ranked = q ? traits.map(t => [t, traitScore(t, q.toLowerCase())]).filter(x => x[1] > 0).sort((a,b) => b[1] - a[1]).map(x => x[0]) : traits;
  return ranked.map(t => { const p = phenoOf(t); return { value: t, label: t, sub: p && p.name && p.name !== t ? p.name : '' }; });
}
function populateTraits() { /* traits are looked up live by the typeahead */ }
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
  const prev = state.run;
  if (!runs.some(r => r.run_id === state.run)) state.run = runs.length ? runs[0].run_id : null;
  if (state.run) $('run').value = state.run;
  $('open').disabled = !state.run;
  if (state.run !== prev) prefetchGenes();
}
function showLanding() { $('view-results').hidden = true; $('view-landing').hidden = false; closeSheet(); setHash({}); setTimeout(() => $('trait').focus(), 50); }
async function openResults() {
  if (!state.run) return;
  $('view-landing').hidden = true; $('view-results').hidden = false;
  setHash({ run: state.run });
  await refreshRun();
}
// Model / run switchers in the results bar, scoped to the trait currently shown.
function populateBarSelectors(r) {
  const same = state.runs.filter(x => (x.trait || x.run_id) === (r.trait || r.run_id));
  const models = uniq(same.map(x => x.model));
  $('bar_model').innerHTML = models.map(m => `<option value="${esc(m)}" ${m === r.model ? 'selected' : ''}>${esc(m)}</option>`).join('');
  const runs = same.filter(x => x.model === r.model);
  $('bar_run').innerHTML = runs.map(x => `<option value="${esc(x.run_id)}" ${x.run_id === r.run_id ? 'selected' : ''}>${esc(x.seed || x.title || x.run_id)}</option>`).join('');
  $('bar_run').parentElement.hidden = runs.length < 2;
  $('bar_model').parentElement.hidden = models.length < 2;
}
function switchRun(runId) { if (!runId || runId === state.run) return; state.run = runId; $('run').value = runId; openResults(); }
function runSummary() {
  const r = state.runs.find(x => x.run_id === state.run);
  if (!r) { $('crumb').innerHTML = '<span class="warn">No run selected.</span>'; $('run-summary').innerHTML = ''; return; }
  populateBarSelectors(r);
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
    $('modal-body').innerHTML = `<div class="controls"><div style="flex:1"><label for="param-search">search parameter (fuzzy)</label><input id="param-search" placeholder="e.g. burn in, seed, chains" style="width:100%" autocomplete="off"></div><div class="muted" id="param-count" style="align-self:end"></div></div>
      <div class="muted" style="margin-bottom:8px"><code>${esc(r.params_path)}</code></div><table><thead><tr><th>Parameter</th><th>Ver.</th><th>Value</th></tr></thead><tbody id="param-rows"></tbody></table>`;
    const render = () => { const q = $('param-search').value.trim();
      const rows = q ? b.params.map(x => [x, fuzzyScore(q, x.parameter)]).filter(x => x[1] > 0).sort((a,c) => c[1] - a[1]).map(x => x[0]) : b.params;
      $('param-rows').innerHTML = rows.map(x => `<tr><td>${esc(x.parameter)}</td><td>${esc(x.version)}</td><td style="white-space:normal;word-break:break-all">${esc(x.value)}</td></tr>`).join('');
      $('param-count').textContent = `${rows.length} of ${b.params.length} parameters`; };
    $('param-search').oninput = render; render(); setTimeout(() => $('param-search').focus(), 50); };
  const mb = $('open-mappings');
  if (mb) mb.onclick = () => openModal(`Phenotype mappings: ${ph.name || r.trait}`, `<div class="muted" style="margin-bottom:8px">${esc(r.trait)} · ${esc(ph.portal_id || '')} · ${esc(ph.trait_group || '')}${ph.trait_type ? ' · ' + esc(ph.trait_type) : ''}${ph.is_dichotomous === '1' || ph.is_dichotomous === 'true' ? ' · dichotomous' : ''}${ph.legacy_trait_group ? ' · legacy group ' + esc(ph.legacy_trait_group) : ''}${ph.description && ph.description !== ph.name ? '<br>' + esc(ph.description) : ''}</div>
      <table><thead><tr><th>Ontology</th><th>ID</th><th>Label</th><th>Predicate</th><th class="num">Conf.</th><th>Justification</th><th>Source</th></tr></thead><tbody>
      ${maps.map(m => `<tr><td>${esc(m.target_ontology)}</td><td><a href="${esc(ontologyUrl(m.target_id))}" target="_blank" rel="noopener">${esc(m.target_id)}</a></td><td class="wrap">${esc(m.target_label)}</td><td>${esc((m.predicate || '').replace('skos:', ''))}</td><td class="num">${conf(m)}</td><td class="muted">${esc(m.justification || '')}</td><td class="muted">${esc(m.source || '')}</td></tr>`).join('')}</tbody></table>`);
}
// Resolvable link for an ontology CURIE (MESH:D003924, MONDO:0005148, EFO:0000275, ...).
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

  drawScatter();
  renderGeneTable();
}
function geneItems(q) {
  const ranked = q ? state.genes.map(g => [g, fuzzyScore(q, g.gene)]).filter(x => x[1] > 0).sort((a,b) => b[1] - a[1]).map(x => x[0]) : state.genes.slice(0, 8);
  return ranked.map(g => ({ value: g.gene, label: g.gene, sub: `combined ${fmt(g.combined)} · log_bf ${fmt(g.log_bf)} · prior ${fmt(g.prior)}` }));
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
  const cols = [['gene_set','Gene set',false],['label','Library',false],['n','N',true],['beta','beta',true],['beta_uncorrected','beta_unc',true],['p_orig','P',true]];
  const rows = sortRows(state.geneSets, s);
  $('gs-table').innerHTML = '<thead><tr>' + cols.map(([c,t,n]) => `<th class="${n?'num':''}" data-col="${c}">${t}${sortMark(s,c)}</th>`).join('') + '</tr></thead><tbody>' +
    rows.map((r,i) => `<tr class="row ${r.gene_set===state.selectedGeneSet?'selected':''}" data-gs="${esc(r.gene_set)}"><td data-full="${esc(r.gene_set)}">${i+1}. ${esc(r.gene_set)}</td><td class="wrap">${esc(r.label)}</td><td class="num">${fmt(r.n)}</td><td class="num">${fmt(r.beta)}</td><td class="num">${fmt(r.beta_uncorrected)}</td><td class="num">${fmt(r.p_orig)}</td></tr>`).join('') + '</tbody>';
  bindSort($('gs-table'), s, ['gene_set','label','p_orig'], () => { $('gs_sort').value = s.col; renderGeneSetTable(); });
  $('gs-table').querySelectorAll('tr.row').forEach(tr => tr.onclick = () => showGeneSet(tr.dataset.gs));
  $('gs-count').textContent = `${rows.length.toLocaleString()} gene sets shown`;
}

// ---------- detail sheet
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
  const d = await api('/api/gene_set', { run: state.run, id, limit: 100000 });
  state.selectedGeneSet = id; state.selectedGene = null; setHash({ run: state.run, gs: id });
  const L = d.loadings;
  openSheet('Gene set', id,
    kv(d, ['label','n','beta','beta_uncorrected','p_orig','z_orig']) +
    `<h3>Gene loadings <span class="muted">${L.length.toLocaleString()} genes in this set (top 40 charted)</span></h3>` +
    `<div class="sheet-controls"><div><label for="gs-metric">gene score</label><select id="gs-metric"><option value="combined">combined</option><option value="prior">prior</option><option value="log_bf">log_bf</option><option value="huge_score">huge_score</option></select></div><div class="ta-wrap" style="flex:1"><label for="gs-gene-filter">filter genes</label><input id="gs-gene-filter" placeholder="fuzzy, e.g. slc2" autocomplete="off"></div></div>` +
    `<div id="loading-plot" style="height:${Math.min(700, 60 + 16 * Math.min(L.length, 40))}px"></div><div id="loading-table"></div>` +
    `<details style="margin-top:10px"><summary class="muted">all columns</summary>${kv(d.extra, Object.keys(d.extra))}</details>`,
    () => renderAcross('gene_set', id, ['beta', 'beta_uncorrected']));
  const draw = () => {
    const metric = $('gs-metric').value, q = $('gs-gene-filter').value.trim();
    let rows = byMetric(L, metric);
    if (q) rows = rows.map(r => [r, fuzzyScore(q, r.gene)]).filter(x => x[1] > 0).sort((a,b) => b[1] - a[1]).map(x => x[0]);
    hBar('loading-plot', rows, 'gene', metric, 'weight', Math.min(700, 60 + 16 * Math.min(rows.length, 40)));
    pagedTable('loading-table', `<tr><th>Gene</th><th class="num">${metric}</th><th class="num">weight</th><th class="num">beta</th><th class="num">combined</th><th class="num">log_bf</th><th class="num">prior</th></tr>`, rows,
      r => `<tr class="row" data-gene="${esc(r.gene)}"><td>${esc(r.gene)}</td><td class="num"><b>${fmt(r[metric])}</b></td><td class="num">${fmt(r.weight)}</td><td class="num">${fmt(r.beta)}</td><td class="num">${fmt(r.combined)}</td><td class="num">${fmt(r.log_bf)}</td><td class="num">${fmt(r.prior)}</td></tr>`,
      tr => tr.onclick = () => showGene(tr.dataset.gene));
  };
  $('gs-metric').onchange = draw; $('gs-gene-filter').oninput = draw; draw();
  setHighlight(L.map(r => r.gene));
  renderGeneSetTable();
}
async function showGene(gene) {
  let d;
  try { d = await api('/api/gene', { run: state.run, id: gene, limit: 100000 }); }
  catch (err) { openSheet('Gene', gene, `<p class="warn">${esc(err.message)} — it may not have passed the build thresholds for this run.</p>`, () => renderAcross('gene', gene, ['combined', 'log_bf', 'prior', 'huge_score'])); return; }
  state.selectedGene = gene; setHash({ run: state.run, gene });
  const G = d.gene_sets;
  openSheet('Gene', gene,
    kv(d, ['combined','log_bf','prior','huge_score','n','chrom','start','end']) +
    `<h3>Gene sets <span class="muted">${G.length.toLocaleString()} sets load on this gene (top 40 charted)</span></h3>` +
    `<div class="sheet-controls"><div><label for="g-metric">gene set score</label><select id="g-metric"><option value="beta">beta</option><option value="beta_uncorrected">beta_uncorrected</option><option value="weight">weight</option></select></div><div style="flex:1"><label for="g-gs-filter">filter gene sets</label><input id="g-gs-filter" placeholder="fuzzy, id or label" autocomplete="off"></div></div>` +
    `<div id="gs-plot"></div><div id="gs-member-table"></div>` +
    `<details style="margin-top:10px"><summary class="muted">all columns</summary>${kv(d.extra, Object.keys(d.extra))}</details>`,
    () => renderAcross('gene', gene, ['combined', 'log_bf', 'prior', 'huge_score']));
  const draw = () => {
    const metric = $('g-metric').value, q = $('g-gs-filter').value.trim();
    let rows = byMetric(G, metric);
    if (q) rows = rows.map(r => [r, Math.max(fuzzyScore(q, r.gene_set), fuzzyScore(q, r.label || '') * 0.98)]).filter(x => x[1] > 0).sort((a,b) => b[1] - a[1]).map(x => x[0]);
    $('gs-plot').style.height = `${Math.min(700, 60 + 16 * Math.min(rows.length, 40))}px`;
    hBar('gs-plot', rows.map(r => ({ ...r, short: r.gene_set.length > 38 ? r.gene_set.slice(0, 36) + '…' : r.gene_set })), 'short', metric, metric === 'weight' ? null : 'weight', Math.min(700, 60 + 16 * Math.min(rows.length, 40)));
    pagedTable('gs-member-table', `<tr><th>Gene set</th><th>Library</th><th class="num">${metric}</th><th class="num">beta</th><th class="num">weight</th><th class="num">beta_unc</th></tr>`, rows,
      r => `<tr class="row" data-gs="${esc(r.gene_set)}"><td data-full="${esc(r.gene_set)}">${esc(r.gene_set)}</td><td class="wrap">${esc(r.label)}</td><td class="num"><b>${fmt(r[metric])}</b></td><td class="num">${fmt(r.beta)}</td><td class="num">${fmt(r.weight)}</td><td class="num">${fmt(r.beta_uncorrected)}</td></tr>`,
      tr => tr.onclick = () => showGeneSet(tr.dataset.gs));
  };
  $('g-metric').onchange = draw; $('g-gs-filter').oninput = draw; draw();
  setHighlight([gene]);
}
// ---------- wiring
let t1, t2;
$('model').onchange = () => { populateTraits(); populateRuns(); };
attachTypeahead($('trait'), traitItems, () => populateRuns());
$('trait').addEventListener('input', populateRuns);
$('trait').addEventListener('keydown', e => { if (e.key === 'Enter') { populateRuns(); if (state.run) openWithGene(); } });
attachTypeahead($('gene_landing'), q => { const runGenes = state.genes.length ? geneItems(q) : []; return runGenes; }, () => {});
$('run').onchange = e => { state.run = e.target.value; prefetchGenes(); };
async function prefetchGenes() { if (!state.run) return; try { const b = await api('/api/genes', { run: state.run, sort: 'combined', limit: 20000 }); state.genes = b.genes; } catch (e) { state.genes = []; } }
$('open').onclick = openWithGene;
async function openWithGene() { await openResults(); const g = $('gene_landing').value.trim(); if (g) { const best = bestGene(g); $('gene_search').value = best; showGene(best); } }
$('back').onclick = showLanding;
$('bar_model').onchange = () => { const r = state.runs.find(x => x.run_id === state.run); const m = $('bar_model').value;
  const same = state.runs.filter(x => (x.trait || x.run_id) === (r.trait || r.run_id) && x.model === m);
  const keep = same.find(x => x.seed === r.seed) || same[0]; if (keep) switchRun(keep.run_id); };
$('bar_run').onchange = () => switchRun($('bar_run').value);
['min_prior','min_log_bf','min_combined'].forEach(id => $(id).oninput = () => { clearTimeout(t1); t1 = setTimeout(loadGenes, 350); });
['min_beta','min_beta_uncorrected','gs_search'].forEach(id => $(id).oninput = () => { clearTimeout(t2); t2 = setTimeout(loadGeneSets, 350); });
$('gs_sort').onchange = () => { const c = $('gs_sort').value; state.gsSort = { col: c, desc: !['gene_set','label','p_orig'].includes(c) }; renderGeneSetTable(); };
attachTypeahead($('gene_search'), geneItems, (_, v) => showGene(v));
$('gene_search').addEventListener('keydown', e => { if (e.key === 'Enter') { const v = $('gene_search').value.trim(); if (v) showGene(bestGene(v)); } });
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
    <p class="lede">Browse PIGEAN results across traits, models, and genes.{api_note}</p>
    <div class="field ta-wrap"><label for="trait">Trait</label><input id="trait" placeholder="Search traits by name or id" autocomplete="off"></div>
    <div class="row">
      <div class="field"><label for="model">Model</label><select id="model"></select></div>
      <div class="field"><label for="run">Run <span id="run-count" class="muted"></span></label><select id="run"></select></div>
    </div>
    <div class="field ta-wrap"><label for="gene_landing">Gene <span class="muted">(optional)</span></label><input id="gene_landing" placeholder="e.g. TCF7L2" autocomplete="off"></div>
    <div class="actions"><button id="open" class="primary" type="button" disabled>Open results</button><span id="landing-note" class="hint"></span></div>
  </div>
</div>
<div id="view-results" class="shell" hidden>
  <div class="bar">
    <button id="back" type="button" title="back to search">◀ Search</button>
    <div class="crumb" id="crumb"></div>
    <div class="right">
      <div><label for="bar_model">Model</label><select id="bar_model"></select></div>
      <div><label for="bar_run">Run</label><select id="bar_run"></select></div>
    </div>
  </div>
  <div id="run-summary"></div>
  <div class="grid">
    <section class="panel">
      <h2>Genes: direct (log_bf) vs indirect (prior)</h2>
      <div class="controls">
        <div class="ta-wrap" style="flex:1"><label for="gene_search">search gene</label><input id="gene_search" placeholder="e.g. TCF7L2 — opens the gene card" style="width:100%" autocomplete="off"></div>
        <div><button id="clear-highlight" type="button">Clear highlight</button></div>
      </div>
      <details class="adv"><summary>Advanced filters</summary><div class="controls">
        <div><label>min prior</label><input id="min_prior" type="number" step="0.1"></div>
        <div><label>min log_bf</label><input id="min_log_bf" type="number" step="0.1"></div>
        <div><label>min combined</label><input id="min_combined" type="number" step="0.1"></div>
      </div></details>
      <div id="scatter"></div>
      <div class="muted" id="scatter-count"></div>
      <div class="muted" id="gene-table-count" style="margin-top:8px"></div>
      <div class="scroll" style="max-height:300px;margin-top:4px"><table id="gene-table"></table></div>
    </section>
    <section class="panel">
      <h2>Top gene sets</h2>
      <div class="controls">
        <div style="flex:1"><label>search gene set (id or library)</label><input id="gs_search" placeholder="e.g. insulin" style="width:100%"></div>
        <div><label for="gs_sort">sort by</label><select id="gs_sort"><option value="beta">beta</option><option value="beta_uncorrected">beta_uncorrected</option><option value="n">N</option><option value="p_orig">P (asc)</option><option value="gene_set">name</option><option value="label">library</option></select></div>
      </div>
      <details class="adv"><summary>Advanced filters</summary><div class="controls">
        <div><label>min beta</label><input id="min_beta" type="number" step="0.01"></div>
        <div><label>min beta_unc</label><input id="min_beta_uncorrected" type="number" step="0.01"></div>
      </div></details>
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
    Complete HTML document for the Explorer UI.

    Args:
        title: Page title.
        plotly_src: URL (or data: URI) of plotly.min.js.
        api_base: Absolute URL of a `pigean.portal serve` instance for a static deployment
            (e.g. a bucket-hosted page calling `http://localhost:8765`); empty means same origin.
    """
    api_note = f" API: <code>{html.escape(api_base)}</code>" if api_base else ""
    body = BODY.replace("{title}", html.escape(title)).replace("{api_note}", api_note)
    return render_document(title=title, plotly_src=plotly_src, api_base=api_base, css=CSS, body=body, script=SCRIPT)
