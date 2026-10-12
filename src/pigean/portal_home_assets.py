"""Home-page entity search, sharing the Explorer's run navigation and styling."""

CSS = r"""
#view-landing { padding:28px 0 44px; align-items:flex-start; }
.landing { margin:4vh auto 0; transition:none; }
.landing.has-evidence { width:min(1100px, 94vw); margin-top:0; }
.home-tabs { margin:24px 0; gap:16px; }
.home-tabs button { font-size:16px; padding:8px 2px 12px; }
.home-tabs button[hidden] { display:none; }
.home-tabs button:focus-visible, .home-match:focus-visible { outline:2px solid var(--accent); outline-offset:3px; }
.home-form { display:grid; grid-template-columns:minmax(0, 1fr) 170px auto; gap:12px; align-items:end; }
.home-form .field { margin:0; }
.home-form button.primary { padding:10px 18px; }
.home-form label, .home-evidence label { text-transform:none; letter-spacing:0; font-size:12px; }
.home-hint { line-height:1.6; margin:14px 0 0; }
.home-matches { margin-top:16px; }
.home-match { display:flex; width:100%; gap:12px; align-items:center; justify-content:space-between; text-align:left;
  padding:12px 0; border:0; border-bottom:1px solid var(--line); border-radius:0; background:#fff; }
.home-match strong { display:block; overflow-wrap:anywhere; }
.home-match small { display:block; color:var(--muted); font-weight:400; margin-top:3px; }
.home-match .match-count { white-space:nowrap; color:var(--muted); font-size:12px; font-weight:400; }
.home-evidence { border-top:1px solid var(--line); padding-top:20px; margin-top:22px; }
.home-evidence h2 { font-size:23px; overflow-wrap:anywhere; margin-bottom:4px; }
.home-evidence-head { display:flex; justify-content:space-between; gap:20px; align-items:start; margin-bottom:16px; }
.home-evidence-head select { width:auto; min-width:170px; }
.home-evidence-note { max-width:80ch; line-height:1.5; margin:0 0 12px; }
.home-evidence .trait-link { font:inherit; color:var(--accent); text-align:left; }
.home-evidence .trait-label { white-space:normal; min-width:180px; }
.landing .home-evidence tr.row { display:table-row; }
.home-evidence .score-bar { display:inline-block; width:60px; height:5px; background:var(--accent-soft); margin-right:10px; vertical-align:middle; }
.home-evidence .score-bar i { display:block; height:100%; background:var(--accent); }
.home-factor-source { margin:6px 0 10px; line-height:1.6; }
@media(max-width:600px) {
  #view-landing { padding:12px 0; }
  .landing { padding:22px 18px; margin-top:0; }
  .landing h1 { font-size:28px; }
  .landing .row { grid-template-columns:minmax(0,1fr); }
  .home-tabs { gap:18px; }
  .home-form { grid-template-columns:minmax(0,1fr) auto; }
  .home-form .query-field { grid-column:1/-1; }
  .home-evidence-head { flex-direction:column; gap:10px; }
  .home-match { align-items:start; }
  .home-match .match-count { max-width:90px; white-space:normal; text-align:right; }
}
"""

BODY = r"""
<div id="home-entity-panel" role="tabpanel" aria-labelledby="home-tab-gene" hidden>
  <form id="home-form" class="home-form">
    <div class="field query-field"><label id="home-query-label" for="home-query">Gene</label>
      <input id="home-query" type="search" placeholder="Search genes, e.g. TCF7L2" autocomplete="off" aria-describedby="home-hint"></div>
    <div class="field"><label for="home-model">Model</label><select id="home-model"><option value="">All models</option></select></div>
    <button class="primary" type="submit">Search</button>
  </form>
  <p id="home-hint" class="muted home-hint"></p>
  <div id="home-search-status" class="muted home-hint" role="status" aria-live="polite"></div>
  <div id="home-matches" class="home-matches" aria-label="Search matches"></div>
  <section id="home-evidence" class="home-evidence" aria-label="Trait evidence" hidden>
    <div class="home-evidence-head">
      <div><h2 id="home-entity-title"></h2><div id="home-factor-source" class="muted home-factor-source" hidden></div>
        <p id="home-evidence-count" class="muted home-hint" role="status" aria-live="polite"></p></div>
      <div id="home-metric-field"><label for="home-metric">Rank traits by</label><select id="home-metric"></select></div>
    </div>
    <p id="home-evidence-note" class="muted home-evidence-note"></p>
    <div id="home-trait-table"></div>
  </section>
</div>
"""

SCRIPT = r"""
const HOME_MODES = ['trait', 'gene', 'gene_set', 'factor'];
const HOME_LABELS = {gene:'Gene', gene_set:'Geneset', factor:'Factor'};
const HOME_METRICS = {
  gene: [['combined','Combined'], ['log_bf','Direct (log_bf)'], ['prior','Indirect (prior)'], ['huge_score','HuGE score']],
  gene_set: [['beta','Adjusted effect (beta)'], ['beta_uncorrected','Uncorrected effect'], ['p_orig','P value']],
  factor: [['beta','Beta'], ['beta_uncorrected','Uncorrected beta'], ['nnls_loading','NNLS loading'],
    ['joint_fraction','Joint fraction'], ['joint_coefficient','Joint coefficient'], ['marginal_fraction','Marginal fraction'],
    ['marginal_coefficient','Marginal coefficient'], ['p_value','P value'], ['graph_weight','Graph loading']]
};
const home = {mode:'trait', request:0, timer:null, views:Object.fromEntries(['gene','gene_set','factor'].map(kind =>
  [kind, {query:'', model:'', selected:null, rows:[], metric:'', loaded:false}]))};
function homeView() { return home.views[home.mode]; }
function homeHash() {
  if (home.mode === 'trait') { setHash({}); return; }
  const v = homeView();
  setHash({view:home.mode, id:v.selected?.id, source:v.selected?.run_id, model:v.model});
}
function selectHomeTab(mode, focus = false) {
  if (!HOME_MODES.includes(mode) || (mode === 'factor' && $('home-tab-factor').hidden)) mode = 'trait';
  ++home.request; clearTimeout(home.timer); home.mode = mode;
  HOME_MODES.forEach(kind => {
    const b = $(`home-tab-${kind}`), active = kind === mode;
    b.classList.toggle('active', active); b.setAttribute('aria-selected', String(active)); b.tabIndex = active ? 0 : -1;
  });
  $('home-trait-panel').hidden = mode !== 'trait'; $('home-entity-panel').hidden = mode === 'trait';
  $('home-entity-panel').setAttribute('aria-labelledby', `home-tab-${mode}`);
  $('home-matches').replaceChildren(); $('home-search-status').textContent = '';
  document.querySelector('.landing').classList.remove('has-evidence');
  if (mode !== 'trait') {
    const v = homeView();
    $('home-query-label').textContent = HOME_LABELS[mode]; $('home-query').value = v.query;
    $('home-query').placeholder = {gene:'Search genes, e.g. TCF7L2', gene_set:'Search geneset identifiers or libraries', factor:'Search factor IDs or mechanism labels'}[mode];
    const models = uniq(state.runs.filter(r => mode !== 'factor' || r.factor_graph_available).map(r => r.model));
    $('home-model').innerHTML = '<option value="">All models</option>' + models.map(m => `<option value="${esc(m)}">${esc(m)}</option>`).join('');
    $('home-model').value = v.model;
    $('home-hint').textContent = mode === 'factor'
      ? 'Find a mechanism in a linked EAGGL graph. Factors are specific to their source run.'
      : `Search across all traits${v.model ? ' in this model' : ''}, then choose a ${mode === 'gene' ? 'gene' : 'geneset'} to see its evidence.`;
    $('home-evidence').hidden = !v.selected;
    if (v.selected) { if (v.loaded) renderHomeEvidence(); else loadHomeEvidence(); }
    else if (v.query) searchHome();
  }
  homeHash();
  if (focus) $(`home-tab-${mode}`).focus();
}
async function restoreHome(h) {
  selectHomeTab(h.view || 'trait');
  if (home.mode === 'trait') return;
  const v = homeView(); v.model = h.model || ''; $('home-model').value = v.model;
  if (h.id) await chooseHomeEntity({id:h.id, label:h.id, run_id:h.source || ''});
}
async function searchHome() {
  const v = homeView(); if (!v) return;
  const request = ++home.request, kind = home.mode, q = $('home-query').value.trim();
  v.query = q; v.model = $('home-model').value; v.selected = null; v.rows = []; v.loaded = false;
  $('home-evidence').hidden = true; $('home-matches').replaceChildren();
  document.querySelector('.landing').classList.remove('has-evidence'); homeHash();
  if (!q) { $('home-search-status').textContent = ''; return; }
  $('home-search-status').textContent = 'Searching…';
  try {
    const body = await api('/api/search', {kind, q, model:v.model, limit:21});
    if (request !== home.request) return;
    const matches = body.matches.slice(0,20);
    $('home-search-status').textContent = matches.length
      ? `${body.matches.length > 20 ? 'First 20' : matches.length} matching ${kind === 'gene_set' ? 'geneset' : kind}${matches.length === 1 ? '' : 's'}. Choose one to see traits.`
      : kind === 'factor' ? 'No matching factors in linked graphs. Try a factor ID or another mechanism label.'
        : 'No matches in the loaded results. Try another identifier or model; build filters may exclude this entry.';
    $('home-matches').innerHTML = matches.map((m,i) => `<button type="button" class="home-match" data-match="${i}">
      <span><strong>${esc(kind === 'factor' ? m.label : m.id)}</strong><small>${esc(kind === 'factor' ? `${m.id} · ${m.model} · ${m.trait || m.run_id} · ${m.seed || ''}` : m.label || '')}</small></span>
      <span class="match-count">${m.n_traits} trait${m.n_traits === 1 ? '' : 's'}${m.n_runs ? ` / ${m.n_runs} runs` : ''}</span></button>`).join('');
    $('home-matches').querySelectorAll('button').forEach(b => b.onclick = () => chooseHomeEntity(matches[+b.dataset.match]));
  } catch (err) { if (request === home.request) $('home-search-status').textContent = `Search failed: ${err.message}. Select Search to retry.`; }
}
async function chooseHomeEntity(match) {
  clearTimeout(home.timer);
  const v = homeView(); v.selected = match; v.query = match.id; v.loaded = false;
  $('home-query').value = match.id; $('home-matches').replaceChildren(); $('home-search-status').textContent = '';
  homeHash(); await loadHomeEvidence();
}
async function loadHomeEvidence() {
  const v = homeView(), kind = home.mode, selected = v.selected, request = ++home.request;
  if (!selected) return;
  $('home-evidence').hidden = false; document.querySelector('.landing').classList.add('has-evidence');
  $('home-entity-title').textContent = selected.label && kind === 'factor' ? selected.label : selected.id;
  $('home-evidence-count').textContent = 'Loading trait evidence…'; $('home-trait-table').replaceChildren();
  $('home-metric-field').hidden = true; $('home-factor-source').hidden = true; $('home-evidence-note').textContent = '';
  try {
    const body = await api(kind === 'factor' ? '/api/factor' : kind === 'gene' ? '/api/gene_across' : '/api/gene_set_across',
      {id:selected.id, model:v.model, run:kind === 'factor' ? selected.run_id : ''});
    if (request !== home.request) return;
    v.rows = body.rows; v.loaded = true;
    if (kind === 'factor') { selected.label = body.label; selected.relevance = body.relevance; }
    renderHomeEvidence();
  } catch (err) {
    if (request !== home.request) return;
    $('home-evidence-count').textContent = `Could not load trait evidence: ${err.message}`;
    $('home-trait-table').innerHTML = '<button type="button" id="home-retry">Retry</button>';
    $('home-retry').onclick = loadHomeEvidence;
  }
}
function renderHomeEvidence() {
  const v = homeView(), kind = home.mode, selected = v.selected;
  document.querySelector('.landing').classList.add('has-evidence'); $('home-evidence').hidden = false;
  $('home-entity-title').textContent = kind === 'factor' ? selected.label : selected.id;
  const metrics = HOME_METRICS[kind].filter(([key]) => kind !== 'factor' || v.rows.some(r => r[key] != null));
  if (!metrics.some(([key]) => key === v.metric)) v.metric = metrics[0]?.[0] || '';
  $('home-metric-field').hidden = !metrics.length;
  $('home-metric').innerHTML = metrics.map(([key,label]) => `<option value="${key}">${esc(label)}</option>`).join('');
  $('home-metric').value = v.metric;
  const rows = sortRows(v.rows, {col:v.metric, desc:!['p_orig','p_value'].includes(v.metric)});
  const traits = uniq(rows.map(r => r.trait || r.run_id)).length;
  $('home-evidence-count').textContent = kind === 'factor' ? `${traits} linked trait${traits === 1 ? '' : 's'} in this graph`
    : `${traits} trait${traits === 1 ? '' : 's'} across ${rows.length} run${rows.length === 1 ? '' : 's'}`;
  $('home-evidence-note').textContent = kind === 'factor'
    ? 'Evidence retained in the exported EAGGL graph, including its trait provenance. Export thresholds and limits may omit traits. Blank scores were not supplied.'
    : 'Scores show the strength of evidence in each run. Only results retained by the portal’s build filters are included.';
  $('home-factor-source').hidden = kind !== 'factor';
  if (kind === 'factor') {
    const source = state.runs.find(r => r.run_id === selected.run_id);
    $('home-factor-source').innerHTML = `${esc(selected.id)} · ${esc(source?.model || '')} · ${esc(source?.trait || '')} · ${esc(source?.seed || '')}` +
      ` <button type="button" id="home-open-graph" class="linkbtn">Open mechanism graph</button>`;
    $('home-open-graph').onclick = () => openHomeRun(selected.run_id, true);
  }
  if (!rows.length) {
    $('home-trait-table').innerHTML = `<p class="muted">${kind === 'factor' ? 'This factor has no trait links in the exported graph. Open its mechanism graph to explore the available genes and genesets.' : 'No trait evidence is retained for this entry in the selected model. Try All models.'}</p>`;
    return;
  }
  const columns = kind === 'factor' ? metrics : HOME_METRICS[kind];
  const max = rows.reduce((value, r) => Math.max(value, Math.abs(r[v.metric] || 0)), 1e-12);
  const traitCell = r => {
    const text = esc(r.phenotype_name || r.trait || r.run_title || r.run_id);
    if (kind !== 'factor') return `<button class="linkbtn trait-link" type="button" data-run="${esc(r.run_id)}">${text}</button>`;
    if (!r.run_ids.length) return text;
    return `${text}<br>` + r.run_ids.map(id => {
      const run = state.runs.find(x => x.run_id === id);
      return `<button class="linkbtn trait-link" type="button" data-run="${esc(id)}">Open ${esc(run?.seed || id)}</button>`;
    }).join(' · ');
  };
  pagedTable('home-trait-table', `<tr><th>Trait</th>${kind === 'factor' ? '' : '<th>Model / run</th>'}${columns.map(([key,label]) => `<th class="num">${esc(label)}</th>`).join('')}</tr>`, rows,
    r => `<tr class="row"><td class="trait-label">${traitCell(r)}<div class="muted">${esc(r.phenotype_name ? r.trait : r.trait_group || '')}</div></td>` +
      (kind === 'factor' ? '' : `<td>${esc(r.model)}<div class="muted">${esc(r.seed || r.run_id)}</div></td>`) +
      columns.map(([key]) => `<td class="num">${key === v.metric && r[key] != null && !['p_orig','p_value'].includes(key) ? `<span class="score-bar" aria-hidden="true"><i style="width:${Math.abs(r[key]) / max * 100}%"></i></span>` : ''}${r[key] == null ? '—' : fmt(r[key])}</td>`).join('') + '</tr>',
    tr => tr.querySelectorAll('[data-run]').forEach(b => b.onclick = () => openHomeRun(b.dataset.run)));
}
async function openHomeRun(runId, graph = false) {
  const kind = home.mode, selected = homeView()?.selected;
  try {
    state.run = runId; $('model').value = ''; $('trait').value = ''; populateRuns(); $('run').value = runId;
    await openResults();
    if (graph) selectEvidenceTab('mechanisms');
    else if (kind === 'gene') await showGene(selected.id);
    else if (kind === 'gene_set') await showGeneSet(selected.id);
  } catch (err) { openModal('Could not open results', esc(err.message)); }
}
HOME_MODES.forEach(mode => {
  const button = $(`home-tab-${mode}`);
  button.onclick = () => selectHomeTab(mode);
  button.onkeydown = e => {
    if (!['ArrowLeft','ArrowRight','Home','End'].includes(e.key)) return;
    e.preventDefault();
    const modes = HOME_MODES.filter(k => !$(`home-tab-${k}`).hidden), i = modes.indexOf(mode);
    const next = e.key === 'Home' ? 0 : e.key === 'End' ? modes.length - 1 : (i + (e.key === 'ArrowRight' ? 1 : -1) + modes.length) % modes.length;
    selectHomeTab(modes[next], true);
  };
});
$('home-form').onsubmit = e => { e.preventDefault(); clearTimeout(home.timer); searchHome(); };
$('home-query').oninput = () => {
  ++home.request; clearTimeout(home.timer);
  homeView().query = $('home-query').value; homeView().selected = null; homeView().loaded = false;
  $('home-evidence').hidden = true; $('home-matches').replaceChildren(); $('home-search-status').textContent = '';
  document.querySelector('.landing').classList.remove('has-evidence'); homeHash();
  home.timer = setTimeout(searchHome, 250);
};
$('home-model').onchange = () => {
  clearTimeout(home.timer);
  const v = homeView(); v.model = $('home-model').value; homeHash();
  if (v.selected && home.mode !== 'factor') { v.loaded = false; loadHomeEvidence(); }
  else searchHome();
};
$('home-metric').onchange = () => { homeView().metric = $('home-metric').value; renderHomeEvidence(); };
"""
