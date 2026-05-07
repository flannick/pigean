from __future__ import annotations

import base64
import gzip
import html
import json


CSS = r"""
:root {
  --ink: #1f2933;
  --muted: #65758b;
  --line: #d9e1ea;
  --panel: rgba(255,255,255,0.9);
  --soft: #f4f7f8;
  --accent: #0f766e;
  --accent-soft: #dff4f1;
  --warn: #9a5b1f;
  --shadow: 0 18px 45px rgba(31,41,51,0.10);
}
* { box-sizing: border-box; }
body { margin: 0; color: var(--ink); font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: radial-gradient(circle at top left, rgba(15,118,110,0.16), transparent 30rem), linear-gradient(135deg, #fbfaf5 0%, #edf4f2 46%, #f8fbfb 100%); }
.shell { max-width: 1480px; margin: 0 auto; padding: 34px 28px 64px; }
.hero, .panel { background: var(--panel); border: 1px solid rgba(217,225,234,0.9); border-radius: 24px; box-shadow: var(--shadow); }
.hero { display: grid; grid-template-columns: minmax(0,1.35fr) minmax(280px,0.65fr); gap: 22px; padding: 30px; margin-bottom: 20px; }
h1 { margin: 0 0 10px; font-size: clamp(30px, 4vw, 52px); letter-spacing: -0.045em; line-height: 0.98; }
h2 { margin: 0 0 12px; font-size: 23px; letter-spacing: -0.02em; }
h3 { margin: 0 0 10px; font-size: 16px; }
p { line-height: 1.5; }
.lede, .note { color: var(--muted); }
.stats { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 12px; }
.stat { background: var(--soft); border: 1px solid var(--line); border-radius: 18px; padding: 16px; }
.stat strong { display: block; font-size: 25px; letter-spacing: -0.02em; }
.stat span { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }
.panel { padding: 22px; margin: 18px 0; }
.controls { display: grid; grid-template-columns: minmax(260px,1fr) minmax(240px,0.8fr); gap: 16px; align-items: end; }
label { display: block; margin-bottom: 7px; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }
select, input { width: 100%; border: 1px solid var(--line); border-radius: 14px; padding: 10px 12px; background: white; color: var(--ink); font-size: 14px; }
select.compact { width: auto; padding: 7px 10px; border-radius: 10px; }
.column-filter-row th { top: 30px; background: #fff; cursor: default; }
.column-filter { width: 5ch; min-width: 5ch; max-width: 5ch; padding: 5px 4px; border-radius: 8px; font-size: 11px; }
.numeric-filter { display: inline-flex; gap: 3px; align-items: center; justify-content: flex-end; }
.numeric-filter select { width: 4.5ch; min-width: 4.5ch; padding: 5px 2px; border-radius: 8px; font-size: 11px; }
.tabs { display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0 14px; }
button { border: 1px solid var(--line); background: #fff; color: var(--ink); border-radius: 12px; padding: 8px 12px; cursor: pointer; font-weight: 650; }
button.active { background: var(--accent); border-color: var(--accent); color: #fff; }
.pill { display: inline-flex; gap: 7px; align-items: center; border-radius: 999px; padding: 7px 10px; background: var(--accent-soft); color: #0d5f59; font-size: 12px; font-weight: 700; }
.badge { border-radius: 999px; background: #f4eadf; color: var(--warn); padding: 4px 8px; font-size: 11px; font-weight: 800; }
.warning { background: #fff7ed; border: 1px solid #fed7aa; color: #7c2d12; border-radius: 14px; padding: 10px 12px; margin: 8px 0; }
.table-tools { display: grid; grid-template-columns: minmax(220px,1fr) auto auto; gap: 12px; align-items: center; margin: 12px 0; }
.table-wrap { overflow: auto; border: 1px solid var(--line); border-radius: 18px; background: #fff; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { position: sticky; top: 0; z-index: 1; background: #f7faf9; color: #405265; text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; cursor: pointer; }
th, td { border-bottom: 1px solid #e8eef2; padding: 9px 10px; vertical-align: top; }
th.number, td.number { text-align: right; font-variant-numeric: tabular-nums; }
tr:hover td { background: #fbfdfc; }
.path { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; overflow-wrap: anywhere; color: #546275; }
.empty { padding: 18px; color: var(--muted); font-style: italic; }
.details-row td { background: #fbfaf5; }
.details-box { padding: 14px; border-left: 4px solid var(--accent); }
.subgrid { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 12px; }
.factor-grid { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 12px; }
.subpanel { border: 1px solid var(--line); border-radius: 16px; padding: 12px; background: rgba(255,255,255,0.75); }
.inline-popover { min-width: 360px; max-width: 760px; margin-top: 8px; padding: 10px; border: 1px solid var(--line); border-radius: 14px; background: #fff; box-shadow: 0 12px 24px rgba(31,41,51,0.10); }
.pager { display: flex; gap: 8px; align-items: center; justify-content: flex-end; margin-top: 10px; }
.factor-graph-frame { width: 100%; min-height: 720px; border: 1px solid var(--line); border-radius: 12px; background: #fff; }
@media (max-width: 900px) { .hero, .controls, .subgrid, .factor-grid { grid-template-columns: 1fr; } .shell { padding: 20px 14px 44px; } .table-tools { grid-template-columns: 1fr; } }
"""


JS = r"""
const DATA_PAYLOAD_GZIP_BASE64 = "__DASHBOARD_DATA_GZIP_BASE64__";
let DATA = null;
const state = { runId: "", section: "genes", modeId: "", searches: {}, columnFilters: {}, columnFilterOps: {}, sorts: {}, pages: {}, pageSizes: {}, openRows: {}, focus: null };
function byId(id) { return document.getElementById(id); }
function esc(value) { return String(value ?? "").replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[ch])); }
function fmt(value) { if (value === null || value === undefined || value === "") return "NA"; if (typeof value === "number") { if (!Number.isFinite(value)) return "NA"; if (Math.abs(value) >= 1000) return value.toFixed(0); if (Math.abs(value) >= 10) return value.toFixed(2); return value.toPrecision(3); } return esc(value); }
function isNumber(value) { return typeof value === "number" && Number.isFinite(value); }
async function decodePayload() { const binary = Uint8Array.from(atob(DATA_PAYLOAD_GZIP_BASE64), c => c.charCodeAt(0)); if (typeof DecompressionStream === "undefined") throw new Error("This browser does not support embedded gzip decoding via DecompressionStream."); const stream = new Blob([binary]).stream().pipeThrough(new DecompressionStream("gzip")); return JSON.parse(await new Response(stream).text()); }
function rowSearchText(row) { return String(row.__search_text || Object.values(row).join(" ")); }
function rowMatches(row, search) { if (!search) return true; return rowSearchText(row).toLowerCase().includes(search.toLowerCase()); }
function columnFiltersMatch(row, columns, id) { const filters = state.columnFilters[id] || {}; const ops = state.columnFilterOps[id] || {}; for (const c of columns) { const raw = filters[c.key]; if (raw === undefined || raw === "") continue; const value = c.filterValue ? c.filterValue(row) : row[c.key]; if (c.numeric) { const target = Number(raw); const actual = Number(value); if (!Number.isFinite(target) || !Number.isFinite(actual)) return false; const op = ops[c.key] || ">="; if (op === ">=" && actual < target) return false; if (op === "<=" && actual > target) return false; } else if (!String(value ?? "").toLowerCase().includes(String(raw).toLowerCase())) return false; } return true; }
function sortedRows(rows, key) { if (!key) return rows; const desc = key.startsWith("-"); const col = desc ? key.slice(1) : key; return [...rows].sort((a,b) => { const av = a[col], bv = b[col]; const cmp = isNumber(av) && isNumber(bv) ? av - bv : String(av ?? "").localeCompare(String(bv ?? "")); return desc ? -cmp : cmp; }); }
function getSearch(id, opts) { return state.searches[id] ?? opts.inheritedSearch ?? ""; }
function restoreFocus() { if (!state.focus || typeof CSS === "undefined" || !CSS.escape) return; const input = document.querySelector(`[data-focus-key="${CSS.escape(state.focus)}"]`); if (!input) return; input.focus(); const len = input.value.length; try { input.setSelectionRange(len, len); } catch {} }
function tableHtml(id, rows, columns, opts = {}) {
  const search = getSearch(id, opts);
  const sortKey = state.sorts[id] || "";
  const pageSize = state.pageSizes[id] || opts.pageSize || 20;
  const filtered = sortedRows((rows || []).filter(r => rowMatches(r, search) && columnFiltersMatch(r, columns, id)), sortKey);
  const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize));
  const page = Math.min(state.pages[id] || 1, pageCount);
  state.pages[id] = page;
  const shown = filtered.slice((page - 1) * pageSize, page * pageSize);
  const header = columns.map(c => `<th class="${c.numeric ? "number" : ""}" data-table="${esc(id)}" data-col="${esc(c.key)}">${esc(c.label || c.key)}</th>`).join("");
  const filterHeader = columns.map(c => {
    const filters = state.columnFilters[id] || {};
    const ops = state.columnFilterOps[id] || {};
    const value = filters[c.key] || "";
    if (c.noFilter) return `<th></th>`;
    if (c.numeric) return `<th class="number"><span class="numeric-filter"><select data-column-filter-op-table="${esc(id)}" data-column-filter-op-col="${esc(c.key)}" data-focus-key="op:${esc(id)}:${esc(c.key)}"><option value=">=" ${(ops[c.key] || ">=") === ">=" ? "selected" : ""}>&gt;=</option><option value="<=" ${ops[c.key] === "<=" ? "selected" : ""}>&lt;=</option></select><input class="column-filter" type="text" inputmode="decimal" value="${esc(value)}" data-column-filter-table="${esc(id)}" data-column-filter-col="${esc(c.key)}" data-focus-key="col:${esc(id)}:${esc(c.key)}"></span></th>`;
    return `<th><input class="column-filter" type="text" value="${esc(value)}" data-column-filter-table="${esc(id)}" data-column-filter-col="${esc(c.key)}" data-focus-key="col:${esc(id)}:${esc(c.key)}"></th>`;
  }).join("");
  const body = shown.map((row, ix) => {
    const key = `${id}:${row.id || row.gene || row.gene_set || row.factor || ix}`;
    const cells = columns.map(c => `<td class="${c.numeric ? "number" : ""}">${c.render ? c.render(row, key) : fmt(row[c.key])}</td>`).join("");
    const detail = opts.detailRenderer && state.openRows[key] ? `<tr class="details-row"><td colspan="${columns.length}">${opts.detailRenderer(row, key, search)}</td></tr>` : "";
    return `<tr>${cells}</tr>${detail}`;
  }).join("") || `<tr><td colspan="${columns.length}" class="empty">No rows available.</td></tr>`;
  const inherited = opts.inheritedSearch && state.searches[id] === undefined ? ` <span class="note">inherited table filter: ${esc(opts.inheritedSearch)}</span>` : "";
  return `<div class="table-tools"><span class="note">${filtered.length} matched / ${(rows || []).length} rows${inherited}</span><label class="note">Rows <select class="compact" data-page-size-table="${esc(id)}">${[10,20,50,100,250].map(n => `<option value="${n}" ${n === pageSize ? "selected" : ""}>${n}</option>`).join("")}</select></label></div><div class="table-wrap"><table><thead><tr>${header}</tr><tr class="column-filter-row">${filterHeader}</tr></thead><tbody>${body}</tbody></table></div><div class="pager"><button data-page-table="${esc(id)}" data-page-delta="-1" ${page <= 1 ? "disabled" : ""}>Prev</button><span class="note">Page ${page} / ${pageCount}</span><button data-page-table="${esc(id)}" data-page-delta="1" ${page >= pageCount ? "disabled" : ""}>Next</button></div>`;
}
function smallTable(rows, cols, id, inheritedSearch = "") { return tableHtml(id, rows || [], cols, {pageSize: 20, inheritedSearch}); }
function warningsHtml(items) { return (items || []).map(w => `<div class="warning">${esc(w)}</div>`).join(""); }
function selectedRun() { return (DATA.pigean_runs || []).find(r => r.run_id === state.runId) || (DATA.pigean_runs || [])[0] || null; }
function eagglForRun(runId) { return Object.values(DATA.eaggl_runs || {}).filter(r => r.run_id === runId); }
function selectedEaggl(runId) { const runs = eagglForRun(runId); return runs.find(r => r.mode_id === state.modeId) || runs[0] || null; }
function toggleOpen(key) { state.openRows[key] = !state.openRows[key]; render(); }
function renderControls() { const runs = DATA.pigean_runs || []; if (!state.runId && runs.length) state.runId = runs[0].run_id; const runOptions = runs.map(r => `<option value="${esc(r.run_id)}" ${r.run_id === state.runId ? "selected" : ""}>${esc(r.title || r.run_id)}</option>`).join(""); const modes = eagglForRun(state.runId); if (!state.modeId && modes.length) state.modeId = modes[0].mode_id; const modeControl = modes.length <= 1 ? `<span class="pill">${esc(modes[0]?.title || modes[0]?.mode_id || "No EAGGL run")}</span>` : `<select id="modeSelect">${modes.map(m => `<option value="${esc(m.mode_id)}" ${m.mode_id === state.modeId ? "selected" : ""}>${esc(m.title || m.mode_id)}</option>`).join("")}</select>`; return `<section class="panel"><div class="controls"><div><label>PIGEAN run</label><select id="runSelect">${runOptions}</select></div><div><label>EAGGL mode</label>${modeControl}</div></div><div class="tabs"><button data-section="genes" class="${state.section === "genes" ? "active" : ""}">Genes</button><button data-section="gene_sets" class="${state.section === "gene_sets" ? "active" : ""}">Gene sets</button><button data-section="factors" class="${state.section === "factors" ? "active" : ""}">Factors</button><button data-section="status" class="${state.section === "status" ? "active" : ""}">Status</button></div></section>`; }
function geneDetails(row, key, search) { const run = selectedRun(); const sets = (run.gene_expansions || {})[row.gene] || []; return `<div class="details-box"><div class="subgrid"><div class="subpanel"><h3>Supporting gene sets</h3>${smallTable(sets, [{key:"gene_set",label:"Gene set"},{key:"label",label:"Label"},{key:"beta",label:"Beta",numeric:true},{key:"beta_uncorrected",label:"Beta uncorr",numeric:true},{key:"n",label:"N",numeric:true}], `${key}:sets`, search)}</div><div class="subpanel"><h3>Provenance</h3><p><strong>Combined:</strong> ${fmt(row.combined)}</p><p><strong>Direct:</strong> ${fmt(row.log_bf)}</p><p><strong>Indirect:</strong> ${fmt(row.prior)}</p><p><strong>HuGE/GWAS:</strong> ${fmt(row.huge_score)}</p><p><strong>Location:</strong> ${esc(row.chrom || "NA")}:${fmt(row.start)}-${fmt(row.end)}</p></div></div></div>`; }
function geneSetDetails(row, key, search) { const run = selectedRun(); const genes = (run.gene_set_expansions || {})[row.gene_set] || []; return `<div class="details-box"><div class="subgrid"><div class="subpanel"><h3>Member genes passing dashboard filter</h3>${smallTable(genes, [{key:"gene",label:"Gene"},{key:"combined",label:"Combined",numeric:true},{key:"log_bf",label:"Direct",numeric:true},{key:"prior",label:"Indirect",numeric:true}], `${key}:genes`, search)}</div><div class="subpanel"><h3>Provenance</h3><p><strong>Beta:</strong> ${fmt(row.beta)}</p><p><strong>Beta uncorrected:</strong> ${fmt(row.beta_uncorrected)}</p><p><strong>P:</strong> ${fmt(row.p_orig)}</p><p><strong>Z:</strong> ${fmt(row.z_orig)}</p><p><strong>Filter:</strong> ${esc(row.filter_reason || "NA")}</p></div></div></div>`; }
function renderGenes(run) { const cols = [{key:"gene",label:"Gene"},{key:"combined",label:"Combined",numeric:true},{key:"log_bf",label:"Direct",numeric:true},{key:"prior",label:"Indirect",numeric:true},{key:"huge_score",label:"HuGE/GWAS",numeric:true},{key:"n",label:"N",numeric:true},{key:"chrom",label:"Chr"},{key:"provenance",label:"Provenance",render:(r,k)=>`<button data-open-row="${esc(k)}">${state.openRows[k] ? "Hide" : "Show"}</button>`}]; return `<section class="panel"><h2>PIGEAN genes</h2><p class="note">Default embedded genes pass combined support ≥ ${fmt(DATA.thresholds.gene_combined)}.</p>${warningsHtml(run.warnings)}${tableHtml(`genes:${run.run_id}`, run.genes, cols, {detailRenderer: geneDetails})}</section>`; }
function renderGeneSets(run) { const cols = [{key:"gene_set",label:"Gene set"},{key:"label",label:"Label"},{key:"beta",label:"Beta",numeric:true},{key:"beta_uncorrected",label:"Beta uncorr",numeric:true},{key:"p_orig",label:"P",numeric:true},{key:"z_orig",label:"Z",numeric:true},{key:"n",label:"N",numeric:true},{key:"provenance",label:"Provenance",render:(r,k)=>`<button data-open-row="${esc(k)}">${state.openRows[k] ? "Hide" : "Show"}</button>`}]; return `<section class="panel"><h2>PIGEAN gene sets</h2><p class="note">Default embedded gene sets pass beta_uncorrected ≥ ${fmt(DATA.thresholds.gene_set_beta_uncorrected)} when available.</p>${tableHtml(`gene_sets:${run.run_id}`, run.gene_sets, cols, {detailRenderer: geneSetDetails})}</section>`; }
function factorInlineDetails(f, table, search, rowKey) { const rows = table === "genes" ? (f.genes || []) : table === "gene_sets" ? (f.gene_sets || []) : (f.phenotypes || []); const label = table === "genes" ? "genes" : table === "gene_sets" ? "gene sets" : "phenotypes"; const key = `${rowKey}:${table}`; let cols; if (table === "genes") cols = [{key:"gene",label:"Gene"},{key:"loading",label:"Loading",numeric:true},{key:"relative_loading",label:"Relative",numeric:true},{key:"combined",label:"Combined",numeric:true}]; else if (table === "gene_sets") cols = [{key:"gene_set",label:"Gene set"},{key:"label",label:"Label"},{key:"loading",label:"Loading",numeric:true},{key:"beta",label:"Beta",numeric:true}]; else cols = [{key:"trait",label:"Trait"},{key:"joint_fraction",label:"Joint frac",numeric:true},{key:"marginal_fraction",label:"Marginal",numeric:true},{key:"trait_neff",label:"Trait neff",numeric:true}]; return `<button data-open-row="${esc(key)}">${state.openRows[key] ? "Hide" : "Show"} ${esc(label)} (${rows.length})</button>${state.openRows[key] ? `<div class="inline-popover">${smallTable(rows, cols, key, search)}</div>` : ""}`; }
function renderFactors(runId) { const eaggl = selectedEaggl(runId); if (!eaggl) return `<section class="panel"><h2>EAGGL factors</h2><div class="empty">No EAGGL run was supplied for this PIGEAN run.</div></section>`; const graph = eaggl.factor_graph_available && eaggl.factor_graph_html ? `<details class="panel"><summary><strong>Embedded factor graph</strong></summary><iframe class="factor-graph-frame" srcdoc="${esc(eaggl.factor_graph_html)}"></iframe><p class="path">${esc(eaggl.paths.factor_graph_html || "")}</p></details>` : `<details class="panel"><summary><strong>Embedded factor graph</strong></summary><div class="empty">No factor graph was generated for this EAGGL run.</div></details>`; const rows = (eaggl.factors || []).map(f => ({...f, mass:f.combined_mass_fraction, n_genes:(f.genes || []).length, n_gene_sets:(f.gene_sets || []).length, n_traits:(f.phenotypes || []).length, __search_text:[f.factor, f.label, ...(f.genes || []).map(g=>g.gene), ...(f.gene_sets || []).map(gs=>`${gs.gene_set} ${gs.label || ""}`), ...(f.phenotypes || []).map(p=>p.trait)].join(" ")})); const cols = [{key:"factor",label:"Factor"},{key:"label",label:"Mechanism"},{key:"factor_tier",label:"Tier"},{key:"lambda",label:"Lambda",numeric:true},{key:"mass",label:"Mass",numeric:true},{key:"top_genes",label:"Top gene loadings",filterValue:r=>(r.genes || []).map(g=>`${g.gene} ${g.label || ""}`).join(" "),render:(r,k)=>factorInlineDetails(r,"genes",getSearch(`factors:${runId}:${eaggl.mode_id}`,{}),k)},{key:"top_gene_sets",label:"Top gene set loadings",filterValue:r=>(r.gene_sets || []).map(gs=>`${gs.gene_set} ${gs.label || ""}`).join(" "),render:(r,k)=>factorInlineDetails(r,"gene_sets",getSearch(`factors:${runId}:${eaggl.mode_id}`,{}),k)},{key:"phewas",label:"PheWAS",filterValue:r=>(r.phenotypes || []).map(p=>p.trait).join(" "),render:(r,k)=>factorInlineDetails(r,"phenotypes",getSearch(`factors:${runId}:${eaggl.mode_id}`,{}),k)}]; return `${graph}<section class="panel"><h2>EAGGL factors</h2><p class="note">Default embedded factor rows keep genes/gene sets within ${fmt(DATA.thresholds.factor_loading_within_max)} of each factor-specific max loading and phenotypes with N_eff ≥ ${fmt(DATA.thresholds.trait_neff)}.</p>${warningsHtml(eaggl.warnings)}${tableHtml(`factors:${runId}:${eaggl.mode_id}`, rows, cols)}</section>`; }
function renderStatus(run) { const rows = []; for (const r of DATA.pigean_runs || []) rows.push({kind:"PIGEAN", run:r.run_id, mode:"", status:(r.warnings || []).length ? "missing or partial" : "loaded", notes:(r.warnings || []).join("; ")}); for (const r of Object.values(DATA.eaggl_runs || {})) rows.push({kind:"EAGGL", run:r.run_id, mode:r.mode_id, status:(r.warnings || []).length ? "missing or partial" : "loaded", notes:(r.warnings || []).join("; ")}); return `<section class="panel"><h2>Dashboard status</h2>${tableHtml("status", rows, [{key:"kind",label:"Kind"},{key:"run",label:"Run"},{key:"mode",label:"Mode"},{key:"status",label:"Status"},{key:"notes",label:"Notes"}], {pageSize: 20})}</section>`; }
function render() { const run = selectedRun(); if (!run) { byId("app").innerHTML = `<section class="panel"><div class="empty">No PIGEAN or EAGGL runs were supplied.</div></section>`; return; } let body = renderControls(); if (state.section === "genes") body += renderGenes(run); if (state.section === "gene_sets") body += renderGeneSets(run); if (state.section === "factors") body += renderFactors(run.run_id); if (state.section === "status") body += renderStatus(run); byId("app").innerHTML = body; bind(); restoreFocus(); }
function bind() { const runSel = byId("runSelect"); if (runSel) runSel.onchange = () => { state.runId = runSel.value; state.modeId = ""; state.pages = {}; render(); }; const modeSel = byId("modeSelect"); if (modeSel) modeSel.onchange = () => { state.modeId = modeSel.value; state.pages = {}; render(); }; document.querySelectorAll("button[data-section]").forEach(btn => btn.onclick = () => { state.section = btn.dataset.section; state.pages = {}; render(); }); document.querySelectorAll("button[data-open-row]").forEach(btn => btn.onclick = () => toggleOpen(btn.dataset.openRow)); document.querySelectorAll("input[data-search-table]").forEach(inp => { inp.onfocus = () => { state.focus = `search:${inp.dataset.searchTable}`; }; inp.oninput = () => { state.focus = `search:${inp.dataset.searchTable}`; state.searches[inp.dataset.searchTable] = inp.value; state.pages[inp.dataset.searchTable] = 1; render(); }; }); document.querySelectorAll("input[data-column-filter-table]").forEach(inp => { inp.onfocus = () => { state.focus = `col:${inp.dataset.columnFilterTable}:${inp.dataset.columnFilterCol}`; }; inp.oninput = () => { const id = inp.dataset.columnFilterTable, col = inp.dataset.columnFilterCol; state.focus = `col:${id}:${col}`; state.columnFilters[id] = state.columnFilters[id] || {}; state.columnFilters[id][col] = inp.value; state.pages[id] = 1; render(); }; }); document.querySelectorAll("select[data-column-filter-op-table]").forEach(sel => { sel.onfocus = () => { state.focus = `op:${sel.dataset.columnFilterOpTable}:${sel.dataset.columnFilterOpCol}`; }; sel.onchange = () => { const id = sel.dataset.columnFilterOpTable, col = sel.dataset.columnFilterOpCol; state.focus = `op:${id}:${col}`; state.columnFilterOps[id] = state.columnFilterOps[id] || {}; state.columnFilterOps[id][col] = sel.value; state.pages[id] = 1; render(); }; }); document.querySelectorAll("select[data-page-size-table]").forEach(sel => sel.onchange = () => { state.pageSizes[sel.dataset.pageSizeTable] = Number(sel.value) || 20; state.pages[sel.dataset.pageSizeTable] = 1; render(); }); document.querySelectorAll("button[data-page-table]").forEach(btn => btn.onclick = () => { const id = btn.dataset.pageTable; state.pages[id] = Math.max(1, (state.pages[id] || 1) + Number(btn.dataset.pageDelta || 0)); render(); }); document.querySelectorAll("th[data-table]").forEach(th => th.onclick = () => { const table = th.dataset.table, col = th.dataset.col; state.sorts[table] = state.sorts[table] === col ? `-${col}` : col; render(); }); }
async function boot() { try { DATA = await decodePayload(); } catch (error) { byId("app").innerHTML = `<section class="panel"><div class="empty">Could not load embedded dashboard data: ${esc(error.message || error)}</div></section>`; return; } byId("summaryStats").innerHTML = `<div class="stat"><strong>${(DATA.pigean_runs || []).length}</strong><span>PIGEAN runs</span></div><div class="stat"><strong>${Object.keys(DATA.eaggl_runs || {}).length}</strong><span>EAGGL runs</span></div><div class="stat"><strong>${DATA.gene_set_membership_count || 0}</strong><span>gene-set memberships</span></div><div class="stat"><strong>${(DATA.warnings || []).length}</strong><span>warnings</span></div>`; render(); }
boot();
"""


def render_html(payload: dict, *, title: str = "PIGEAN/EAGGL Dashboard") -> str:
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded = base64.b64encode(gzip.compress(data_json, compresslevel=9)).decode("ascii")
    script = JS.replace("__DASHBOARD_DATA_GZIP_BASE64__", encoded)
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>{CSS}</style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div>
        <h1>{safe_title}</h1>
        <p class="lede">Standalone post-processing dashboard for supplied PIGEAN and EAGGL outputs. Missing optional inputs are recorded in the dashboard rather than treated as fatal.</p>
      </div>
      <div id="summaryStats" class="stats"></div>
    </section>
    <div id="app"><section class="panel"><div class="empty">Loading embedded dashboard data...</div></section></div>
  </main>
  <script>{script}</script>
</body>
</html>
"""
