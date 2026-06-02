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
  --panel: #ffffff;
  --soft: #f4f7f8;
  --accent: #0f766e;
  --accent-2: #9a5b1f;
  --accent-soft: #dff4f1;
  --warn: #9a5b1f;
  --shadow: 0 18px 45px rgba(31,41,51,0.10);
}
* { box-sizing: border-box; }
body { margin: 0; color: var(--ink); font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: radial-gradient(circle at top left, rgba(15,118,110,0.18), transparent 32rem), linear-gradient(135deg, #fbfaf5 0%, #edf4f2 45%, #f8fbfb 100%); }
.shell { max-width: 1480px; margin: 0 auto; padding: 36px 28px 64px; }
.hero { display: grid; grid-template-columns: 1.3fr 0.7fr; gap: 24px; margin-bottom: 24px; align-items: stretch; }
.hero-card, .panel { background: rgba(255,255,255,0.88); border: 1px solid rgba(217,225,234,0.85); border-radius: 24px; box-shadow: var(--shadow); }
.hero-card { padding: 32px; }
h1 { margin: 0 0 10px; font-size: clamp(30px, 4vw, 54px); letter-spacing: -0.045em; line-height: 0.98; }
h2 { margin: 0 0 12px; font-size: 24px; letter-spacing: -0.02em; }
h3 { margin: 0 0 10px; font-size: 17px; }
p { line-height: 1.5; }
.lede { color: var(--muted); max-width: 880px; font-size: 16px; }
.note { color: var(--muted); font-size: 13px; }
.stats { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 12px; padding: 18px; }
.stat { background: var(--soft); border: 1px solid var(--line); border-radius: 18px; padding: 16px; }
.stat strong { display: block; font-size: 26px; letter-spacing: -0.02em; }
.stat span { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }
.panel { padding: 22px; margin: 18px 0; }
.controls { display: grid; grid-template-columns: minmax(260px,1fr) minmax(240px,0.7fr); gap: 18px; align-items: end; }
.run-summary { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 16px; }
label { display: block; margin-bottom: 7px; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }
select, input { width: 100%; border: 1px solid var(--line); border-radius: 14px; padding: 12px 14px; background: white; color: var(--ink); font-size: 14px; }
select.compact { width: auto; padding: 7px 10px; border-radius: 10px; }
.column-filter-row th { top: 30px; background: #fff; cursor: default; }
.column-filter { width: 100%; min-width: 10ch; padding: 5px 6px; border-radius: 8px; font-size: 11px; }
.numeric-filter { display: inline-flex; gap: 4px; align-items: center; justify-content: flex-end; width: 100%; }
.numeric-filter .column-filter { width: 10ch; min-width: 10ch; max-width: 10ch; }
.numeric-filter select { width: 9ch; min-width: 9ch; padding: 5px 4px; border-radius: 8px; font-size: 11px; }
.tabs { display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0 14px; }
button { border: 1px solid var(--line); background: #fff; color: var(--ink); border-radius: 12px; padding: 8px 12px; cursor: pointer; font-weight: 650; }
button.active { background: var(--accent); border-color: var(--accent); color: #fff; }
.pill { display: inline-flex; gap: 7px; align-items: center; border-radius: 999px; padding: 7px 10px; background: var(--accent-soft); color: #0d5f59; font-size: 12px; font-weight: 700; }
.badge { border-radius: 999px; background: #f4eadf; color: var(--warn); padding: 4px 8px; font-size: 11px; font-weight: 800; }
.warning { background: #fff7ed; border: 1px solid #fed7aa; color: #7c2d12; border-radius: 14px; padding: 10px 12px; margin: 8px 0; }
.table-tools { display: grid; grid-template-columns: minmax(220px,1fr) auto; gap: 12px; align-items: center; margin: 12px 0; }
.table-wrap { overflow: auto; border: 1px solid var(--line); border-radius: 18px; background: #fff; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { position: sticky; top: 0; z-index: 1; background: #f7faf9; color: #405265; text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; cursor: pointer; }
th, td { border-bottom: 1px solid #e8eef2; padding: 9px 10px; vertical-align: top; }
th.number, td.number { text-align: right; font-variant-numeric: tabular-nums; }
.column-info { display: inline-flex; align-items: center; justify-content: center; width: 16px; height: 16px; margin-left: 5px; border-radius: 999px; border: 1px solid #9fb1c1; background: #fff; color: #405265; font-size: 10px; font-weight: 900; line-height: 1; vertical-align: middle; cursor: help; padding: 0; }
.column-info:hover { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); }
.metric-summary { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0 12px; }
.metric-summary .pill { background: #eef5f4; color: #315a56; }
tr:hover td { background: #fbfdfc; }
.path { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; overflow-wrap: anywhere; color: #546275; }
.empty { padding: 18px; color: var(--muted); font-style: italic; }
.details-row td { background: #fbfaf5; }
.details-box { padding: 14px; border-left: 4px solid var(--accent); }
.subgrid { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 12px; }
.subpanel { border: 1px solid var(--line); border-radius: 16px; padding: 12px; background: rgba(255,255,255,0.75); }
.factor-details { border-left-color: var(--accent-2); }
.factor-tabs { display: flex; gap: 8px; flex-wrap: wrap; margin: 0 0 12px; }
.factor-details .table-wrap { width: 100%; }
.details-row > td { padding: 0; }
.details-row .details-box { margin: 0; width: 100%; }
.compact-tabs { margin-top: 0; }
.full-width-subpanel { width: 100%; }
.pager { display: flex; gap: 8px; align-items: center; justify-content: flex-end; margin-top: 10px; }
.factor-graph-accordion { margin: 16px 0; border: 1px solid var(--line); border-radius: 18px; background: #fff; overflow: hidden; }
.factor-graph-accordion summary { cursor: pointer; padding: 13px 16px; font-weight: 800; color: var(--ink); background: #f7faf9; }
.factor-graph-body { padding: 12px; }
.factor-graph-frame { width: 100%; min-height: 720px; border: 1px solid var(--line); border-radius: 12px; background: #fff; }
.heatmap-controls { display: flex; gap: 16px; align-items: end; flex-wrap: wrap; margin: 12px 0; }
.heatmap-controls label { min-width: 180px; }
.heatmap-controls .heatmap-regex-label { min-width: 260px; flex: 1; }
.heatmap-regex { width: 100%; min-width: 24ch; padding: 7px 9px; border-radius: 9px; }
.heatmap-wrap { overflow: auto; max-height: 760px; border: 1px solid var(--line); border-radius: 18px; background: #fff; }
.loading-heatmap { border-collapse: separate; border-spacing: 0; table-layout: fixed; width: max-content; min-width: 100%; }
.loading-heatmap th, .loading-heatmap td { padding: 0; border-bottom: 1px solid #edf2f5; border-right: 1px solid #edf2f5; }
.loading-heatmap thead th { height: 150px; position: sticky; top: 0; z-index: 3; vertical-align: bottom; }
.loading-heatmap .heat-row-label { position: sticky; left: 0; z-index: 2; width: 220px; max-width: 220px; padding: 6px 8px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; background: #f7faf9; cursor: default; text-transform: none; letter-spacing: 0; font-size: 12px; }
.loading-heatmap thead .heat-row-label { z-index: 4; text-transform: uppercase; letter-spacing: 0.06em; font-size: 11px; vertical-align: bottom; }
.loading-heatmap .heat-factor { width: 24px; min-width: 24px; cursor: pointer; vertical-align: bottom; }
.loading-heatmap .heat-factor span { display: inline-block; writing-mode: vertical-rl; transform: rotate(180deg); max-height: 136px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; padding: 6px 2px; text-transform: none; letter-spacing: 0; font-size: 10px; }
.loading-heatmap .heat-factor:hover { background: var(--accent-soft); }
.loading-heatmap .heat-cell { width: 18px; min-width: 18px; height: 18px; text-align: center; }
.loading-heatmap .heat-cell span { display: none; }
.heatmap-tooltip { position: fixed; display: none; z-index: 10000; max-width: 360px; pointer-events: none; background: rgba(31,41,51,0.96); color: #fff; border-radius: 10px; padding: 8px 10px; font-size: 12px; line-height: 1.4; box-shadow: 0 12px 32px rgba(31,41,51,0.28); }
.phi-metric-heatmap { margin: 12px 0 16px; border: 1px solid var(--line); border-radius: 18px; background: #fff; overflow: auto; }
.phi-metric-heatmap table { min-width: 760px; }
.phi-metric-heatmap th { position: static; cursor: default; }
.phi-metric-heatmap .metric-composite { border-left: 3px solid var(--accent); border-right: 3px solid var(--accent); }
.phi-metric-cell { text-align: right; font-variant-numeric: tabular-nums; }
.phi-metric-cell.best { font-weight: 900; color: #063f3a; outline: 2px solid rgba(15,118,110,0.35); outline-offset: -2px; }
.phi-metric-row-active td, .phi-metric-row-active th { background: #f0fbf8; }
@media (max-width: 900px) { .hero, .controls, .run-summary, .subgrid { grid-template-columns: 1fr; } .shell { padding: 20px 14px 44px; } .table-tools { grid-template-columns: 1fr; } }
"""


JS = r"""
const DATA_PAYLOAD_GZIP_BASE64 = "__DASHBOARD_DATA_GZIP_BASE64__";
let DATA = null;
const state = {
  runId: "",
  pigeanGroupId: "",
  section: "genes",
  eagglSection: "factors",
  groupId: "",
  modeId: "",
  geneLoadingSource: "",
  heatmapMetric: "loading",
  columnFilters: {},
  columnFilterOps: {},
  sorts: {},
  pages: {},
  pageSizes: {},
  openRows: {},
  factorDetailTabs: {},
  heatmapSorts: {},
  heatmapRegexFilters: {},
  focus: null,
};
function byId(id) { return document.getElementById(id); }
function esc(value) { return String(value ?? "").replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[ch])); }
function fmt(value) { if (value === null || value === undefined || value === "") return "NA"; if (typeof value === "number") { if (!Number.isFinite(value)) return "NA"; if (Math.abs(value) >= 1000) return value.toFixed(0); if (Math.abs(value) >= 10) return value.toFixed(2); return value.toPrecision(3); } return esc(value); }
function fmtText(value) { if (value === null || value === undefined || value === "") return "NA"; if (typeof value === "number") return fmt(value); return String(value); }
function isNumber(value) { return typeof value === "number" && Number.isFinite(value); }
async function decodePayload() { const binary = Uint8Array.from(atob(DATA_PAYLOAD_GZIP_BASE64), c => c.charCodeAt(0)); if (typeof DecompressionStream === "undefined") throw new Error("This browser does not support embedded gzip decoding via DecompressionStream."); const stream = new Blob([binary]).stream().pipeThrough(new DecompressionStream("gzip")); return JSON.parse(await new Response(stream).text()); }
function regexMatch(value, raw) { try { return new RegExp(String(raw), "i").test(String(value ?? "")); } catch { return false; } }
function columnFiltersMatch(row, columns, id, presetFilters = {}) { const filters = {...presetFilters, ...(state.columnFilters[id] || {})}; const ops = state.columnFilterOps[id] || {}; for (const c of columns) { const raw = filters[c.key]; if (raw === undefined || raw === "") continue; const value = c.filterValue ? c.filterValue(row) : row[c.key]; if (c.numeric) { const target = Number(raw); const actual = Number(value); if (!Number.isFinite(target) || !Number.isFinite(actual)) return false; const op = ops[c.key] || ">="; if (op === ">=" && actual < target) return false; if (op === "<=" && actual > target) return false; } else if (!regexMatch(value, raw)) return false; } return true; }
function sortedRows(rows, key) { if (!key) return rows; const desc = key.startsWith("-"); const col = desc ? key.slice(1) : key; return [...rows].sort((a,b) => { const av = a[col], bv = b[col]; const cmp = isNumber(av) && isNumber(bv) ? av - bv : String(av ?? "").localeCompare(String(bv ?? "")); return desc ? -cmp : cmp; }); }
function restoreFocus() { if (!state.focus || typeof CSS === "undefined" || !CSS.escape) return; const input = document.querySelector(`[data-focus-key="${CSS.escape(state.focus)}"]`); if (!input) return; input.focus(); const len = input.value.length; try { input.setSelectionRange(len, len); } catch {} }
function preserveScroll(fn) { const x = window.scrollX, y = window.scrollY; fn(); window.scrollTo(x, y); requestAnimationFrame(() => window.scrollTo(x, y)); }
function tableHtml(id, rows, columns, opts = {}) {
  const sortKey = state.sorts[id] || "";
  const pageSize = state.pageSizes[id] || opts.pageSize || 20;
  const presetFilters = opts.presetFilters || {};
  const filtered = sortedRows((rows || []).filter(r => columnFiltersMatch(r, columns, id, presetFilters)), sortKey);
  const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize));
  const page = Math.min(state.pages[id] || 1, pageCount);
  state.pages[id] = page;
  const shown = filtered.slice((page - 1) * pageSize, page * pageSize);
  const header = columns.map(c => {
    const info = c.definition ? `<button class="column-info" data-column-info="${esc(c.definition)}" aria-label="Definition for ${esc(c.label || c.key)}" title="Click for definition">i</button>` : "";
    return `<th class="${c.numeric ? "number" : ""}" data-table="${esc(id)}" data-col="${esc(c.key)}">${esc(c.label || c.key)}${info}</th>`;
  }).join("");
  const filterHeader = columns.map(c => {
    const filters = {...presetFilters, ...(state.columnFilters[id] || {})};
    const ops = state.columnFilterOps[id] || {};
    const value = filters[c.key] || "";
    if (c.noFilter) return `<th></th>`;
    if (c.numeric) return `<th class="number"><span class="numeric-filter"><select data-column-filter-op-table="${esc(id)}" data-column-filter-op-col="${esc(c.key)}" data-focus-key="op:${esc(id)}:${esc(c.key)}"><option value=">=" ${(ops[c.key] || ">=") === ">=" ? "selected" : ""}>&gt;=</option><option value="<=" ${ops[c.key] === "<=" ? "selected" : ""}>&lt;=</option></select><input class="column-filter" type="text" inputmode="decimal" value="${esc(value)}" data-column-filter-table="${esc(id)}" data-column-filter-col="${esc(c.key)}" data-focus-key="col:${esc(id)}:${esc(c.key)}"></span></th>`;
    return `<th><input class="column-filter" type="text" value="${esc(value)}" data-column-filter-table="${esc(id)}" data-column-filter-col="${esc(c.key)}" data-focus-key="col:${esc(id)}:${esc(c.key)}"></th>`;
  }).join("");
  const body = shown.map((row, ix) => {
    const key = `${id}:${row.id || row.gene || row.gene_set || row.factor || ix}`;
    const cells = columns.map(c => `<td class="${c.numeric ? "number" : ""}">${c.render ? c.render(row, key) : fmt(row[c.key])}</td>`).join("");
    const showDetail = opts.detailRenderer && (opts.hasDetail ? opts.hasDetail(row, key) : state.openRows[key]);
    const detail = showDetail ? `<tr class="details-row"><td colspan="${columns.length}">${opts.detailRenderer(row, key)}</td></tr>` : "";
    return `<tr>${cells}</tr>${detail}`;
  }).join("") || `<tr><td colspan="${columns.length}" class="empty">No rows available.</td></tr>`;
  return `<div class="table-tools"><span class="note">${filtered.length} matched / ${(rows || []).length} rows</span><label class="note">Rows <select class="compact" data-page-size-table="${esc(id)}">${[10,20,50,100,250].map(n => `<option value="${n}" ${n === pageSize ? "selected" : ""}>${n}</option>`).join("")}</select></label></div><div class="table-wrap"><table><thead><tr>${header}</tr><tr class="column-filter-row">${filterHeader}</tr></thead><tbody>${body}</tbody></table></div><div class="pager"><button data-page-table="${esc(id)}" data-page-delta="-1" ${page <= 1 ? "disabled" : ""}>Prev</button><span class="note">Page ${page} / ${pageCount}</span><button data-page-table="${esc(id)}" data-page-delta="1" ${page >= pageCount ? "disabled" : ""}>Next</button></div>`;
}
function smallTable(rows, cols, id, opts = {}) { return tableHtml(id, rows || [], cols, {pageSize: 20, ...opts}); }
function warningsHtml(items) { return (items || []).map(w => `<div class="warning">${esc(w)}</div>`).join(""); }
function pigeanGroups() { return DATA.pigean_groups || []; }
function selectedPigeanGroup() { const groups = pigeanGroups(); if (!groups.length) return null; return groups.find(g => g.group_id === state.pigeanGroupId) || groups[0] || null; }
function runsForSelectedPigeanGroup() { const group = selectedPigeanGroup(); const all = DATA.pigean_runs || []; if (!group) return all; const allowed = new Set(group.run_ids || []); return all.filter(r => allowed.has(r.run_id)); }
function selectedRun() { const runs = runsForSelectedPigeanGroup(); const selected = runs.find(r => r.run_id === state.runId) || runs[0] || null; if (selected) state.runId = selected.run_id; const group = selectedPigeanGroup(); if (group) state.pigeanGroupId = group.group_id; return selected; }
function selectedEagglRunId(fallbackRunId) { const group = selectedPigeanGroup(); return (group && (DATA.eaggl_groups || {})[group.group_id]) ? group.group_id : fallbackRunId; }
function groupsForRun(runId) { return (DATA.eaggl_groups || {})[runId] || []; }
function hasMeaningfulGroups(runId) { return groupsForRun(runId).some(g => (g.mode_ids || []).length > 1); }
function selectedGroup(runId) { const groups = groupsForRun(runId); if (!hasMeaningfulGroups(runId)) return null; return groups.find(g => g.group_id === state.groupId) || groups[0] || null; }
function eagglForRun(runId) { const runs = Object.values(DATA.eaggl_runs || {}).filter(r => r.run_id === runId); const group = selectedGroup(runId); if (!group) return runs; const allowed = new Set(group.mode_ids || []); return runs.filter(r => allowed.has(r.mode_id)); }
function isSelectedPhiRun(run) { const value = String((run?.selected_phi_metrics || {}).selected ?? "").toLowerCase(); return value === "1" || value === "1.0" || value === "true" || value === "yes"; }
function selectedEaggl(runId) { const runs = eagglForRun(runId); const explicit = runs.find(r => r.mode_id === state.modeId); if (explicit) return explicit; return runs.find(isSelectedPhiRun) || runs[0] || null; }
function geneLoadingSources(eaggl) { return Object.values(eaggl?.gene_loading_sources || {}); }
function selectedGeneSource(eaggl) { const sources = geneLoadingSources(eaggl); const defaultId = DATA.default_gene_loading_source || "full_direct"; return sources.find(s => s.id === state.geneLoadingSource) || sources.find(s => s.id === defaultId) || sources.find(s => s.id === "full_direct") || sources.find(s => s.id === "discovery") || sources.find(s => s.id === "full_via_gene_sets") || sources[0] || null; }
function factorGenesFromSource(eaggl, factor) { const source = selectedGeneSource(eaggl); return (source?.by_factor || {})[factor.factor] || factor.genes || []; }
function geneDetails(row, key) { const run = selectedRun(); const sets = (run.gene_expansions || {})[row.gene] || []; return `<div class="details-box"><div class="subgrid"><div class="subpanel"><h3>Supporting gene sets</h3>${smallTable(sets, [{key:"gene_set",label:"Gene set"},{key:"label",label:"Label"},{key:"beta",label:"Beta",numeric:true},{key:"beta_uncorrected",label:"Beta uncorr",numeric:true},{key:"n",label:"N",numeric:true}], `${key}:sets`)}</div><div class="subpanel"><h3>Provenance</h3><p><strong>Combined:</strong> ${fmt(row.combined)}</p><p><strong>Direct:</strong> ${fmt(row.log_bf)}</p><p><strong>Indirect:</strong> ${fmt(row.prior)}</p><p><strong>HuGE/GWAS:</strong> ${fmt(row.huge_score)}</p><p><strong>Location:</strong> ${esc(row.chrom || "NA")}:${fmt(row.start)}-${fmt(row.end)}</p></div></div></div>`; }
function geneSetDetails(row, key) { const run = selectedRun(); const genes = (run.gene_set_expansions || {})[row.gene_set] || []; return `<div class="details-box"><div class="subgrid"><div class="subpanel"><h3>Member genes passing dashboard filter</h3>${smallTable(genes, [{key:"gene",label:"Gene"},{key:"combined",label:"Combined",numeric:true},{key:"log_bf",label:"Direct",numeric:true},{key:"prior",label:"Indirect",numeric:true}], `${key}:genes`)}</div><div class="subpanel"><h3>Provenance</h3><p><strong>Beta:</strong> ${fmt(row.beta)}</p><p><strong>Beta uncorrected:</strong> ${fmt(row.beta_uncorrected)}</p><p><strong>P:</strong> ${fmt(row.p_orig)}</p><p><strong>Z:</strong> ${fmt(row.z_orig)}</p><p><strong>Filter:</strong> ${esc(row.filter_reason || "NA")}</p></div></div></div>`; }
function renderControls(run, eaggl) {
  const groups = pigeanGroups();
  const selectedGroup = selectedPigeanGroup();
  const runs = runsForSelectedPigeanGroup();
  if (!state.runId && runs.length) state.runId = runs[0].run_id;
  const groupControl = groups.length ? `<div><label>PIGEAN group</label><select id="pigeanGroupSelect">${groups.map(g => `<option value="${esc(g.group_id)}" ${g.group_id === (selectedGroup?.group_id || "") ? "selected" : ""}>${esc(g.title || g.group_id)}</option>`).join("")}</select></div>` : "";
  const runOptions = runs.map(r => `<option value="${esc(r.run_id)}" ${r.run_id === state.runId ? "selected" : ""}>${esc(r.title || r.run_id)}</option>`).join("");
  const runControl = `<div><label>PIGEAN run</label><select id="runSelect">${runOptions}</select></div>`;
  const contextControl = `<div><label>Search context</label><div class="pill">${esc(run.trait_id || "supplied outputs")} · ${esc((run.warnings || []).length ? "partial" : "loaded")}</div></div>`;
  const controls = groups.length ? `<div class="controls">${groupControl}${runControl}</div><div class="controls">${contextControl}</div>` : `<div class="controls">${runControl}${contextControl}</div>`;
  return `<section class="panel">${controls}<div class="run-summary"><div class="stat"><strong>${(run.genes || []).length.toLocaleString()}</strong><span>genes in output</span></div><div class="stat"><strong>${(run.gene_sets || []).length.toLocaleString()}</strong><span>gene sets in output</span></div><div class="stat"><strong>${eaggl ? (eaggl.factors || []).length.toLocaleString() : "0"}</strong><span>EAGGL factors</span></div></div><p class="lede">${esc(run.summary || "PIGEAN run supplied on the dashboard command line.")}</p><p class="path">${esc(run.paths?.gene_stats || "missing gene stats")}<br>${esc(run.paths?.gene_set_stats || "missing gene-set stats")}</p>${warningsHtml(run.warnings)}</section>`;
}
function renderPigeanTables(run) { return `<section class="panel"><div class="tabs"><button data-section="genes" class="${state.section === "genes" ? "active" : ""}">Top genes</button><button data-section="gene_sets" class="${state.section === "gene_sets" ? "active" : ""}">Top gene sets</button><button data-section="status" class="${state.section === "status" ? "active" : ""}">Status</button></div><div id="pigean-table-region">${renderActivePigeanTable(run)}</div></section>`; }
function renderActivePigeanTable(run) { return state.section === "gene_sets" ? renderGeneSets(run) : state.section === "status" ? renderStatus(run) : renderGenes(run); }
function renderGenes(run) { const cols = [{key:"gene",label:"Gene"},{key:"combined",label:"Combined",numeric:true},{key:"log_bf",label:"Direct",numeric:true},{key:"prior",label:"Indirect",numeric:true},{key:"huge_score",label:"HuGE/GWAS",numeric:true},{key:"n",label:"N",numeric:true},{key:"chrom",label:"Chr"},{key:"provenance",label:"Provenance",noFilter:true,render:(r,k)=>`<button data-open-row="${esc(k)}">${state.openRows[k] ? "Hide" : "Show"}</button>`}]; return `<h2>PIGEAN genes</h2><p class="note">Default embedded genes pass combined support ≥ ${fmt(DATA.thresholds.gene_combined)}.</p>${tableHtml(`genes:${run.run_id}`, run.genes, cols, {detailRenderer: geneDetails})}`; }
function renderGeneSets(run) { const cols = [{key:"gene_set",label:"Gene set"},{key:"label",label:"Label"},{key:"beta",label:"Beta",numeric:true},{key:"beta_uncorrected",label:"Beta uncorr",numeric:true},{key:"p_orig",label:"P",numeric:true},{key:"z_orig",label:"Z",numeric:true},{key:"n",label:"N",numeric:true},{key:"provenance",label:"Provenance",noFilter:true,render:(r,k)=>`<button data-open-row="${esc(k)}">${state.openRows[k] ? "Hide" : "Show"}</button>`}]; return `<h2>PIGEAN gene sets</h2><p class="note">Default embedded gene sets pass beta_uncorrected ≥ ${fmt(DATA.thresholds.gene_set_beta_uncorrected)} when available.</p>${tableHtml(`gene_sets:${run.run_id}`, run.gene_sets, cols, {detailRenderer: geneSetDetails})}`; }
function factorDetailRows(eaggl, f, table) { if (table === "genes") return {title:"Top gene loadings", rows:factorGenesFromSource(eaggl, f), cols:[{key:"gene",label:"Gene"},{key:"loading",label:"Loading",numeric:true},{key:"cosine_loading",label:"Cosine",numeric:true},{key:"euclidean_distance",label:"Euclidean",numeric:true},{key:"combined",label:"Gene combined",numeric:true}]}; return {title:"Top gene set loadings", rows:f.gene_sets || [], cols:[{key:"gene_set",label:"Gene set"},{key:"label",label:"Label"},{key:"loading",label:"Loading",numeric:true},{key:"cosine_loading",label:"Cosine",numeric:true},{key:"euclidean_distance",label:"Euclidean",numeric:true},{key:"beta",label:"Beta",numeric:true}]}; }
function factorInlineDetails(eaggl, f, table, rowKey) { return `<button data-open-row-tab="${esc(rowKey)}" data-tab="${esc(table)}">${state.openRows[rowKey] && state.factorDetailTabs[rowKey] === table ? "Hide" : "Show"}</button>`; }
function factorHasDetail(f, rowKey) { return Boolean(state.openRows[rowKey]); }
function nestedFilterOptions(parentTableId, nestedTable) { const parentKey = nestedTable === "genes" ? "top_genes" : "top_gene_sets"; const value = (state.columnFilters[parentTableId] || {})[parentKey] || ""; if (!value) return {}; const childKey = nestedTable === "genes" ? "gene" : "gene_set"; return {presetFilters: {[childKey]: value}}; }
function factorRowDetails(eaggl, f, rowKey) { const requestedTab = state.factorDetailTabs[rowKey] || "genes"; const tab = requestedTab === "gene_sets" ? "gene_sets" : "genes"; const detail = factorDetailRows(eaggl, f, tab); const parentTableId = rowKey.split(":").slice(0, -1).join(":"); const labels = {genes:"Genes", gene_sets:"Gene sets"}; const tabs = ["genes", "gene_sets"].map(name => `<button data-factor-detail-tab="${esc(rowKey)}" data-tab="${esc(name)}" class="${tab === name ? "active" : ""}">${labels[name]}</button>`).join(""); return `<div class="details-box factor-details"><div class="factor-tabs">${tabs}</div>${smallTable(detail.rows, detail.cols, `${rowKey}:${tab}:table`, nestedFilterOptions(parentTableId, tab))}</div>`; }
function selectedPhiMetric(eaggl, key) { const value = (eaggl.selected_phi_metrics || {})[key]; return Number.isFinite(Number(value)) ? Number(value) : value; }
function factorMetricSummary(eaggl) { const m = eaggl.selected_phi_metrics || {}; const pieces = []; if (m.phi !== undefined) pieces.push(`Selected phi: ${fmt(m.phi)}`); if (m.phi_composite_score !== undefined) pieces.push(`Composite: ${fmt(m.phi_composite_score)}`); if (m.selection_reason) pieces.push(`Reason: ${esc(m.selection_reason)}`); if (m.selection_rank !== undefined) pieces.push(`Rank: ${fmt(m.selection_rank)}`); return pieces.length ? `<div class="metric-summary">${pieces.map(p => `<span class="pill">${p}</span>`).join("")}</div>` : ""; }
function metricValue(item, metric) { const value = metric === "cosine_loading" ? item.cosine_loading : metric === "euclidean_distance" ? item.euclidean_distance : item.loading; return Number.isFinite(Number(value)) ? Number(value) : 0; }
function loadingMatrixRows(eaggl, kind) { const idKey = kind === "genes" ? "gene" : "gene_set"; const rowsById = new Map(); for (const factor of eaggl.factors || []) { const factorId = factor.factor; for (const item of (kind === "genes" ? factorGenesFromSource(eaggl, factor) : factor.gene_sets || [])) { const id = item[idKey] || item.id; if (!id) continue; if (!rowsById.has(id)) rowsById.set(id, {id, [idKey]: id, label: item.label, combined: item.combined, log_bf: item.log_bf, prior: item.prior, beta: item.beta, beta_uncorrected: item.beta_uncorrected, max_loading: 0, values:{}}); const row = rowsById.get(id); const values = {loading: metricValue(item, "loading"), cosine_loading: metricValue(item, "cosine_loading"), euclidean_distance: metricValue(item, "euclidean_distance")}; row.values[factorId] = values; row.max_loading = Math.max(row.max_loading || 0, values.loading || 0); } } return [...rowsById.values()]; }
function heatColor(value, maxValue) { if (!Number.isFinite(value) || value <= 0 || !Number.isFinite(maxValue) || maxValue <= 0) return "rgba(244,247,248,0.95)"; const t = Math.min(1, Math.max(0, value / maxValue)); const alpha = 0.12 + 0.82 * t; return `rgba(15, 118, 110, ${alpha.toFixed(3)})`; }
function compileHeatmapRegex(tableId) { const raw = state.heatmapRegexFilters[tableId] || ""; if (!raw) return {raw, regex:null, error:""}; try { return {raw, regex:new RegExp(raw, "i"), error:""}; } catch (error) { return {raw, regex:null, error:error.message || "invalid regex"}; } }
function loadingHeatmap(eaggl, kind) { const rows = loadingMatrixRows(eaggl, kind); const idKey = kind === "genes" ? "gene" : "gene_set"; const factors = eaggl.factors || []; const tableId = `heatmap:${kind}:${eaggl.run_id}:${eaggl.mode_id}:${selectedGeneSource(eaggl)?.id || "default"}`; const metric = state.heatmapMetric || "loading"; const sortFactor = state.heatmapSorts[tableId] || ""; const regexState = compileHeatmapRegex(tableId); const filteredRows = regexState.regex ? rows.filter(row => regexState.regex.test(`${row[idKey] || ""} ${row.label || ""}`)) : rows; const maxValue = Math.max(0, ...filteredRows.flatMap(row => Object.values(row.values || {}).map(v => Number(v[metric]) || 0))); const sorted = [...filteredRows].sort((a,b) => sortFactor ? (metricValue({[metric]:(b.values[sortFactor] || {})[metric]}, metric) - metricValue({[metric]:(a.values[sortFactor] || {})[metric]}, metric)) : ((b.max_loading || 0) - (a.max_loading || 0) || String(a[idKey]).localeCompare(String(b[idKey])))); const title = kind === "genes" ? "EAGGL gene loading heatmap" : "EAGGL gene-set loading heatmap"; const rowLabel = kind === "genes" ? "Gene" : "Gene set"; const metricLabel = metric === "cosine_loading" ? "cosine" : metric === "euclidean_distance" ? "euclidean" : "raw"; const regexNote = regexState.error ? `<span class="warning">Invalid regex: ${esc(regexState.error)}</span>` : `<span class="note">${filteredRows.length} matched / ${rows.length} embedded rows · click a column to sort</span>`; const controls = `<div class="heatmap-controls"><label>Loading metric <select id="heatmapMetricSelect"><option value="loading" ${metric === "loading" ? "selected" : ""}>Raw</option><option value="cosine_loading" ${metric === "cosine_loading" ? "selected" : ""}>Cosine</option><option value="euclidean_distance" ${metric === "euclidean_distance" ? "selected" : ""}>Euclidean</option></select></label><label class="heatmap-regex-label">${rowLabel} regex <input class="heatmap-regex" type="text" value="${esc(regexState.raw)}" placeholder="e.g. PPARG|PDX1" data-heatmap-regex-table="${esc(tableId)}" data-focus-key="heatmap-regex:${esc(tableId)}"></label>${regexNote}</div>`; const head = factors.map(f => `<th class="heat-factor" data-heatmap-table="${esc(tableId)}" data-factor="${esc(f.factor)}" title="${esc(f.label || f.factor)}"><span>${esc(f.label || f.factor)}</span></th>`).join(""); const body = sorted.map(row => `<tr><th class="heat-row-label" title="${esc(row.label || row[idKey])}">${esc(row[idKey])}</th>${factors.map(f => { const vals = row.values[f.factor] || {}; const value = Number(vals[metric]) || 0; const tip = `${rowLabel}: ${row[idKey]} | Factor: ${f.label || f.factor} | Value (${metricLabel}): ${fmtText(value)}`; return `<td class="heat-cell" data-heatmap-tip="${esc(tip)}" title="${esc(tip)}" style="background:${heatColor(value, maxValue)}"><span>${value > 0 ? fmt(value) : ""}</span></td>`; }).join("")}</tr>`).join("") || `<tr><td class="empty" colspan="${factors.length + 1}">No rows match the current regex.</td></tr>`; return `<h2>${title}</h2><p class="note">Rows are dashboard-embedded ${kind === "genes" ? "genes" : "gene sets"}; columns are factors and show labels on hover.</p>${controls}<div class="heatmap-wrap"><table class="loading-heatmap"><thead><tr><th class="heat-row-label">${rowLabel}</th>${head}</tr></thead><tbody>${body}</tbody></table></div><div id="heatmapTooltip" class="heatmap-tooltip"></div>`; }
function phenotypeRows(eaggl) { const rows = []; for (const factor of eaggl.factors || []) { for (const pheno of factor.phenotypes || []) rows.push({...pheno, factor_label: factor.label || factor.factor}); } rows.sort((a,b) => (Number(b.nnls_loading) || 0) - (Number(a.nnls_loading) || 0) || (Number(b.beta) || -1e300) - (Number(a.beta) || -1e300)); return rows; }
function renderPhenotypes(eaggl) { const rows = phenotypeRows(eaggl); const has = key => rows.some(r => r[key] !== null && r[key] !== undefined && r[key] !== ""); const cols = [{key:"trait",label:"Trait"},{key:"factor",label:"Factor"},{key:"factor_label",label:"Factor label"},{key:"nnls_loading",label:"NNLS",numeric:true},{key:"cosine_loading",label:"Cosine",numeric:true},{key:"euclidean_distance",label:"Euclidean",numeric:true}]; if (has("beta")) cols.push({key:"beta",label:"Beta",numeric:true}); if (has("beta_uncorrected")) cols.push({key:"beta_uncorrected",label:"Beta uncorr",numeric:true}); if (has("beta_tilde")) cols.push({key:"beta_tilde",label:"Beta tilde",numeric:true}); if (has("se")) cols.push({key:"se",label:"SE",numeric:true}); if (has("z")) cols.push({key:"z",label:"Z",numeric:true}); if (has("p_value")) cols.push({key:"p_value",label:"P",numeric:true}); cols.push({key:"trait_neff",label:"Trait neff",numeric:true},{key:"trait_response_source",label:"Response"},{key:"is_anchor",label:"Anchor"}); return `<h2>EAGGL phenotype loadings</h2><p class="note">Rows can come from a premerged trait_factor_links.out.gz, or the dashboard can merge trait_factor_links.nnls.out.gz with factor_trait_pigean_enrichments.out.gz during construction. NNLS is the native fixed-W phenotype projection; beta statistics come from the PIGEAN multi-Y enrichment workflow.</p>${tableHtml(`phenotypes:${eaggl.run_id}:${eaggl.mode_id}`, rows, cols, {pageSize: 20})}`; }
const PHI_HEATMAP_METRICS = [
  {key:"phi", label:"Phi"},
  {key:"phi_composite_score", label:"Composite", composite:true},
  {key:"factor_size_score", label:"Size"},
  {key:"nonoverlap_score", label:"Nonoverlap"},
  {key:"entity_concentration_score", label:"Concentr."},
  {key:"coverage_score", label:"Coverage"},
  {key:"reconstruction_score", label:"Recon."},
  {key:"coherence_score", label:"Coherence"},
  {key:"factor_balance_score", label:"Balance"},
  {key:"annotation_bridge_qc_score", label:"Bridge QC"},
  {key:"num_factors", label:"Factors"},
  {key:"modal_factor_count", label:"Modal K"}
];
function metricNumber(value) { const n = Number(value); return Number.isFinite(n) ? n : null; }
function groupMetricHeatmap(runId, activeModeId) {
  const group = selectedGroup(runId);
  if (!group) return "";
  const runs = eagglForRun(runId);
  const rows = runs.map(r => ({...r, metrics:{...(r.selected_phi_metrics || {}), ...(((group.metrics_by_mode || {})[r.mode_id]) || {})}}));
  const available = PHI_HEATMAP_METRICS.filter(m => rows.some(r => metricNumber(r.metrics[m.key]) !== null));
  if (!available.length) return "";
  const maxByMetric = {};
  for (const metric of available) {
    const values = rows.map(r => metricNumber(r.metrics[metric.key])).filter(v => v !== null);
    maxByMetric[metric.key] = values.length ? Math.max(...values) : null;
  }
  const head = `<tr><th>EAGGL run</th>${available.map(m => `<th class="${m.composite ? "metric-composite" : ""}">${esc(m.label)}</th>`).join("")}</tr>`;
  const body = rows.map(r => {
    const active = r.mode_id === activeModeId;
    const cells = available.map(m => {
      const value = metricNumber(r.metrics[m.key]);
      const selected = isSelectedPhiRun(r) && m.composite;
      const best = m.key !== "phi" && value !== null && maxByMetric[m.key] !== null && Math.abs(value - maxByMetric[m.key]) <= 1e-12;
      const text = value === null ? "NA" : `${selected ? "★ " : ""}${fmt(value)}`;
      return `<td class="phi-metric-cell ${best ? "best" : ""} ${m.composite ? "metric-composite" : ""}">${text}</td>`;
    }).join("");
    return `<tr class="${active ? "phi-metric-row-active" : ""}"><th>${esc(r.title || r.mode_id)}</th>${cells}</tr>`;
  }).join("");
  return `<div class="phi-metric-heatmap"><table><thead>${head}</thead><tbody>${body}</tbody></table></div>`;
}
function factorGraphKey(eaggl) { const source = selectedGeneSource(eaggl); return eaggl ? `${eaggl.run_id}::${eaggl.mode_id}::${source?.id || "default"}` : ""; }
function detachMountedFactorGraph(eaggl) { const graph = document.getElementById("factor-graph-mounted"); if (!graph || graph.dataset.graphKey !== factorGraphKey(eaggl)) return null; graph.remove(); return graph; }
function selectedFactorGraph(eaggl) { const source = selectedGeneSource(eaggl); if (source?.factor_graph_available) return {html: source.factor_graph_html, path: source.factor_graph_html_path, label: source.label}; return {html: eaggl?.factor_graph_html || "", path: eaggl?.paths?.factor_graph_html || "", label: source?.label || "default"}; }
function renderFactorGraphSlot(eaggl) { const graph = selectedFactorGraph(eaggl); if (!graph.html) return `<details class="factor-graph-accordion"><summary>Factor graph</summary><div class="factor-graph-body empty">No factor graph was generated for this EAGGL run and gene-loading source.</div></details>`; return `<div id="factor-graph-slot" data-graph-key="${esc(factorGraphKey(eaggl))}"></div>`; }
function createFactorGraphNode(eaggl) { const graph = selectedFactorGraph(eaggl); const graphFrame = graph.html ? `<iframe class="factor-graph-frame" srcdoc="${esc(graph.html)}"></iframe>` : ""; const wrapper = document.createElement("div"); wrapper.id = "factor-graph-mounted"; wrapper.dataset.graphKey = factorGraphKey(eaggl); wrapper.innerHTML = graphFrame ? `<details class="factor-graph-accordion" open><summary>Factor graph</summary><div class="factor-graph-body"><p class="note">Interactive graph generated from ${esc(graph.label)} gene loadings.</p>${graphFrame}<p class="path">${esc(graph.path || "")}</p></div></details>` : `<details class="factor-graph-accordion"><summary>Factor graph</summary><div class="factor-graph-body empty">No factor graph was generated for this EAGGL run and gene-loading source.</div></details>`; return wrapper; }
function mountFactorGraph(eaggl, existingGraph) { const slot = document.getElementById("factor-graph-slot"); if (!slot) return; slot.replaceWith(existingGraph || createFactorGraphNode(eaggl)); }
function renderEagglTableRegion(eaggl) {
  const anchorCols = (eaggl.anchor_traits || []).map(a => ({
    key:a.column,
    label:`${a.trait} relevance`,
    numeric:true,
    definition:`Anchor-specific factor relevance/capture for anchor trait ${a.trait}.`
  }));
  const rows = (eaggl.factors || []).map(f => ({
    ...f,
    mass:f.combined_mass_fraction,
    n_genes:factorGenesFromSource(eaggl, f).length,
    n_gene_sets:(f.gene_sets || []).length,
    n_traits:(f.phenotypes || []).length,
    selected_phi:selectedPhiMetric(eaggl, "phi"),
    per_factor_size_score:f.factor_size_score,
  }));
  const cols = [
    {key:"label",label:"Mechanism",definition:"Factor label assigned from top loadings; badge shows the internal Factor identifier.",render:r=>`${esc(r.label || r.factor)} <span class="badge">${esc(r.factor)}</span>`},
    {key:"top_genes",label:"Top gene loadings",definition:"Expand to inspect top genes for this factor under the selected dashboard gene-loading source.",filterValue:r=>factorGenesFromSource(eaggl, r).map(g=>`${g.gene} ${g.label || ""}`).join(" "),render:(r,k)=>factorInlineDetails(eaggl,r,"genes",k)},
    {key:"top_gene_sets",label:"Top gene set loadings",definition:"Expand to inspect top gene-set/annotation loadings for this factor.",filterValue:r=>(r.gene_sets || []).map(gs=>`${gs.gene_set} ${gs.label || ""}`).join(" "),render:(r,k)=>factorInlineDetails(eaggl,r,"gene_sets",k)},
    {key:"lambda",label:"Lambda",numeric:true,definition:"NMF factor strength/ARD scale reported by EAGGL for this factor."},
    {key:"anchor_any_joint",label:"Anchor joint relevance",numeric:true,definition:"Factor relevance to the aggregate anchor support surface after joint factor capture/projection."},
    {key:"anchor_any_marginal",label:"Anchor marginal relevance",numeric:true,definition:"Factor relevance to the aggregate anchor support surface from marginal factor overlap/capture."},
    ...anchorCols,
    {key:"mass",label:"Mass",numeric:true,definition:"Combined mass fraction from factors.out: the factor's share of total retained factor loading mass."},
    {key:"factor_gene_mass",label:"Gene mass",numeric:true,definition:"Sum of capped nonnegative gene loadings for this factor; used by the composite factor-size score when available."},
    {key:"per_factor_size_score",label:"Size score",numeric:true,definition:"Per-factor score for closeness to the target factor gene mass. It is 1 at the target and decreases for smaller or larger factors."},
    {key:"gene_effective_support",label:"Gene Neff",numeric:true,definition:"Effective number of genes supporting this factor based on its gene loading distribution."},
    {key:"gene_max_jaccard",label:"Gene overlap",numeric:true,definition:"Maximum soft gene-loading Jaccard overlap with another factor; lower indicates less redundant factors."},
    {key:"factor_coherence_score",label:"Coherence",numeric:true,definition:"Per-factor within-factor target enrichment score used by the composite coherence component when available."},
    {key:"within_factor_target_mean",label:"Target mean",numeric:true,definition:"Mean target value among entities weighted by this factor's loadings."},
    {key:"factor_gene_bridge_count",label:"Gene bridges",numeric:true,definition:"Count of genes flagged as bridge-like for this factor by phi-selection support metrics, when available."},
    {key:"factor_annotation_bridge_count",label:"Annot bridges",numeric:true,definition:"Count of annotations flagged as bridge-like for this factor by phi-selection support metrics, when available."},
    {key:"selected_phi",label:"Phi",numeric:true,definition:"Candidate phi selected for this EAGGL run."}
  ];
  const factorTableId = `factors:${eaggl.run_id}:${eaggl.mode_id}`;
  if (state.eagglSection === "genes") return loadingHeatmap(eaggl, "genes");
  if (state.eagglSection === "gene_sets") return loadingHeatmap(eaggl, "gene_sets");
  if (state.eagglSection === "phenotypes") return renderPhenotypes(eaggl);
  return `<h2>EAGGL factors</h2>${factorMetricSummary(eaggl)}<p class="note">Factor detail subtables inherit parent filters. Click column info icons for metric definitions. Run-level phi-selection metrics are shown in the selected-run summary and phi-sweep heatmap; this table keeps factor-level metrics and anchor-specific relevance.</p>${tableHtml(factorTableId, rows, cols, {hasDetail: factorHasDetail, detailRenderer:(row,key)=>factorRowDetails(eaggl,row,key)})}`;
}
function renderEagglPanel(runId) {
  const eaggl = selectedEaggl(runId);
  const modes = eagglForRun(runId);
  if (!eaggl) return `<section class="panel"><h2>EAGGL factors</h2><div class="empty">No EAGGL run was supplied for this PIGEAN run.</div></section>`;
  const groups = groupsForRun(runId);
  const meaningfulGroups = hasMeaningfulGroups(runId);
  const group = selectedGroup(runId);
  const sources = geneLoadingSources(eaggl);
  const currentSource = selectedGeneSource(eaggl);
  if (currentSource && state.geneLoadingSource !== currentSource.id) state.geneLoadingSource = currentSource.id;
  if (group && state.groupId !== group.group_id) state.groupId = group.group_id;
  const groupControl = meaningfulGroups ? `<div><label>EAGGL group</label><select id="groupSelect">${groups.map(g => `<option value="${esc(g.group_id)}" ${g.group_id === (group?.group_id || "") ? "selected" : ""}>${esc(g.title || g.group_id)}</option>`).join("")}</select></div>` : "";
  const modeControl = modes.length > 1 ? `<div><label>EAGGL run</label><select id="modeSelect">${modes.map(m => `<option value="${esc(m.mode_id)}" ${m.mode_id === eaggl.mode_id ? "selected" : ""}>${esc(m.title || m.mode_id)}</option>`).join("")}</select></div>` : `<div><label>EAGGL run</label><div class="pill">${esc(eaggl.title || eaggl.mode_id)}</div></div>`;
  const graphInfo = selectedFactorGraph(eaggl);
  const sourceControl = sources.length > 1 ? `<div><label>Gene loading source</label><select id="geneLoadingSourceSelect">${sources.map(s => `<option value="${esc(s.id)}" ${s.id === state.geneLoadingSource ? "selected" : ""}>${esc(s.label)}</option>`).join("")}</select></div>` : "";
  const groupedControls = meaningfulGroups
    ? `<div class="controls">${groupControl}</div>${groupMetricHeatmap(runId, eaggl.mode_id)}<div class="controls">${modeControl}${sourceControl}<div><span class="pill">${esc(graphInfo.html ? "Factor graph available" : "No factor graph")}</span></div></div>`
    : `<div class="controls">${modeControl}${sourceControl}<div><span class="pill">${esc(graphInfo.html ? "Factor graph available" : "No factor graph")}</span></div></div>`;
  return `<section class="panel" id="eaggl-panel">${groupedControls}<p class="lede">${esc(eaggl.summary || "EAGGL run supplied on the dashboard command line.")}</p><p class="path">${esc(eaggl.paths?.factors || "missing factors")}<br>${esc(currentSource?.path || "")}</p>${warningsHtml(eaggl.warnings)}${renderFactorGraphSlot(eaggl)}<div class="tabs"><button data-eaggl-section="factors" class="${state.eagglSection === "factors" ? "active" : ""}">Factors</button><button data-eaggl-section="genes" class="${state.eagglSection === "genes" ? "active" : ""}">Genes</button><button data-eaggl-section="gene_sets" class="${state.eagglSection === "gene_sets" ? "active" : ""}">Gene sets</button><button data-eaggl-section="phenotypes" class="${state.eagglSection === "phenotypes" ? "active" : ""}">Phenotypes</button></div><div id="eaggl-table-region">${renderEagglTableRegion(eaggl)}</div></section>`;
}
function renderStatus(run) { const rows = []; for (const r of DATA.pigean_runs || []) rows.push({kind:"PIGEAN", run:r.run_id, mode:"", status:(r.warnings || []).length ? "missing or partial" : "loaded", notes:(r.warnings || []).join("; ")}); for (const r of Object.values(DATA.eaggl_runs || {})) rows.push({kind:"EAGGL", run:r.run_id, mode:r.mode_id, status:(r.warnings || []).length ? "missing or partial" : "loaded", notes:(r.warnings || []).join("; ")}); return `<h2>Dashboard status</h2>${tableHtml("status", rows, [{key:"kind",label:"Kind"},{key:"run",label:"Run"},{key:"mode",label:"Mode"},{key:"status",label:"Status"},{key:"notes",label:"Notes"}], {pageSize: 20})}`; }
function render() { const run = selectedRun(); if (!run) { byId("app").innerHTML = `<section class="panel"><div class="empty">No PIGEAN or EAGGL runs were supplied.</div></section>`; return; } const eagglRunId = selectedEagglRunId(run.run_id); const eaggl = selectedEaggl(eagglRunId); const existingGraph = detachMountedFactorGraph(eaggl); byId("app").innerHTML = renderControls(run, eaggl) + renderPigeanTables(run) + renderEagglPanel(eagglRunId); bind(); bindHeatmapTooltips(); mountFactorGraph(eaggl, existingGraph); restoreFocus(); }
function updateTabClasses(attr, active) { document.querySelectorAll(`button[${attr}]`).forEach(btn => btn.classList.toggle("active", btn.dataset[attr.replace(/^data-/, "").replace(/-([a-z])/g, (_, ch) => ch.toUpperCase())] === active)); }
function moveHeatmapTooltip(event, tip) { tip.style.left = `${Math.min(window.innerWidth - 380, event.clientX + 14)}px`; tip.style.top = `${Math.min(window.innerHeight - 80, event.clientY + 14)}px`; }
function bindHeatmapTooltips() { const tip = byId("heatmapTooltip"); if (!tip) return; document.querySelectorAll(".heat-cell[data-heatmap-tip]").forEach(cell => { cell.onmouseenter = event => { tip.textContent = cell.dataset.heatmapTip || ""; tip.style.display = tip.textContent ? "block" : "none"; moveHeatmapTooltip(event, tip); }; cell.onmousemove = event => moveHeatmapTooltip(event, tip); cell.onmouseleave = () => { tip.style.display = "none"; }; }); }
function refreshPigeanTable() { const region = byId("pigean-table-region"); const run = selectedRun(); if (!region || !run) return false; region.innerHTML = renderActivePigeanTable(run); bind(); restoreFocus(); return true; }
function refreshEagglTable() { const region = byId("eaggl-table-region"); const run = selectedRun(); const eaggl = run ? selectedEaggl(selectedEagglRunId(run.run_id)) : null; if (!region || !eaggl) return false; region.innerHTML = renderEagglTableRegion(eaggl); bind(); bindHeatmapTooltips(); restoreFocus(); return true; }
function refreshTableForKey(key) { if (String(key).startsWith("factors:") || String(key).startsWith("heatmap:") || String(key).startsWith("phenotypes:") || String(key).includes(":gene_by_")) return refreshEagglTable(); if (String(key).startsWith("genes:") || String(key).startsWith("gene_sets:")) return refreshPigeanTable(); return refreshPigeanTable() || refreshEagglTable(); }
function toggleOpen(key) { state.openRows[key] = !state.openRows[key]; preserveScroll(() => { if (!refreshTableForKey(key)) render(); }); }
function toggleOpenWithTab(key, tab) { if (state.openRows[key] && state.factorDetailTabs[key] === tab) state.openRows[key] = false; else { state.openRows[key] = true; state.factorDetailTabs[key] = tab; } preserveScroll(() => { if (!refreshEagglTable()) render(); }); }
function bind() { const pigeanGroupSel = byId("pigeanGroupSelect"); if (pigeanGroupSel) pigeanGroupSel.onchange = () => { state.pigeanGroupId = pigeanGroupSel.value; state.runId = ""; state.groupId = ""; state.modeId = ""; state.geneLoadingSource = ""; state.pages = {}; state.openRows = {}; render(); }; const runSel = byId("runSelect"); if (runSel) runSel.onchange = () => { state.runId = runSel.value; state.groupId = ""; state.modeId = ""; state.geneLoadingSource = ""; state.pages = {}; state.openRows = {}; render(); }; const groupSel = byId("groupSelect"); if (groupSel) groupSel.onchange = () => { state.groupId = groupSel.value; state.modeId = ""; state.geneLoadingSource = ""; state.pages = {}; state.openRows = {}; render(); }; const modeSel = byId("modeSelect"); if (modeSel) modeSel.onchange = () => { state.modeId = modeSel.value; state.geneLoadingSource = ""; state.pages = {}; state.openRows = {}; render(); }; const sourceSel = byId("geneLoadingSourceSelect"); if (sourceSel) sourceSel.onchange = () => { state.geneLoadingSource = sourceSel.value; state.pages = {}; state.openRows = {}; render(); }; const heatMetric = byId("heatmapMetricSelect"); if (heatMetric) heatMetric.onchange = () => { state.heatmapMetric = heatMetric.value; refreshEagglTable(); }; document.querySelectorAll("input[data-heatmap-regex-table]").forEach(inp => { inp.onfocus = () => { const id = inp.dataset.heatmapRegexTable; state.focus = `heatmap-regex:${id}`; }; inp.oninput = () => { const id = inp.dataset.heatmapRegexTable; state.focus = `heatmap-regex:${id}`; state.heatmapRegexFilters[id] = inp.value; refreshEagglTable(); }; }); document.querySelectorAll("button[data-section]").forEach(btn => btn.onclick = () => { state.section = btn.dataset.section; state.pages = {}; updateTabClasses("data-section", state.section); refreshPigeanTable(); }); document.querySelectorAll("button[data-eaggl-section]").forEach(btn => btn.onclick = () => { state.eagglSection = btn.dataset.eagglSection; state.pages = {}; updateTabClasses("data-eaggl-section", state.eagglSection); refreshEagglTable(); }); document.querySelectorAll("button[data-column-info]").forEach(btn => btn.onclick = event => { event.preventDefault(); event.stopPropagation(); alert(btn.dataset.columnInfo || "No definition available."); }); document.querySelectorAll("button[data-open-row]").forEach(btn => btn.onclick = event => { event.preventDefault(); toggleOpen(btn.dataset.openRow); }); document.querySelectorAll("button[data-open-row-tab]").forEach(btn => btn.onclick = event => { event.preventDefault(); toggleOpenWithTab(btn.dataset.openRowTab, btn.dataset.tab); }); document.querySelectorAll("button[data-factor-detail-tab]").forEach(btn => btn.onclick = event => { event.preventDefault(); state.factorDetailTabs[btn.dataset.factorDetailTab] = btn.dataset.tab; preserveScroll(() => refreshEagglTable()); }); document.querySelectorAll("input[data-column-filter-table]").forEach(inp => { inp.onfocus = () => { state.focus = `col:${inp.dataset.columnFilterTable}:${inp.dataset.columnFilterCol}`; }; inp.oninput = () => { const id = inp.dataset.columnFilterTable, col = inp.dataset.columnFilterCol; state.focus = `col:${id}:${col}`; state.columnFilters[id] = state.columnFilters[id] || {}; state.columnFilters[id][col] = inp.value; state.pages[id] = 1; refreshTableForKey(id); }; }); document.querySelectorAll("select[data-column-filter-op-table]").forEach(sel => { sel.onfocus = () => { state.focus = `op:${sel.dataset.columnFilterOpTable}:${sel.dataset.columnFilterOpCol}`; }; sel.onchange = () => { const id = sel.dataset.columnFilterOpTable, col = sel.dataset.columnFilterOpCol; state.focus = `op:${id}:${col}`; state.columnFilterOps[id] = state.columnFilterOps[id] || {}; state.columnFilterOps[id][col] = sel.value; state.pages[id] = 1; refreshTableForKey(id); }; }); document.querySelectorAll("select[data-page-size-table]").forEach(sel => sel.onchange = () => { state.pageSizes[sel.dataset.pageSizeTable] = Number(sel.value) || 20; state.pages[sel.dataset.pageSizeTable] = 1; refreshTableForKey(sel.dataset.pageSizeTable); }); document.querySelectorAll("button[data-page-table]").forEach(btn => btn.onclick = () => { const id = btn.dataset.pageTable; state.pages[id] = Math.max(1, (state.pages[id] || 1) + Number(btn.dataset.pageDelta || 0)); refreshTableForKey(id); }); document.querySelectorAll("th[data-table]").forEach(th => th.onclick = () => { const table = th.dataset.table, col = th.dataset.col; state.sorts[table] = state.sorts[table] === col ? `-${col}` : col; refreshTableForKey(table); }); document.querySelectorAll("th[data-heatmap-table]").forEach(th => th.onclick = () => { state.heatmapSorts[th.dataset.heatmapTable] = th.dataset.factor; refreshEagglTable(); }); }
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
      <div class="hero-card">
        <h1>{safe_title}</h1>
        <p class="lede">Standalone post-processing dashboard for supplied PIGEAN and EAGGL outputs. Missing optional inputs are recorded in the dashboard rather than treated as fatal.</p>
      </div>
      <div id="summaryStats" class="hero-card stats"></div>
    </section>
    <div id="app"><section class="panel"><div class="empty">Loading embedded dashboard data...</div></section></div>
  </main>
  <script>{script}</script>
</body>
</html>
"""
