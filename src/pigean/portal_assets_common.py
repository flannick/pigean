"""Pieces shared by the Explorer (`portal_assets`) and the Comparer (`portal_compare_assets`):
the stylesheet, the JavaScript helpers (API client, formatting, fuzzy scoring, typeahead,
paginated tables, modal, ontology links) and the HTML document wrapper.
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
.landing .row { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.landing select, .landing input { width:100%; font-size:14px; padding:10px 12px; border-radius:10px; font-family:inherit; }
.landing select { appearance:auto; }
.ta-wrap { position:relative; }
.ta { position:absolute; left:0; right:0; top:calc(100% + 4px); background:#fff; border:1px solid var(--line); border-radius:10px; box-shadow:0 12px 30px rgba(31,41,51,.12); z-index:30; overflow:hidden; }
.ta[hidden] { display:none; }
.ta div { padding:7px 12px; cursor:pointer; display:flex; gap:10px; align-items:baseline; font-size:13px; }
.ta div b { font-weight:600; min-width:6ch; }
.ta div span { color:var(--muted); font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.ta div.active, .ta div:hover { background:var(--accent-soft); }
.landing .actions { display:flex; gap:12px; align-items:center; margin-top:18px; }
.landing button.primary { background:var(--accent); color:#fff; border-color:var(--accent); font-size:15px; padding:10px 18px; }
.landing button.primary:hover { background:#0b5f59; }
.landing .hint { color:var(--muted); font-size:12px; }
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
.pager { display:flex; gap:8px; align-items:center; justify-content:flex-end; margin:6px 0 2px; font-size:12px; color:var(--muted); }
.pager button { padding:3px 9px; font-weight:500; }
.pager button:disabled { opacity:.4; cursor:default; }
.sheet-controls { display:flex; gap:10px; align-items:end; flex-wrap:wrap; margin:6px 0 8px; }
details.adv { margin-bottom:10px; }
details.adv summary { cursor:pointer; color:var(--muted); font-size:12px; user-select:none; }
details.adv .controls { margin:8px 0 0; }
"""

# Shared JavaScript. Expects a page-level `state` object with a `runs` array (used by phenoOf).
COMMON_SCRIPT = r"""
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
// Small typeahead: `items(q)` returns ranked [{value, label, sub}] (already limited); `pick(value)` on choose.
function attachTypeahead(input, items, pick, limit = 8) {
  const box = document.createElement('div'); box.className = 'ta'; box.hidden = true; input.parentElement.appendChild(box);
  let list = [], active = -1;
  const render = () => { box.innerHTML = list.map((it, i) => `<div class="${i===active?'active':''}" data-i="${i}"><b>${esc(it.label)}</b>${it.sub ? `<span>${esc(it.sub)}</span>` : ''}</div>`).join(''); box.hidden = !list.length;
    box.querySelectorAll('div').forEach(d => { d.onmousedown = e => { e.preventDefault(); choose(+d.dataset.i); }; }); };
  const choose = i => { if (i < 0 || i >= list.length) return; input.value = list[i].value; box.hidden = true; list = []; pick(list, input.value); };
  const refresh = () => { list = items(input.value.trim()).slice(0, limit); active = list.length ? 0 : -1; render(); };
  input.addEventListener('input', refresh); input.addEventListener('focus', refresh);
  input.addEventListener('blur', () => setTimeout(() => { box.hidden = true; }, 120));
  input.addEventListener('keydown', e => {
    if (box.hidden && (e.key === 'ArrowDown')) { refresh(); return; }
    if (e.key === 'ArrowDown') { active = Math.min(active + 1, list.length - 1); render(); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { active = Math.max(active - 1, 0); render(); e.preventDefault(); }
    else if (e.key === 'Enter') { if (!box.hidden && active >= 0) { choose(active); e.preventDefault(); e.stopPropagation(); } }
    else if (e.key === 'Escape') { box.hidden = true; }
  });
}
const traitScore = (t, q) => { const p = phenoOf(t); return Math.max(fuzzyScore(q, t), p ? fuzzyScore(q, p.name || '') * 0.98 : 0, p && (p.portal_id || '').toLowerCase() === q ? 1000 : 0); };
const phenoOf = t => { const r = state.runs.find(x => x.trait === t && x.phenotype); return r ? r.phenotype : null; };
function setHash(obj) { const h = new URLSearchParams(); Object.entries(obj).forEach(([k,v]) => { if (v) h.set(k, v); }); const next = '#' + h.toString(); if (location.hash !== next) history.replaceState(null, '', next); }
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
const KV_NAMES = { label: 'library' };
function kv(obj, keys) { return '<div class="kv">' + keys.filter(k => obj[k] !== undefined && obj[k] !== null && obj[k] !== '').map(k => `<div><span>${esc(KV_NAMES[k] || k)}</span>${esc(typeof obj[k]==='number'?fmt(obj[k]):obj[k])}</div>`).join('') + '</div>'; }
// Paginated table: renders `rows` into #{id} 25 at a time with a prev/next pager.
function pagedTable(id, headerHtml, rows, rowHtml, onRow, pageSize = 25) {
  const host = $(id); let page = 0;
  const draw = () => {
    const n = Math.max(1, Math.ceil(rows.length / pageSize)); page = Math.min(page, n - 1);
    const slice = rows.slice(page * pageSize, (page + 1) * pageSize);
    host.innerHTML = `<div class="pager"><span>${rows.length ? page * pageSize + 1 : 0}–${Math.min(rows.length, (page + 1) * pageSize)} of ${rows.length.toLocaleString()}</span><button type="button" data-p="first" ${page===0?'disabled':''}>«</button><button type="button" data-p="prev" ${page===0?'disabled':''}>‹</button><span>page ${page + 1} / ${n}</span><button type="button" data-p="next" ${page>=n-1?'disabled':''}>›</button><button type="button" data-p="last" ${page>=n-1?'disabled':''}>»</button></div>` +
      `<div class="scroll"><table><thead>${headerHtml}</thead><tbody>${slice.map(rowHtml).join('')}</tbody></table></div>`;
    host.querySelectorAll('.pager button').forEach(b => b.onclick = () => { page = { first: 0, prev: page - 1, next: page + 1, last: n - 1 }[b.dataset.p]; draw(); });
    if (onRow) host.querySelectorAll('tr.row').forEach(onRow);
  };
  draw();
}
const byMetric = (rows, metric) => rows.slice().sort((a, b) => ((b[metric] ?? -Infinity) - (a[metric] ?? -Infinity)) || ((b.weight ?? 0) - (a.weight ?? 0)));
function hBar(el, rows, labelKey, metric, colorKey, height) {
  const top = rows.slice(0, 40).reverse();
  if (!top.length) { Plotly.purge(el); return; }
  Plotly.react(el, [{ type: 'bar', orientation: 'h', y: top.map(r => r[labelKey]), x: top.map(r => r[metric] ?? 0),
      marker: { color: top.map(r => r[metric] ?? 0), colorscale: 'Viridis', opacity: colorKey ? top.map(r => 0.45 + 0.55 * Math.min(1, Math.max(0, r[colorKey] ?? 0))) : 0.9 },
      customdata: top.map(r => colorKey ? (r[colorKey] ?? 0) : 0), hovertemplate: `%{y}: ${metric} %{x:.3f}${colorKey ? `, ${colorKey} %{customdata:.2f}` : ''}<extra></extra>` }],
    { height, margin: { l: 110, r: 10, t: 4, b: 30 }, xaxis: { title: metric }, yaxis: { automargin: true, tickfont: { size: 10 } } }, { responsive: true, displaylogo: false });
}

"""


def render_document(*, title: str, plotly_src: str, api_base: str, css: str, body: str, script: str) -> str:
    """Wrap a page body + script in the portal's HTML skeleton (API base injected for static hosting)."""
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{html.escape(title)}</title>\n<style>{CSS}{css}</style>\n"
        f"<script src=\"{html.escape(plotly_src, quote=True)}\"></script>\n</head>\n<body>\n"
        + body
        + f"<script>window.PIGEAN_PORTAL_API_BASE = {json.dumps(api_base)};</script>\n"
        + f"<script>{COMMON_SCRIPT}</script>\n<script>{script}</script>\n</body>\n</html>\n"
    )
