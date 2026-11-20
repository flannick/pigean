"""Offline views over the saved EAGGL tree, factor inventory and phi decision."""

STYLE = r"""
:root {--ink:#1b2c45;--muted:#536377;--line:#d8e1ea;--soft:#f4f7fa;--accent:#1262ad;--accent-soft:#e9f3fc;--shadow:0 2px 8px rgba(33,58,82,.045)}
body {background:#f4f7fa;font-family:Roboto,Arial,sans-serif}
.shell {max-width:none;margin:0;padding:8px 7% 24px}
.hero-card,.panel,.subpanel {background:white;border:1px solid var(--line);border-radius:6px;box-shadow:var(--shadow)}
.hero-card {padding:22px}.hero {gap:16px}.stat,.table-wrap,button,select {border-radius:5px}
h1 {font-size:clamp(26px,3vw,38px);line-height:1.18;letter-spacing:-.035em}
h2 {font-size:23px;font-weight:550}.pill {color:#215783}.panel {margin:8px 0;padding:10px 8px}
.hbar,.hnav,.hrow,.hheading {display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.hbar,.hheading {justify-content:space-between}.hbar {padding-top:16px;border-bottom:1px solid var(--line);gap:16px}
.hbar label {text-transform:none;letter-spacing:0;font-size:14px;color:var(--ink)}
.hbar select,.hheading select {width:auto;font:inherit;background:#fff;border:1px solid var(--line);padding:7px 10px}
.hnav {gap:4px}.hnav button {font:inherit;background:none;border:0;border-radius:0;border-bottom:3px solid transparent;padding:12px}
.hnav button[aria-pressed=true] {color:#125aa1;border-color:#1262ad;font-weight:600}
.hmeta {font-size:14px;color:var(--muted);line-height:1.5;margin:5px 0 14px}
.hgrid {display:grid;grid-template-columns:minmax(0,1.55fr) minmax(285px,.95fr);gap:16px;align-items:start}
.hgrid>.panel {min-width:0}.hinspect {position:sticky;top:12px;max-height:calc(100vh - 24px);overflow:auto}
.hscroll {overflow:auto;max-height:70vh}.htree,.htree ul {list-style:none;margin:0;padding:0}
.htree ul {margin-left:10px;padding-left:17px;border-left:1px solid #91a8bf}
.htree li {position:relative}.htree ul>li:before {content:"";position:absolute;left:-17px;top:17px;width:13px;border-top:1px solid #91a8bf}
.htree summary {cursor:pointer;padding:6px 0;line-height:1.5}.htree summary::marker {color:#1475bf}
.hlink {background:none;border:0;border-radius:3px;color:#125fa7;text-align:left;padding:4px 2px;font:inherit;text-decoration:underline;text-underline-offset:2px}
.hleaf {padding:3px 0 3px 12px;line-height:1.55}.hsmall {color:var(--muted);font-size:.86em}
.hflag,.hpick {display:inline-block;border-radius:4px;padding:2px 6px;font-weight:600;font-size:12px;background:#fff0db;color:#874600}
.hpick {background:#e8f2fc;color:#195690}.hselected,.htree [aria-current=true],.hgrid [data-hitem][aria-current=true] {background:#edf5fd}
.hinspect h3 {margin-top:16px}.hinspect td:first-child {overflow-wrap:anywhere}
.hdata th {cursor:default;text-transform:none;letter-spacing:0;font-size:13px;position:static}
.hdata th,.hdata td {padding:9px 10px}.hdata caption {text-align:left;color:var(--muted);padding-bottom:8px}
.hmeta strong {color:var(--ink)}.hsub {margin:18px 0 8px}
.hbar button:focus-visible,.htree summary:focus-visible,.hlink:focus-visible {outline:2px solid #1262ad;outline-offset:2px}
@media(max-width:780px){.shell{padding-inline:8px}.hgrid {grid-template-columns:minmax(0,1fr);gap:0}.hinspect {position:static;max-height:none}.hbar {align-items:start}.hnav {gap:0}.hnav button {padding:10px 6px;font-size:14px}.htree ul {padding-left:12px}.hscroll{max-height:58vh}}
"""

SCRIPT = r"""
const explorer = {trait:"", cut:0, item:null};
function currentHierarchy() {return DATA.hierarchies[explorer.trait];}
function percent(x) {return (100*x).toFixed(2)+"%";}
function topNames(rows, n=3) {return rows.slice(0,n).map(r=>esc(r[0])).join(" · ");}
function itemLink(key,text) {return `<button type="button" class="hlink" data-hitem="${esc(key)}">${text}</button>`;}
function leafTitle(h,id) {
  const name=h.leafLabels?.["f"+id];
  return "Factor "+id+(name?" · "+name:"");
}
function familyTitle(h,id) {
  const name=h.familyLabels?.["n"+id];
  return name ? name+" · Family "+id : "Family "+id;
}
function flagLabel(f) {return f.flagged ? `<span class="hflag">Flagged · ${esc(f.reason)}</span>` : "Pass";}
function cutLabel(c) {return (c.metric==="min_leaf_cosine" ? "Minimum leaf cosine" : c.metric.replaceAll("_"," "))+" "+c.floor.toFixed(2);}
function nodeHtml(h,id,open=false) {
  const n=h.nodes[id], key=n.factor ? "f"+n.factor : "n"+id;
  if(n.factor) return `<li class="hleaf">${itemLink(key,esc(leafTitle(h,n.factor)))} <span>${topNames(n.genes)}</span></li>`;
  return `<li><details ${open?"open":""}><summary data-hpreview="${key}">${h.familyLabels?.[key]?esc(familyTitle(h,id)):topNames(n.genes)} <span class="hsmall"> · ${n.count} factors</span></summary><ul>${n.children.map(i=>nodeHtml(h,i)).join("")}</ul></details></li>`;
}
function loadingList(title,rows) {
  return `<h3>${title}</h3><table class="hdata"><thead><tr><th>Name</th><th class="number">Loading</th></tr></thead><tbody>${rows.map(([name,value])=>`<tr><td>${esc(name)}</td><td class="number">${esc(Number(value.toPrecision(4)))}</td></tr>`).join("")}</tbody></table>`;
}
function inspector(h) {
  if(!explorer.item) return '<h2>Loading details</h2><p class="hmeta">Choose a family or factor to inspect its loadings.</p>';
  const leaf=explorer.item[0]==="f", id=Number(explorer.item.slice(1));
  const obj=leaf?h.leaves.find(f=>f.id===id):h.nodes[id];
  return `<h2>${leaf?esc(leafTitle(h,id)):esc(familyTitle(h,id))}</h2><div class="hmeta">${leaf?`QC · ${flagLabel(obj)}<br>Effective genes · ${obj.effective.toFixed(1)} · Restart support · ${esc(obj.support||"Unassessed")}`:`${obj.count} descendant factors · OR-combined gene loadings`}</div>${loadingList("Leading discovery genes",obj.genes)}${obj.annotations?.length?loadingList(leaf?"Leading gene sets":"Leading gene sets across descendants",obj.annotations):""}<p class="hmeta">Up to 15 positive loadings per list.</p>`;
}
function treeCard(h) {
  const cut=h.cuts[explorer.cut], flagged=h.leaves.filter(f=>f.flagged);
  const roots=cut?cut.nodes:[h.root];
  const first=h.nodes[roots[0]];
  if(!explorer.item) explorer.item=first.factor?"f"+first.factor:"n"+first.id;
  const options=h.cuts.map((c,i)=>`<option value="${i}" ${i===explorer.cut?"selected":""}>${esc(cutLabel(c))}</option>`).join("");
  return `<div class="hgrid"><section class="panel"><div class="hheading"><h2>Factor hierarchy</h2><select aria-label="Hierarchy view" id="hcut">${options}<option value="-1" ${!cut?"selected":""}>Full tree</option></select></div>
    <p class="hmeta">${cut?roots.length+" overview groups":"One root"} · ${h.leaves.length-flagged.length} passing factors · φ = ${h.phi} · ${esc(h.method)}. ${h.familyLabels?"Mechanism names summarize the leading genes and gene sets.":"Gene labels show the strongest loadings."}</p>
    <div class="hscroll"><ul class="htree">${roots.map(i=>nodeHtml(h,i,explorer.item==="n"+i)).join("")}</ul></div>
    ${flagged.length?`<h3 class="hsub">Outside the hierarchy</h3>${flagged.map(f=>`<div>${itemLink("f"+f.id,esc(leafTitle(h,f.id)))} · ${topNames(f.genes)} ${flagLabel(f)}</div>`).join("")}`:""}</section>
    <aside id="hinspector" class="panel hinspect">${inspector(h)}</aside></div>`;
}
function inventoryCard(h) {
  const cols=[
    {key:"id",label:"Factor",numeric:true,render:f=>"Factor "+f.id},
    ...(h.leafLabels?[{key:"mechanism",label:"Mechanism",render:f=>esc(f.mechanism||"Unlabeled")}]:[]),
    {key:"genes",label:"Leading genes",render:(f,k)=>`<button data-open-row-tab="${esc(k)}" data-tab="genes">${topNames(f.genes)}</button>`,filterValue:f=>f.genes.map(g=>g[0]).join(" ")},
    {key:"annotations",label:"Leading gene sets",render:(f,k)=>`<button data-open-row-tab="${esc(k)}" data-tab="gene_sets">${topNames(f.annotations,1)}</button>`,filterValue:f=>f.annotations.map(g=>g[0]).join(" ")},
    {key:"summary",label:"QC",render:f=>flagLabel(f)},
    {key:"effective",label:"Effective genes",numeric:true},
    {key:"support",label:"Restart support",numeric:true}];
  const rows=h.leaves.map(f=>({...f,mechanism:h.leafLabels?.["f"+f.id]||"",summary:f.flagged?f.reason:"Pass",support:f.support===""?null:Number(f.support)}));
  return `<h2>EAGGL factors</h2><p class="note">${rows.length} fitted leaves · ${rows.filter(f=>f.flagged).length} flagged. Expand a loading column to inspect its leading entries.</p>`+
    tableHtml("factors:"+h.trait+":hierarchy",rows,cols,{detailRenderer:(f,k)=>state.factorDetailTabs[k]==="gene_sets"?loadingList("Leading gene sets",f.annotations):loadingList("Leading genes",f.genes)});
}
function phiCard(h) {
  const d=h.decision,p=d.policy;
  const rows=h.candidates.map(c=>{
    const result=c.selected?"Selected":!c.qualified?"Solver failed":!c.usable?"Leaf QC failed":!c.eligible?"Ineligible":c.within?"Within tolerance":"Over error limit";
    return `<tr class="${c.selected?"hselected":""}"><td>${c.phi}</td><td>${c.factors}</td><td>${c.flags}</td><td>${percent(c.flaggedEnergy)}</td><td>${percent(c.global)}</td><td>${percent(c.balanced)}</td><td>${percent(c.error)}</td><td>${c.selected?'<span class="hpick">Selected</span>':result}</td></tr>`;
  }).join("");
  return `<section class="panel"><h2>Phi selection</h2><p><strong>Selected φ = ${h.phi}</strong> · ${d.factors} fitted factors</p>
    <p class="hmeta">A qualified fit can have up to ${percent(p.maximum_flagged_fraction)} flagged factors and ${percent(p.maximum_flagged_energy)} of squared loading mass in flagged factors. For each eligible fit, the larger of its global and gene-balanced errors sets its comparison error.</p>
    <p class="hmeta">The best eligible error was <strong>${percent(d.minimum_distortion)}</strong>. An allowance of ${Number((100*p.epsilon).toFixed(3))} percentage points set the limit at <strong>${percent(d.minimum_distortion+p.epsilon)}</strong>. The selector chose the fewest fitted factors within that limit. Ties favor lower error, then lower φ.</p>
    <details><summary>Factor QC thresholds</summary><p class="hmeta">Flag a factor for effective gene count below ${p.minimum_effective_genes}, contrast below ${p.minimum_contrast}, or a duplicate profile at cosine ${p.duplicate_cosine}. Restart support requires ${p.minimum_support} matching starts at cosine ${p.support_cosine}. The factor inventory keeps every flagged leaf.</p></details>
    <div class="table-wrap"><table class="hdata"><thead><tr>${["φ","Factors","Flagged","Flagged mass","Global error","Gene-balanced error","Comparison error","Decision"].map(s=>"<th>"+s+"</th>").join("")}</tr></thead><tbody>${rows}</tbody></table></div></section>`;
}
const plotFilters=new Map(), decoded=new WeakMap();
function plotState(h){
  if(!plotFilters.has(h.trait)){
    const cut=h.cuts[0],metric=cut?.metric==="maximum_gene_equivalents"?"gene_equivalents":cut?.metric;
    plotFilters.set(h.trait,{scope:cut?"frontier":"all",metric:["min_leaf_cosine","min_shared_mass","gene_equivalents"].includes(metric)?metric:"merge",
      floor:metric==="gene_equivalents" ? 0.5 : (cut?.floor ?? 0.5),size:metric==="gene_equivalents"?cut.floor:50,flags:false});
  }
  return plotFilters.get(h.trait);
}
function plotted(h){
  const s=plotState(h);
  if(s.scope==="all" && s.flags) return h.views.keys;
  if(s.scope==="leaves")return h.leaves.filter(f=>s.flags||!f.flagged).map(f=>"f"+f.id);
  const keep=new Set();
  function visit(id){
    const n=h.nodes[id], key=n.factor?"f"+n.factor:"n"+id;
    const pass=s.metric==="gene_equivalents"?n.levels[s.metric]<=s.size:n.levels[s.metric==="merge"?h.mergeMetric:s.metric]>=s.floor;
    if(s.scope==="all" || n.factor || pass || s.scope==="above") keep.add(key);
    if(!n.factor && (s.scope==="all" || !pass)) n.children.forEach(visit);
  }
  visit(h.root);
  if(s.flags) h.leaves.filter(f=>f.flagged).forEach(f=>keep.add("f"+f.id));
  return h.views.keys.filter(k=>keep.has(k));
}
function nodeControls(h){
  if(explorer.trait!==h.trait){explorer.trait=h.trait;explorer.cut=0;explorer.item=null;}
  const s=plotState(h),options=(items,v)=>items.map(([key,name])=>`<option value="${key}" ${key===v?"selected":""}>${name}</option>`).join("");
  return `<div class="node-controls"><label>Plot nodes <select id="nscope">${options([["frontier","At the cut"],["leaves","Leaves only"],["above","At or above the cut"],["all","All nodes"]],s.scope)}</select></label>
  <label>Cut rule <select id="nmetric" ${s.scope==="leaves"?"disabled":""}>${options([["min_leaf_cosine","Minimum leaf cosine"],["merge","Minimum merge overlap"],["min_shared_mass","Minimum shared mass"],["gene_equivalents","Maximum gene equivalents"]],s.metric)}</select></label>
  <label>${s.metric==="gene_equivalents"?"Size ceiling":"Overlap floor"} <input id="nlevel" ${s.scope==="leaves"?"disabled":""} type="${s.metric==="gene_equivalents"?"number":"range"}" min="${s.metric==="gene_equivalents"?"0.01":"0"}" ${s.metric==="gene_equivalents"?'':'max="1"'} step="0.01" value="${s.metric==="gene_equivalents"?s.size:s.floor}"> <output id="nvalue">${s.metric==="gene_equivalents"?s.size:s.floor}</output></label>
  <label><input id="nflags" type="checkbox" ${s.flags?"checked":""}> Include flagged factors</label><span id="ncount" class="note">${plotted(h).length} of ${h.views.keys.length} nodes</span></div>`;
}
function updateGraph(h){
  document.querySelector(".factor-graph-frame")?.contentWindow?.postMessage({type:"eaggl-hierarchy-cut",factors:plotted(h)},"*");
}
function wireGraph(h){
  const frame=document.querySelector(".factor-graph-frame");
  if(frame){frame.onload=()=>updateGraph(h);updateGraph(h);}
}
function researchTabs(){
  return ["hierarchy","phi"].map(t=>`<button data-eaggl-section="${t}" class="${state.eagglSection===t?"active":""}">${t==="hierarchy"?"Hierarchy":"Phi selection"}</button>`).join("");
}
function unpackBytes(encoded){
  if(Uint8Array.fromBase64)return Uint8Array.fromBase64(encoded);
  const raw=atob(encoded),bytes=new Uint8Array(raw.length);
  for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);
  return bytes;
}
function numbers(block,field){
  if(!decoded.has(block))decoded.set(block,{});
  const cache=decoded.get(block);
  if(!cache[field]) cache[field]=new Float32Array(unpackBytes(block[field]).buffer);
  return cache[field];
}
const matrixPreferences=new Map();
function matrixPrefs(h,kind){
  const k=h.trait+"/"+kind;
  if(!matrixPreferences.has(k))matrixPreferences.set(k,{page:0,query:"",sort:-1,metric:kind==="phenotypes"?"z":"loading"});
  return matrixPreferences.get(k);
}
function logP(z){
  const x=Math.abs(z)/Math.SQRT2,t=1/(1+.5*x);
  const poly=[.17087277,-.82215223,1.48851587,-1.13520398,.27886807,-.18628806,.09678418,.37409196,1.00002368];
  const correction=poly.reduce((p,c)=>p*t+c,0);
  return Math.max(0,(x*x+1.26551223-t*correction-Math.log(t))/Math.LN10);
}
function nodeMatrix(h,kind){
  const v=h.views,phe=kind==="phenotypes",block=phe?v.phewas:v[kind], pref=matrixPrefs(h,kind);
  if(!block)return '<p class="empty">No matched phenotype reference was supplied for this hierarchy.</p>';
  const tableId="node:"+h.trait+":"+kind;
  const keys=new Set(plotted(h));
  let cols=v.keys.map((k,i)=>keys.has(k)?i:-1).filter(i=>i>=0);
  const base=numbers(block,phe?"beta":"data"),se=phe?numbers(block,"se"):null,N=v.keys.length;
  const metric=pref.metric,query=compileNameRegex(pref.query),rows=[];
  if(!cols.includes(pref.sort))pref.sort=-1;
  function value(j,i,norm){
    const off=j*N+i,x=base[off];
    if(phe){
      if(metric==="beta")return x;
      const z=se[off]>0?x/se[off]:NaN;
      return metric==="logp"?logP(z):z;
    }
    if(metric==="cosine")return norm>0?x/Math.sqrt(norm):0;
    if(metric==="distance")return Math.sqrt(Math.max(0,1+norm-2*x));
    return x;
  }
  for(let j=0;j<block.rows.length;j++){
    if(query.error || (query.regex && !query.regex.test(block.rows[j])))continue;
    let norm=0;
    if(!phe && metric!=="loading")for(let col=0;col<N;col++)norm+=base[j*N+col]**2;
    let score=-Infinity;
    for(const c of pref.sort<0?cols:[pref.sort]){
      const x=value(j,c,norm);
      if(Number.isFinite(x))score=Math.max(score,phe?Math.abs(x):metric==="distance"?-x:x);
    }
    rows.push({j,norm,score});
  }
  if(!phe && state.factorWeightOrder[tableId]){
    cols=columnsByWeight(cols,rows,(row,col)=>base[row.j*N+col]);
  }
  rows.sort((a,b)=>b.score-a.score||a.j-b.j);
  const pages=Math.max(1,Math.ceil(rows.length/25));pref.page=Math.max(0,Math.min(pages-1,pref.page));
  const labels=phe?[["z","Signed Z"],["beta","Regression coefficient"],["logp","−log₁₀ P (two-sided)"]]:[["loading","Loading"],["cosine","Cosine"],["distance","Euclidean distance"]];
  const cell=(j,c,norm)=>{
    const x=value(j,c,norm);
    if(!Number.isFinite(x))return '<td title="Not estimable">—</td>';
    const scale=phe?(metric==="z"?5:1):1, a=Math.min(0.82,Math.abs(x)/scale*.8);
    const tip=phe?`${block.rows[j]} · ${v.labels[c]} · β ${base[j*N+c].toPrecision(4)} · HC3 SE ${se[j*N+c].toPrecision(4)} · ${block.hits[j]} hit genes`:`${block.rows[j]} · ${v.labels[c]} · ${x.toPrecision(5)}`;
    return `<td title="${esc(tip)}" style="background:rgba(${x<0?'210,126,51':'65,143,203'},${a})">${x.toFixed(2)}</td>`;
  };
  const title=phe?"Factor PheWAS":kind==="genes"?"Gene loadings":"Gene-set loadings";
  const note=phe?"Each column uses a separate marginal regression on discovery genes, adjusted for direct anchor evidence. The response is Combined > 1 in the phenotype reference. P values are nominal, two-sided HC3 tests. A dash marks an unestimable result.":"Each column shows the loadings for one displayed factor or family.";
  return `<h2>${title}</h2><p class="note">${note}</p><div class="node-controls matrix-toolbar"><label>Find ${phe?"phenotype":kind==="genes"?"gene":"gene set"} <input id="nquery" type="search" placeholder="Name or regex" aria-invalid="${!!query.error}" value="${esc(pref.query)}"></label><label>Metric <select id="nmeasure">${labels.map(([key,name])=>`<option value="${key}" ${key===metric?"selected":""}>${name}</option>`).join("")}</select></label>${phe?"":weightOrderControl(tableId,kind)}<div class="matrix-pages"><button data-npage="-1" ${pref.page===0?"disabled":""}>Previous</button><span>${rows.length ? pref.page*25+1 : 0}–${Math.min((pref.page+1)*25,rows.length)} of ${rows.length.toLocaleString()}</span><button data-npage="1" ${pref.page+1===pages?"disabled":""}>Next</button></div></div>
  ${query.error?'<p class="matrix-error" role="status">Invalid regular expression. Check the brackets or parentheses.</p>':""}
  <div class="table-wrap"><table class="node-matrix" data-kind="${kind}"><thead><tr><th>${phe?"Phenotype":kind==="genes"?"Gene":"Gene set"}</th>${cols.map(c=>`<th><button title="Sort by this node" data-nsort="${c}">${esc(v.labels[c])}${pref.sort===c?" ↓":""}</button></th>`).join("")}</tr></thead><tbody>${rows.slice(pref.page*25,pref.page*25+25).map(({j,norm})=>`<tr><th>${esc(block.rows[j])}</th>${cols.map(c=>cell(j,c,norm)).join("")}</tr>`).join("")}</tbody></table></div>`;
}
function researchTable(h){
  if(state.eagglSection==="hierarchy")return treeCard(h);
  if(state.eagglSection==="phi")return phiCard(h);
  if(["genes","gene_sets","phenotypes"].includes(state.eagglSection))return nodeMatrix(h,state.eagglSection);
  return inventoryCard(h);
}
function activeResearch(){
  const run=selectedRun(), e=run && selectedEaggl(selectedEagglRunId(run.run_id));
  return e?.research?DATA.hierarchies[e.run_id]:null;
}
function refreshNodes(h){
  refreshEagglTable();
  byId("ncount").textContent=plotted(h).length+" of "+h.views.keys.length+" nodes";
  updateGraph(h);
}
document.addEventListener("change",e=>{
  if(!["nscope","nlevel","nmetric","nflags","hcut","nmeasure"].includes(e.target.id))return;
  const h=activeResearch();if(!h)return;
  const s=plotState(h),id=e.target.id;
  if(id==="nscope")s.scope=e.target.value;
  if(id==="nmetric"){s.metric=e.target.value;if(s.scope==="all")s.scope="frontier";}
  if(id==="nlevel"){const x=Number(e.target.value);if(!Number.isFinite(x)||!e.target.checkValidity()||!e.target.value)return;if(s.metric==="gene_equivalents")s.size=x;else s.floor=x;if(s.scope==="all")s.scope="frontier";}
  if(id==="nflags")s.flags=e.target.checked;
  if(id==="hcut"){explorer.cut=Number(e.target.value);explorer.item=null;}
  if(id==="nmeasure")matrixPrefs(h,state.eagglSection).metric=e.target.value;
  if(["nscope","nmetric","nlevel"].includes(id)){
    const controls=byId("nscope").closest(".node-controls");controls.outerHTML=nodeControls(h);byId(id)?.focus();
  }
  refreshNodes(h);
});
document.addEventListener("input",e=>{
  if(e.target.id==="nlevel"){byId("nvalue").textContent=e.target.value;return;}
  if(e.target.id!=="nquery")return;
  const h=activeResearch(),p=matrixPrefs(h,state.eagglSection);p.query=e.target.value;p.page=0;
  const pos=e.target.selectionStart;byId("eaggl-table-region").innerHTML=researchTable(h);
  byId("nquery").focus();byId("nquery").setSelectionRange(pos,pos);
});
document.addEventListener("click",e=>{
  const btn=e.target.closest("[data-hitem],[data-hpreview],[data-npage],[data-nsort]");if(!btn)return;
  const h=activeResearch();if(!h)return;
  if(btn.hasAttribute("data-npage")||btn.hasAttribute("data-nsort")){
    const p=matrixPrefs(h,state.eagglSection);if(btn.hasAttribute("data-npage"))p.page+=Number(btn.dataset.npage);
    else {p.sort=p.sort===Number(btn.dataset.nsort)?-1:Number(btn.dataset.nsort);p.page=0;}
    byId("eaggl-table-region").innerHTML=researchTable(h);return;
  }
  explorer.item=btn.dataset.hitem||btn.dataset.hpreview;
  const panel=byId("hinspector");panel.innerHTML=inspector(h);
  if(btn.dataset.hitem && matchMedia("(max-width:780px)").matches)panel.scrollIntoView({block:"start"});
});
"""
STYLE += """
.node-controls {display:flex;align-items:end;flex-wrap:wrap;gap:10px 18px;margin:12px 0}
.node-controls label {display:grid;gap:4px;text-transform:none;letter-spacing:0}
.node-controls input[type=checkbox]{width:auto}.node-controls label:has(input[type=checkbox]){display:inline-flex;align-items:center}
.node-matrix th:first-child {position:sticky;left:0;background:white;z-index:2;max-width:300px;overflow-wrap:anywhere}
.node-matrix td {text-align:right;font-variant-numeric:tabular-nums}
.matrix-toolbar {align-items:end;gap:8px 14px}.matrix-toolbar input,.matrix-toolbar select {box-sizing:border-box;width:100%;height:32px;border-radius:4px;padding:4px 8px;font:inherit}
.matrix-toolbar label {font-size:13px}.matrix-toolbar label:first-child {width:min(100%,260px)}
.heatmap-controls label.weight-order,.matrix-toolbar label.weight-order {display:inline-flex;align-items:center;gap:6px;font-size:13px;text-transform:none;letter-spacing:0}.heatmap-controls .weight-order input,.matrix-toolbar .weight-order input {height:auto;width:auto;margin:0}
.matrix-pages {display:flex;align-items:center;gap:10px;margin-left:auto;min-height:32px;font-size:13px;color:var(--muted);font-variant-numeric:tabular-nums}
.matrix-pages button {padding:5px 9px;font-weight:500}.matrix-pages button:disabled {color:#8996a2;background:#f7f9fa;cursor:default}
.node-matrix[data-kind="genes"] th:first-child {min-width:8ch;white-space:nowrap;overflow-wrap:normal}
.matrix-error {color:#9c321c;font-size:13px}.node-matrix tbody th {font-size:12px;text-transform:none;letter-spacing:0}
.factor-graph-body {padding:4px}.factor-graph-accordion {border-radius:5px}.factor-graph-accordion summary {background:white;padding:8px}.factor-graph-frame {border:0;border-radius:0}
.node-matrix th,.node-matrix td {padding:5px 7px}.node-matrix thead th {white-space:nowrap}.node-matrix th button{border:0;background:none;padding:3px;font-size:12px}
"""
