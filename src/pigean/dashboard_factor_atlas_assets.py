"""Compact offline factor map and overlays over the dashboard's saved matrices."""
STYLE = """
.factor-view-switch {display:flex;gap:5px;flex-wrap:wrap;margin:0 0 12px}
.factor-view-switch button[aria-pressed=true] {color:white;background:#145e9f;border-color:#145e9f}
.atlas-controls {display:flex;align-items:end;flex-wrap:wrap;gap:12px}
.atlas-controls label {display:grid;gap:4px;color:#475569;font-size:14px}
.atlas-controls label {text-transform:none;letter-spacing:normal}.atlas-controls input {min-width:min(300px,74vw)}
.atlas-grid {display:grid;grid-template-columns:minmax(0,3fr) minmax(200px,1fr);gap:18px;margin-top:12px}
.atlas-grid svg {width:100%;display:block;background:#fafcfe;border:1px solid #dbe4eb;border-radius:5px}
.atlas-grid circle {cursor:pointer}.atlas-grid circle:focus {outline:none;stroke:#131d27;stroke-width:3}
.atlas-side {min-width:0;overflow-wrap:anywhere}.atlas-side h3 {margin:0 0 8px}
.atlas-side button {padding:4px 7px;font:inherit;text-align:left}
.atlas-list {display:grid;gap:4px}.atlas-list button {display:flex;justify-content:space-between;gap:10px;width:100%;border:0;background:#f4f7fa}
.atlas-legend {display:flex;align-items:center;gap:6px 12px;flex-wrap:wrap;font-size:.9em;color:#49596a;margin-top:8px}
.atlas-key {display:inline-block;width:.75em;height:.75em;border-radius:50%;margin-right:4px}
.atlas-ramp {display:inline-block;width:90px;height:9px;border-radius:4px}
.atlas-detail {margin:12px 0;line-height:1.5}
@media(max-width:750px){.atlas-grid{grid-template-columns:1fr}.atlas-controls{align-items:start}}
"""

SCRIPT = r"""
let atlasMode=false;
const atlasState={kind:"genes",query:"",pick:null,zoom:1,cx:390,cy:220};
const atlasIndexes=new Map();
const atlasPalette=["#0072b2","#d55e00","#009e73","#91558a","#b48011","#4a6696"];
function atlasIndex(kind){
  if(!atlasIndexes.has(kind)){
    const byTrait=new Map(), names=new Set();
    for(const trait of new Set(DATA.factor_atlas.points.map(p=>p.trait))){
      const block=DATA.hierarchies[trait].views[kind];
      if(!block)continue;
      const rows=new Map();
      block.rows.forEach((name,i)=>{names.add(name);rows.set(name,i);});
      byTrait.set(trait,rows);
    }
    atlasIndexes.set(kind,{byTrait,names:[...names].sort()});
  }
  return atlasIndexes.get(kind);
}
function atlasValue(point,kind,name){
  const row=atlasIndex(kind).byTrait.get(point.trait)?.get(name);
  if(row==null)return null;
  const view=DATA.hierarchies[point.trait].views, block=view[kind];
  const v=numbers(block,kind==="phewas"?"beta":"data")[row*view.keys.length+point.column];
  return Number.isFinite(v)?v:null;
}
function atlasNumber(value){return value==null?"Unavailable":Number(value.toPrecision(3)).toString();}
function atlasShell(){
  const kinds=[["genes","Gene"],["gene_sets","Gene set"],["phewas","Phenotype"]];
  return `<div class="atlas-controls"><label>Color factors by<select id="atlas-kind">${kinds.filter(([key])=>atlasIndex(key).names.length).map(([key,label])=>`<option value="${key}">${label}</option>`).join("")}</select></label>
    <label>Search <input id="atlas-search" type="search" list="atlas-names" autocomplete="off" placeholder="Search by name"><datalist id="atlas-names"></datalist></label>
    <button type="button" id="atlas-clear">Clear search</button></div>
    <div class="atlas-grid"><div><svg id="atlas-map" role="group" aria-label="Cross-trait factor atlas" viewBox="0 0 780 440"></svg>
    <div id="atlas-legend" class="atlas-legend"></div>
    <div class="factor-view-switch" aria-label="Map zoom"><button type="button" id="atlas-in" aria-label="Zoom in">+</button><button type="button" id="atlas-out" aria-label="Zoom out">−</button><button type="button" id="atlas-fit">Fit all</button></div></div><aside class="atlas-side"><h3>Factor atlas</h3>
    <div class="note" id="atlas-caption"></div><div id="atlas-detail" class="atlas-detail"></div><div class="atlas-list" id="atlas-rank"></div></aside></div>`;
}
function atlasFactorTitle(p){
  const name=DATA.hierarchies[p.trait].leafLabels?.[p.key];
  return p.factor+(name?" · "+name:"");
}
function atlasDetail(p,name){
  if(!p){byId("atlas-detail").textContent="Select a point to inspect its factor.";return;}
  const leaf=DATA.hierarchies[p.trait].leaves.find(l=>"f"+l.id===p.key);
  const value=name?atlasNumber(atlasValue(p,atlasState.kind,name)):null;
  byId("atlas-detail").innerHTML=`<strong>${esc(p.trait)} · ${esc(atlasFactorTitle(p))}</strong><br>Family ${p.family}${p.redundant?" · Repeated profile":""}<br>
    ${topNames(leaf.genes,4)}${name?`<br>${esc(name)} · ${esc(value)}`:""}`;
}
function drawAtlas(){
  const atlas=DATA.factor_atlas, pts=atlas.points, kind=atlasState.kind, q=atlasState.query.trim().toLowerCase();
  const names=atlasIndex(kind).names;
  const norm=s=>s.toLowerCase().replace(/[_-]/g," ");
  const matches=q?names.filter(n=>norm(n).includes(norm(q))):[];
  const name=names.includes(atlasState.query.trim())?atlasState.query.trim():matches.find(n=>n.toLowerCase()===q);
  const suggestions=matches.slice().sort((a,b)=>Number(!norm(a).startsWith(norm(q)))-Number(!norm(b).startsWith(norm(q)))).slice(0,24);
  byId("atlas-names").innerHTML=suggestions.map(n=>`<option value="${esc(n)}"></option>`).join("");
  const traits=[...new Set(pts.map(p=>p.trait))].sort();
  const anchor=selectedEagglRunId(state.runId);
  const visible=pts.map((p,i)=>i).filter(i=>anchor===null || pts[i].trait===anchor);
  if(atlasState.pick!==null && !visible.includes(atlasState.pick))atlasState.pick=null;
  const palette=t=>atlasPalette[traits.indexOf(t)%atlasPalette.length];
  const vals=pts.map(p=>name?atlasValue(p,kind,name):null);
  const max=vals.reduce((m,v)=>Math.max(m,Math.abs(v??0)),0);
  const ranked=visible.map(i=>({p:pts[i],v:vals[i]})).filter(({v})=>v!=null&&v!==0).sort((a,b)=>Math.abs(b.v)-Math.abs(a.v));
  const signed=kind==="phewas", units=signed?"Marginal Factor PheWAS β":"Raw loading";
  const color=v=>{
    const base=[236,241,244],end=v<0?[181,78,31]:[14,94,162];
    const t=max?Math.min(1,Math.abs(v)/max):0;
    return "rgb("+base.map((b,i)=>Math.round(b+(end[i]-b)*t)).join(",")+")";
  };
  const xs=pts.map(p=>p.x),ys=pts.map(p=>p.y);
  const bounds=a=>a.reduce(([lo,hi],v)=>[Math.min(lo,v),Math.max(hi,v)],[Infinity,-Infinity]);
  const [x0,x1]=bounds(xs),[y0,y1]=bounds(ys);
  const s=Math.min(742/Math.max(1e-8,x1-x0),402/Math.max(1e-8,y1-y0));
  const position=p=>[390+(p.x-(x0+x1)/2)*s,220-(p.y-(y0+y1)/2)*s];
  const svg=byId("atlas-map");
  function zoomTo(z,center){
    atlasState.zoom=Math.max(1,Math.min(8,z));
    const width=780/atlasState.zoom,height=440/atlasState.zoom;
    atlasState.cx=Math.max(width/2,Math.min(780-width/2,center[0]));
    atlasState.cy=Math.max(height/2,Math.min(440-height/2,center[1]));
    svg.setAttribute("viewBox",[atlasState.cx-width/2,atlasState.cy-height/2,width,height].join(" "));
    for(const c of svg.querySelectorAll("circle")){
      c.setAttribute("r",(atlasState.pick===+c.dataset.point?7:5)/Math.sqrt(atlasState.zoom));
    }
    byId("atlas-in").disabled=atlasState.zoom>=8;byId("atlas-out").disabled=atlasState.zoom===1;
  }
  const zoomCenter=()=>pts[atlasState.pick]?position(pts[atlasState.pick]):[atlasState.cx,atlasState.cy];
  byId("atlas-in").onclick=()=>zoomTo(atlasState.zoom*2,zoomCenter());
  byId("atlas-out").onclick=()=>zoomTo(atlasState.zoom/2,zoomCenter());
  byId("atlas-fit").onclick=()=>zoomTo(1,[390,220]);
  zoomTo(atlasState.zoom,[atlasState.cx,atlasState.cy]);
  const order=visible.slice().sort((a,b)=>(Math.abs(vals[a]??0)-Math.abs(vals[b]??0)));
  byId("atlas-map").innerHTML=order.map(i=>{
    const p=pts[i],v=vals[i],missing=name&&v==null, selected=atlasState.pick===i;
    const [x,y]=position(p);
    const title=p.trait+" · "+atlasFactorTitle(p)+(name?" · "+name+" · "+atlasNumber(v):"");
    return `<circle data-point="${i}" tabindex="0" role="button" aria-label="${esc(title)}" cx="${x}" cy="${y}" r="${(selected?7:5)/Math.sqrt(atlasState.zoom)}" fill="${missing?"white":name?color(v):palette(p.trait)}" stroke="${selected?"#1c2733":missing?"#a1acb7":name?"#677786":"white"}" stroke-width="${selected?2.5:1}"><title>${esc(title)}</title></circle>`;
  }).join("");
  const key=byId("atlas-legend"), caption=byId("atlas-caption");
  caption.textContent=visible.length+" consensus leaves · "+(anchor===null?traits.length+" anchor traits":anchor)+" · "+atlas.method+".";
  if(name){
    const missing=visible.filter(i=>vals[i]==null).length;
    if(!max)key.textContent=missing===visible.length?"No available measurements":units+" · All measured values are zero · "+missing+" unavailable (hollow)";
    else key.innerHTML=`<strong>${esc(units)}</strong> ${signed?atlasNumber(-max):"0"} <i class="atlas-ramp" style="background:linear-gradient(to right,${signed?"#b54e1f,#ecf1f4":"#ecf1f4"},#0e5ea2)"></i> ${atlasNumber(max)} · ${visible.length-missing} measured · ${missing} unavailable (hollow)`;
  }else{
    key.innerHTML=traits.filter(t=>anchor===null || t===anchor).map(t=>`<span><i class="atlas-key" style="background:${palette(t)}"></i>${esc(t)}</span>`).join("");
    if(q)caption.textContent+=matches.length?" Choose a matching name.":" No matching name.";
  }
  byId("atlas-rank").innerHTML=name?(ranked.length?`<strong>${signed?"Largest absolute effects":"Highest loadings"}</strong>`+ranked.slice(0,6).map(({p,v})=>`<button type="button" data-atlas-pick="${p.index}"><span>${esc(p.trait)} · ${esc(atlasFactorTitle(p))}</span><b>${atlasNumber(v)}</b></button>`).join(""):"No nonzero values for this selection."):"";
  const select=i=>{atlasState.pick=i;if(atlasState.zoom>1)zoomTo(atlasState.zoom,position(pts[i]));drawAtlas();};
  svg.onclick=e=>{if(e.target.hasAttribute("data-point"))select(+e.target.dataset.point);};
  svg.onkeydown=e=>{if((e.key==="Enter"||e.key===" ")&&e.target.hasAttribute("data-point")){e.preventDefault();const i=+e.target.dataset.point;select(i);svg.querySelector('[data-point="'+i+'"]').focus();}};
  svg.onpointerover=svg.onfocusin=e=>{if(e.target.hasAttribute("data-point"))atlasDetail(pts[+e.target.dataset.point],name);};
  svg.onpointerleave=()=>atlasDetail(pts[atlasState.pick],name);
  byId("atlas-rank").onclick=e=>{const b=e.target.closest("button");if(b)select(+b.dataset.atlasPick);};
  atlasDetail(pts[atlasState.pick],name);
}
function wireAtlas(wrapper){
  if(!wrapper?.querySelector("[data-factor-view]"))return;
  const toggles=wrapper.querySelectorAll("[data-factor-view]");
  function activate(on){
    atlasMode=on;
    wrapper.querySelector(".atlas-area").hidden=!on;
    wrapper.querySelector(".mds-area").hidden=on;
    toggles.forEach(b=>b.setAttribute("aria-pressed",String((b.dataset.factorView==="atlas")===on)));
    if(!on)return;
    mountAtlasControls(wrapper);
  }
  toggles.forEach(b=>b.onclick=()=>activate(b.dataset.factorView==="atlas"));
  activate(atlasMode);
}
function mountAtlasControls(wrapper){
    if(!byId("atlas-kind"))wrapper.querySelector(".atlas-area").innerHTML=atlasShell();
    const kind=byId("atlas-kind"),search=byId("atlas-search");
    kind.value=atlasState.kind;search.value=atlasState.query;
    kind.onchange=()=>{atlasState.kind=kind.value;search.value=atlasState.query="";atlasState.pick=null;drawAtlas();search.focus();};
    search.oninput=()=>{atlasState.query=search.value;atlasState.pick=null;drawAtlas();};
    byId("atlas-clear").onclick=()=>{search.value="";search.oninput();search.focus();};
    drawAtlas();
}
"""
