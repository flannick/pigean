"""Evidence views for the PIGEAN tables in the standalone dashboard."""

STYLE = """
.evidence-plot,.locus-plot {width:100%;display:block;background:white}
.evidence-plot {max-height:310px}.evidence-plot text,.locus-plot text {font:12px Roboto,Arial,sans-serif;fill:#44546a}
.evidence-plot .axis-title {font-size:14px;fill:#24384c}.evidence-plot circle:hover {stroke:#0b416c;stroke-width:2;fill:#2379be}
.locus-wrap {overflow:auto}.locus-plot {min-width:550px}.locus-plot text.focus {font-weight:bold;fill:#165d9b}
.gene-disclosure {white-space:nowrap;font-weight:500;color:#14598e}.locus-signals {overflow:auto;max-height:230px}
"""

SCRIPT = r"""
function supportScatter(rows,cols,tableId,xKey,yKey,xName,yName,nameKey){
  const shown=rows.filter(r=>columnFiltersMatch(r,cols,tableId) && Number.isFinite(r[xKey]) && Number.isFinite(r[yKey]));
  if(!shown.length)return '<p class="note">No matching rows have both plotted values.</p>';
  const width=Math.max(310,Math.min(1180,window.innerWidth-60)),height=252,left=63,right=width-24,top=17,bottom=201;
  function domain(key){
    let lo=0,hi=0;
    for(const r of shown){lo=Math.min(lo,r[key]);hi=Math.max(hi,r[key]);}
    if(lo===hi)hi=lo+1;
    const pad=(hi-lo)*.035;
    return [lo-pad,hi+pad];
  }
  const [xmin,xmax]=domain(xKey),[ymin,ymax]=domain(yKey);
  const x=v=>left+(v-xmin)/(xmax-xmin)*(right-left), y=v=>bottom-(v-ymin)/(ymax-ymin)*(bottom-top);
  function ticks(lo,hi){
    const raw=(hi-lo)/5,unit=10**Math.floor(Math.log10(raw)),step=[1,2,5,10].find(i=>i*unit>=raw)*unit;
    const out=[];for(let i=Math.ceil(lo/step);i<=Math.floor(hi/step);i++)out.push(Number((i*step).toPrecision(5)));return out;
  }
  const xt=ticks(xmin,xmax).map(v=>`<line x1="${x(v)}" x2="${x(v)}" y1="${bottom}" y2="${bottom+4}" stroke="#69798b"/><text x="${x(v)}" y="${bottom+19}" text-anchor="middle">${v}</text>`).join("");
  const yt=ticks(ymin,ymax).map(v=>`<line x1="${left}" x2="${right}" y1="${y(v)}" y2="${y(v)}" stroke="#e3e8ee"/><text x="${left-8}" y="${y(v)+4}" text-anchor="end">${v}</text>`).join("");
  const points=shown.map(r=>`<circle cx="${x(r[xKey]).toFixed(2)}" cy="${y(r[yKey]).toFixed(2)}" r="3" fill="#358bd0" fill-opacity=".53"><title>${esc(r[nameKey])} · ${esc(xName)} ${esc(r[xKey])} · ${esc(yName)} ${esc(r[yKey])}</title></circle>`).join("");
  return `<svg class="evidence-plot" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(yName)} versus ${esc(xName)}, ${shown.length} matching points">${yt}${xt}<path d="M${left} ${top}V${bottom}H${right}" stroke="#687c8c" fill="none"/>${points}<text class="axis-title" x="${(left+right)/2}" y="245" text-anchor="middle">${esc(xName)}</text><text class="axis-title" transform="translate(16 ${(top+bottom)/2}) rotate(-90)" text-anchor="middle">${esc(yName)}</text></svg>`;
}
function locusSketch(row,hit,loci){
  const [chrom,pos]=hit, same=row.chrom===chrom && Number.isFinite(row.start) && Number.isFinite(row.end);
  const left=Math.max(1,(same?Math.min(pos,row.start):pos)-50000),right=(same?Math.max(pos,row.end):pos)+50000;
  const width=920,x=v=>20+880*(v-left)/(right-left);
  const intervals=(loci.intervals[chrom]||[]).filter(g=>g[2]>=left && g[1]<=right);
  const ends=[],bars=[];
  for(const [gene,start,end] of intervals){
    const x1=x(Math.max(start,left)),x2=x(Math.min(end,right)),active=gene===row.gene;
    const labelX=Math.max(20,Math.min(x1,900-gene.length*7.5));
    let lane=ends.findIndex(v=>v+8<Math.min(x1,labelX));
    if(lane<0)lane=ends.length;
    ends[lane]=Math.max(x2,labelX+gene.length*7.5);
    const y=55+lane*27;
    bars.push(`<g><title>${esc(gene)} · chr${esc(chrom)}:${start.toLocaleString()}–${end.toLocaleString()}</title><rect x="${x1}" y="${y}" width="${Math.max(2,x2-x1)}" height="${active?6:4}" fill="${active?"#1767aa":"#93acc3"}"/><text x="${labelX}" y="${y+18}" class="${active?"focus":""}">${esc(gene)}</text></g>`);
  }
  const height=68+Math.max(1,ends.length)*27;
  return `<div class="locus-wrap"><svg class="locus-plot" viewBox="0 0 ${width} ${height}" role="img" aria-label="Lead SNP and nearby gene intervals"><text x="20" y="15">chr${esc(chrom)} · ${left.toLocaleString()} bp</text><text x="900" y="15" text-anchor="end">${right.toLocaleString()} bp</text><path d="M20 36H900" stroke="#9eafbd"/><path d="M${x(pos)-5} 26h10l-5 9Z" fill="#a65113"/><line x1="${x(pos)}" x2="${x(pos)}" y1="35" y2="${height-6}" stroke="#a65113" stroke-dasharray="3 3"/>${bars.join("")}</svg></div>`;
}
function locusDetails(row){
  const loci=selectedRun().loci||{},hits=loci.signals?.[row.gene]||[];
  if(!hits.length)return `<div class="details-box">${loci.signals?"No linked HuGE signal was recorded for this gene.":"This run has no HuGE signal archive attached."}</div>`;
  const [chrom,pos,p]=hits[0],bp=v=>Number(v).toLocaleString("en-US");
  let distance="";
  if(chrom===row.chrom && Number.isFinite(row.start) && Number.isFinite(row.end)){
    distance=pos<row.start?`${bp(row.start-pos)} bp before the gene start`:pos>row.end?`${bp(pos-row.end)} bp beyond the gene end`:"inside the gene interval";
  }
  return `<div class="details-box"><h3>Lead SNP for ${esc(row.gene)}</h3><p><strong>chr${esc(chrom)}:${bp(pos)}</strong>${loci.genome_build?" ("+esc(loci.genome_build)+")":""} · GWAS P = ${esc(p)}${distance?" · "+distance:""}</p>
    <p class="note">Lowest GWAS P among the HuGE signals linked to this gene. The marker shows the signal lead. Bars show gene start and end coordinates.</p>${locusSketch(row,hits[0],loci)}
    <details><summary>${hits.length} linked signal${hits.length===1?"":"s"}</summary><div class="locus-signals"><table><thead><tr><th>Lead SNP</th><th>GWAS P</th><th>HuGE gene–signal BF</th></tr></thead><tbody>${hits.map(([c,v,q,b])=>`<tr><td>chr${esc(c)}:${bp(v)}</td><td>${esc(q)}</td><td>${esc(Number(b.toPrecision(4)))}</td></tr>`).join("")}</tbody></table></div></details></div>`;
}
function geneSupportDetails(row,key){
  if(state.factorDetailTabs[key]==="locus")return locusDetails(row);
  const run=selectedRun(),sets=(run.gene_expansions||{})[row.gene]||[];
  if(!sets.length)return `<div class="details-box">${run.membership_loaded?"No matching scored gene-set memberships for this gene.":"No annotation membership inputs were supplied for this run."}</div>`;
  return `<div class="details-box"><h3>Supporting gene sets for ${esc(row.gene)}</h3><p class="note">Memberships in fitted sets, ordered by joint effect. The coefficients describe each set across its genes.${DATA.thresholds.max_provenance_rows_per_entry>=0?" Up to "+DATA.thresholds.max_provenance_rows_per_entry+" sets are shown.":""}</p>${smallTable(sets,[{key:"gene_set",label:"Gene set"},{key:"label",label:"Library"},{key:"beta",label:"Joint β",numeric:true},{key:"beta_uncorrected",label:"Marginal β",numeric:true},{key:"n",label:"N",numeric:true}],key+":sets")}</div>`;
}
function evidenceButton(k,tab,label){
  const open=Boolean(state.openRows[k] && state.factorDetailTabs[k]===tab);
  return `<button class="gene-disclosure" data-open-row-tab="${esc(k)}" data-tab="${tab}" aria-expanded="${open}">${open?"Hide ":"Show "}${label}</button>`;
}
"""
