"""Static scientific maps and an offline node-inspection gallery for HIC30."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from abaqus_scripts.inp_geom import Deck
from analyze_hic30_node_response import clean_json, PART_NAMES, FAMILY_OF


def face_arrays(deck):
    return {part:[np.asarray([deck.nodes[n] for n in deck.elems[e]])[:,[1,0]]
                  for e in deck.elsets[name]]
            for part,name in [(0,'Hood_Inner-1-2'),(1,'Hood_Outer-1-2')]}


def draw_map(ax,arrays,values,part,impact,faces,title,selected):
    xyz=arrays['ref_xyz']; labels=arrays['ref_part']
    ax.add_collection(PolyCollection(faces[part],facecolors='#f2f4f6',edgecolors='#c3ccd2',
                                     linewidths=.09,rasterized=True))
    use=(labels==part)|((part==0)&(labels==2))
    finite=use&np.isfinite(values)
    other=use&~finite
    ax.scatter(xyz[other,1],xyz[other,0],s=.5,c='#c6cdd3',linewidths=0,rasterized=True)
    order=np.flatnonzero(finite)
    order=order[np.argsort(np.abs(values[order]),kind='stable')]
    ax.scatter(xyz[order,1],xyz[order,0],c=values[order],cmap='RdBu_r',vmin=-1,vmax=1,
               s=3.5,linewidths=0,rasterized=True)
    ax.scatter(impact[1],impact[0],marker='*',s=190,c='#f8de46',edgecolors='#132a3b',linewidths=1,zorder=6)
    for row in selected.itertuples():
        node=int(row.node_index)
        if not use[node]:
            continue
        ax.scatter(xyz[node,1],xyz[node,0],s=140,facecolors='none',edgecolors='#161b22',linewidths=1.3,zorder=7)
        ax.annotate(str(int(row.rank)),(xyz[node,1],xyz[node,0]),xytext=(7,7),
                    textcoords='offset points',fontsize=8,weight='bold',zorder=8,
                    bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1.5))
    ax.set(title=title,xlabel='X2 (mm)',ylabel='X1 (mm)',aspect='equal',xlim=(-720,990),ylim=(-530,1060))
    ax.tick_params(labelsize=8)
    ax.grid(alpha=.12)


def make_figure(arrays,metadata,candidates,il,faces):
    loc=int(arrays['locations'][il]); impact=arrays['impacts'][il]
    selected=candidates[candidates.location==loc]
    all_top=selected[selected['mode']=='all']
    within_top=selected[selected['mode']=='within']
    fig,axes=plt.subplots(2,2,figsize=(13,11.3),layout='constrained')
    draw_map(axes[0,0],arrays,arrays['score_all'][:,il],1,impact,faces,
             'All 12 designs: outer panel',all_top)
    draw_map(axes[0,1],arrays,arrays['score_all'][:,il],0,impact,faces,
             'All 12 designs: inner panel + other structure',all_top)
    draw_map(axes[1,0],arrays,arrays['score_within'][:,il],0,impact,faces,
             'Small edits: within-family association (A and C)',within_top)
    ax=axes[1,1]
    colors=['#22749b','#dc812d','#2f966b','#8268a0']
    if len(within_top):
        top=within_top.sort_values('rank').iloc[0]
        node=int(top.node_index)
        x=arrays['geometry_within'][:,node]
        y=arrays['response_within'][:,il]
        for f in range(4):
            take=np.asarray([FAMILY_OF[int(a)]==f for a,_ in arrays['within_pairs']])
            ax.scatter(x[take],y[take],s=65,facecolors=colors[f] if f in (0,2) else 'none',
                       edgecolors=colors[f],label=f'Family {"ABCD"[f]}'+(' (one pair; not scored)' if f in (1,3) else ''),zorder=3)
        for i,(a,b) in enumerate(arrays['within_pairs']):
            ax.annotate(f'{a}-{b}',(x[i],y[i]),xytext=(4,4+(i%2)*8),textcoords='offset points',fontsize=7)
        p=top.permutation_global_scan_fraction
        q=top.leave_one_design_out_q10
        stability=f'{q:.2f}' if np.isfinite(q) else 'not estimable after every deletion'
        ax.set_title(f'Top small-edit candidate: reference node {int(top.node_id)}\n'
                     f'Within-family rank association = {top.rho_within:.2f}',fontsize=11)
        tied=int(top.within_identical_rank_group_nodes)
        ax.text(.02,.02,f'{tied} node-centered patches share this geometry rank pattern\n'
                f'Design-deletion 10th percentile: {stability}\n'
                f'Global scan permutation reference: {p:.3f}',transform=ax.transAxes,fontsize=8,
                va='bottom',bbox=dict(facecolor='white',alpha=.88,edgecolor='#d6dde2'))
    else:
        ax.text(.5,.5,'No positive within-family candidate',ha='center',transform=ax.transAxes)
    ax.set(xlabel='Local exact geometry displacement RMS (mm)',ylabel='Acceleration-history difference RMS (g)')
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.2)
    ax.legend(loc='upper right',fontsize=8)
    scalar=plt.cm.ScalarMappable(norm=Normalize(-1,1),cmap='RdBu_r')
    fig.colorbar(scalar,ax=[axes[0,0],axes[0,1],axes[1,0]],shrink=.7,pad=.025,
                 label='Rank association: larger local geometry change vs larger history difference')
    fig.suptitle(f'Impact location {loc}  |  HIC range {arrays["range_percent"][il]:.1f}%\n'
                 f'Impact (X1, X2) = ({impact[0]:.1f}, {impact[1]:.1f}) mm',fontsize=16,weight='bold')
    all_ids=', '.join(f'{int(r.rank)}: {int(r.node_id)}' for r in all_top.itertuples())
    within_ids=', '.join(f'{int(r.rank)}: {int(r.node_id)}' for r in within_top.itertuples())
    fig.supxlabel('Reference mesh: design 0. Each color scores a 32-node, same-part neighborhood; star = impact.\n'
                  f'Candidate center IDs — all-design: {all_ids}; small-edit: {within_ids}.\n'
                  'Gray = unchanged or unsupported. Associations are exploratory; co-changing nodes are not separately identifiable.',fontsize=9)
    return fig


def encode_scores(values):
    # Store location-major signed integers; 32767 marks undefined associations.
    data=np.where(np.isfinite(values),np.rint(values*1000),32767).T.astype('<i2')
    return base64.b64encode(data.tobytes()).decode('ascii')


def write_gallery(out,arrays,metadata,candidates):
    payload=dict(locations=arrays['locations'],impacts=arrays['impacts'],hic=arrays['range_percent'],
        xyz=np.round(arrays['ref_xyz'],3),node_ids=arrays['ref_ids'],parts=arrays['ref_part'],
        radius=np.round(arrays['patch_radii_mm'].max(0),1),
        all=encode_scores(arrays['score_all']),within=encode_scores(arrays['score_within']),
        loo=encode_scores(arrays['loo_q10']),candidates=candidates.to_dict(orient='records'))
    template=r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>HIC30: nodes associated with history differences</title>
<style>
body{margin:0;background:#edf2f5;color:#173044;font:15px system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:24px}
h1{font-size:27px;margin:0 0 10px}p{line-height:1.5}select,button{font:inherit;padding:8px;border:1px solid #adbcc7;border-radius:6px;background:white}
.controls,.card{background:white;border-radius:10px;padding:16px}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}canvas{width:100%;display:block;touch-action:none}
h2{font-size:18px;margin:0 0 8px}.muted{color:#597083}.legend{background:linear-gradient(90deg,#2166ac,#f7f7f7,#b2182b);height:14px;width:260px;border:1px solid #c7d1da}
.row{display:flex;justify-content:space-between;width:260px;font-size:12px}.note{min-height:42px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:9px;border-bottom:1px solid #dae2e8}
a{color:#156397}#tip{position:fixed;pointer-events:none;background:#112d40;color:white;padding:10px;border-radius:6px;font-size:13px;display:none;z-index:10;max-width:330px}
@media(max-width:850px){main{padding:12px}.grid{grid-template-columns:1fr}.controls{gap:8px}table{font-size:12px}}
</style></head><body><main>
<h1>Which geometry changes accompany different acceleration histories?</h1>
<p>__COUNT__ locations retained by HIC range ≥ __THRESHOLD__%. The maps use the simulation responses directly, with no trained-model attention.
Hover over a node to read its ID and score; zoom or drag the mesh to inspect a region.</p>
<div class="controls"><button id="prev">← Previous</button><label>Impact <select id="location"></select></label><button id="next">Next →</button>
<label>Comparison <select id="mode"><option value="all">All 12 designs</option><option value="within">Small edits within families A/C</option></select></label>
<button id="in">Zoom +</button><button id="out">Zoom −</button><button id="reset">Reset view</button></div>
<p id="status"></p><p id="method" class="muted note"></p>
<div class="legend"></div><div class="row"><span>−1: inverse association</span><span>0</span><span>+1: positive association</span></div>
<p class="muted">Gray nodes have no resolved variation or an unsupported local patch. Gray does not mean physically unimportant.</p>
<div class="grid"><div class="card"><h2>Outer panel</h2><canvas id="outer" width="700" height="640"></canvas></div>
<div class="card"><h2>Inner panel and other structure</h2><canvas id="inner" width="700" height="640"></canvas></div></div>
<div class="card" style="margin-top:16px"><h2>Strongest positive candidates, at least 75 mm apart within each part</h2>
<table><thead><tr><th>Rank</th><th>Reference node</th><th>Part</th><th>Association</th><th>Identical-pattern patches</th><th>Deletion check</th><th>Global permutation reference</th></tr></thead><tbody id="rows"></tbody></table>
<p id="downloads"></p></div>
<p>Each score belongs to a <b>node-centered neighborhood</b>, not the isolated causal effect of moving that node. All-design comparisons include broad shape-family differences.
The small-edit score uses exact matching node IDs within families; families B and D each have only one pair and cannot identify a within-family correlation.</p>
<p><a href="all___COUNT___location_maps.pdf">__COUNT__-page PDF</a> · <a href="README.md">Method and findings</a> · <a href="top_regions.csv">All candidate regions</a> ·
<a href="top_region_node_membership.csv">Actual patch member node IDs for all 12 designs</a> · <a href="top_region_pair_evidence.csv">Pairwise evidence</a></p>
<p class="muted">Yellow star = impact. Numbered circles = candidate patch centers. The reference mesh is design 0.
The score summarizes all 1,000 time samples. Larger positive scores indicate stronger association, not a predicted change in g or proof of causality.</p>
</main><div id="tip"></div><script>const D=__DATA__;
const el=id=>document.getElementById(id),L=el('location'),M=el('mode'),N=D.node_ids.length;
function decode(b){const raw=atob(b),bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);return new Int16Array(bytes.buffer)}
const scores={all:decode(D.all),within:decode(D.within)},loo=decode(D.loo);delete D.all;delete D.within;delete D.loo;
D.locations.forEach((loc,i)=>L.add(new Option(loc,i)));let zoom=1,pan=[0,0],selected=-1,drag=null;
const fmt=(x,k=2)=>x==null?'not estimable':Number(x).toFixed(k),value=(a,i)=>a[Number(L.value)*N+i]===32767?null:a[Number(L.value)*N+i]/1000;
function rgb(t){let low=t<0?[33,102,172]:[178,24,43],u=Math.min(1,Math.abs(t));return `rgb(${low.map(x=>Math.round(247+(x-247)*u)).join(',')})`}
function project(p){let s=Math.min(580/1710,525/1590)*zoom;return [350+(p[1]-135)*s+pan[0],320-(p[0]-265)*s+pan[1]]}
function dot(g,x,y,r,c){g.beginPath();g.arc(x,y,r,0,Math.PI*2);g.fillStyle=c;g.fill()}
function visible(i,part){return part===1?D.parts[i]===1:D.parts[i]!==1}
function candidates(){return D.candidates.filter(r=>r.location===D.locations[Number(L.value)]&&r.mode===M.value).sort((a,b)=>a.rank-b.rank)}
function draw(id,part){const c=el(id),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);g.save();g.beginPath();g.rect(35,25,635,560);g.clip();
const scored=[];for(let i=0;i<N;i++)if(visible(i,part)){const p=project(D.xyz[i]),v=value(scores[M.value],i);if(v==null)dot(g,p[0],p[1],Math.max(.65,.65*zoom),'#c6cdd3');else scored.push([i,v])}
scored.sort((a,b)=>Math.abs(a[1])-Math.abs(b[1]));for(const [i,v] of scored){const p=project(D.xyz[i]);dot(g,p[0],p[1],Math.max(.8,.8*zoom),rgb(v))}
let impact=D.impacts[Number(L.value)],p=project(impact);g.beginPath();for(let i=0;i<10;i++){const a=-Math.PI/2+i*Math.PI/5,r=i%2?5:13,x=p[0]+r*Math.cos(a),y=p[1]+r*Math.sin(a);i?g.lineTo(x,y):g.moveTo(x,y)}g.closePath();g.fillStyle='#f8de46';g.fill();g.strokeStyle='#122a3b';g.lineWidth=1.5;g.stroke();
g.font='bold 13px system-ui';for(const r of candidates()){let i=r.node_index;if(!visible(i,part))continue;const p=project(D.xyz[i]);g.strokeStyle='#142c3b';g.lineWidth=2;g.beginPath();g.arc(p[0],p[1],10,0,Math.PI*2);g.stroke();g.fillStyle='#142c3b';g.fillText(r.rank,p[0]+12,p[1]-9)}
if(selected>=0&&visible(selected,part)){const p=project(D.xyz[selected]);g.strokeStyle='#050505';g.lineWidth=2;g.strokeRect(p[0]-7,p[1]-7,14,14)}g.restore();
g.strokeStyle='#bfcbd4';g.lineWidth=1;g.strokeRect(35,25,635,560);g.fillStyle='#52697c';g.font='13px system-ui';g.fillText('X2 (mm) →',310,620);g.fillText('X1 (mm) ↑',40,17);
for(let v=-500;v<=1000;v+=250){let p=project([265,v]);if(p[0]>35&&p[0]<670)g.fillText(v,p[0]-12,603);p=project([v,135]);if(p[1]>25&&p[1]<585)g.fillText(v,0,p[1]+4)}}
function render(){let il=Number(L.value),loc=D.locations[il];el('status').textContent=`Location ${loc} · HIC range ${fmt(D.hic[il],1)}% · Impact (X1, X2) = (${fmt(D.impacts[il][0],1)}, ${fmt(D.impacts[il][1],1)}) mm`;
el('method').textContent=M.value==='all'?'All 12 designs, 66 dependent pairs: rank association between local same-part geometric mismatch and acceleration-history RMS difference. Broad geometry-family effects are included.':'Within-family-centered rank association: larger exact local node displacements versus larger history differences. Families A and C supply ordering information; no comparison of unrelated node numbers across meshes.';
el('rows').innerHTML=candidates().map(r=>`<tr data-node="${r.node_index}"><td>${r.rank}</td><td>${r.node_id}</td><td>${r.part}</td><td>${fmt(M.value==='all'?r.rho_all:r.rho_within)}</td><td>${M.value==='all'?'—':r.within_identical_rank_group_nodes}</td><td>${M.value==='all'?'—':fmt(r.leave_one_design_out_q10)}</td><td>${M.value==='all'?'—':fmt(r.permutation_global_scan_fraction,3)}</td></tr>`).join('');
el('downloads').innerHTML=`<a href="plots/location_${String(loc).padStart(3,'0')}.png">Full location figure and pair evidence</a> · <a href="top_nodes/location_${String(loc).padStart(3,'0')}_${M.value}.csv">Top 300 nodes (CSV)</a> · <a href="node_scores/location_${String(loc).padStart(3,'0')}.csv.gz">Every node score (compressed CSV)</a>`;
draw('outer',1);draw('inner',0);el('rows').querySelectorAll('tr').forEach(row=>row.onclick=()=>{selected=Number(row.dataset.node);draw('outer',1);draw('inner',0)})}
L.onchange=M.onchange=()=>{selected=-1;render()};el('prev').onclick=()=>{L.value=(Number(L.value)+D.locations.length-1)%D.locations.length;render()};el('next').onclick=()=>{L.value=(Number(L.value)+1)%D.locations.length;render()};
el('in').onclick=()=>{zoom=Math.min(8,zoom*1.4);render()};el('out').onclick=()=>{zoom=Math.max(.65,zoom/1.4);render()};el('reset').onclick=()=>{zoom=1;pan=[0,0];render()};
for(const [id,part] of [['outer',1],['inner',0]]){const c=el(id);c.onpointerdown=e=>{drag=[e.clientX,e.clientY,...pan];c.setPointerCapture(e.pointerId)};c.onpointerup=()=>drag=null;c.onpointercancel=()=>drag=null;
c.onpointermove=e=>{if(drag){pan=[drag[2]+(e.clientX-drag[0])*c.width/c.clientWidth,drag[3]+(e.clientY-drag[1])*c.height/c.clientHeight];draw('outer',1);draw('inner',0);el('tip').style.display='none';return}
const r=c.getBoundingClientRect(),x=(e.clientX-r.left)*c.width/r.width,y=(e.clientY-r.top)*c.height/r.height;let best=-1,distance=60;for(let i=0;i<N;i++)if(visible(i,part)){const p=project(D.xyz[i]),d=(p[0]-x)**2+(p[1]-y)**2;if(d<distance){distance=d;best=i}}
const tip=el('tip');if(best<0){tip.style.display='none';return}const v=value(scores[M.value],best),q=value(loo,best);tip.innerHTML=`<b>Reference node ${D.node_ids[best]}</b><br>${['Inner panel','Outer panel','Other structure'][D.parts[best]]}<br>(X1, X2, X3): ${D.xyz[best].map(v=>fmt(v,1)).join(', ')} mm<br>Association: ${v==null?'undefined / unsupported':fmt(v,3)}<br>Maximum patch radius: ${D.radius[best]} mm${M.value==='within'?'<br>Design-deletion q10: '+fmt(q,3):''}`;tip.style.display='block';tip.style.left=Math.min(e.clientX+12,window.innerWidth-350)+'px';tip.style.top=Math.max(5,Math.min(e.clientY+12,window.innerHeight-180))+'px'};
c.onpointerleave=()=>el('tip').style.display='none';c.onwheel=e=>{e.preventDefault();zoom=Math.max(.65,Math.min(8,zoom*(e.deltaY<0?1.15:1/1.15)));render()}}
render();</script></body></html>'''
    data=json.dumps(clean_json(payload),separators=(',',':'),allow_nan=False).replace('</','<\\/')
    template=template.replace('__COUNT__',str(len(arrays['locations']))).replace('__THRESHOLD__',f'{metadata["threshold_percent"]:g}')
    (out/'gallery.html').write_text(template.replace('__DATA__',data),encoding='utf-8')


def render(out,data_dir):
    out=Path(out)
    arrays=dict(np.load(out/'analysis_arrays.npz',allow_pickle=False))
    metadata=json.loads((out/'analysis_metadata.json').read_text())
    candidates=pd.read_csv(out/'top_regions.csv')
    deck=Deck(Path(data_dir)/'inp_files/HoodImpact_1.inp')
    faces=face_arrays(deck)
    (out/'plots').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    with PdfPages(out/f'all_{len(arrays["locations"])}_location_maps.pdf') as pdf:
        for il,loc in enumerate(arrays['locations']):
            fig=make_figure(arrays,metadata,candidates,il,faces)
            fig.savefig(out/'plots'/f'location_{int(loc):03d}.png',dpi=170)
            pdf.savefig(fig,dpi=140)
            plt.close(fig)
            print(f'Map {il+1}/{len(arrays["locations"])}: location {loc}',flush=True)
    write_gallery(out,arrays,metadata,candidates)
    print(f'Offline gallery: {out / "gallery.html"}',flush=True)
