"""One-map-per-location figures and a simple geometry-feature inspection page."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd

from abaqus_scripts.inp_geom import Deck
from analyze_hic30_node_response import clean_json
from analyze_hic30_feature_response import FEATURES


def label(name):
    if name.startswith('panel_separation'):
        return f'Panel separation proxy ({name.rsplit("_",1)[1]} nodes)'
    if 'nearest_XY_distance' in name:
        return f'Distance to {name.split("_")[0]} panel in XY'
    panel, rest = name.split('_',1)
    feature,k = rest.rsplit('_',1)
    return f'{panel.capitalize()} {feature.replace("_"," ")} ({k} nodes)'


def encode(values):
    # CV values below -1 are explicitly marked, not presented as exactly -1.
    quantized=np.where(np.isfinite(values),np.rint(np.clip(values,-1,1)*1000),32767)
    quantized[values < -1]=-1001
    return base64.b64encode(quantized.T.astype('<i2').tobytes()).decode('ascii')


def gallery(out,a,metadata,top,ceiling=None):
    payload=dict(locations=a['locations'],xy=np.round(a['ref_xyz'][:,:2],3),ids=a['ref_ids'],
        impacts=a['impacts'],hic=a['range_percent'],features=[label(n) for n in FEATURES],
        best=base64.b64encode(a['best_feature'].T.astype('u1').tobytes()).decode('ascii'),
        corr=encode(a['association']),design=encode(a['design_skill']),family=encode(a['family_skill']),
        top=top.to_dict(orient='records'),impact_family=a['impact_family_skill'],
        top5=np.nanquantile(a['association'],.95,axis=0),
        ceiling=None if ceiling is None else ceiling.shape_label_ceiling.to_numpy(),
        tied=None if ceiling is None else ceiling.sites_at_shape_label_ceiling.to_numpy())
    html=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Geometry features that track acceleration differences</title><style>
body{margin:0;background:#edf2f5;color:#193044;font:16px system-ui,sans-serif}main{max-width:1150px;margin:auto;padding:24px}
h1{font-size:28px}p{line-height:1.5}button,select{font:inherit;padding:9px;border:1px solid #a8bac7;border-radius:5px;background:white}
.controls{display:flex;align-items:center;flex-wrap:wrap;gap:10px}.card{background:white;border-radius:10px;padding:20px;margin:18px 0}
canvas{display:block;width:100%;max-width:850px;margin:auto;touch-action:none}a{color:#126297}.muted{color:#526c7e}
table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:9px;text-align:left;border-bottom:1px solid #dde4e9}
#tip{position:fixed;display:none;background:#142d40;color:white;border-radius:6px;padding:10px;pointer-events:none;z-index:5;max-width:360px}
#gradient{height:15px;width:280px;background:linear-gradient(90deg,#f7fbff,#2171b5,#08306b)}.legend{display:flex;width:280px;justify-content:space-between;font-size:13px}
@media(max-width:700px){main{padding:12px}table{font-size:12px}.card{padding:10px}}
</style></head><body><main><h1>Where does geometry carry information about the different acceleration curves?</h1>
<p>All 38,691 structural node sites are considered. No node is excluded because it stayed still or is far from the impact.
The features describe heights, slopes, shape and separation of the two panels around each site.</p>
<div class="card"><div class="controls"><button id="prev">Previous</button><label>Impact <select id="location"></select></label><button id="next">Next</button>
<label>Show <select id="mode"><option value="corr">Feature–curve correlation</option><option value="design">Held-out design prediction check</option><option value="family">Held-out geometry-family prediction check</option></select></label>
<button id="in">Zoom +</button><button id="out">Zoom −</button><button id="reset">Reset</button>
<label><input type="checkbox" id="highlight" checked> Highlight top 5% of correlations</label></div>
<p id="status"></p><p id="explanation"></p><p id="ceiling"></p><div id="gradient"></div><div class="legend" id="legend"></div>
<canvas id="mesh" width="850" height="800"></canvas><p class="muted">Yellow star = impact. Circles = representative high-correlation sites. Hover for node IDs and the feature behind the score; drag to pan.</p></div>
<div class="card"><h2>Features behind the highlighted regions</h2><table><thead><tr><th>Node</th><th>Feature</th><th>Correlation</th><th>Design check</th><th>Family check</th></tr></thead><tbody id="rows"></tbody></table>
<p id="downloads"></p><p id="baseline"></p></div>
<p><b>An unchanged node can still have useful surrounding features.</b> For example, an outer-panel node may stay fixed while the inner panel below it moves.
A feature that stays identical in every design cannot distinguish their curves by itself. In highlight mode, gray includes the remaining 95% of sites.
Otherwise gray indicates no varying feature in this feature bank. Neither means physical unimportance; the top 5% is a display ranking, not a significance threshold.</p>
<p>Correlation measures a relationship in these 12 designs. The prediction checks withhold responses before choosing a feature and fitting a small linear predictor.
Positive check scores beat the training-average curve; negative scores are worse. A weak check can indicate insufficient data, confounding or an inadequate feature/predictor.</p>
<p><a href="plots/why_the_maps_tie.png">Why the maps tie (summary)</a> · <a href="plots/curves_by_shape.png">Curves grouped by hood shape</a> · <a href="all_46_feature_maps.pdf">46 simple maps (PDF)</a> · <a href="README.md">Method and findings</a> · <a href="top_regions.csv">Candidate regions CSV</a> ·
<a href="candidate_feature_values.csv">Actual feature values across all designs</a></p>
</main><div id="tip"></div><script>const D=__DATA__;
const el=id=>document.getElementById(id),L=el('location'),M=el('mode'),C=el('mesh'),N=D.ids.length;
function bytes(s){const raw=atob(s),b=new Uint8Array(raw.length);for(let i=0;i<b.length;i++)b[i]=raw.charCodeAt(i);return b}
const scores={corr:new Int16Array(bytes(D.corr).buffer),design:new Int16Array(bytes(D.design).buffer),family:new Int16Array(bytes(D.family).buffer)},best=bytes(D.best);
delete D.corr;delete D.design;delete D.family;delete D.best;
D.locations.forEach((v,i)=>L.add(new Option(v,i)));let zoom=1,pan=[0,0],drag=null;
const index=i=>Number(L.value)*N+i,fmt=(v,d=2)=>v==null?'undefined':Number(v).toFixed(d),score=i=>scores[M.value][index(i)];
const formatScore=(v,mode)=>v===32767?'no varying feature':mode==='corr'?fmt(v/1000,3):v===-1001?'< −100% (see CSV)':fmt(v/10,1)+'%';
function project(xy){const scale=.415*zoom;return [430+(xy[1]-135)*scale+pan[0],395-(xy[0]-265)*scale+pan[1]]}
function color(v){if(v===32767)return '#c7d0d6';const x=Math.max(-1,Math.min(1,v/1000));
if(M.value==='corr'&&el('highlight').checked&&x<D.top5[Number(L.value)]-.0005)return '#d4dde3';
if(M.value==='corr'){const lo=[247,251,255],hi=[8,48,107];return `rgb(${lo.map((k,i)=>Math.round(k+(hi[i]-k)*x)).join(',')})`}
const hi=x<0?[190,52,55]:[20,122,91];return `rgb(${hi.map(k=>Math.round(247+(k-247)*Math.abs(x))).join(',')})`}
function draw(){const g=C.getContext('2d');g.clearRect(0,0,C.width,C.height);g.save();g.beginPath();g.rect(45,25,770,700);g.clip();
const order=Array.from({length:N},(_,i)=>i).sort((a,b)=>Math.abs(score(a))-Math.abs(score(b)));
for(const i of order){const [x,y]=project(D.xy[i]);g.fillStyle=color(score(i));g.beginPath();g.arc(x,y,Math.max(.8,.8*zoom),0,Math.PI*2);g.fill()}
const [x,y]=project(D.impacts[Number(L.value)]);g.beginPath();for(let j=0;j<10;j++){const a=-Math.PI/2+j*Math.PI/5,r=j%2?6:15;j?g.lineTo(x+r*Math.cos(a),y+r*Math.sin(a)):g.moveTo(x+r*Math.cos(a),y+r*Math.sin(a))}g.closePath();g.fillStyle='#f6d338';g.fill();g.strokeStyle='#102938';g.stroke();
g.font='bold 14px system-ui';for(const r of D.top.filter(r=>r.location===D.locations[Number(L.value)])){const p=project(D.xy[r.node_index]);g.strokeStyle='#0c2537';g.lineWidth=2;g.beginPath();g.arc(p[0],p[1],10,0,Math.PI*2);g.stroke();g.fillStyle='#0c2537';g.fillText(r.rank,p[0]+12,p[1]-9)}g.restore();
g.strokeStyle='#bfcdd7';g.lineWidth=1;g.strokeRect(45,25,770,700);g.fillStyle='#526a7b';g.font='14px system-ui';g.fillText('X2 (mm)',380,775);g.fillText('X1 (mm)',48,18);
for(let v=-500;v<=1000;v+=250){let p=project([265,v]);if(p[0]>45&&p[0]<815)g.fillText(v,p[0]-15,748);p=project([v,135]);if(p[1]>25&&p[1]<725)g.fillText(v,4,p[1]+4)}}
function render(){const il=Number(L.value),loc=D.locations[il],mode=M.value;el('status').textContent=`Location ${loc} · HIC range ${fmt(D.hic[il],1)}%`;
el('explanation').textContent=mode==='corr'?'Darker blue means that at least one measured feature here tracks more of the differences between the complete acceleration curves. The top-5% display cutoff here is '+fmt(D.top5[il],3)+'. Uncheck the highlight option to see all scores.':mode==='design'?'Does a feature from this region help predict a design whose response was withheld? Other designs from the same geometry family can remain in training.':'Does a feature from this region help predict a completely withheld geometry family? This is the harder check for transferring to a different hood shape.';
el('gradient').style.background=mode==='corr'?'linear-gradient(90deg,#f7fbff,#2171b5,#08306b)':'linear-gradient(90deg,#be3437,#f7f7f7,#147a5b)';el('legend').innerHTML=mode==='corr'?'<span>0: no linear relationship</span><span>1: strongest</span>':'<span>−100% or worse</span><span>0</span><span>+100%</span>';
el('rows').innerHTML=D.top.filter(r=>r.location===loc).map(r=>`<tr><td>${r.node_id}</td><td>${D.features[best[index(r.node_index)]]}</td><td>${fmt(r.association,3)}</td><td>${fmt(r.held_out_design_skill*100,1)}%</td><td>${fmt(r.held_out_family_skill*100,1)}%</td></tr>`).join('');
el('downloads').innerHTML=`<a href="plots/location_${String(loc).padStart(3,'0')}.png">This location map</a> · <a href="node_scores/location_${String(loc).padStart(3,'0')}.csv.gz">All node scores and feature names</a>`;
el('ceiling').innerHTML=D.ceiling?`<b>Caution:</b> a feature that only labels the 4 hood shapes (A/B/C/D) already reaches ${fmt(D.ceiling[il],3)} here, and ${D.tied[il].toLocaleString()} of ${N.toLocaleString()} sites score within 0.01 of that. The highlighted sites are interchangeable in this data, not a unique region. See <a href="plots/why_the_maps_tie.png">why the maps tie</a>.`:'';
el('baseline').textContent=`For comparison, the same feature bank centered at the impact has a held-out-family score of ${fmt(D.impact_family[il]*100,1)}%. Positive is better than the training-average curve.`;draw()}
L.onchange=M.onchange=render;el('highlight').onchange=draw;el('prev').onclick=()=>{L.value=(Number(L.value)+D.locations.length-1)%D.locations.length;render()};el('next').onclick=()=>{L.value=(Number(L.value)+1)%D.locations.length;render()};
el('in').onclick=()=>{zoom=Math.min(8,zoom*1.4);draw()};el('out').onclick=()=>{zoom=Math.max(.65,zoom/1.4);draw()};el('reset').onclick=()=>{zoom=1;pan=[0,0];draw()};
C.onpointerdown=e=>{drag=[e.clientX,e.clientY,...pan];C.setPointerCapture(e.pointerId)};C.onpointerup=C.onpointercancel=()=>drag=null;
C.onpointermove=e=>{if(drag){pan=[drag[2]+(e.clientX-drag[0])*C.width/C.clientWidth,drag[3]+(e.clientY-drag[1])*C.height/C.clientHeight];draw();el('tip').style.display='none';return}
const r=C.getBoundingClientRect(),x=(e.clientX-r.left)*C.width/r.width,y=(e.clientY-r.top)*C.height/r.height;let nearest=-1,distance=70;
for(let i=0;i<N;i++){const p=project(D.xy[i]),d=(p[0]-x)**2+(p[1]-y)**2;if(d<distance){nearest=i;distance=d}}
const tip=el('tip');if(nearest<0){tip.style.display='none';return}tip.innerHTML=`<b>Reference node ${D.ids[nearest]}</b><br>${D.features[best[index(nearest)]]}<br>Correlation: ${formatScore(scores.corr[index(nearest)],'corr')}<br>Design prediction check: ${formatScore(scores.design[index(nearest)],'design')}<br>Family prediction check: ${formatScore(scores.family[index(nearest)],'family')}`;
tip.style.display='block';tip.style.left=Math.max(5,Math.min(e.clientX+12,window.innerWidth-370))+'px';tip.style.top=Math.max(5,Math.min(e.clientY+12,window.innerHeight-170))+'px'};
C.onpointerleave=()=>el('tip').style.display='none';C.onwheel=e=>{e.preventDefault();zoom=Math.max(.65,Math.min(8,zoom*(e.deltaY<0?1.15:1/1.15)));draw()};render();
</script></body></html>'''
    (out/'gallery.html').write_text(html.replace('__DATA__',json.dumps(clean_json(payload),separators=(',',':'))),encoding='utf-8')


def render(out,data_dir):
    out=Path(out); a=dict(np.load(out/'feature_response_arrays.npz',allow_pickle=False))
    metadata=json.loads((out/'analysis_metadata.json').read_text())
    top=pd.read_csv(out/'top_regions.csv')
    ceiling=pd.read_csv(out/'identifiability_summary.csv') if (out/'identifiability_summary.csv').exists() else None
    if ceiling is not None:
        np.testing.assert_array_equal(ceiling.location,a['locations'])
    deck=Deck(Path(data_dir)/'inp_files/HoodImpact_1.inp')
    faces=[np.array([deck.nodes[n] for n in deck.elems[e]])[:,[1,0]] for e in deck.elsets['Hood_Inner-1-2']]
    (out/'plots').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    with PdfPages(out/'all_46_feature_maps.pdf') as pdf:
        for il,loc in enumerate(a['locations']):
            fig,ax=plt.subplots(figsize=(9.5,10),layout='constrained')
            value=a['association'][:,il]; xyz=a['ref_xyz']; rows=top[top.location==loc]
            ax.add_collection(PolyCollection(faces,facecolors='none',edgecolors='#b9c5ce',linewidths=.15,rasterized=True))
            threshold=np.nanquantile(value,.95)
            ax.scatter(xyz[:,1],xyz[:,0],c='#d4dde3',s=1.5,linewidths=0,rasterized=True)
            order=np.flatnonzero(value>=threshold)
            order=order[np.argsort(value[order],kind='stable')]
            dots=ax.scatter(xyz[order,1],xyz[order,0],c=value[order],cmap='Blues',vmin=0,vmax=1,s=3.5,linewidths=0,rasterized=True)
            ax.scatter(a['impacts'][il,1],a['impacts'][il,0],marker='*',s=200,c='#f6d338',edgecolors='#163143',zorder=4)
            for row in rows.itertuples():
                node=int(row.node_index)
                ax.scatter(xyz[node,1],xyz[node,0],s=180,facecolors='none',edgecolors='#0b2436',linewidths=1.5,zorder=5)
                ax.annotate(str(row.rank),(xyz[node,1],xyz[node,0]),xytext=(9,9),textcoords='offset points',weight='bold',zorder=6,
                            bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1))
            ax.set(aspect='equal',xlim=(-720,990),ylim=(-530,1060),xlabel='X2 (mm)',ylabel='X1 (mm)')
            ax.grid(alpha=.12)
            fig.colorbar(dots,ax=ax,shrink=.65,label='Feature–curve association (0 to 1)')
            fig.suptitle(f'Impact location {loc} | HIC range {a["range_percent"][il]:.1f}%\n'
                         'Which surrounding geometry features track the curve differences?',fontsize=14,weight='bold')
            lines=[f'All sites scored; top 5% highlighted (association ≥ {threshold:.3f}); gray = remaining sites. Yellow star = impact.']
            lines += [f'{r.rank}. Node {r.node_id}: {label(r.best_feature)} — association {r.association:.3f}' for r in rows.itertuples()]
            first=rows.iloc[0]
            lines += [f'First candidate prediction check: design {100*first.held_out_design_skill:+.1f}%; geometry family {100*first.held_out_family_skill:+.1f}%.',
                      'Positive check = better than the training-average curve. Correlation alone does not establish usefulness.',
                      'Each site summarizes both panels nearby; its own coordinates need not change. Reference drawing: design 0.']
            if ceiling is not None:
                c=ceiling.iloc[il]
                lines.append(f'Caution: a pure 4-shape label already scores {c.shape_label_ceiling:.3f}; '
                             f'{int(c.sites_at_shape_label_ceiling):,} of {len(value):,} sites are within 0.01 of it, so highlighted sites are interchangeable.')
            fig.supxlabel('\n'.join(lines),fontsize=8.5)
            fig.savefig(out/'plots'/f'location_{int(loc):03d}.png',dpi=160)
            pdf.savefig(fig,dpi=140)
            plt.close(fig)
            print(f'Feature map {il+1}/{len(a["locations"])}: location {loc}',flush=True)
    gallery(out,a,metadata,top,ceiling)
    print(f'Gallery: {out/"gallery.html"}',flush=True)
