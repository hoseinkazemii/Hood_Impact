"""Post-hoc geometry/response associations and a self-contained pair explorer.

Consumes the geometry and response audits; does not modify or train a network.
All-design analyses include held-out designs and are explicitly exploratory.
"""

import argparse
from itertools import combinations, permutations, product
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


FAMILIES = ((0, 1, 2, 3), (4, 5), (6, 7, 8, 9), (10, 11))
WITHIN_PAIRS = [pair for family in FAMILIES for pair in combinations(family, 2)]
FAMILY_OF = {design: family for family, designs in enumerate(FAMILIES) for design in designs}


def correlation(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 3:
        return np.nan
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 1e-12 else np.nan


def spatial_association(geometry, response):
    """Average within-pair rank association, plus pair/location rank residuals.

    A zero geometry field has no rank information. Its individual correlation
    is undefined; report the number of informative pairs rather than imply zero.
    The two-way statistic uses standard double demeaning across the complete
    pair/location table; this removes additive pair and location effects.
    """
    x, y = rankdata(geometry, axis=1), rankdata(response, axis=1)
    per_pair = np.array([correlation(a, b) for a, b in zip(x, y)])
    xc = x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + x.mean()
    yc = y - y.mean(1, keepdims=True) - y.mean(0, keepdims=True) + y.mean()
    return {
        "mean_within_pair_spearman": float(np.nanmean(per_pair)) if np.isfinite(per_pair).any() else None,
        "pair_and_location_rank_residual_correlation": correlation(xc, yc),
        "informative_pairs": int(np.isfinite(per_pair).sum()),
        "per_pair_spearman": per_pair,
    }


def family_rank_residual(values):
    """Rank and center dissimilarities within each design family.

    Two-design families have one pair and provide no ordering information.
    They remain available for direct effect maps, but cannot support this test.
    """
    values = np.asarray(values)
    if values.shape != (len(WITHIN_PAIRS),):
        raise ValueError("Expected the 14 within-family pair dissimilarities")
    output = np.zeros_like(values, dtype=float)
    for family in range(4):
        indices = [i for i, (a, _) in enumerate(WITHIN_PAIRS) if FAMILY_OF[a] == family]
        ranks = rankdata(values[indices])
        output[indices] = ranks - ranks.mean()
    return output


def family_label_permutation(geometry, response):
    """Exact 24x24 design-label permutations within the two four-design families.

    Shared designs stay shared: this does not shuffle 14 dependent pairs as if
    they were independent. The two singleton pair families cannot be tested.
    This is a small observational exchangeability diagnostic, not causal proof.
    """
    x, y = family_rank_residual(geometry), family_rank_residual(response)
    observed = correlation(x, y)
    lookup = {pair: index for index, pair in enumerate(WITHIN_PAIRS)}
    null = []
    for first, third in product(permutations(FAMILIES[0]), permutations(FAMILIES[2])):
        mapping = dict(zip(FAMILIES[0] + FAMILIES[2], first + third))
        mapping.update({4: 4, 5: 5, 10: 10, 11: 11})
        indices = [lookup[tuple(sorted((mapping[a], mapping[b])))] for a, b in WITHIN_PAIRS]
        null.append(correlation(x, y[indices]))
    null = np.asarray(null)
    return {
        "observed_family_centered_rank_correlation": observed,
        "num_design_label_permutations": len(null),
        "null_p05": float(np.nanpercentile(null, 5)) if np.isfinite(null).any() else None,
        "null_p95": float(np.nanpercentile(null, 95)) if np.isfinite(null).any() else None,
        "fraction_permutations_at_least_observed": float(np.mean(null >= observed - 1e-12)) if np.isfinite(observed) else None,
        "informative_families": ["A (0-3)", "C (6-9)"],
    }


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path, payload):
    Path(path).write_text(json.dumps(clean_json(payload), indent=2, allow_nan=False), encoding="utf-8")


def write_explorer(path, payload):
    """Offline canvas explorer; all plotted values are embedded in this file."""
    template = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hood geometry and design response</title>
<style>
body{font:15px system-ui,sans-serif;margin:0;background:#f3f5f7;color:#192b3c}main{max-width:1400px;margin:auto;padding:24px}
h1{font-size:27px;margin:0 0 8px}p{line-height:1.5}.muted{color:#586675}select,button{font:inherit;padding:7px;border:1px solid #b8c5cf;border-radius:5px;background:white}
.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;background:white;padding:15px;border-radius:8px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}.card{background:white;border-radius:8px;padding:16px}
h2{font-size:17px;margin:0 0 8px}canvas{width:100%;height:auto;display:block}.stat{font-size:16px;line-height:1.65}#note{min-height:42px}
@media(max-width:850px){.grid{grid-template-columns:1fr}main{padding:12px}}.legend{font-size:13px;color:#536273}
</style><main>
<h1>Do small geometry changes produce different impact responses?</h1>
<p class="muted">Explore all 12 designs through their 14 within-family pairs and 142 matching impact locations. All-design results are post-hoc, including held-out designs 4, 5 and validation design 11. No model was fitted for this analysis.</p>
<div class="controls"><label>Design pair <select id="pair"></select></label><label>Impact location <select id="location"></select></label>
<button id="corner">Location 1</button><button id="sensitive">Location 142</button><label><input id="hicMode" type="checkbox"> Show HIC difference on map</label></div>
<p class="stat" id="note"></p>
<div class="grid">
<section class="card"><h2>1. Geometry changes for this design pair</h2><canvas id="geometry" width="660" height="485"></canvas>
<p class="legend">Inner panel in gray. Changed points: purple = inside a saved attention patch; orange = outside. Hollow circles = saved training-only anchors. Star = selected impact. Distances are nearest-node differences &gt; 0.5 mm, not material displacements.</p></section>
<section class="card"><h2>2. Where the simulation responses differ</h2><canvas id="effects" width="660" height="485"></canvas>
<p class="legend">Same pair at every point; brighter/larger dots mean a larger response difference. Click a dot to select that impact. This is an effect map, not a node-importance map.</p></section>
<section class="card"><h2>3. Acceleration histories at the selected impact</h2><canvas id="histories" width="660" height="310"></canvas>
<p class="legend" id="curveLegend"></p></section>
<section class="card"><h2>4. Signed design difference through time</h2><canvas id="delta" width="660" height="310"></canvas>
<p class="legend">Higher-numbered design minus lower-numbered design. Dashed red line, when available, is the trained model. Matching the difference requires both the correct size and direction.</p></section>
</div><p class="muted">A small response difference at a corner and a large difference elsewhere can arise from the same geometry change. These whole-design simulations establish response sensitivity to design, but do not isolate a particular node's causal effect or prove that an attention mechanism will learn it.</p>
</main><script>const D=__DATA__;
const el=id=>document.getElementById(id), P=el('pair'), L=el('location');
D.pairs.forEach((p,i)=>{let o=new Option(`${p.family}: designs ${p.a} and ${p.b}${p.a===4?' (held out)':p.b===11?' (train / validation)':''}`,i);P.add(o)});
for(let i=0;i<D.impact_xy.length;i++)L.add(new Option(i+1,i));P.value=D.pairs.findIndex(p=>p.a===4&&p.b===5);L.value=141;
const rms=a=>Math.sqrt(a.reduce((s,x)=>s+x*x,0)/a.length),sub=(a,b)=>a.map((v,i)=>v-b[i]);
const fmt=v=>Number(v).toFixed(2),color=t=>`rgb(${Math.round(55+200*t)},${Math.round(40+135*t)},${Math.round(130-90*t)})`;
function ctx(id){const c=el(id),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);g.font='12px system-ui';return g}
function mapping(g){const b=D.bounds, x=v=>55+(v-b[0])/(b[1]-b[0])*555,y=v=>430-(v-b[2])/(b[3]-b[2])*395;
g.strokeStyle='#dce3e8';g.lineWidth=1;g.strokeRect(55,35,555,395);g.fillStyle='#607080';g.fillText('X2 / Y (mm)',280,472);
for(let v=Math.ceil(b[0]/250)*250;v<=b[1];v+=250)g.fillText(v,x(v)-12,449);
for(let v=Math.ceil(b[2]/250)*250;v<=b[3];v+=250)g.fillText(v,12,y(v)+4);
g.fillText('X1 / X (mm)',8,19);return{x,y}}
function dot(g,x,y,r,c){g.fillStyle=c;g.beginPath();g.arc(x,y,r,0,Math.PI*2);g.fill()}
function star(g,m,p){let x=m.x(p[1]),y=m.y(p[0]);g.strokeStyle='#0b2738';g.lineWidth=2;g.beginPath();for(let i=0;i<10;i++){let a=-Math.PI/2+i*Math.PI/5,r=i%2?4:10;let X=x+r*Math.cos(a),Y=y+r*Math.sin(a);if(!i)g.moveTo(X,Y);else g.lineTo(X,Y)}g.closePath();g.fillStyle='#fff';g.fill();g.stroke()}
function base(g,m,p){for(const q of p.base)dot(g,m.x(q[1]),m.y(q[0]),.65,'#d0d8de')}
function graph(id,series,labels,zero=false){let g=ctx(id),all=series.flatMap(s=>s.values),lo=Math.min(0,...all),hi=Math.max(...all);let pad=Math.max((hi-lo)*.08,1);lo-=pad;hi+=pad;
const x=t=>55+(t-D.times[0])/(D.times.at(-1)-D.times[0])*565,y=v=>260-(v-lo)/(hi-lo)*230;
g.strokeStyle='#e5eaee';g.fillStyle='#607080';for(let i=0;i<5;i++){let v=lo+(hi-lo)*i/4;g.beginPath();g.moveTo(55,y(v));g.lineTo(620,y(v));g.stroke();g.fillText(fmt(v),5,y(v)+4)}
for(let t=0;t<=25;t+=5)g.fillText(t,x(t/1000)-5,280);g.fillText('Time (ms)',300,304);g.fillText(zero?'Difference (g)':'Acceleration (g)',7,17);
if(zero){g.strokeStyle='#9aa6ae';g.beginPath();g.moveTo(55,y(0));g.lineTo(620,y(0));g.stroke()}
series.forEach(s=>{g.strokeStyle=s.color;g.lineWidth=s.width||2;g.setLineDash(s.dashed?[6,4]:[]);g.beginPath();s.values.forEach((v,i)=>{if(i)g.lineTo(x(D.times[i]),y(v));else g.moveTo(x(D.times[i]),y(v))});g.stroke()});g.setLineDash([]);
labels.forEach((label,i)=>{g.fillStyle=series[i].color;g.fillText(label,350,18+i*16)})}
function render(){const p=D.pairs[+P.value],l=+L.value,impact=D.impact_xy[l],a=D.curves[p.a][l],b=D.curves[p.b][l],d=sub(b,a),H=D.hic[p.b][l]-D.hic[p.a][l];
let g=ctx('geometry'),m=mapping(g);base(g,m,p);p.points.forEach((q,i)=>dot(g,m.x(q[1]),m.y(q[0]),2,p.covered[i]?'#7554a6':'#e68a37'));
g.strokeStyle='#596a78';g.lineWidth=.8;for(const q of D.anchors){g.beginPath();g.arc(m.x(q[1]),m.y(q[0]),3.5,0,Math.PI*2);g.stroke()}star(g,m,impact);
g=ctx('effects');m=mapping(g);base(g,m,p);let effects=D.impact_xy.map((_,i)=>el('hicMode').checked?Math.abs(D.hic[p.b][i]-D.hic[p.a][i]):rms(sub(D.curves[p.b][i],D.curves[p.a][i]))),max=Math.max(...effects);
effects.forEach((v,i)=>dot(g,m.x(D.impact_xy[i][1]),m.y(D.impact_xy[i][0]),4+4*v/(max||1),color(v/(max||1))));star(g,m,impact);g.fillStyle='#334a5a';g.fillText(`Map range: 0 to ${fmt(max)} ${el('hicMode').checked?'HIC15':'g RMS'}`,275,19);
let series=[{values:a,color:'#2277b0'},{values:b,color:'#e1872d'}],delta=[{values:d,color:'#242d4f'}],names=[`Design ${p.a}: simulation`,`Design ${p.b}: simulation`],dn=['Simulation difference'];
let extra='';if(D.predictions[p.a]&&D.predictions[p.b]){const pa=D.predictions[p.a][l],pb=D.predictions[p.b][l],pd=sub(pb,pa);series.push({values:pa,color:'#2277b0',dashed:true},{values:pb,color:'#e1872d',dashed:true});delta.push({values:pd,color:'#c74250',dashed:true});dn.push('Model difference');extra=` Model difference: <b>${fmt(rms(pd))} g RMS</b>.`}
graph('histories',series,names);graph('delta',delta,dn,true);el('curveLegend').textContent=`Blue = design ${p.a}; orange = design ${p.b}. Solid = simulation; dashed = model, when available.`;
el('note').innerHTML=`Location <b>${l+1}</b>, designs <b>${p.a} → ${p.b}</b>: simulation difference <b>${fmt(rms(d))} g RMS</b>; HIC15 difference <b>${H>=0?'+':''}${fmt(H)}</b>.${extra}<br>Saved change patches cover <b>${fmt(100*p.coverage)}%</b> of this pair's changed points. Geometry map stays fixed when you switch impact location.`;}
P.onchange=L.onchange=el('hicMode').onchange=render;el('corner').onclick=()=>{L.value=0;render()};el('sensitive').onclick=()=>{L.value=141;render()};
el('effects').onclick=e=>{const c=e.currentTarget,r=c.getBoundingClientRect(),X=(e.clientX-r.left)*c.width/r.width,Y=(e.clientY-r.top)*c.height/r.height,b=D.bounds;let best=0,dist=Infinity;D.impact_xy.forEach((p,i)=>{let x=55+(p[1]-b[0])/(b[1]-b[0])*555,y=430-(p[0]-b[2])/(b[3]-b[2])*395,d=(x-X)**2+(y-Y)**2;if(d<dist){dist=d;best=i}});L.value=best;render()};render();
</script></html>'''
    embedded = json.dumps(clean_json(payload), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    Path(path).write_text(template.replace("__DATA__", embedded), encoding="utf-8")


def association_analysis(root, responses):
    geometry = pd.read_csv(root / "geometry/pair_location_geometry.csv")
    pair_summary = pd.read_csv(root / "geometry/pairwise_geometry.csv")
    curves = responses["curves"]
    keys = [f"{a}-{b}" for a, b in WITHIN_PAIRS]
    response = np.array([np.sqrt(np.mean((curves[b]-curves[a])**2, axis=-1)) for a, b in WITHIN_PAIRS])
    hic = np.array([np.abs(responses["hic"][b]-responses["hic"][a]) for a, b in WITHIN_PAIRS])
    def matrix(column):
        return geometry.pivot(index="pair", columns="location_id", values=column).loc[keys].to_numpy()
    rows, pair_rows, permutation_rows = [], [], []
    train_mask = np.array([a not in (4,5,11) and b not in (4,5,11) for a,b in WITHIN_PAIRS])
    for radius in (50,100,200,400):
        # A rank should not be determined by sub-micron text-rounding noise.
        # This descriptor measures thresholded changed fraction directly.
        feature = matrix(f"local_{radius}mm_changed_fraction")
        for scope, selection in (("all_within_family", np.ones(14, bool)), ("training_within_family", train_mask)):
            for target_name, target in (("history", response), ("hic15", hic)):
                result = spatial_association(feature[selection], target[selection])
                rows.append({"radius_mm": radius, "scope": scope, "target": target_name,
                             **{k:v for k,v in result.items() if k != "per_pair_spearman"}})
                for pair, rho in zip(np.array(keys)[selection], result["per_pair_spearman"]):
                    pair_rows.append({"radius_mm":radius,"scope":scope,"target":target_name,"pair":pair,"spearman":rho})
        for location in (1,142):
            permutation_rows.append({"geometry_descriptor":f"fraction_changed_within_{radius}mm", "location_id":location,
                                     **family_label_permutation(feature[:,location-1], response[:,location-1])})
    for descriptor in ("nn_rms_mm", "aligned_rms_mm"):
        feature = pair_summary.set_index("pair").loc[keys,descriptor].to_numpy()
        for location in (1,142):
            permutation_rows.append({"geometry_descriptor":descriptor,"location_id":location,
                                     **family_label_permutation(feature,response[:,location-1])})
    distances = matrix("min_changed_distance_xy_mm")
    distance_rows = []
    for index, key in enumerate(keys):
        rho = correlation(rankdata(distances[index]), rankdata(response[index]))
        distance_rows.append({"pair":key,"distance_vs_response_spearman":rho,
                              "distance_at_1_mm":distances[index,0],"distance_at_142_mm":distances[index,141],
                              "response_at_1_g":response[index,0],"response_at_142_g":response[index,141]})
    pd.DataFrame(rows).to_csv(root / "spatial_associations.csv",index=False)
    pd.DataFrame(pair_rows).to_csv(root / "pair_spatial_associations.csv",index=False)
    pd.DataFrame(distance_rows).to_csv(root / "distance_response_associations.csv",index=False)
    write_json(root / "design_label_permutations.json",permutation_rows)
    # Save merged per-pair/per-impact evidence for arbitrary user inspection.
    evidence = geometry.copy()
    evidence["history_difference_rms_g"] = [float(np.sqrt(np.mean((curves[b,l-1]-curves[a,l-1])**2)))
        for a,b,l in zip(evidence.design_i,evidence.design_j,evidence.location_id)]
    evidence["absolute_hic15_difference"] = [float(abs(responses["hic"][b,l-1]-responses["hic"][a,l-1]))
        for a,b,l in zip(evidence.design_i,evidence.design_j,evidence.location_id)]
    evidence.to_csv(root / "matched_geometry_response.csv",index=False)
    summary = {"spatial_associations":rows,"distance_response":distance_rows,
               "design_label_permutations":permutation_rows,
               "method":"Geometry descriptor is fraction of structural points differing by >0.5mm within a fixed XY radius. Radius sweep is descriptive, not tuned into a model.",
               "interpretation":["Spatial correlations pool 142 dependent impact locations; no independent-point significance tests.",
                                 "Two-way rank residuals remove patterns shared by all impact locations and offsets specific to a pair.",
                                 "Within-family label permutations retain shared-design dependence; only A and C have enough designs to order pairs.",
                                 "All-design comparisons include held-out data and do not establish a future test score.",
                                 "These associations neither identify causal nodes nor establish that an attention architecture will generalize."]}
    write_json(root / "association_summary.json",summary)
    plot_associations(root,pd.DataFrame(rows),pd.DataFrame(distance_rows),evidence)
    return evidence


def plot_associations(root,table,distance,evidence):
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout="constrained")
    selected=table[(table.scope=="all_within_family") & (table.target=="history")]
    ax=axes[0]
    ax.plot(selected.radius_mm,selected.mean_within_pair_spearman,"o-",label="Mean within-pair spatial correlation")
    ax.plot(selected.radius_mm,selected.pair_and_location_rank_residual_correlation,"s-",label="After removing common location pattern")
    ax.axhline(0,color="gray",lw=.8);ax.set(xlabel="Radius around impact (mm)",ylabel="Rank correlation",ylim=(-.2,1),title="Does more nearby change mean more response change?")
    ax.legend(fontsize=8,loc="upper left")
    for ax,pair in zip(axes[1:],("4-5","6-7")):
        local=evidence[evidence.pair==pair]
        ax.scatter(local.min_changed_distance_xy_mm,local.history_difference_rms_g,c="#adc4d2",s=17)
        for loc,color in ((1,"#2878b5"),(142,"#d8782b")):
            row=local[local.location_id==loc].iloc[0]
            ax.scatter(row.min_changed_distance_xy_mm,row.history_difference_rms_g,c=color,s=70)
            ax.annotate(str(loc),(row.min_changed_distance_xy_mm,row.history_difference_rms_g),xytext=(6,5),textcoords="offset points")
        rho=distance.set_index("pair").loc[pair,"distance_vs_response_spearman"]
        ax.set(xlabel="Distance to nearest changed point (XY mm)",ylabel="Simulation difference (g RMS)",title=f"Designs {pair}: distance correlation {rho:.2f}")
    fig.suptitle("Geometry and response are spatially related, but proximity is not a complete explanation",fontsize=15)
    for suffix in ("png","pdf"):
        fig.savefig(root/f"geometry_response_associations.{suffix}",dpi=190)
    plt.close(fig)


def build_explorer_data(root,run_dir,data_dir,responses):
    from abaqus_scripts.inp_geom import Deck
    clouds=np.load(root/"geometry/within_family_changed_nodes.npz",allow_pickle=False)
    summary=pd.read_csv(root/"geometry/pairwise_geometry.csv").set_index("pair")
    atlas=np.load(run_dir/"change_atlas.npz",allow_pickle=False)["anchors_mm"]
    bases={}
    for family in FAMILIES:
        design=family[0]
        deck=Deck(data_dir/"inp_files"/f"HoodImpact_{142*design+1}.inp")
        base=np.array([deck.nodes[n] for n in sorted(deck.elset_nodes("Hood_Inner-1-2"))])
        bases[FAMILY_OF[design]]=base[::max(1,len(base)//4500)]
    pairs=[]
    for a,b in WITHIN_PAIRS:
        points=np.concatenate([clouds[f"pair_{a}_{b}_d{d}_xyz"] for d in (a,b)])
        covered=np.concatenate([clouds[f"pair_{a}_{b}_d{d}_covered"] for d in (a,b)])
        pairs.append({"a":a,"b":b,"family":"ABCD"[FAMILY_OF[a]],"points":np.round(points,3),
                      "covered":covered.astype(int),"base":np.round(bases[FAMILY_OF[a]],2),
                      "coverage":summary.loc[f"{a}-{b}","changed_coverage_fraction"]})
    predictions={}
    frame=pd.read_csv(run_dir/"test_acceleration_histories.csv")
    for design in (4,5):
        curves=[]
        for loc in range(1,143):
            values=frame[frame.run_number==142*design+loc].sort_values("time")
            if len(values)!=len(responses["times"]) or not np.allclose(values.time,responses["times"],atol=1e-8,rtol=0):
                raise ValueError("Saved predictions and simulation grids do not match")
            curves.append(values.acceleration_pred_g.to_numpy())
        predictions[design]=np.round(curves,5)
    all_points=np.concatenate(list(bases.values()))
    bounds=[float(all_points[:,1].min()-50),float(all_points[:,1].max()+50),
            float(all_points[:,0].min()-50),float(all_points[:,0].max()+50)]
    return {"times":responses["times"],"curves":np.round(responses["curves"],5),"hic":responses["hic"],
            "impact_xy":responses["impact_xy"],"pairs":pairs,"anchors":atlas,"bounds":bounds,"predictions":predictions}


def plot_key_pair(root,payload):
    p=next(p for p in payload["pairs"] if (p["a"],p["b"])==(4,5))
    curves,impacts,times=payload["curves"],payload["impact_xy"],payload["times"]*1000
    fig,axes=plt.subplots(2,2,figsize=(13,10),layout="constrained")
    base,changed=np.asarray(p["base"]),np.asarray(p["points"])
    ax=axes[0,0];ax.scatter(base[:,1],base[:,0],s=.8,c="#cbd4dc",rasterized=True)
    ax.scatter(changed[:,1],changed[:,0],s=10,c="#d8782b",label="Changed inner-panel points >0.5 mm")
    for loc,color in ((1,"#2878b5"),(142,"#a83256")):
        xy=impacts[loc-1];ax.scatter(xy[1],xy[0],marker="*",s=170,c=color,zorder=4)
        ax.annotate(f"Impact {loc}",(xy[1],xy[0]),xytext=(8,9),textcoords="offset points",color=color)
    ax.set(aspect="equal",xlabel="X2 / Y (mm)",ylabel="X1 / X (mm)",title="Only 102 inner-panel nodes move >0.5 mm")
    ax.legend(fontsize=8,loc="lower left")
    ax.text(.02,.98,"Outer panel and connectivity unchanged\n97.1% of changed points are inside saved patches",transform=ax.transAxes,va="top",fontsize=9)
    difference=np.sqrt(np.mean((curves[5]-curves[4])**2,axis=-1));ax=axes[0,1]
    ax.scatter(base[:,1],base[:,0],s=.5,c="#d6dce2",rasterized=True)
    m=ax.scatter(impacts[:,1],impacts[:,0],c=difference,cmap="magma",s=38,vmin=0)
    for loc in (1,142):
        xy=impacts[loc-1];ax.annotate(str(loc),(xy[1],xy[0]),xytext=(7,5),textcoords="offset points",fontsize=11,
            bbox=dict(facecolor="white",alpha=.9,edgecolor="none",pad=1))
    ax.set(aspect="equal",xlabel="X2 / Y (mm)",ylabel="X1 / X (mm)",title="The same geometric change affects impacts differently")
    fig.colorbar(m,ax=ax,label="Design 5 minus 4: history difference (g RMS)",shrink=.7)
    for ax,loc in zip(axes[1],(1,142)):
        truth=curves[5,loc-1]-curves[4,loc-1]
        pred=payload["predictions"][5][loc-1]-payload["predictions"][4][loc-1]
        ax.plot(times,truth,label="Simulation difference",color="#273b58",lw=2)
        ax.plot(times,pred,"--",label="Model difference",color="#d8782b",lw=2)
        ax.axhline(0,color="gray",lw=.7)
        ax.set(xlabel="Time (ms)",ylabel="Acceleration difference (g)",ylim=(-32,32),
            title=f"Impact {loc}: true {np.sqrt(np.mean(truth**2)):.2f} g vs model {np.sqrt(np.mean(pred**2)):.2f} g RMS")
        ax.legend(fontsize=9);ax.grid(alpha=.2)
    fig.suptitle("Designs 4 and 5: changed geometry is captured, but its response effect is underpredicted",fontsize=15)
    for suffix in ("png","pdf"):
        fig.savefig(root/f"designs_4_5_geometry_and_response.{suffix}",dpi=190)
    plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir",type=Path,default=Path("runs/change_geometry_analysis"))
    parser.add_argument("--run-dir",type=Path,default=Path("runs/mesh_change_attention/20260922_170037_3194307_1704"))
    parser.add_argument("--data-dir",type=Path,default=Path("Data/HoodImpact_1704_EuroNCAP"))
    args=parser.parse_args(argv)
    responses=dict(np.load(args.analysis_dir/"responses/response_arrays.npz",allow_pickle=False))
    association_analysis(args.analysis_dir,responses)
    payload=build_explorer_data(args.analysis_dir,args.run_dir,args.data_dir,responses)
    write_explorer(args.analysis_dir/"geometry_response_explorer.html",payload)
    plot_key_pair(args.analysis_dir,payload)
    print("Saved association tables, key-pair figure and offline explorer:",args.analysis_dir)


if __name__=="__main__":
    main()
