"""Map observed geometry/acceleration associations at HIC-filtered locations.

No neural model is fitted or queried. Scores describe node-centered geometric
neighborhoods, not the causal influence of independently moving one node.
"""
from __future__ import annotations

import argparse
import base64
from itertools import combinations, permutations, product
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from abaqus_scripts.inp_geom import Deck
from mesh_design_clusters import GEOMETRY_CLUSTERS

ROOT = Path(__file__).resolve().parent
FAMILIES = tuple(tuple(v) for v in GEOMETRY_CLUSTERS.values())
ALL_PAIRS = np.asarray(list(combinations(range(12), 2)))
WITHIN_PAIRS = np.asarray([p for family in FAMILIES for p in combinations(family, 2)])
FAMILY_OF = {d: f for f, ids in enumerate(FAMILIES) for d in ids}
PART_NAMES = ('Inner panel', 'Outer panel', 'Other structure')


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value), indent=2, allow_nan=False), encoding='utf-8')


def rank_columns(values, groups=None):
    """Rank and center within pair groups; normalize columns to unit norm.

    Constant columns, and singleton groups such as the B/D within-family pair,
    supply no ordering information. Return a zero vector plus a validity mask.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Expected a finite pair-by-feature matrix')
    if groups is None:
        groups = np.zeros(len(values), dtype=int)
    groups = np.asarray(groups)
    if groups.shape != (len(values),):
        raise ValueError('Each pair must have a group')
    ranked = np.zeros_like(values)
    for g in np.unique(groups):
        use = groups == g
        r = rankdata(values[use], axis=0, method='average')
        ranked[use] = r - r.mean(axis=0)
    norm = np.linalg.norm(ranked, axis=0)
    valid = norm > 1e-12
    return np.divide(ranked, norm, out=np.zeros_like(ranked), where=valid), valid


def score_fields(geometry, response, groups=None):
    x, valid_x = rank_columns(geometry, groups)
    y, valid_y = rank_columns(response, groups)
    rho = x.T @ y
    rho[~valid_x, :] = np.nan
    rho[:, ~valid_y] = np.nan
    return rho


def within_permutation_indices(pairs=WITHIN_PAIRS):
    """All 24 x 24 whole-design relabelings in A and C, never shuffle pairs."""
    lookup = {tuple(p): i for i, p in enumerate(pairs)}
    output = []
    for a, c in product(permutations(FAMILIES[0]), permutations(FAMILIES[2])):
        mapping = dict(zip(FAMILIES[0] + FAMILIES[2], a + c))
        mapping.update({4: 4, 5: 5, 10: 10, 11: 11})
        output.append([lookup[tuple(sorted((mapping[int(i)], mapping[int(j)])))] for i, j in pairs])
    return np.asarray(output)


def permutation_maxima(geometry, response, supported):
    """Exploratory exact reference; maxima cover all nodes, then all locations.

    Family-constrained design permutations preserve the pair dependence.
    This is not proof of exchangeability or a causal significance test.
    """
    groups = np.asarray([FAMILY_OF[int(a)] for a, _ in WITHIN_PAIRS])
    x, valid = rank_columns(geometry, groups)
    y, _ = rank_columns(response, groups)
    valid &= supported
    if not valid.any():
        return np.zeros((576, response.shape[1]))
    # All identical rank profiles have identical tests; compress exactly.
    x = np.unique(x[:, valid].T, axis=0).T
    indices = within_permutation_indices()
    maxima = np.empty((len(indices), response.shape[1]))
    for start in range(0, len(indices), 32):
        ix = indices[start:start+32]
        permuted = y[ix].transpose(1, 0, 2).reshape(len(y), -1)
        values = (x.T @ permuted).reshape(x.shape[1], len(ix), response.shape[1])
        maxima[start:start+len(ix)] = values.max(axis=0)
    return maxima


def leave_design_out(geometry, response):
    """Recompute within-family ranks after removing each informative design."""
    values = []
    for design in (*FAMILIES[0], *FAMILIES[2]):
        select = np.all(WITHIN_PAIRS != design, axis=1)
        groups = np.asarray([FAMILY_OF[int(a)] for a, _ in WITHIN_PAIRS[select]])
        values.append(score_fields(geometry[select], response[select], groups))
    values = np.stack(values)
    count = np.isfinite(values).sum(0)
    # Require every one-design deletion to be estimable. Missing is not zero.
    complete = count == len(values)
    q10 = np.quantile(np.nan_to_num(values, nan=-2.), .1, axis=0)
    q10[~complete] = np.nan
    fraction_positive = np.where(complete, np.mean(values > 0, axis=0), np.nan)
    return q10, fraction_positive, count


def identical_rank_profiles(geometry, groups, supported):
    """Identify spatially separate patches that these comparisons cannot rank apart."""
    x,valid=rank_columns(geometry,groups)
    valid &= supported
    group_id=np.full(geometry.shape[1],-1,dtype=int)
    group_size=np.zeros(geometry.shape[1],dtype=int)
    profiles,inverse,counts=np.unique(x[:,valid].T,axis=0,return_inverse=True,return_counts=True)
    group_id[valid]=inverse
    group_size[valid]=counts[inverse]
    return group_id,group_size,len(profiles)


def load_geometry(data_dir, k, max_radius, resolution):
    decks, parts, xyz, ids = {}, {}, {}, {}
    audit = []
    for d in range(12):
        deck = Deck(data_dir/'inp_files'/f'HoodImpact_{142*d+1}.inp')
        decks[d] = deck
        head = deck.impactor_node_ids()
        structural = set(deck.nodes) - head
        inner = deck.elset_nodes('Hood_Inner-1-2') & structural
        outer = deck.elset_nodes('Hood_Outer-1-2') & structural
        if inner & outer:
            raise ValueError('Inner/outer nodes overlap; require an explicit part mapping')
        parts[d] = (inner, outer, structural-inner-outer)
        ids[d] = [np.asarray(sorted(p), dtype=np.int64) for p in parts[d]]
        xyz[d] = [np.asarray([deck.nodes[int(n)] for n in ns]) for ns in ids[d]]
        audit.append(dict(design=d, canonical_run=142*d+1, headform_nodes=len(head),
                          part_nodes=[len(ns) for ns in ids[d]]))
    ref_ids = np.concatenate(ids[0])
    ref_xyz = np.concatenate(xyz[0])
    ref_part = np.repeat(np.arange(3), [len(v) for v in ids[0]])
    trees = {d: [cKDTree(p) for p in xyz[d]] for d in range(12)}
    neighbors, radii = {}, []
    for d in range(12):
        neighbors[d] = []
        radius = []
        for part in range(3):
            reference = xyz[0][part][:, :2]
            distances, indices = cKDTree(xyz[d][part][:, :2]).query(reference, k=min(k,len(ids[d][part])), workers=4)
            neighbors[d].append(np.asarray(indices).reshape(len(reference), -1))
            radius.extend(np.asarray(distances).reshape(len(reference), -1)[:, -1])
        radii.append(radius)
    radii = np.asarray(radii)
    supported = np.max(radii, axis=0) <= max_radius
    all_fields = np.empty((len(ALL_PAIRS),len(ref_ids)))
    within_fields = np.empty((len(WITHIN_PAIRS),len(ref_ids)))
    within_lookup = {tuple(pair): i for i,pair in enumerate(WITHIN_PAIRS)}
    for ip,(a,b) in enumerate(ALL_PAIRS):
        values, exact_values = [], []
        is_within = (int(a),int(b)) in within_lookup
        if is_within:
            if decks[a].elems != decks[b].elems:
                raise ValueError(f'Within-family connectivity mismatch: {a}-{b}')
        for part in range(3):
            da = trees[b][part].query(xyz[a][part],workers=4)[0]
            db = trees[a][part].query(xyz[b][part],workers=4)[0]
            values.append(np.sqrt((np.mean(da[neighbors[a][part]]**2,axis=1)+
                                   np.mean(db[neighbors[b][part]]**2,axis=1))/2))
            if is_within:
                if not np.array_equal(ids[a][part],ids[b][part]):
                    raise ValueError(f'Within-family node IDs mismatch: {a}-{b}, part {part}')
                delta2=np.sum((xyz[b][part]-xyz[a][part])**2,axis=1)
                exact_values.append(np.sqrt((np.mean(delta2[neighbors[a][part]],axis=1)+
                                             np.mean(delta2[neighbors[b][part]],axis=1))/2))
        all_fields[ip] = np.concatenate(values)
        if is_within:
            within_fields[within_lookup[(int(a),int(b))]] = np.concatenate(exact_values)
        if (ip+1)%11==0:
            print(f'Geometry fields: {ip+1}/{len(ALL_PAIRS)} design pairs',flush=True)
    # Rank only changes resolved above a stated physical precision, not roundoff.
    all_fields = np.round(all_fields/resolution)*resolution
    within_fields = np.round(within_fields/resolution)*resolution
    return dict(decks=decks,ids=ids,xyz=xyz,neighbors=neighbors,
                ref_ids=ref_ids,ref_xyz=ref_xyz,ref_part=ref_part,
                all_fields=all_fields,within_fields=within_fields,supported=supported,
                radii=radii,audit=audit)


def load_histories(data_dir, run_dir, threshold):
    table = pd.read_csv(run_dir/'hic_location_variation.csv')
    columns = [f'hic_design_{d}' for d in range(12)]
    values = table[columns].to_numpy()
    ranges = 100*np.ptp(values,axis=1)/values.mean(axis=1)
    locations = table.loc[ranges>=threshold,'location'].to_numpy(int)
    if not len(locations):
        raise ValueError('No locations pass the requested HIC range')
    full=[]
    reference=None
    for d in range(12):
        own=[]
        for loc in locations:
            frame=pd.read_csv(data_dir/'output_history_acc'/f'HoodImpact_{142*d+loc}_SAE1000_interp1000.csv')
            times=frame.Time.to_numpy(float)
            if reference is None:
                reference=times
            if not np.array_equal(times,reference):
                raise ValueError('Histories do not have the same full time grid')
            a=frame['A(in g)'].to_numpy(float)
            if not np.isfinite(a).all():
                raise ValueError('Nonfinite acceleration')
            own.append(a)
        full.append(own)
    full=np.asarray(full)
    if np.any(np.diff(reference)<=0):
        raise ValueError('Invalid time grid')
    from hic15 import batched_hic
    recomputed=batched_hic(reference,np.abs(full))
    expected=values[np.isin(table.location,locations)].T
    np.testing.assert_allclose(recomputed,expected,rtol=2e-6,atol=1e-3)
    impacts=pd.read_csv(data_dir/'ImpactCoords_1704.csv')[['X1','X2']].to_numpy().reshape(12,142,2)
    np.testing.assert_allclose(impacts, np.broadcast_to(impacts[:1], impacts.shape),rtol=0,atol=.001)
    def differences(curves,pairs):
        return np.sqrt(np.mean((curves[pairs[:,1]]-curves[pairs[:,0]])**2,axis=-1))
    return dict(locations=locations,times=reference,full=full,impacts=impacts[0,locations-1],hic=recomputed,
                range_percent=ranges[np.isin(table.location,locations)],
                response_all=differences(full,ALL_PAIRS),response_within=differences(full,WITHIN_PAIRS),
                response_sampled=differences(full[:,:,::16],ALL_PAIRS),
                response_within_sampled=differences(full[:,:,::16],WITHIN_PAIRS))


def spatial_candidates(score,xyz,part,count=3,separation=75.):
    """Positive, finite, spatially distinct candidates. Never promote NaNs."""
    order=np.flatnonzero(np.isfinite(score)&(score>0))
    order=order[np.argsort(-score[order],kind='stable')]
    chosen=[]
    for i in order:
        if all(part[i]!=part[j] or np.linalg.norm(xyz[i,:2]-xyz[j,:2])>=separation for j in chosen):
            chosen.append(int(i))
            if len(chosen)==count:
                break
    return chosen


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'Data/HoodImpact_1704_EuroNCAP')
    parser.add_argument('--run-dir',type=Path,default=ROOT/'runs/mesh_impact_history/hic_filtered/20260925_191414_3218656_1704')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'runs/hic30_node_response_maps')
    parser.add_argument('--threshold',type=float,default=30.)
    parser.add_argument('--patch-nodes',type=int,default=32)
    parser.add_argument('--max-patch-radius-mm',type=float,default=75.)
    parser.add_argument('--geometry-resolution-mm',type=float,default=.05)
    parser.add_argument('--compute-only',action='store_true')
    parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args(argv)
    if args.patch_nodes<1 or args.max_patch_radius_mm<=0 or args.geometry_resolution_mm<=0:
        parser.error('Patch size, radius, and geometry resolution must be positive')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.render_only:
        from render_hic30_node_response import render
        render(args.output_dir,args.data_dir)
        return
    with threadpool_limits(limits=4):
        run_analysis(args)
    if not args.compute_only:
        from render_hic30_node_response import render
        render(args.output_dir,args.data_dir)


def run_analysis(args):
    out=args.output_dir
    print('Loading and checking full simulation histories and the HIC filter.',flush=True)
    response=load_histories(args.data_dir,args.run_dir,args.threshold)
    print(f'{len(response["locations"])} locations pass HIC range >= {args.threshold:g}%.',flush=True)
    geometry=load_geometry(args.data_dir,args.patch_nodes,args.max_patch_radius_mm,args.geometry_resolution_mm)
    x=geometry['all_fields']; xw=geometry['within_fields']
    supported=geometry['supported']
    groups=np.asarray([FAMILY_OF[int(a)] for a,_ in WITHIN_PAIRS])
    score_all=score_fields(x,response['response_all'])
    score_within=score_fields(xw,response['response_within'],groups)
    score_sampled=score_fields(x,response['response_sampled'])
    score_within_sampled=score_fields(xw,response['response_within_sampled'],groups)
    q10,positive,count=leave_design_out(xw,response['response_within'])
    score_a=score_fields(xw[groups==0],response['response_within'][groups==0])
    score_c=score_fields(xw[groups==2],response['response_within'][groups==2])
    profile_id,profile_size,profile_count=identical_rank_profiles(xw,groups,supported)
    for field in [score_all,score_within,score_sampled,score_within_sampled,q10,positive,score_a,score_c]:
        field[~supported]=np.nan
    print('Computing 576 family-preserving design permutations; scanning all nodes and locations.',flush=True)
    null_max=permutation_maxima(xw,response['response_within'],supported)
    global_null_max=null_max.max(axis=1)
    np.savez_compressed(out/'analysis_arrays.npz',**{k:response[k] for k in response},
        ref_ids=geometry['ref_ids'],ref_xyz=geometry['ref_xyz'],ref_part=geometry['ref_part'],
        patch_radii_mm=geometry['radii'],geometry_all=x,geometry_within=xw,
        all_pairs=ALL_PAIRS,within_pairs=WITHIN_PAIRS,supported=supported,
        score_all=score_all,score_within=score_within,score_sampled=score_sampled,
        score_within_sampled=score_within_sampled,loo_q10=q10,loo_positive_fraction=positive,
        loo_valid_count=count,score_family_a=score_a,score_family_c=score_c,
        within_profile_id=profile_id,within_profile_size=profile_size,
        permutation_location_max=null_max,permutation_global_max=global_null_max)
    location_rows=[]; candidates=[]; membership=[]; pair_rows=[]
    (out/'node_scores').mkdir(exist_ok=True)
    (out/'top_nodes').mkdir(exist_ok=True)
    for il,loc in enumerate(response['locations']):
        table=pd.DataFrame(dict(node_id=geometry['ref_ids'],part=[PART_NAMES[p] for p in geometry['ref_part']],
            X1_mm=geometry['ref_xyz'][:,0],X2_mm=geometry['ref_xyz'][:,1],X3_mm=geometry['ref_xyz'][:,2],
            patch_supported=supported,max_patch_radius_mm=geometry['radii'].max(0),
            rho_all=score_all[:,il],rho_within=score_within[:,il],rho_family_A=score_a[:,il],rho_family_C=score_c[:,il],
            within_identical_rank_group=profile_id,within_identical_rank_group_nodes=profile_size,
            leave_one_design_out_valid_count=count[:,il],
            leave_one_design_out_q10=q10[:,il],leave_one_design_out_positive_fraction=positive[:,il],
            rho_all_stride16=score_sampled[:,il],rho_within_stride16=score_within_sampled[:,il]))
        table.to_csv(out/'node_scores'/f'location_{loc:03d}.csv.gz',index=False,compression='gzip')
        table.nlargest(300,'rho_all').to_csv(out/'top_nodes'/f'location_{loc:03d}_all.csv',index=False)
        table.nlargest(300,'rho_within').to_csv(out/'top_nodes'/f'location_{loc:03d}_within.csv',index=False)
        chosen_by_mode={}
        for mode,score in [('all',score_all[:,il]),('within',score_within[:,il])]:
            selected=spatial_candidates(score,geometry['ref_xyz'],geometry['ref_part'])
            chosen_by_mode[mode]=selected
            for rank,node_index in enumerate(selected,1):
                row=table.iloc[node_index].to_dict()
                row.update(location=int(loc),mode=mode,rank=rank,node_index=node_index,
                    permutation_location_scan_fraction=(float(np.mean(null_max[:,il]>=score[node_index]-1e-12)) if mode=='within' else np.nan),
                    permutation_global_scan_fraction=(float(np.mean(global_null_max>=score[node_index]-1e-12)) if mode=='within' else np.nan))
                candidates.append(row)
                part=int(geometry['ref_part'][node_index])
                part_index=int(np.sum(geometry['ref_part'][:node_index]==part))
                for d in range(12):
                    ni=geometry['neighbors'][d][part][part_index]
                    for own_index in ni:
                        pos=geometry['xyz'][d][part][own_index]
                        own_node=int(geometry['ids'][d][part][own_index])
                        base=FAMILIES[FAMILY_OF[d]][0]
                        displacement=pos-np.asarray(geometry['decks'][base].nodes[own_node])
                        membership.append(dict(location=int(loc),mode=mode,rank=rank,reference_node_id=int(geometry['ref_ids'][node_index]),
                            design=d,node_id=own_node,part=PART_NAMES[part],
                            X1_mm=pos[0],X2_mm=pos[1],X3_mm=pos[2],family_base_design=base,
                            delta_X1_from_family_base_mm=displacement[0],delta_X2_from_family_base_mm=displacement[1],
                            delta_X3_from_family_base_mm=displacement[2],displacement_from_family_base_mm=float(np.linalg.norm(displacement))))
                pairs=ALL_PAIRS if mode=='all' else WITHIN_PAIRS
                gf=x if mode=='all' else xw
                rf=response['response_all'] if mode=='all' else response['response_within']
                for ip,(a,b) in enumerate(pairs):
                    pair_rows.append(dict(location=int(loc),mode=mode,rank=rank,reference_node_id=int(geometry['ref_ids'][node_index]),
                        design_a=int(a),design_b=int(b),geometry_change_mm=gf[ip,node_index],history_difference_rms_g=rf[ip,il]))
        top=chosen_by_mode['within'][0] if chosen_by_mode['within'] else None
        location_rows.append(dict(location=int(loc),X1_mm=response['impacts'][il,0],X2_mm=response['impacts'][il,1],
            hic_range_percent=response['range_percent'][il],mean_pair_history_difference_g=response['response_all'][:,il].mean(),
            within_top_node_id=int(geometry['ref_ids'][top]) if top is not None else None,
            within_top_rho=float(score_within[top,il]) if top is not None else None,
            within_top_loo_q10=float(q10[top,il]) if top is not None else None,
            within_location_scan_fraction=float(np.mean(null_max[:,il]>=score_within[top,il]-1e-12)) if top is not None else None,
            within_global_scan_fraction=float(np.mean(global_null_max>=score_within[top,il]-1e-12)) if top is not None else None))
    pd.DataFrame(location_rows).to_csv(out/'location_summary.csv',index=False)
    pd.DataFrame(candidates).to_csv(out/'top_regions.csv',index=False)
    pd.DataFrame(membership).to_csv(out/'top_region_node_membership.csv',index=False)
    pd.DataFrame(pair_rows).to_csv(out/'top_region_pair_evidence.csv',index=False)
    metadata=dict(threshold_percent=args.threshold,locations=response['locations'],num_locations=len(response['locations']),
        design_ids=list(range(12)),families=GEOMETRY_CLUSTERS,reference_design=0,reference_run=1,
        patch_nodes=args.patch_nodes,max_patch_radius_mm=args.max_patch_radius_mm,
        geometry_resolution_mm=args.geometry_resolution_mm,full_time_points=len(response['times']),
        source_run=str(args.run_dir.resolve()),source_data=str(args.data_dir.resolve()),
        geometry_audit=geometry['audit'],reference_nodes=len(geometry['ref_ids']),supported_nodes=int(supported.sum()),
        within_informative_nodes=int(np.isfinite(score_within[:,0]).sum()),
        all_informative_nodes=int(np.isfinite(score_all[:,0]).sum()),
        part_counts={name:int(np.sum(geometry['ref_part']==p)) for p,name in enumerate(PART_NAMES)},
        part_within_informative={name:int(np.sum(np.isfinite(score_within[:,0])&(geometry['ref_part']==p))) for p,name in enumerate(PART_NAMES)},
        patch_radius_mm_percentiles=np.percentile(geometry['radii'],[0,50,95,100]),
        permutation_count=len(global_null_max),permutation_global_max_p95=float(np.quantile(global_null_max,.95)),
        within_distinct_geometry_rank_profiles=profile_count,
        location_summaries=location_rows,
        score_all_definition='Spearman rank correlation across 66 design pairs: local symmetric same-part nearest-node mismatch vs full-history acceleration RMS difference.',
        score_within_definition='Correlation of within-family-centered ranks across 14 pairs: local exact node-ID displacement RMS vs history RMS; only A and C have pair-ordering information.',
        limitations=['Observational associations, not causal single-node effects.',
          'All 12 designs including former validation/test are used; these are exploratory maps.',
          'A score belongs to a node-centered patch, not an independently perturbed node.',
          'All-design nearest-node differences can respond to remeshing/density and broad geometry families.',
          'B and D each have only one within-family pair, so cannot identify which regions explain their pair difference.',
          'Co-changing geometry regions may be statistically indistinguishable.',
          'Permutation fractions are exploratory conditional relabeling references; exchangeability is not established.'])
    write_json(out/'analysis_metadata.json',metadata)
    print(pd.DataFrame(location_rows).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
