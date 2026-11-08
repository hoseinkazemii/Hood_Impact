"""Export the 12 EuroNCAP hoods and all pair differences as portable 3D files.

Uses the existing viewer's distance fields across different mesh families;
these are quantized to 0.01 mm. Same-family displacement uses original decks.
Run with .venv/Scripts/python.exe export_hood_cluster_3d.py.
Vendor Three.js r147, OrbitControls, GLTFExporter and LICENSE in OUT/vendor first.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import struct
import zipfile
from pathlib import Path

import numpy as np
import pyvista as pv

from mesh_design_pair_differences_1704 import load_design, triangles

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "figures/design_pair_mesh_differences_1704"
OUT = ROOT / "figures/hood_cluster_3d_exports"
PARTS = ("outer", "inner", "hinges")
RAMP = np.array([[183,211,246], [134,182,239], [85,152,231], [42,120,214],
                 [28,92,171], [16,66,129], [10,42,85]], dtype=float)
GREY = np.array([201,205,210], dtype=np.uint8)


def difference_colors(values):
    t = np.clip((np.log10(np.maximum(values, 0.001)) + 3) / 5, 0, 1) * 6
    low = np.minimum(t.astype(int), 5)
    colors = RAMP[low] + (RAMP[low + 1] - RAMP[low]) * (t - low)[:, None]
    colors[values <= 0.001] = GREY
    return np.rint(colors).astype(np.uint8)


def vertex_normals(xyz, tris):
    p = xyz[tris]
    face = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    normals = np.zeros_like(xyz)
    for j in range(3):
        np.add.at(normals, tris[:, j], face)
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-20)
    return normals.astype("<f4")


class GLB:
    """Minimal glTF 2.0 writer with independent named hood parts."""

    def __init__(self, extras=None):
        self.bin = bytearray()
        self.doc = dict(asset=dict(version="2.0", generator="Hood cluster 3D exporter"),
                        scene=0, scenes=[dict(nodes=[0])],
                        nodes=[dict(name="Abaqus_mm_Zup_to_glTF_metres_Yup", children=[],
                                    rotation=[-2**-0.5, 0, 0, 2**-0.5], scale=[.001]*3)],
                        meshes=[], materials=[dict(name="Hood vertex colors", doubleSided=True,
                        pbrMetallicRoughness=dict(baseColorFactor=[1,1,1,1], metallicFactor=0,
                                                 roughnessFactor=.8))],
                        buffers=[], bufferViews=[], accessors=[], extras=extras or {})

    def accessor(self, values, kind, component, target, normalized=False):
        a = np.ascontiguousarray(values)
        while len(self.bin) % 4:
            self.bin.append(0)
        view = len(self.doc["bufferViews"])
        self.doc["bufferViews"].append(dict(buffer=0, byteOffset=len(self.bin),
                                             byteLength=a.nbytes, target=target))
        self.bin.extend(a.tobytes())
        entry = dict(bufferView=view, componentType=component,
                     count=len(a), type=kind)
        if normalized:
            entry["normalized"] = True
        if kind == "VEC3" and component == 5126:
            entry.update(min=a.min(0).tolist(), max=a.max(0).tolist())
        idx = len(self.doc["accessors"])
        self.doc["accessors"].append(entry)
        return idx

    def add(self, g, colors=None, offset=(0,0,0), label=None, extras=None):
        xyz, tris, tri_part = g["xyz"], g["triangles"], g["tri_part"]
        attrs = dict(POSITION=self.accessor(xyz.astype("<f4"), "VEC3", 5126, 34962),
                     NORMAL=self.accessor(g["normals"], "VEC3", 5126, 34962))
        if colors is None:
            colors = np.tile(GREY, (len(xyz), 1))
        rgba = np.column_stack((colors, np.full(len(colors), 255, dtype=np.uint8)))
        attrs["COLOR_0"] = self.accessor(rgba, "VEC4", 5121, 34962, normalized=True)
        parent = len(self.doc["nodes"])
        self.doc["nodes"].append(dict(name=label or f"Design_{g['design']:02d}",
                                     children=[], translation=list(offset),
                                     extras=extras or dict(design=g["design"], cluster=g["cluster"])))
        self.doc["nodes"][0]["children"].append(parent)
        for code, name in enumerate(PARTS):
            ind = tris[tri_part == code].astype("<u4").ravel()
            if not len(ind):
                continue
            mi = len(self.doc["meshes"])
            self.doc["meshes"].append(dict(name=name, primitives=[dict(
                attributes=attrs, indices=self.accessor(ind, "SCALAR", 5125, 34963),
                material=0, mode=4)]))
            ni = len(self.doc["nodes"])
            self.doc["nodes"].append(dict(name=f"Design_{g['design']:02d}_{name}", mesh=mi))
            self.doc["nodes"][parent]["children"].append(ni)

    def save(self, path):
        self.doc["buffers"] = [dict(byteLength=len(self.bin))]
        jb = json.dumps(self.doc, separators=(",", ":")).encode()
        jb += b" " * (-len(jb) % 4)
        self.bin.extend(b"\x00" * (-len(self.bin) % 4))
        header = struct.pack("<III", 0x46546C67, 2, 28 + len(jb) + len(self.bin))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(header + struct.pack("<I4s", len(jb), b"JSON") + jb +
                         struct.pack("<I4s", len(self.bin), b"BIN\x00") + self.bin)


def surface(g):
    faces = np.concatenate([[len(c), *c] for c in g["conn"]])
    mesh = pv.PolyData(g["xyz"], faces)
    mesh.point_data["node_id"] = g["node_ids"]
    mesh.point_data["part_id"] = g["part"].astype(np.uint8)
    mesh.cell_data["part_id"] = g["elem_part"].astype(np.uint8)
    mesh.cell_data["element_id"] = g["elem_ids"]
    mesh.field_data["part_names"] = np.array(PARTS)
    mesh.field_data["design"] = np.array([g["design"]])
    mesh.field_data["cluster"] = np.array([g["cluster"]])
    return mesh


def make_viewer(out, meta):
    html = (SOURCE / "viewer.html").read_text(encoding="utf-8")
    html = re.sub(r'<link[^>]+>', '', html)
    for name in ("three.min.js", "OrbitControls.js"):
        url = ("https://cdn.jsdelivr.net/npm/three@0.147.0/build/three.min.js" if name == "three.min.js"
               else "https://cdn.jsdelivr.net/npm/three@0.147.0/examples/js/controls/OrbitControls.js")
        js = (out / "vendor" / name).read_text(encoding="utf-8")
        html = html.replace(f'<script src="{url}"></script>', '<script>' + js.replace('</script', '<\\/script') + '</script>')
    js = (out / "vendor/GLTFExporter.js").read_text(encoding="utf-8")
    html = html.replace('<script>\n(() => {', '<script>' + js + '</script>\n<script>\n(() => {', 1)
    html = html.replace('const r = await fetch("viewer_data.b64.txt");\n    if (!r.ok) throw new Error("viewer_data.b64.txt: HTTP " + r.status);\n    const b64 = (await r.text()).trim(), bin = atob(b64);',
                        'const b64 = "' + (SOURCE / "viewer_data.b64.txt").read_text().strip() + '", bin = atob(b64);')
    html = html.replace('const r = await fetch("viewer_meta.json"); if (!r.ok) throw new Error("viewer_meta.json: HTTP " + r.status);\n      META = await r.json();',
                        'META = ' + json.dumps(meta, separators=(",", ":")) + ';')
    html = html.replace('All 66 pairs of the 12 Euro-NCAP hood designs, compared node by node.',
                        'All 12 EuroNCAP designs in clusters A (0–3), B (4–5), C (6–9), D (10–11). All 66 pairs, using node displacement or closest-surface distance as labelled.')
    html = html.replace('<div class="toolbar">', '''<div class="toolbar">
        <div class="tgroup"><span class="eyebrow">Export</span>
          <button class="btn primary" id="exportGLB">Save displayed 3D (.glb)</button>
          <button class="btn" id="exportPNG">Save image (.png)</button>
          <label>Preset <select id="clusterPreset" class="btn">
            <option value="">Choose comparison</option>
            <optgroup label="Within a cluster"><option value="0,1">A: 0 vs 1</option>
            <option value="4,5">B: 4 vs 5</option><option value="6,7">C: 6 vs 7</option>
            <option value="10,11">D: 10 vs 11</option></optgroup>
            <optgroup label="Between clusters"><option value="0,4">A–B: 0 vs 4</option>
            <option value="0,6">A–C: 0 vs 6</option><option value="0,10">A–D: 0 vs 10</option>
            <option value="4,6">B–C: 4 vs 6</option><option value="4,10">B–D: 4 vs 10</option>
            <option value="6,10">C–D: 6 vs 10</option></optgroup></select></label>
        </div>''', 1)
    html = html.replace('  // ---------- boot ----------', r'''
  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob), a = document.createElement("a");
    a.href = url; a.download = filename; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  }
  $("clusterPreset").addEventListener("change", e => {
    if (e.target.value) { const [a,b] = e.target.value.split(",").map(Number); select(a,b); fitAll(VIEWS[state.view]); }
  });
  $("exportPNG").addEventListener("click", () => {
    renderer.render(scene,camera);
    renderer.domElement.toBlob(b => downloadBlob(b, `design_${state.shown}_vs_${state.other}.png`));
  });
  $("exportGLB").addEventListener("click", () => {
    if (!cur) return;
    const btn = $("exportGLB"); btn.disabled = true; btn.textContent = "Exporting…";
    const root = new THREE.Group(); root.name = `Design_${state.shown}_vs_${state.other}`;
    root.rotation.x = -Math.PI/2; root.scale.setScalar(0.001);
    root.userData = {shown:state.shown, reference:state.other, method:cur.fld.p.method,
      threshold_mm:state.thr, colorScale:state.scale, sourceUnits:"mm", outputUnits:"m",
      center_mm:META.center_mm, note:"Colors baked; enable/disable parts using named meshes"};
    for (let code=0; code<3; code++) if (state.parts[code]) {
      const g = new THREE.BufferGeometry(), indices=[];
      g.setAttribute("position",cur.fg.geom.getAttribute("position").clone());
      g.setAttribute("color",cur.fg.geom.getAttribute("color").clone());
      for(let k=0;k<cur.fg.triPart.length;k++) if(cur.fg.triPart[k]===code)
        indices.push(cur.fg.tris[3*k],cur.fg.tris[3*k+1],cur.fg.tris[3*k+2]);
      g.setIndex(indices); g.computeVertexNormals();
      const m = new THREE.Mesh(g, new THREE.MeshStandardMaterial({vertexColors:true,
        side:THREE.DoubleSide, metalness:0, roughness:0.8}));
      m.name=PART_NAMES[code]; root.add(m);
    }
    const done=()=>{root.traverse(o=>{if(o.isMesh){o.geometry.dispose();o.material.dispose();}});
      btn.disabled=false;btn.textContent="Save displayed 3D (.glb)";};
    new THREE.GLTFExporter().parse(root, result=>{
      downloadBlob(new Blob([result],{type:"model/gltf-binary"}),`design_${state.shown}_vs_${state.other}.glb`); done();
    }, error=>{console.error(error);alert("Export failed: "+error.message);done();}, {binary:true});
  });
  // ---------- boot ----------''')
    assert 'await fetch("viewer_' not in html, 'Viewer still requires a server'
    assert '<script src=' not in html
    html = '<!doctype html>\n<html lang="en" data-theme="light">\n' + html + '\n</html>\n'
    (out / "hood_clusters_interactive.html").write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()
    out = args.out_dir
    for p in ("designs/glb", "designs/stl", "designs/vtp", "clusters", "within_clusters", "between_clusters", "previews"):
        (out / p).mkdir(parents=True, exist_ok=True)
    meta = json.loads((SOURCE / "viewer_meta.json").read_text())
    packed = gzip.decompress((SOURCE / "viewer_data.bin").read_bytes())
    def array(e):
        return np.frombuffer(packed, dtype=e["dtype"], count=e["length"], offset=e["offset"])
    cluster_of = {d:c for c, members in meta["clusters"].items() for d in members}
    designs = []
    for d in range(12):
        g = load_design(ROOT / "Data/HoodImpact_1704_EuroNCAP", d)
        g["cluster"] = cluster_of[d]
        g["triangles"], g["tri_part"] = triangles(g)
        g["normals"] = vertex_normals(g["xyz"], g["triangles"])
        f = meta["families"][meta["designs"][d]["family"]]
        assert np.array_equal(g["node_ids"], array(f["node_ids"]))
        assert np.array_equal(g["triangles"].ravel(), array(f["triangles"]))
        designs.append(g)
        glb = GLB(); glb.add(g); glb.save(out / f"designs/glb/cluster_{g['cluster']}_design_{d:02d}.glb")
        mesh = surface(g)
        mesh.save(out / f"designs/vtp/design_{d:02d}.vtp")
        mesh.triangulate().save(out / f"designs/stl/design_{d:02d}_assembled_mm.stl")
        for code, part in enumerate(PARTS):
            mesh.extract_cells(np.flatnonzero(g["elem_part"] == code)).extract_surface().triangulate().save(
                out / f"designs/stl/design_{d:02d}_{part}_mm.stl")
        print(f"Exported design {d} (cluster {g['cluster']})", flush=True)
    for c, members in meta["clusters"].items():
        glb = GLB(dict(layout="Display offsets only; original component coordinates in mm"))
        for i, d in enumerate(members):
            glb.add(designs[d], offset=(2100*i,0,0))
        glb.save(out / f"clusters/cluster_{c}_all_designs.glb")
    overview = GLB()
    for i, (c, members) in enumerate(meta["clusters"].items()):
        overview.add(designs[members[0]], offset=(2100*(i%2),2100*(i//2),0), label=f"Cluster_{c}_design_{members[0]}")
    overview.save(out / "clusters/four_cluster_representatives.glb")
    manifest = []
    for pair in meta["pairs"]:
        a,b = pair["a"], pair["b"]
        ca,cb = cluster_of[a], cluster_of[b]
        category = "within_clusters" if ca == cb else "between_clusters"
        folder = out / category / (f"cluster_{ca}" if ca == cb else f"clusters_{ca}_{cb}")
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"design_{a:02d}_vs_{b:02d}"
        glb = GLB(dict(design_a=a, design_b=b, method=pair["method"],
                       threshold_mm=.001, color_range_mm=[.001,100], color_scale="log10",
                       legend=dict(gray="<=0.001 mm", blue="0.001 to 100 mm, light to dark"),
                       surface_distance_quantum_mm=.01 if pair["method"] == "surface" else 0))
        for slot, (shown, other) in enumerate(((a,b),(b,a))):
            g = designs[shown]
            if pair["method"] == "exact":
                assert np.array_equal(g["node_ids"], designs[other]["node_ids"])
                delta = designs[other]["xyz"] - g["xyz"]
                values = np.linalg.norm(delta, axis=1)
                values[values <= meta["sparse_tolerance_mm"]] = 0
            else:
                delta = None
                values = array(pair["dist_a"] if shown == a else pair["dist_b"]).astype(float) * meta["distance_quantum_mm"]
            colors = difference_colors(values)
            glb.add(g, colors, offset=(slot*2100,0,0), label=f"Design_{shown:02d}_distance_to_{other:02d}",
                    extras=dict(shown=shown, reference=other, cluster=cluster_of[shown], method=pair["method"]))
            mesh = surface(g)
            mesh.point_data["difference_mm"] = values
            mesh.point_data["difference_rgb"] = colors
            if delta is not None:
                mesh.point_data["displacement_to_reference_mm"] = delta
            mesh.field_data["reference_design"] = np.array([other])
            mesh.field_data["method"] = np.array([pair["method"]])
            mesh.field_data["surface_distance_quantum_mm"] = np.array([.01 if delta is None else 0])
            mesh.set_active_scalars("difference_mm")
            mesh.save(folder / f"{stem}_on_{shown:02d}.vtp")
        glb.save(folder / f"{stem}.glb")
        manifest.append(dict(a=a,b=b,cluster_a=ca,cluster_b=cb,method=pair["method"],
                             glb=(folder / f"{stem}.glb").relative_to(out).as_posix()))
        print(f"Exported {a:02d}–{b:02d}: {category} / {pair['method']}", flush=True)
    make_viewer(out, meta)
    for name in ("pair_summary.csv", "pair_summary.json", "viewer_meta.json"):
        shutil.copyfile(SOURCE / name, out / name)
    for src in (ROOT / "figures/design_mesh_geometries_1704").glob("*.png"):
        shutil.copyfile(src, out / "previews" / src.name)
    (out / "manifest.json").write_text(json.dumps(dict(clusters=meta["clusters"], pairs=manifest), indent=2))
    (out / "README.txt").write_text('''HOOD DESIGN CLUSTERS — 3D EXPORTS

START HERE: open hood_clusters_interactive.html in Chrome or Edge. It works
offline; no server, account or external resources are required. Rotate with
left drag; pan with right drag; zoom with the mouse wheel. Choose a pair in
the matrix or presets. Hide outer skin to expose the inner reinforcement.
Top, underside and isometric cameras are available. Zoom to changes reveals
small local edits. Export buttons save the visible colored geometry as GLB,
or the current viewport as PNG. GLB export bakes colors and visible parts;
point markers, ghost wireframe, interface text and legends are not included.

CLUSTERS (existing project grouping; no new clustering performed)
A: designs 0,1,2,3. B: 4,5. C: 6,7,8,9. D: 10,11.
12 designs; 14 pairs within clusters; 52 pairs between clusters.

FILES
designs/glb/: 12 individual designs, independent outer/inner/hinge objects.
clusters/: four groups with all members side by side, plus four-cluster overview.
within_clusters/: all 14 colored pairs grouped by cluster.
between_clusters/: all 52 colored pairs grouped by cluster combination.
Each comparison GLB shows both directions side by side (lower design ID at
left). Both are measured independently when meshes differ. Hide the outer
objects in your 3D editor to expose local inner-panel changes.
Each pair also has two VTP files, one per measured design, with difference_mm
and original node/element/part IDs. Open in ParaView and color by difference_mm;
use a shared logarithmic range 0.001–100 mm when comparing magnitudes.
designs/vtp/: unmodified meshes with original quadrilateral/triangle shells.
designs/stl/: assembled and individual part surfaces, in mm. STL is geometry
only and does not preserve difference colors, IDs or scalar fields.
previews/: existing static reference images (these are not the 3D exports).
pair_summary.csv/json: existing full-precision summary statistics.
manifest.json: paths, methods, design pairs and cluster assignments.

GEOMETRY AND UNITS
Source: first impact deck per design from Data/HoodImpact_1704_EuroNCAP.
Original hood shell geometry: outer skin, inner panel, hinges. Headform
excluded by element sets. No smoothing, decimation or vertical exaggeration.
Original quadrilaterals are split into two triangles in GLB and STL.
VTP/STL: original absolute XYZ, Z up, mm. Shells have no modeled thickness.
GLB: standard metres and Y up via a single explicit root rotation/scale.
Designs remain in original source coordinates beneath that root. Group and
pair GLBs use 2100 mm display offsets; offsets are separate node transforms.
The viewer's Save displayed 3D exports centered geometry; its center_mm
metadata records the source origin offset. GLB positions use float32.

DIFFERENCES AND COLORS
Designs 0–5 share one mesh; 6–11 share another. Same-family comparisons use
exact matching node IDs and original deck displacement. Values <=0.0001 mm
are suppressed as coordinate-rounding noise. VTP also retains the original
displacement vector (reference minus shown). For different mesh families,
values are absolute closest-surface distance, matched by part, from the
existing viewer data; quantization is 0.01 mm (rounding error up to 0.005 mm).
These surface distances include a discretization floor and are directional;
they are not displacement vectors. Original summaries retain full precision
and can differ slightly from the quantized fields. This also explains why
thresholds below 0.01 mm cannot resolve cross-mesh changes.
Pair GLB colors match the viewer's light theme: grey <=0.001 mm; light blue
to dark blue on a fixed log10 scale of 0.001–100 mm, saturating above 100 mm.
Named objects preserve part visibility. Scalar data are stored in VTP;
GLB stores baked vertex colors and method metadata.

PROVENANCE
The shared Claude link returned its page shell but its content endpoint
returned HTTP 403. Exports were reconstructed from the local project meshes
and existing comparison viewer; exact identity with the linked artifact
could not be verified. No private Claude session data were accessed.
Three.js r147 and associated controls/exporter are included under MIT;
see vendor/THREE-LICENSE.txt. All resources needed by the HTML are embedded.
''', encoding="utf-8")
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "checksums_sha256.json")
    checksums = {p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out / "checksums_sha256.json").write_text(json.dumps(checksums, indent=2))
    if not args.no_zip:
        zip_path = out.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    z.write(p, Path(out.name) / p.relative_to(out))
        print(f"Package: {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB)", flush=True)
    print(f"Viewer: {out / 'hood_clusters_interactive.html'}", flush=True)


if __name__ == "__main__":
    main()
