"""Save physical-XYZ kNN lists for the 12 fixed 1704 structural meshes."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mesh_neighborhood import NeighborGraphCache
from utils.utils import DataPreprocessor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="Data/HoodImpact_1704_EuroNCAP")
    parser.add_argument("--cache-dir")
    parser.add_argument("--k", type=int, default=256, help="Total neighbors, including the center when selected at zero distance.")
    parser.add_argument("--impactor-nodes", type=int, default=286)
    parser.add_argument("--designs", type=int, nargs="+", default=list(range(12)), help="Zero-based design IDs.")
    args = parser.parse_args(argv)
    if args.k < 1 or args.impactor_nodes < 0 or any(design not in range(12) for design in args.designs):
        parser.error("Require k > 0, impactor-nodes >= 0, and design IDs in 0..11")
    root = Path(args.data_root).expanduser().resolve()
    cache = NeighborGraphCache(directory=args.cache_dir or root / "neighbor_graphs")
    loader = DataPreprocessor(SimpleNamespace(data_format="euroncap1704", inp_dir=str(root / "inp_files")))
    designs = []
    for design in sorted(set(args.designs)):
        first_run, last_run = 142 * design + 1, 142 * (design + 1)
        mesh = loader.load_mesh_geometry(first_run)
        last = loader.load_mesh_geometry(last_run)
        structure = mesh[args.impactor_nodes:]
        if len(structure) < 2:
            raise ValueError(f"Design {design} has fewer than two structural nodes")
        if not np.array_equal(structure, last[args.impactor_nodes:]):
            raise ValueError(f"Design {design}: structural XYZ/order differs between runs {first_run} and {last_run}; "
                             "a single design graph cannot represent both. Check the headform boundary.")
        graph = cache.neighbors(torch.from_numpy(structure), args.k)
        path = cache.path_for(structure, args.k)
        designs.append({"design_id": design, "reference_run": first_run, "checked_last_run": last_run,
                        "total_nodes": len(mesh), "structural_nodes": len(structure),
                        "shape": list(graph.shape), "file": path.name, "bytes": path.stat().st_size,
                        "geometry_digest": cache.geometry_digest(structure)})
        print(f"Design {design:2d}: {tuple(graph.shape)} | built={cache.builds} | disk loads={cache.disk_hits}", flush=True)
    manifest = {"algorithm": cache.ALGORITHM, "requested_k": args.k,
                "impactor_nodes_excluded": args.impactor_nodes, "data_root": str(root),
                "directory": str(cache.directory.resolve()), "indices": "zero-based rows of the structural slice; add impactor_nodes for full-mesh rows",
                "storage_dtype": "int32", "runtime_dtype": "torch.int64", "designs": designs,
                "built": cache.builds, "loaded_from_disk": cache.disk_hits,
                "unique_files": len({item["file"] for item in designs})}
    cache.directory.mkdir(parents=True, exist_ok=True)
    (cache.directory / f"manifest_k{args.k}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {manifest['unique_files']} graphs to {cache.directory.resolve()}", flush=True)
    return manifest


if __name__ == "__main__":
    main()
