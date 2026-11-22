"""Solve the pilot decks with Abaqus (1 core per job) and extract each ODB to <name>_results.npz.

    python run_pilot.py B1 B2 B3 B4 [--root D:\\Papers\\Hood_Impact\\LocalEditBenchmarks\\pilot]
                        [--lanes 3] [--designs base e1_deepen] [--drop-odb]

Each design solves in its own folder. abq2025.bat always exits 0, so the
solver status comes from the .sta file: "completed", or "aborted" (the burst
runs in B2 may stop past the pressure peak; the extractor still reads every
saved frame). <name>_run.json records the status and wall time. Finished
designs (results file present) are skipped, so the command can be re-run.

License: --abaqus-env copies an abaqus_v6.env into every job folder (Abaqus
reads it from the working directory, ahead of the site settings). The default
is the FlexNet research-license file; the license server must be reachable
(UIUC VPN off campus), which is checked before any job starts.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path

ABAQUS = r"C:\SIMULIA\Commands\abq2025.bat"
EXTRACTOR = Path(__file__).resolve().with_name("extract_odb.py")
FOLDERS = {"B1": "B1_plate", "B2": "B2_elbow", "B3": "B3_heatsink", "B4": "B4_cantilever"}
SCRATCH = (".com", ".prt", ".sim", ".mdl", ".stt", ".res", ".abq", ".pac", ".sel", ".023", ".lck")
DEFAULT_ROOT = Path(r"D:\Papers\Hood_Impact\LocalEditBenchmarks\pilot")
DEFAULT_ENV = Path(r"D:\Papers\Hood_Impact\abaqus_license\abaqus_v6.env")


def license_reachable(env_file):
    """True if the FlexNet server named in the env file accepts a TCP connection (or none is named)."""
    match = re.search(r'abaquslm_license_file\s*=\s*["\']?(\d+)@([\w.\-]+)', Path(env_file).read_text())
    if not match:
        return True
    try:
        socket.create_connection((match.group(2), int(match.group(1))), timeout=5).close()
        return True
    except OSError:
        return False


def solver_status(sta):
    text = sta.read_text(errors="ignore") if sta.exists() else ""
    if "THE ANALYSIS HAS COMPLETED SUCCESSFULLY" in text:
        return "completed"
    if "THE ANALYSIS HAS NOT BEEN COMPLETED" in text:
        return "aborted"
    return "unknown"


def run(folder, abaqus, drop_odb, env_file=None):
    name = folder.name
    results = folder / f"{name}_results.npz"
    if results.exists():
        return name, "skipped (results exist)", 0.0
    if env_file:
        shutil.copyfile(env_file, folder / "abaqus_v6.env")
    start = time.time()
    with open(folder / f"{name}_abaqus.log", "w") as log:
        subprocess.run([abaqus, f"job={name}", f"input={name}.inp", "cpus=1", "ask_delete=OFF", "interactive"],
                       cwd=folder, stdout=log, stderr=subprocess.STDOUT)
        status = solver_status(folder / f"{name}.sta")
        solve_seconds = time.time() - start
        odb = folder / f"{name}.odb"
        if odb.exists() and (folder / f"{name}.sta").exists():  # no .sta: the input processor failed
            subprocess.run([abaqus, "python", str(EXTRACTOR), "--", str(odb), str(results), status],
                           cwd=folder, stdout=log, stderr=subprocess.STDOUT)
    for suffix in SCRATCH + ((".odb",) if drop_odb and results.exists() else ()):
        path = folder / f"{name}{suffix}"
        if path.exists():
            path.unlink()
    record = {"status": status, "solve_seconds": round(solve_seconds, 1), "results": results.exists()}
    (folder / f"{name}_run.json").write_text(json.dumps(record, indent=2))
    return name, status, solve_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmarks", nargs="+", choices=sorted(FOLDERS))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--lanes", type=int, default=3, help="concurrent single-core jobs")
    parser.add_argument("--designs", nargs="*", help="only these design names")
    parser.add_argument("--abaqus", default=ABAQUS)
    parser.add_argument("--drop-odb", action="store_true", help="delete the .odb after a successful extraction")
    parser.add_argument("--abaqus-env", default=str(DEFAULT_ENV),
                        help="abaqus_v6.env copied into each job folder ('none' = use the site settings)")
    args = parser.parse_args()
    env_file = None if args.abaqus_env.lower() == "none" else Path(args.abaqus_env)
    if env_file and not license_reachable(env_file):
        raise SystemExit(f"license server in {env_file} is not reachable (UIUC VPN connected?)")
    folders = []
    for key in args.benchmarks:  # listed order; put the slow B2 first for better packing
        folders += sorted(p.parent for p in (args.root / FOLDERS[key]).glob("*/*.inp")
                          if not args.designs or p.parent.name in args.designs)
    print(f"{len(folders)} jobs, {args.lanes} lanes")
    with cf.ThreadPoolExecutor(max_workers=args.lanes) as pool:
        futures = {pool.submit(run, folder, args.abaqus, args.drop_odb, env_file): folder for folder in folders}
        for future in cf.as_completed(futures):
            name, status, seconds = future.result()
            print(f"  {futures[future].parent.name}/{name}: {status} ({seconds / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
