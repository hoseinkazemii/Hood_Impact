# Regenerating hood-impact acceleration histories with Abaqus

Goal: produce more training samples for the acceleration-history ML model by
running Abaqus on the **12 valid hood designs** we already have, instead of
designing new hoods. This folder contains the scripts to do that, starting with
a **verification step**: reproduce the delivered 60 samples exactly, to prove
the simulation pipeline is correct before generating *new* impact locations.

## The key realization

`Data/HoodImpact_60_IndustryLike/inp_files/HoodImpact_<n>.inp` are **not just
meshes** — each is a *complete, runnable Abaqus/Explicit job*. Each deck already
contains everything:

| Ingredient | Where in the `.inp` |
|---|---|
| Hood mesh (S3R/S4 shells) | `*NODE`, `*ELEMENT, TYPE=S3R/S4` |
| Rigid headform (impactor) | `*RIGID BODY, REF NODE=…, ELSET="Rigid Body1_2-1-3"` |
| Materials (Al hood, steel layer) | `*MATERIAL`, `*ELASTIC`, `*PLASTIC` |
| Shell thicknesses | `*SHELL SECTION` (1.5 / 0.9 / 1.0 mm) |
| Fasteners, couplings, hinges | `*CONNECTOR SECTION`, `*COUPLING` |
| **11.1 m/s impact velocity** | `*INITIAL CONDITIONS, TYPE=VELOCITY` → `…,3,-11100.` |
| Boundary conditions | `*BOUNDARY` (Fixed Displacement 1/2) |
| Contact | `*CONTACT` + surface-to-surface pairs |
| Step (explicit, 25 ms) | `*DYNAMIC, EXPLICIT` → `, 0.025` |
| **Acceleration output** | `*OUTPUT, HISTORY, FREQUENCY=1` / `*NODE OUTPUT, NSET="Nodal Probe12"` / `A` |

So we **do not rebuild any physics**. We solve the existing deck and read the
headform acceleration back out. (Rebuilding a partner's proprietary FE model —
mesh, fasteners, couplings — from scratch to match byte-for-byte is effectively
impossible; reusing their deck is the correct and reliable approach.)

### Units (from the deck header — 3DEXPERIENCE R2025x)
Tonne · millimetre · second ⇒ force = N, stress = MPa, **velocity = mm/s**
(11100 mm/s = 11.1 m/s), **acceleration = mm/s²**.

### Where the acceleration comes from
HIC is computed from the **resultant acceleration of the rigid headform
reference node** (`NSET="Nodal Probe12"`, which is also the Initial-Velocity and
Rigid-Body reference node, and whose coordinates equal the impact location).

⚠️ The reference-node **label is not constant**: it is `44166` for designs 1–30
and `45368` for designs 31–60. The extractor therefore finds the acceleration by
its `A1/A2/A3` history outputs, never by a hardcoded node number.

## Post-processing (reverse-engineered from the delivered files)
The delivered `HoodImpact_<n>_SAE1000_interp1000.csv` are the raw headform
acceleration, then:
1. resample the raw, slightly non-uniform history onto a **uniform fine grid**
   (`dt = 5e-7 s`, finer than the raw spacing — otherwise the sharp initial
   contact spikes alias);
2. **SAE-J211 CFC-1000** zero-phase filter on A1, A2, A3 (“SAE1000”), with
   *constant* edge padding (pre-impact acceleration ≈ 0, so this avoids the
   edge ringing that otherwise injects a spurious initial spike);
3. resample to **1000 points** over `[0, t_final]` (“interp1000”);
4. `A(mm/s2) = sqrt(A1²+A2²+A3²)`, `A(in g) = A(mm/s2) / 9800`.

**Explicit start-up transient.** The first 1–2 increments of an explicit step
with an applied initial velocity into a zero-gap HARD contact are numerically
meaningless (huge values followed by exact zeros in the raw `.csv`). They sit
far above the CFC1000 band, but their net impulse slightly lifts the filtered
onset. `postprocess_acc.py` zeroes a short leading window (`--clip-startup-us`,
default 10 µs) to remove it; `--clip-startup-us 0` gives the purest
reproduction. This affects only the first ~0.15 ms and is HIC-irrelevant.

Verified on run 1: peak **147.34 vs 147.32 g** (0.01 %), HIC15 **1283.0 vs
1283.5** (0.04 %), RMSE **0.25 g** over the whole curve.

Verified against the delivered data: `A(mm/s2)/A(in g) == 9800` (to 1e-7), and
HIC15 = `dt·(a_avg)^2.5` (a in g, 15 ms window) — same formula as
`analyze_inp_designs.py`.

## Scripts
- **`run_hood_impact.py`** — driver: solve a deck on 1 CPU → extract → post-process.
- **`extract_acc_odb.py`** — runs inside Abaqus; pulls raw `Time,A1,A2,A3` from the `.odb`.
- **`postprocess_acc.py`** — plain Python (numpy/scipy); SAE1000 + interp1000 + HIC + compare/plot.

## How to verify (do this first)

From `d:\Papers\Hood_Impact\Code` in PowerShell, with your ML Python env active
(needs numpy/scipy/pandas/matplotlib):

```powershell
python abaqus_scripts\run_hood_impact.py --runs 1 --compare --plot
```

This will:
1. solve `HoodImpact_1.inp` with `abq2025 … cpus=1` in `abaqus_scripts\work\run_1\`,
2. extract the raw headform acceleration,
3. write `HoodImpact_1_SAE1000_interp1000.csv`, compute HIC, and print a
   comparison vs the delivered reference + save `HoodImpact_1_acc_overlay.png`.

**Success criterion:** the overlay curve sits on top of the delivered one, and
the peak-g / HIC differences are small (a few %). That confirms the deck, the
probe node, the units, and the post-processing are all correct.

Re-run only the post-processing without re-solving:
```powershell
python abaqus_scripts\run_hood_impact.py --runs 1 --skip-solve --compare --plot
```

Solve a few designs (sequential, 1 core each):
```powershell
python abaqus_scripts\run_hood_impact.py --runs 1,6,31
```

> A full explicit solve of these ~40k-element decks to 25 ms on one core takes
> roughly tens of minutes to a couple of hours. Verify one run before batching.

## Next steps (after verification passes)
1. **New impact locations (same 12 designs).** A new sample = the same deck with
   the rigid headform moved to a new point on the hood. Mechanically this is a
   rigid translation of *all impactor nodes* (the nodes used by the
   `"Rigid Body1_2-1-3"` element set **plus** the reference node) by
   `new_location − current_location`, positioned just above the hood surface
   along the impact direction. The probe/initial-velocity sets need no change
   (they already point at the reference node). I can add a
   `make_inp_for_new_location.py` that does this and a sampler that scatters,
   say, 50 valid locations per design over the hood.
2. **Parallel processing.** Swap the sequential loop in `run_hood_impact.py` for
   multiple concurrent single-core (or multi-core) jobs — you asked to defer this
   until the single-core pipeline is verified.
