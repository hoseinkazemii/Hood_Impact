# Geometry and response analysis of the 12 hood designs

This is an observational audit of run `20260922_170037_3194307_1704`, not a new
architecture or training run. All 12 designs, including test designs 4/5 and
validation design 11, are included as explicitly requested post-hoc analysis.
No atlas, scaler, checkpoint or training setting is changed.

Open [the interactive pair/location explorer](runs/change_geometry_analysis/geometry_response_explorer.html)
or [the illustrated report](runs/change_geometry_analysis/report.pdf).
The [written findings](runs/change_geometry_analysis/report.md) explain the
architecture, evidence, and limits. Generated artifacts are ignored by Git.

## Findings

The saved model uses 32 fixed training-derived change patches of 256 nodes,
two attention blocks per patch, one pooled token per patch, 16 global and
16 impact-local tokens, three latent mixing blocks and two temporal decoder
blocks (width 128, four heads, 1,632,409 parameters). The saved run used plain
acceleration MSE, not a design-difference loss. Atlas scoring used changes
within training families A and C; the atlas was never refitted for this audit.

Small-support geometry changes coexist with substantial response changes:
32.15% of design-weighted response variance is within families. Location 142
has the greatest within-family spread of the 142 impact locations. Only
0.116–0.578% of structural nodes differ by more than 0.5 mm within a family,
but individual displacements reach 16–24.6 mm. These are localized modifications,
not uniformly sub-millimeter displacements. Shell connectivity matches within
every family; changed nodes above that threshold are on the inner panel.

The saved patches cover 97.1% of changed nodes for test designs 4/5 and
96.7–100% across all sibling pairs. A frozen token-swap check shows that the
change branch carries almost all of the model's existing design difference.
The branch is active, but the resulting differences are much too small,
including in selected training pairs. Missing changed nodes therefore does
not explain this run's failure by itself.

Spatial associations support a partial relationship: the mean within-pair
rank correlation between nearby changed-node fraction and response difference
is 0.38 at 100 mm and 0.46 at 200 mm, falling to 0.30 and 0.32 after removing
shared location patterns and pair offsets. Whole-hood geometry distance does
not order within-family response differences well at location 142. These
descriptive findings support geometry/impact interaction, not a guarantee that
attention to the changed nodes will learn the simulation response.

## Reproduce

From the repository root, using the existing Python environment:

```bash
python analysis_change_responses.py --run-dir runs/mesh_change_attention/20260922_170037_3194307_1704
python analysis_change_geometry.py
python analyze_change_geometry_response.py
```

The commands select the run above and default to `Data/HoodImpact_1704_EuroNCAP`.
Use `--help` for path overrides. The integration script consumes the first
two scripts' outputs, then writes matched tables, association summaries,
figures, and the offline HTML explorer. It never trains a network.

The audited frozen-model intervention scripts and their outputs are preserved
in `runs/change_geometry_analysis/run_audit/`. They strictly reload the saved
checkpoint, use actual impact decks, and verify original test predictions
before swapping tokens. Such swaps measure model pathways, not FE causality.

Validation:

```bash
python -m pytest tests/test_analysis_change_geometry.py tests/test_analysis_change_responses.py tests/test_change_geometry_response_analysis.py
```

## Definitions and limits

- History difference: RMS over the same 63 saved times of one design's
  acceleration minus another's, in g. Signed curves use higher design ID minus
  lower design ID; RMS is unsigned.
- Spread: square root of mean population variance across designs and time.
  Total variance is decomposed exactly into design-weighted within-family and
  between-family terms. Spread and pairwise RMS are different statistics.
- Changed-node fraction: symmetric structural nearest-node distances above
  0.5 mm, after excluding 286 headform nodes. Locality is physical XY distance;
  the geometric mismatch itself is in XYZ. Exact node-ID displacement is also
  checked when node IDs and connectivity match.
- Frozen-patch coverage: fraction of changed points present in the union of
  saved 256-node patches for that input design. Coverage is not a guarantee
  that attention assigns meaningful weight to each covered point.
- HIC15: exhaustive admissible windows, acceleration in g and time in seconds.
  Sampled and full-resolution HIC are saved separately; the model comparisons
  use its saved sampled grid.
- The 66 pairs share 12 designs, and impact locations are spatially dependent.
  No independent-pair or independent-location significance claims are made.
  Within-family design-label permutations preserve pair dependence, but are
  exploratory and have little power with only four designs in A and C.
- Whole-design comparisons do not establish the causal importance of individual
  nodes or separate deterministic sensitivity from solver variability. Controlled
  feature perturbations and repeated simulations would address those questions.
