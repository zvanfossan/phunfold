# phunfold — phonon band unfolding for domain-wall supercells (VASP + phonopy)

A self-contained Python workflow that takes a large BaTiO3 supercell with
ferroelectric domain walls, computes its phonons from VASP finite-displacement
forces, and **unfolds** the supercell spectrum onto the 5-atom perovskite
Brillouin zone. Wall-localised vibrations then show up as extra (often flat)
branches with partial spectral weight, which the code can also flag by how much
of each mode lives on the wall atoms.

The unfolding engine is pure NumPy (no phonopy needed at that stage). phonopy is
used only for what it is best at: generating symmetry-reduced displacements and
turning forces into force constants.

```
phunfold/
  phunfold/structure.py   POSCAR I/O, masses, lattice conventions
  phunfold/dynmat.py      dynamical matrix: native engine (FORCE_CONSTANTS) or phonopy wrapper
  phunfold/unfold.py      atom mapping, unfolding weights, k-path, spectral function
  phunfold/domainwall.py  180-degree domain-wall supercell builder
  phunfold/vasp.py        INCAR/KPOINTS/POTCAR per stage, displacement dirs, OUTCAR Born parser
  phunfold/plot.py        spectral-function and fat-band plots
  phunfold/__main__.py    CLI:  phunfold {build-dw, setup, setup-disp, unfold, plot}
  tests/                  exact validation against a toy perovskite force field
  examples/               end-to-end dry run + SLURM array script
```

Install (on the cluster): `pip install -e .`  (add `phonopy` to the same
environment for steps 2 and 5–6).

---

## Theory (what the code computes)

Let the supercell contain N = |det M| reference cells, with
`supercell.lattice = M @ reference.lattice` (lattice vectors as rows). Each
supercell atom s is assigned to a reference sublattice κ(s) and an ideal site
x_s = R_s + τ_κ (nearest site of the same species). For a supercell mode J with
r-gauge eigenvector e_{sα}(K,J) — phonopy's convention, phase
exp(2πi K·(r_t − r_s)) in D(K) — the weight on reference wave vector k is

    W_J(k) = (1/N) Σ_κ Σ_α | Σ_{s∈κ} e_{sα}(K,J) · exp(2πi [K·r_s − k·x_s]) |²

(Allen et al., PRB 87, 085322 (2013); phonon form: Ikeda et al., PRB 95,
024305 (2017), the method behind `phonopy.unfolding`). Two implementation
choices:

* **D is diagonalised at K = k itself** (the supercell dynamical matrix is
  defined for any wave vector), so there is no need to fold k into the
  supercell BZ and find G. The phase reduces to exp(2πi k·(r_s − x_s)): only
  each atom's small displacement from its ideal site enters.
* That displacement phase is **kept** by default. For a perfectly periodic
  crystal whose atoms sit off the ideal sites it makes the weights exactly 0/1;
  `phonopy.unfolding` drops it (`--no-distortion-phase` reproduces that).
  The difference is O(k·δ), about 1 % here.

Checks built into every run: 0 ≤ W ≤ 1 and Σ_J W_J(k) = 3 × (atoms in the
reference cell) = 15, printed as `sum-rule error`.

Plotted quantity: A(k, ω) = Σ_J W_J(k) L(ω − ω_J), with L a Lorentzian of width
`--sigma`. With `dw_info.json` present, each mode also gets
`wall_participation` (fraction of Σ|e|² on atoms within `--wall-width` Å of a
wall) and an inverse participation ratio.

---

## Step-by-step VASP workflow

The example commands use phonopy ≥ 4 syntax. **phonopy 4.0 (May 2026) split
the CLI:** setup steps (`-d`, `--rd`, `-f`, `--fz`) moved to `phonopy-init`.
For phonopy 2.x/3.x, replace `phonopy-init` with `phonopy`, and use
`phonopy-load` where `phonopy` is used for the phonon step.

### Step 0 — Convergence and settings (bulk 5-atom cell)

| Setting | Recommendation for BaTiO3 |
|---|---|
| Functional | **PBEsol** (`GGA = PS`). PBE badly over-estimates tetragonality and the double-well depth. LDA under-binds. Stay consistent across all steps. |
| PAW | `Ba_sv`, `Ti_sv` (or `Ti_pv`), `O` |
| ENCUT | Converge stress/phonons; 550–650 eV is typical. Use the same value everywhere. |
| k-mesh | 8×8×8 Γ-centred for the 5-atom cell (≈ `--kspacing 0.2` Å⁻¹). Scale inversely with cell length for supercells. |
| Electronic | `EDIFF = 1E-8`, `PREC = Accurate`, `LREAL = .FALSE.`, `ADDGRID = .TRUE.`, `LASPH = .TRUE.`, `ISMEAR = 0`, `SIGMA = 0.02` |

`phunfold setup` writes these defaults; override any tag with `--set TAG=VALUE`.

### Step 1 — Relax the bulk polar cell

```bash
phunfold setup --stage relax_bulk --poscar POSCAR_P4mm --dir 01_bulk_relax \
               --potcar-dir $VASP_PP/potpaw_PBE --potcar-variant Ba=Ba_sv Ti=Ti_sv
# run VASP; copy CONTCAR -> POSCAR and run again (removes Pulay stress)
```

Start from tetragonal P4mm with the **polar axis along c** (Ti and O displaced
along z). Keep the standard setting (Ba at the origin). This CONTCAR,
`POSCAR_bulk`, is the reference cell used for both building and unfolding.

### Step 2 — Bulk reference phonons and Born charges (recommended)

```bash
# 2a: bulk phonons, used as the dashed overlay and as a sanity check
mkdir 02_bulk_phonon && cd 02_bulk_phonon && cp ../POSCAR_bulk POSCAR
phonopy-init -d --dim 4 4 4 --pa P -c POSCAR
phunfold setup-disp --potcar-dir ... --potcar-variant Ba=Ba_sv Ti=Ti_sv
#   run every disp-*/ (examples/run_displacements.slurm), then:
phonopy-init -f disp-*/vasprun.xml
phonopy --writefc --full-fc         # -> FORCE_CONSTANTS
# 2b: Born effective charges + eps_inf, for LO-TO splitting (DFPT)
phunfold setup --stage born --poscar POSCAR_bulk --dir 02_born   # IBRION=8, LEPSILON
```

For DFPT (`IBRION = 7/8`, `LEPSILON`), do **not** set `NCORE`/`NPAR`.

### Step 3 — Build the domain-wall supercell

```bash
phunfold build-dw --bulk POSCAR_bulk --ncells 4 --normal a --lateral 1 1 --offset 0
# -> POSCAR_DW (2 x 4 = 8 cells along a, 40 atoms) and dw_info.json
```

* This builds neutral **180° walls**: P ∥ c, wall normal along a. The code
  refuses P ∥ normal, which gives charged head-to-head walls, unless you pass
  `--allow-charged`.
* `--offset 0` puts the wall on the BaO plane (Ba-centred, the low-energy
  180° wall in BaTiO3). `--offset 0.5` puts it on the TiO2 plane, which is
  worth computing to compare energies.
* `--ncells` (cells per domain) sets the wall–wall separation. Converge it:
  energy, relaxed profile and the localised-mode frequencies should not change
  between, for example, 4 and 6.
* `--width w` gives a tanh starting profile; 0 (abrupt) is fine because the
  relaxation sets the real profile.
* **Other walls** (90° {101} walls, inclined walls, cells from MD or other
  tools): the unfolding does not need `build-dw`. Any supercell works if its
  lattice is an integer combination of the reference lattice (non-diagonal M is
  supported and tested) and every atom lies within `max_displacement`
  (default 0.8 Å) of an ideal site. Pass `--supercell-matrix` if
  auto-detection fails. For 90° walls, pick a reference cell and orientation
  that are commensurate with both domains (for example a pseudocubic cell) and
  accept that the weights then mix two tetragonal orientations.

### Step 4 — Relax the domain-wall supercell (ions only)

```bash
phunfold setup --stage relax_dw --poscar POSCAR_DW --dir 03_dw_relax --set NCORE=8 ...
```

* `ISIF = 2`: the lattice stays at the bulk tetragonal value, which is correct
  for 180° walls because both domains have the same strain state.
* **Tight forces: `EDIFFG = -1E-3` eV/Å or tighter.** Residual forces in a
  shallow double-well system produce spurious imaginary branches. Check that
  the final forces in OUTCAR are below about 5×10⁻⁴ eV/Å.
* Check that the wall is still centred where you put it, and that Ti
  displacements flip sign across it. Keep `dw_info.json` next to the relaxed
  cell. It stores the wall planes in fractional coordinates; update it if a
  wall migrated.

Save the result: `cp 03_dw_relax/CONTCAR POSCAR_DW_relaxed`.

### Step 5 — Finite displacements on the DW cell

```bash
mkdir 04_dw_phonon && cd 04_dw_phonon
cp ../POSCAR_DW_relaxed POSCAR && cp ../dw_info.json .
phonopy-init -d --dim 1 2 2 --pa P -c POSCAR      # -> SPOSCAR, POSCAR-001...
phunfold setup-disp --kspacing 0.2 --potcar-dir ... --potcar-variant Ba=Ba_sv Ti=Ti_sv
N=$(ls -d disp-* | wc -l); sbatch --array=0-$N ../examples/run_displacements.slurm
```

* **`--dim`:** the DW cell is already long along the normal (≥ 32 Å), but only
  one cell (≈ 4 Å) wide laterally. Use at least ×2 laterally (≈ 8 Å);
  ×3 (≈ 12 Å) is safer for BaTiO3's long-range polar force constants.
  Compare against the bulk 4×4×4 run to judge convergence.
* **Single-point INCAR** (`single_point` stage): `IBRION = -1`, `NSW = 0`,
  `EDIFF = 1E-8`, `ISYM = 0`, `LREAL = .FALSE.`, `ADDGRID = .TRUE.`. To save
  SCF steps, copy the perfect-cell WAVECAR into each directory and set
  `ISTART = 1`.
* `perfect/` is the undisplaced SPOSCAR. Its residual forces are subtracted with
  `--fz` (step 6). Plus/minus displacements, phonopy's default where symmetry
  requires them, also cancel most of the residual-force and anharmonic error.
* **Cost-saving alternative for large, low-symmetry cells:** use random
  displacements and a fitted force-constant solver instead of the systematic
  set. For example `phonopy-init --rd 150 --amplitude 0.03 --dim 1 2 2 --pa P -c POSCAR`,
  then `--fc-calculator symfc` when building FCs. This typically needs a small
  fraction of the systematic displacement count. Check convergence against the
  number of snapshots.
* An accurately fine-tuned machine-learned potential can supply these forces
  instead of VASP (phonopy works with any force source; `pypolymlp` is built
  in). This makes larger walls and lateral dimensions affordable. Validate it
  against a few DFT displacement sets first.

### Step 6 — Forces → force constants

```bash
phonopy-init --fz perfect/vasprun.xml -f disp-*/vasprun.xml   # -> FORCE_SETS
phonopy --writefc --full-fc                                   # -> FORCE_CONSTANTS
```

(The exact `--fz` argument order is version dependent; see `phonopy-init -h`.
Without `--fz`: `phonopy-init -f disp-*/vasprun.xml`.) Use `--full-fc`; the
native backend reads full or compact FCs, but full is robust to how phonopy
chooses its primitive cell.

### Step 7 — Unfold and plot

```bash
phunfold unfold --unitcell POSCAR --sposcar SPOSCAR --fc FORCE_CONSTANTS \
                --reference ../POSCAR_bulk --dw-info dw_info.json \
                --bulk-dir ../02_bulk_phonon --units cm-1 --out unfolded
```

Outputs: `unfolded.png` (left: A(k,ω) with the bulk bands dashed; right: fat
bands colour-coded by wall participation), `unfolded.npz` (everything, re-plot
with `phunfold plot unfolded.npz --ylim 0 800 --units cm-1`) and `unfolded.dat`
(x, q, ω, W, wall participation per line, for your own plotting).

* **k-path:** the default is
  `Γ–X–M–Γ–Y | Γ–Z–R–A–Z` in reference reduced coordinates.
  **X = (½,0,0) lies along the wall normal and Y = (0,½,0) in the wall plane.
  They are no longer equivalent in the DW cell, so always compare the two.**
  Z is the polar axis. Pass your own path with
  `--path "G 0 0 0, X 1/2 0 0 | ..."`.
* **LO–TO splitting:** the native backend has no non-analytic correction, so
  polar TO/LO near Γ are not split. To include it, use the phonopy backend with
  bulk Born charges tiled onto the DW cell:
  `phunfold unfold --backend phonopy --phonopy-yaml phonopy_disp.yaml --force-sets FORCE_SETS --born-outcar ../02_born/OUTCAR --reference ../POSCAR_bulk`.
  The bulk OUTCAR atom order must match `POSCAR_bulk`.
* **Interpreting the plot:** bulk-like modes follow the dashed bulk bands with
  weight ≈ 1. Wall-localised modes are dispersionless along X, carry a
  reduced, spread-out weight, and show high `wall_participation`. A mode
  localised in one plane has no definite k⊥, so its weight is smeared over the
  whole Γ–X line by construction. Imaginary frequencies are plotted as negative.

### Step 8 — Validate (do this once on your own cluster)

1. `python tests/test_unfold.py`: exact checks (below).
2. **Unfold an unrelaxed, wall-free 8×1×1 bulk supercell** with the same
   `--dim` as step 5. Every weight should be 0 or 1, and the unfolded bands
   should coincide with step 2a. This tests the whole DFT + phonopy pipeline
   on your files.
3. Optionally cross-check against `phonopy.unfolding.Unfolding` with
   `--no-distortion-phase`; the two should agree to numerical precision.

---

## Validation performed

`tests/test_unfold.py` uses an exactly solvable toy perovskite force field
(`tests/toymodel.py`):

* Perfect supercells with diagonal and non-diagonal M (6×1×1, 2×2×2, a
  √2×√2×3 cell rotated 45° about c, a 4×√2×√2 cell rotated about a), with atoms randomly displaced
  0.08 Å from ideal sites: the summed weight of every degenerate group matches
  the primitive multiplicity to 1e-15, and the unfolded frequencies equal the
  primitive dispersion to 1e-6 THz. Without the distortion phase, weights
  deviate from 0/1 by about 0.008, as expected.
* Native dynamical matrix from text FORCE_CONSTANTS + SPOSCAR reproduces the
  exact D(k) to 1e-13 THz, including minimum-image ties (multiplicity up to 8)
  where the force-constant range equals half the supercell.
* `examples/toy_dw_demo.py` runs the complete file-based CLI path (build-dw →
  SPOSCAR/FORCE_CONSTANTS → unfold → plot) on a wall with softened Ti–O bonds
  and recovers bulk-like branches plus flat wall-localised modes.

Not tested here: the phonopy backend and the OUTCAR parser against real VASP
output. They follow phonopy's documented API and VASP's OUTCAR layout, but
check them on your first real run (step 8).

## Python API

```python
from phunfold import NativeBackend, read_poscar, run_unfolding
from phunfold.plot import plot_unfolded

be  = NativeBackend("POSCAR", "SPOSCAR", "FORCE_CONSTANTS")
res = run_unfolding(be, read_poscar("POSCAR_bulk"),
                    "G 0 0 0, X 1/2 0 0, M 1/2 1/2 0, G 0 0 0",
                    wall_axis=0, wall_positions=[0.0, 0.5], wall_width=2.5)
plot_unfolded(res, style="scatter", color_by="wall_participation", units="cm-1")
```
