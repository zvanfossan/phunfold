# phunfold — phonon band unfolding for any commensurate supercell

phunfold takes the phonons of a large supercell and **unfolds** them onto the
Brillouin zone of a small reference cell. Each supercell mode gets a weight
for how much it looks like a reference-cell Bloch wave at k. The result is an
effective band structure where bulk-like branches stay sharp and features from
broken translational symmetry show up as extra or smeared spectral weight.
Typical uses:

* **extended defects**: domain walls, interfaces, stacking faults, surfaces/slabs
* **point defects**: vacancies, substitutions, interstitials
* **disorder**: alloys and special quasirandom structures, mass disorder and isotopes
* **distortions**: relaxed, rattled, or ML-MD snapshot cells; octahedral tilts
  (unfold onto the untilted cell)

What the supercell may differ in, relative to the reference:

| Supercell differs from the reference by… | Handled by |
|---|---|
| Any integer supercell matrix M (non-diagonal, rotated basis) | automatic, or `--supercell-matrix` |
| Rigid rotation and/or origin shift | lattice-metric search + origin search, automatic |
| Strain up to `--strain-tol` (default 5 %) | reference lattice rebuilt as inv(M)·L_sc |
| Atomic displacements up to `--max-displacement` (default 0.8 Å) | displacement phase, exact for periodic distortions |
| Substitutions | `--species-map Sr=Ba,Zr=Ti` or `--any-species` |
| Vacancies / interstitials | `--allow-vacancies` / `--allow-interstitials` |
| No clean reference cell (alloys) | `--reference auto`: average structure of the supercell |

Force constants can come from any code phonopy interfaces with (VASP, QE,
ABINIT, CP2K, CASTEP, FHI-aims, ML potentials…) through phonopy's
`FORCE_CONSTANTS`/`force_constants.hdf5`. You can also plug in your own
eigen-solver (`CallableBackend`, `DynamicalMatrixBackend`), with eigenvectors in
the r-gauge or the R-gauge (`gauge="R"`).

The unfolding engine is pure NumPy. phonopy is only needed to generate
displacements and turn forces into force constants.

```
phunfold/
  phunfold/structure.py    POSCAR I/O (other formats via ASE), supercells, masses
  phunfold/lattice.py      supercell-matrix / orientation search, average reference cell
  phunfold/mapping.py      atom -> site mapping (origin, substitutions, vacancies, interstitials)
  phunfold/dynmat.py       dynamical matrix: native (FORCE_CONSTANTS), phonopy, or user callable
  phunfold/unfold.py       unfolding weights (total / sublattice-resolved), pipeline, results
  phunfold/paths.py        k-path presets and coordinate frames
  phunfold/projections.py  atom groups: species, indices, sites, slabs, spheres, defects
  phunfold/plot.py         spectral-function and fat-band plots
  phunfold/domainwall.py   180° domain-wall builder (worked example)
  phunfold/vasp.py         VASP inputs per stage, displacement dirs, OUTCAR Born parser
  phunfold/__main__.py     CLI: phunfold {unfold, plot, supercell, presets, build-dw, setup, setup-disp}
  tests/                   exact validation against a toy perovskite force field
  examples/                end-to-end dry runs (domain wall; rotated defect cell) + SLURM script
```

Install: `pip install -e .` (add `phonopy` to the same environment for
force-constant generation; `ase` and `h5py` are optional).

---

## Quick start: unfolding any supercell

You need three things:

1. **The supercell** whose phonons you want. Its force constants come from
   phonopy's usual route: displacements on the supercell (`--dim` for extra
   repetitions if needed), forces from your code, `FORCE_CONSTANTS` with
   `--full-fc`.
2. **The reference cell** to unfold onto: a small cell with the same lattice
   type and basis. It doesn't need the same orientation, origin, atom order, or
   exact lattice constants.
3. **A k-path** in the reference cell's coordinates: a preset or your own.

```bash
phunfold unfold --unitcell POSCAR --sposcar SPOSCAR --fc FORCE_CONSTANTS \
                --reference POSCAR_prim --path cubic --out unfolded
```

Common variations:

```bash
# substitutional alloy, with Sr and Zr on the Ba and Ti sites; species-resolved spectra
phunfold unfold ... --species-map Sr=Ba,Zr=Ti --resolve species

# vacancy + substitution; flag modes localised around the defect
phunfold unfold ... --allow-vacancies --species-map Sr=Ba \
                    --project defect=sphere:0.5,0.5,0.5:3.0 Sr=species:Sr

# interface / domain wall / stacking fault at fractional 0 and 0.5 along lattice vector 0
phunfold unfold ... --project interface=slab:0:0,0.5:2.5

# no clean reference cell (e.g. SQS): use the average structure of a 4x4x4 cell
phunfold unfold ... --reference auto --supercell-matrix "4 4 4" --any-species

# path in supercell coordinates, or Cartesian (1/Å, reference-file frame)
phunfold unfold ... --path "G 0 0 0, X 1/2 0 0" --path-frame supercell

# polar material: LO-TO splitting from a VASP LEPSILON run (this cell or the reference cell)
phunfold unfold ... --born ../born/vasprun.xml

# isotopes, very large cells, Quantum ESPRESSO force constants
phunfold unfold ... --mass O=17.999 --fc-threshold 1e-4 --fc-units Ry/bohr^2
```

`phunfold presets` lists the built-in paths: `cubic`, `tetragonal`,
`orthorhombic`, `hexagonal` (γ = 120°), `fcc`, `bcc` (Setyawan–Curtarolo
labels), and `dw-tetragonal`. The fcc and bcc presets are defined in Cartesian
cubic-axis units, so they work with any primitive-vector convention as long as
the cubic axes of your reference file lie along x, y, z.

`phunfold supercell --reference POSCAR_prim --matrix "2 2 2" --rattle 0.01`
builds test supercells. The best validation of a new setup is to unfold a
defect-free supercell made this way: every weight must be 0 or 1, on top of
the primitive bands.

### Orientation choices

If the files are already in the same orientation
(`sc.lattice ≈ M · ref.lattice`) and every atom maps validly, that orientation
is always used. Only otherwise does the code search all proper rotations and
re-basings of the reference lattice, keeping the one whose basis fits the atoms
best. It then prints where the reference a, b, c axes ended up. If the
supercell has lower symmetry than the reference (walls, interfaces, uniaxial
strain), equivalent orientations give physically different path directions,
for example X normal to a wall versus X in its plane. The code warns when
that ambiguity exists; fix it with `--supercell-matrix` or by writing both
cells in the same orientation.

### Python API

```python
from phunfold import NativeBackend, read_structure, run_unfolding
from phunfold.plot import plot_unfolded

be  = NativeBackend("POSCAR", "SPOSCAR", "FORCE_CONSTANTS")
res = run_unfolding(be, read_structure("POSCAR_prim"), path="cubic",
                    mapping_options=dict(species_map={"Sr": "Ba"},
                                         allow_vacancies=True),
                    projections={"defect": "sphere:0.5,0.5,0.5:3.0"},
                    resolve="species")
plot_unfolded(res, style="scatter", color_by="defect", units="cm-1")
plot_unfolded(res, style="spectral", weights=res.extra["w_O"])  # O-sublattice part
```

Your own eigenvectors, for example from another code or a model Hamiltonian:

```python
from phunfold import CallableBackend, run_unfolding
backend = CallableBackend(structure, lambda k: my_solver(k))  # -> (freqs, eigvecs)
res = run_unfolding(backend, reference, path="hexagonal", gauge="R")
```

---

## Theory (what the code computes)

Let the supercell contain N = |det M| reference cells, with
`supercell.lattice = M @ reference.lattice` (lattice vectors as rows, after
any rotation). Each supercell atom s is assigned to a reference sublattice
κ(s) and an ideal site x_s = R_s + τ_κ: the nearest site of a compatible
species. For a supercell mode J with r-gauge eigenvector e_{sα}(K,J)
(phonopy's convention: phase exp(2πi K·(r_t − r_s)) in D(K)), the weight on
reference wave vector k is

    W_J(k) = (1/N) Σ_κ Σ_α | Σ_{s∈κ} e_{sα}(K,J) · exp(2πi [K·r_s − k·x_s]) |²

(Allen et al., PRB 87, 085322 (2013); phonon form: Ikeda et al., PRB 95,
024305 (2017), the method behind `phonopy.unfolding`). Implementation choices:

* **D is diagonalised at K = k itself.** The supercell dynamical matrix is
  defined for any wave vector, so k never has to be folded and no G has to be
  searched for. The phase reduces to exp(2πi k·(r_s − x_s)), which only
  involves each atom's small displacement from its ideal site.
* **The displacement phase is kept** by default. For a periodic crystal whose
  atoms sit off the ideal sites, it makes the weights exactly 0/1.
  `phonopy.unfolding` drops it (`--no-distortion-phase` reproduces that); the
  difference is O(k·δ).
* **Sublattice decomposition:** W_J = Σ_κ W_Jκ. `--resolve` stores these per
  sublattice or per reference species. A substituent counts toward the
  sublattice it occupies, so the "Ba" panel includes Sr on Ba sites.
* **Vacancies** are empty sites. **Interstitials** form one-atom ghost
  sublattices, so their weight spreads uniformly over k.

### Non-analytical correction (polar materials)

`--born FILE` (a `vasprun.xml` or `OUTCAR` from a `LEPSILON = .TRUE.` run)
applies the Gonze–Lee dipole–dipole correction, the same method and default
parameters as phonopy's `gonze` NAC:

1. The dipole–dipole part, an Ewald sum over Born charges Z\* screened by ε∞,
   is evaluated at the commensurate q-points of the force-constant supercell
   and subtracted from the force constants.
2. The short-range remainder is interpolated as usual.
3. The dipole–dipole term is added back analytically at every k. At Γ, the
   q → 0 limit is taken along each path segment, which gives the LO–TO
   splitting.

The Born charges may belong to the cell itself or to the reference cell. In
the second case they are copied onto each supercell atom by sublattice (and
rotated along with the cell if needed). That is a good approximation for walls
and dilute defects; for strongly perturbed regions, compute Z\* of the full
cell. The Ewald width and cutoff are taken from the reference cell's volume,
so a supercell and its reference use the identical dipole sum and their
spectra unfold consistently. Points exactly at Γ with no direction get TO
modes only, as in phonopy.

Built-in checks: 0 ≤ W ≤ 1 and Σ_J W_J(k) = 3·n_atoms/N (= 3·n_ref without
vacancies or interstitials). This is printed as `sum rule … holds to …`.

Plotted quantity: A(k, ω) = Σ_J W_J(k) L(ω − ω_J), with L a Lorentzian of width
`--sigma`. Each `--project` group adds a panel of fat bands coloured by the
fraction of |e|² on that group. An inverse participation ratio per mode is
always stored.

---

## Worked example: BaTiO3 domain walls with VASP (step by step)

The example commands use phonopy ≥ 4 syntax. **phonopy 4.0 (May 2026) split
the CLI:** the setup steps moved to `phonopy-init`, and phonopy 4 needs
Python ≥ 3.11. With phonopy 2.x/3.x (for example on a Python 3.9 cluster
install), translate as follows:

| phonopy ≥ 4 (used below) | phonopy 2.x / 3.x |
|---|---|
| `phonopy-init -d --dim … --pa P -c POSCAR` | `phonopy -d --dim … --pa P -c POSCAR` |
| `phonopy-init --rd 150 --amplitude 0.03 …` | `phonopy --rd 150 --amplitude 0.03 …` |
| `phonopy-init -f disp-*/vasprun.xml` | `phonopy -f disp-*/vasprun.xml` |
| `phonopy-init --fz perfect/vasprun.xml -f disp-*/vasprun.xml` | see `phonopy -h` for `--fz` |
| `phonopy --writefc --full-fc` | `phonopy-load --writefc --full-fc` |
| `phonopy --band …` | `phonopy-load --band …` |

All `phunfold` commands are the same for every phonopy version. phunfold
needs `FORCE_CONSTANTS`, which phonopy writes only when asked (`--writefc`).

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
  tools): the unfolding does not need `build-dw`; see *Quick start* for the
  general case (any integer M, rotated/shifted cells, defects). For 90° walls, pick a reference cell and orientation
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
bands colour-coded by the fraction of each mode on the wall atoms, from the `wall` projection that `dw_info.json` adds), `unfolded.npz` (everything, re-plot
with `phunfold plot unfolded.npz --ylim 0 800 --units cm-1`) and `unfolded.dat`
(x, q, ω, W and every projection per line, for your own plotting).

* **k-path:** the default is
  `dw-tetragonal`, i.e. `Γ–X–M–Γ–Y | Γ–Z–R–A–Z` in reference reduced coordinates.
  **X = (½,0,0) lies along the wall normal and Y = (0,½,0) in the wall plane.
  They are no longer equivalent in the DW cell, so always compare the two.**
  Z is the polar axis. Pass your own path with
  `--path "G 0 0 0, X 1/2 0 0 | ..."`.
* **LO–TO splitting:** add `--born ../02_born/vasprun.xml` (or the
  `OUTCAR` of that run). The bulk Born charges and ε∞ are tiled onto the DW
  cell by sublattice, and the Gonze–Lee dipole–dipole correction is applied
  (see *Non-analytical correction* above). The atom order of the LEPSILON run
  must match `POSCAR_bulk`. With `--bulk-dir`, the dashed bulk overlay gets the
  same correction.
* **Interpreting the plot:** bulk-like modes follow the dashed bulk bands with
  weight ≈ 1. Wall-localised modes are dispersionless along X, carry a
  reduced, spread-out weight, and a large fraction on the `wall` projection. A mode
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
`tests/test_general.py` covers the general-supercell features:

* Supercells randomly rotated and origin-shifted relative to the reference,
  with a polar basis and anisotropic force constants so that a wrong
  orientation would give wrong frequencies. The search recovers an orientation
  that reproduces the primitive dispersion exactly.
* Substitution (Sr on Ba) via `species_map` and via `match_species=False`;
  O vacancy (sum rule 3·39/8); H interstitial (sum rule 3·41/8).
* Reference cell built from the supercell average gives exact unfolding.
  R-gauge eigenvectors give the same weights as r-gauge ones.
  Sublattice-resolved weights sum to the total.
* Projections (species, slab, sphere) and species resolution through
  `run_unfolding`; fcc preset coordinates; supercell path frame.
* Force-constant pruning (same frequencies from about 12 % of the pairs) and
  mass overrides.

`tests/test_nac.py` covers the non-analytical correction:

* At the commensurate q-points of the force-constant supercell, NAC returns
  the raw force-constant D(q) to 5·10⁻⁷ eV/Å². The removed and re-added dipole
  parts cancel, which checks the phase conventions. The 10⁻⁷ comes from the
  G-sphere truncation that phonopy uses as well.
* The q → 0 jump equals 4πe²/V (d·Z)(d·Z)/(d·ε∞·d) exactly and matches the
  small-q limit.
* A 40-atom supercell with tiled Born charges unfolds onto the 5-atom NAC bands
  to 2·10⁻¹⁵, including LO–TO at Γ.
* Born charges and ε∞ are read correctly from `vasprun.xml`.

* `examples/toy_dw_demo.py` runs the complete file-based CLI path (build-dw →
  SPOSCAR/FORCE_CONSTANTS → unfold → plot) on a wall with softened Ti–O bonds
  and recovers bulk-like branches plus flat wall-localised modes.
* `examples/toy_defect_demo.py` runs a 3×3×3 cell with an Sr substitution and
  an O vacancy, rotated 30° and shifted relative to the reference file, through
  the CLI. The orientation search, vacancy, species map, defect projection and
  species-resolved panels all work, and the Sr resonance mode stands out.

Not tested here: the phonopy backend and the OUTCAR parser against real VASP
output. They follow phonopy's documented API and VASP's OUTCAR layout, but
check them on your first real run (step 8).
