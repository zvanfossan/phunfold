# phunfold — phonon band unfolding for any commensurate supercell

phunfold takes the phonons of a large supercell and **unfolds** them onto the
Brillouin zone of a small reference cell. Each supercell mode gets a weight
for how much it looks like a reference-cell Bloch wave at k. The result is an
effective band structure: bulk-like branches stay sharp, and anything caused
by broken translational symmetry (domain walls, interfaces, defects, disorder)
shows up as extra or smeared spectral weight.

phunfold builds the POSCAR files for each step, then reads phonopy's output
and does the unfolding and plotting. VASP runs with **your own**
INCAR/KPOINTS/POTCAR/job script. phonopy generates the displacements and
collects the forces. The unfolding engine itself is pure NumPy and includes
the LO–TO (non-analytical) correction for polar materials.

```
phunfold perovskite   cubic or polar/tetragonal ABX3 starting POSCAR
phunfold build-dw     180° domain-wall supercell POSCAR from a relaxed polar cell
phunfold supercell    any cell × integer matrix (test cells, rattled cells)
phunfold disp-dirs    perfect/ + disp-NNN/ folders after `phonopy -d`, with your input files copied in
phunfold check        verify the whole chain in a bulk phonon folder (PASS/FAIL)
phunfold unfold       unfold a supercell onto a reference cell and plot
phunfold plot         re-plot a saved result
phunfold presets      list k-path presets
```

## Install

```bash
git clone git@github.com:zvanfossan/phunfold.git
cd phunfold
python -m pip install --user -e .     # or into a conda/venv environment
python tests/test_unfold.py && python tests/test_general.py && python tests/test_nac.py
```

phonopy has to be installed in the same Python (`python -m pip install --user
phonopy`). On Python 3.9, pip installs phonopy 2.x, which uses the command
names shown below; for phonopy ≥ 4 see the table at the end of this section.

---

## Workflow: BaTiO3, from bulk to a domain wall

The layout below is a suggestion. Each `relax`, `born` and `phonon` folder is
an ordinary VASP or phonopy folder that you fill with your own inputs.

```
BTO/
  inputs/           INCAR_relax, INCAR_static, INCAR_born, KPOINTS_*, POTCAR, submit.sh
  cubic/  relax/  born/  phonon/
  tet/    relax/  born/  phonon/
  dw/     relax/  phonon/
```

Settings that matter for phonons (use the same functional, ENCUT and PAW set
everywhere): PBEsol (`GGA = PS`) for BaTiO3, `EDIFF = 1E-8`,
`PREC = Accurate`, `LREAL = .FALSE.`, `ADDGRID = .TRUE.`. Relaxations need
tight forces (`EDIFFG = -1E-3` or tighter). Static force runs use
`IBRION = -1`, `NSW = 0`. The Born run uses `LEPSILON = .TRUE.`,
`IBRION = 8`, without `NCORE`/`NPAR`. Scale the k-mesh inversely with cell
size.

### 1. Cubic BaTiO3: structure, relaxation, Born charges

```bash
cd BTO/cubic
phunfold perovskite --a 4.00 --out relax/POSCAR
# relax/: run VASP (ISIF = 3), rerun once from CONTCAR
cp relax/CONTCAR born/POSCAR          # born/: run the LEPSILON calculation
```

### 2. Cubic phonons

```bash
cd phonon
cp ../relax/CONTCAR POSCAR
phonopy -d --dim 2 2 2 --pa P -c POSCAR          # SPOSCAR, POSCAR-001 ...
phunfold disp-dirs --copy ../../inputs/INCAR_static:INCAR ../../inputs/KPOINTS_222:KPOINTS \
                          ../../inputs/POTCAR ../../inputs/submit.sh
#   SOURCE:NAME copies a file under a new name; plain paths keep their name
for d in disp-*; do (cd $d && sbatch submit.sh); done
# when all are finished:
phonopy -f disp-*/vasprun.xml                    # -> FORCE_SETS
```

### 3. Check the whole chain (no extra DFT)

```bash
phunfold check --born ../born/vasprun.xml
```

`check` treats phonopy's 40-atom SPOSCAR as a supercell and unfolds it onto
your 5-atom POSCAR. It then compares the result with the 5-atom dispersion
computed from the same force constants. The two must agree exactly, and
`check` prints **PASS** or **FAIL**. `check.png` shows the 120 folded supercell
branches next to the 15 unfolded ones. Run it first in every bulk phonon
folder: it tests the files, the atom mapping, the k-path and the LO–TO
correction together.

* `FORCE_CONSTANTS` is written automatically from `FORCE_SETS` (phunfold calls
  `phonopy-load --writefc --full-fc`) when it is missing.
* The k-path is picked from the lattice (`cubic` here); change it with
  `--path`.
* `--born` takes the `vasprun.xml` (or `OUTCAR`) of the LEPSILON run directly.
  No BORN file is needed.

**What to expect for cubic BaTiO3:** imaginary (negative) branches at Γ and
along Γ–X–M. That's the ferroelectric instability of the cubic phase. R is
stable, and the LO mode near 650 cm⁻¹ sits at Γ. A 2×2×2 supercell is a first
pass; converge the force constants with 3×3×3 or 4×4×4.

phonopy's own plot, for comparison (it needs a BORN file:
`phonopy-vasp-born ../born/vasprun.xml > BORN`):

```bash
env MPLBACKEND=Agg phonopy-load --band "0 0 0  0 1/2 0  1/2 1/2 0  0 0 0  1/2 1/2 1/2  0 1/2 0" \
    --band-labels '$\Gamma$ X M $\Gamma$ R X' -p -s
```

### 4. Tetragonal P4mm, the reference cell for the domain wall

```bash
cd BTO/tet
phunfold perovskite --a 3.99 --c 4.03 --polar-axis c \
                    --disp B=0.10 X_ap=-0.08 X_eq=-0.06 --out relax/POSCAR
# relax/ (ISIF = 3, keep symmetry; rerun from CONTCAR), then born/ as in step 1
cd phonon && cp ../relax/CONTCAR POSCAR
phonopy -d --dim 2 2 2 --pa P -c POSCAR          # (3 3 3 for production)
phunfold disp-dirs --copy ...                    # as in step 2, run, then
phonopy -f disp-*/vasprun.xml
phunfold check --born ../born/vasprun.xml        # path: 'tetragonal'
```

Check that c/a is about 1.01 and Ti sits about 0.1 Å off-centre. Some soft
or slightly imaginary in-plane (E) modes are normal: at 0 K, DFT tetragonal
BaTiO3 is itself unstable toward polarization rotation.

### 5. The 180° domain wall

```bash
cd BTO/dw
phunfold build-dw --bulk ../tet/relax/CONTCAR --ncells 4 --normal a --offset 0
#   -> POSCAR_DW (40 atoms, 8 cells along a) and dw_info.json
cp POSCAR_DW relax/POSCAR      # relax/: ions only (ISIF = 2), forces < ~5e-4 eV/A
cd phonon
cp ../relax/CONTCAR POSCAR
phonopy -d --dim 1 2 2 --pa P -c POSCAR
phunfold disp-dirs --copy ...                    # run all, then
phonopy -f disp-*/vasprun.xml
phunfold unfold --reference ../../tet/relax/CONTCAR \
                --born ../../tet/born/vasprun.xml \
                --bulk-dir ../../tet/phonon --dw-info ../dw_info.json \
                --out dw_unfolded
```

* `--reference` is the cell to unfold onto, the relaxed tetragonal 5-atom
  cell.
* `--born` should come from the **tetragonal** LEPSILON run. Its charges are
  copied onto every wall atom by sublattice.
* `--bulk-dir` draws the tetragonal bulk bands, dashed, with the same NAC.
* `--dw-info` adds a `wall` projection, and selects the `dw-tetragonal` path
  automatically.

Build-dw options:
* `--offset 0` gives the Ba-centred wall (the low-energy one in BaTiO3), and
  `--offset 0.5` the Ti-centred one.
* `--ncells` sets the wall separation.
* `--lateral` repeats the cell in the wall plane.

The code refuses P ∥ wall normal (a charged head-to-head wall) unless you pass
`--allow-charged`.

**Reading `dw_unfolded.png`:**
- **Left panel:** the unfolded spectral function. Bulk-like modes follow the
  dashed tetragonal bands with weight ≈ 1.
- **Right panel:** each mode coloured by its fraction on atoms within 2.5 Å of
  a wall (`--wall-width`).
- **Wall-localised modes** are dispersionless along Γ–X (X is normal to the
  wall), carry reduced, smeared weight, and are bright in the right panel.
  They can disperse along Γ–Y, which lies in the wall plane.

Converge the wall-mode frequencies with `--ncells 6` and `--dim 1 3 3`.

### Commands for phonopy ≥ 4

phonopy 4.0 (May 2026, Python ≥ 3.11) moved the setup steps to `phonopy-init`:

| phonopy 2.x / 3.x (above) | phonopy ≥ 4 |
|---|---|
| `phonopy -d --dim … --pa P -c POSCAR` | `phonopy-init -d --dim … --pa P -c POSCAR` |
| `phonopy -f disp-*/vasprun.xml` | `phonopy-init -f disp-*/vasprun.xml` |
| `phonopy-load --band …` | `phonopy --band …` |

phunfold works with both. Its automatic FORCE_CONSTANTS step tries
`phonopy-load` first and then `phonopy`.

---

## Beyond domain walls: unfolding any supercell

Same pattern: displacements on the supercell with phonopy, forces from any
code phonopy supports, then

```bash
phunfold unfold --reference POSCAR_prim [options]
```

with `POSCAR`, `SPOSCAR` and `FORCE_SETS`/`FORCE_CONSTANTS` in the current
folder.

| Supercell differs from the reference by… | Handled by |
|---|---|
| Any integer supercell matrix M (non-diagonal, rotated basis) | automatic, or `--supercell-matrix` |
| Rigid rotation and/or origin shift | lattice-metric search + origin search, automatic |
| Strain up to `--strain-tol` (default 5 %) | reference lattice rebuilt as inv(M)·L_sc |
| Atomic displacements up to `--max-displacement` (default 0.8 Å) | displacement phase, exact for periodic distortions |
| Substitutions | `--species-map Sr=Ba,Zr=Ti` or `--any-species` |
| Vacancies / interstitials | `--allow-vacancies` / `--allow-interstitials` |
| No clean reference cell (alloys, SQS) | `--reference auto --supercell-matrix "4 4 4"` |

```bash
# substitution + vacancy, flag modes localised around the defect, split by species
phunfold unfold --reference POSCAR_prim --species-map Sr=Ba --allow-vacancies \
    --project defect=sphere:0.5,0.5,0.5:3.0 Sr=species:Sr --resolve species

# interface or stacking fault at fractional 0 and 0.5 along lattice vector 0
phunfold unfold --reference POSCAR_prim --project interface=slab:0:0,0.5:2.5

# custom path, in supercell coordinates; isotopes; very large cells; QE units
phunfold unfold --reference POSCAR_prim --path "G 0 0 0, X 1/2 0 0" --path-frame supercell
phunfold unfold --reference POSCAR_prim --mass O=17.999 --fc-threshold 1e-4 --fc-units Ry/bohr^2
```

Projection specs: `species:Ti,O`, `atoms:0-19`, `sites:0`,
`slab:AXIS:F1,F2:WIDTH`, `sphere:F1,F2,F3:RADIUS`, `interstitials`,
`substituted`. Path presets (`phunfold presets`): `cubic`, `tetragonal`,
`orthorhombic`, `hexagonal`, `fcc`, `bcc`, `dw-tetragonal`. `--path auto`, the
default, picks one from the reference lattice.

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

## Python API

```python
from phunfold import NativeBackend, read_structure, run_unfolding
from phunfold.plot import plot_unfolded
from phunfold.nac import read_born

born, eps = read_born("../born/vasprun.xml")               # for this cell's atoms
be  = NativeBackend("POSCAR", "SPOSCAR", "FORCE_CONSTANTS",
                    nac=dict(born=born, epsilon=eps))
res = run_unfolding(be, read_structure("POSCAR_prim"), path="cubic",
                    mapping_options=dict(species_map={"Sr": "Ba"}, allow_vacancies=True),
                    projections={"defect": "sphere:0.5,0.5,0.5:3.0"}, resolve="species")
plot_unfolded(res, style="scatter", color_by="defect", units="cm-1")
```

Structures: `phunfold.builders.perovskite`, `phunfold.domainwall.build_180_wall`,
`phunfold.make_supercell`, `phunfold.builders.make_disp_dirs`. Your own
eigen-solver plugs in through `phunfold.CallableBackend(structure, fn)`.

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

`tests/test_general.py` also checks the POSCAR builders (cubic and polar
perovskite → domain wall) and `disp-dirs`. The `check` command applies the
same exact-agreement test to your own data.

* `examples/toy_dw_demo.py` runs the complete file-based CLI path (build-dw →
  SPOSCAR/FORCE_CONSTANTS → unfold → plot) on a wall with softened Ti–O bonds
  and recovers bulk-like branches plus flat wall-localised modes.
* `examples/toy_defect_demo.py` runs a 3×3×3 cell with an Sr substitution and
  an O vacancy, rotated 30° and shifted relative to the reference file, through
  the CLI. The orientation search, vacancy, species map, defect projection and
  species-resolved panels all work, and the Sr resonance mode stands out.

On real data: for cubic BaTiO3 (VASP, PBEsol, 2×2×2 supercell, LEPSILON Born
charges), the phunfold dispersion with NAC matches phonopy's NAC band
structure, and `phunfold check` passes. Not yet exercised on real output: the
`--backend phonopy` path and reading Born charges from `OUTCAR`
(`vasprun.xml` is the tested route).
