"""phunfold <command> [options]      (or: python -m phunfold ...)

Structures (POSCAR files for each step; run VASP with your own inputs)
  perovskite  cubic or polar/tetragonal ABX3 starting cell
  build-dw    180-degree domain-wall supercell from a relaxed polar bulk cell
  supercell   tile any cell with an integer matrix (test cells, rattled cells)
  disp-dirs   one folder per phonopy displacement, your own input files copied in

Analysis
  check       verify the pipeline in a bulk phonon folder: unfold phonopy's
              SPOSCAR onto POSCAR and compare (folded vs unfolded, PASS/FAIL)
  unfold      unfold the phonons of any supercell onto a reference cell
  plot        re-plot a saved .npz
  presets     list the k-path presets

Run any command with -h for its options.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np


# ---------------------------------------------------------------- parsing
def _parse_matrix(text):
    """'2 0 0 0 2 0 0 0 2', '2 2 2' (diagonal), or JSON nested list."""
    if text is None:
        return None
    text = text.strip()
    if text.startswith("["):
        return np.array(json.loads(text), dtype=int)
    vals = [int(v) for v in text.replace(",", " ").split()]
    if len(vals) == 3:
        return np.diag(vals)
    if len(vals) == 9:
        return np.array(vals).reshape(3, 3)
    raise SystemExit("matrix needs 3 (diagonal) or 9 integers")


def _parse_species_map(text):
    """'Sr=Ba,Zr=Ti+Hf' -> {'Sr': 'Ba', 'Zr': ['Ti', 'Hf']}"""
    if not text:
        return None
    out = {}
    for item in text.split(","):
        k, v = item.split("=")
        opts = v.split("+")
        out[k.strip()] = opts[0].strip() if len(opts) == 1 else [
            o.strip() for o in opts]
    return out


def _parse_masses(items):
    return {k: float(v) for k, v in (x.split("=") for x in items)} if items \
        else None


# ------------------------------------------------------------- structures
def cmd_perovskite(a):
    from .builders import perovskite, parse_disp, write_structure

    s = perovskite(a.a, c=a.c, species=tuple(a.species),
                   polar_axis=a.polar_axis, disp=parse_disp(a.disp))
    kind = "polar" if a.disp else ("tetragonal" if a.c else "cubic")
    formula = f"{a.species[0]}{a.species[1]}{a.species[2]}3"
    write_structure(s, a.out, f"{formula} {kind}")
    print(f"wrote {a.out}: {kind} {formula} cell, "
          f"a = {a.a}" + (f", c = {a.c}" if a.c else "") + " A")


def cmd_build_dw(a):
    from .structure import read_structure, write_poscar
    from .domainwall import build_180_wall, write_info

    polar = read_structure(a.bulk)
    ref = read_structure(a.ref) if a.ref else None
    sc, info = build_180_wall(polar, ncells=a.ncells, normal=a.normal,
                              lateral=tuple(a.lateral), offset=a.offset,
                              width=a.width, ref=ref,
                              allow_charged=a.allow_charged)
    write_poscar(sc, a.out, comment=f"180deg DW {a.ncells}+{a.ncells} cells")
    write_info(info, a.info)
    print(f"wrote {a.out}: {sc.natoms} atoms, supercell matrix "
          f"{info['supercell_matrix']}; walls at fractional "
          f"{info['wall_positions']} along lattice vector {info['normal_axis']}")
    print(f"wrote {a.info}  (keep it next to the wall calculation; "
          "'phunfold unfold' uses it for the wall projection)")


def cmd_supercell(a):
    from .structure import read_structure, write_poscar, make_supercell, rattle

    prim = read_structure(a.reference)
    M = _parse_matrix(a.matrix)
    sc = make_supercell(prim, M)
    if a.rattle:
        sc = rattle(sc, a.rattle, a.seed)
    write_poscar(sc, a.out, comment=f"supercell {M.tolist()}")
    print(f"wrote {a.out}: {sc.natoms} atoms, M = {M.tolist()}")


def cmd_disp_dirs(a):
    from .builders import make_disp_dirs

    made = make_disp_dirs(a.root, a.pattern, a.sposcar, copy=a.copy,
                          perfect=not a.no_perfect)
    n = len(made) - (0 if a.no_perfect else 1)
    copied = (", copied " + " ".join(
        (c.partition(":")[2] or os.path.basename(c)) for c in a.copy)
        if a.copy else "")
    print(f"created {n} displacement folders"
          f"{'' if a.no_perfect else ' + perfect/'}{copied}")
    print("submit, e.g.:  for d in " + ("perfect " if not a.no_perfect else "")
          + "disp-*; do (cd $d && sbatch submit.sh); done")


def cmd_presets(a):
    from .paths import PRESETS

    for name, p in PRESETS.items():
        print(f"{name:14s} {p['path']}")
        for lab, q in p["points"].items():
            unit = " (2pi/a_conv, cubic axes)" if p["frame"] == "cubic_units" \
                else ""
            print(f"{'':16s}{lab:3s} {np.round(q, 4).tolist()}{unit}")


# --------------------------------------------------------------- helpers
def _ensure_force_constants(fc_path):
    """Write FORCE_CONSTANTS with phonopy if only FORCE_SETS exists."""
    if os.path.exists(fc_path):
        return
    d = os.path.dirname(fc_path) or "."
    if not os.path.exists(os.path.join(d, "FORCE_SETS")):
        raise SystemExit(f"{fc_path} not found, and no FORCE_SETS in {d} to "
                         "make it from. Run 'phonopy -f disp-*/vasprun.xml' "
                         "(phonopy-init -f for phonopy >= 4) first.")
    tried = []
    for cmd in (["phonopy-load", "--writefc", "--full-fc"],
                ["phonopy", "--writefc", "--full-fc"]):
        try:
            r = subprocess.run(cmd, cwd=d, capture_output=True, text=True)
        except FileNotFoundError:
            tried.append(f"{cmd[0]}: not found")
            continue
        if r.returncode == 0 and os.path.exists(fc_path):
            print(f"[phunfold] wrote {fc_path} with '{' '.join(cmd)}' "
                  f"in {d}")
            return
        tried.append(f"{' '.join(cmd)}: {(r.stderr or r.stdout).strip()[-300:]}")
    raise SystemExit("could not create FORCE_CONSTANTS automatically:\n  "
                     + "\n  ".join(tried) + "\nRun phonopy with --writefc "
                     "--full-fc yourself in " + d)


def _mapping_options(a):
    return dict(species_map=_parse_species_map(a.species_map),
                match_species=not a.any_species,
                allow_vacancies=a.allow_vacancies,
                allow_interstitials=a.allow_interstitials,
                max_displacement=a.max_displacement,
                origin=None if a.no_origin_search else "auto",
                strain_tol=a.strain_tol,
                allow_rotation=not a.no_rotation)


def _born_for_cell(a, cell, reference, M):
    """Born charges for ``cell`` from a LEPSILON vasprun.xml/OUTCAR given for
    either the cell itself or the reference cell (then tiled by sublattice)."""
    from .nac import read_born, tile_born
    from .unfold import Unfolder

    born, eps = read_born(a.born)
    if len(born) == cell.natoms:
        how = "given for this cell"
        z, e = born, eps
    elif len(born) == reference.natoms:
        unf = Unfolder(cell, reference, supercell_matrix=M,
                       **_mapping_options(a))
        z, e = tile_born(unf, born, eps)
        how = f"tiled from the {reference.natoms}-atom reference by sublattice"
    else:
        raise SystemExit(f"--born has {len(born)} atoms; expected {cell.natoms} "
                         f"(this cell) or {reference.natoms} (reference cell)")
    print(f"[phunfold] NAC: Born charges {how}; eps_inf diag = "
          f"{np.round(np.diag(e), 3).tolist()}")
    return born, eps, z, e


def _bulk_bands(bulk_dir, path, born=None, eps=None):
    """Bands of the bulk reference on the same reduced-q path, with NAC when
    Born charges for that cell are available."""
    from .dynmat import NativeBackend
    from .structure import read_structure

    fc = os.path.join(bulk_dir, "FORCE_CONSTANTS")
    _ensure_force_constants(fc)
    pos = read_structure(os.path.join(bulk_dir, "POSCAR"))
    nac = None
    if born is not None and len(born) == pos.natoms:
        nac = dict(born=born, epsilon=eps, ref_volume=pos.volume)
    nb = NativeBackend(pos, os.path.join(bulk_dir, "SPOSCAR"), fc, nac=nac)
    k = path["q"] @ nb.structure.recip
    f, _ = nb.solve(k, q_directions=list(path["leg_direction"]))
    return f


def _resolve_path_arg(a, reference):
    from .paths import detect_preset

    if a.path != "auto":
        return a.path
    if a.dw_info and os.path.exists(a.dw_info):
        name = "dw-tetragonal"
    else:
        name = detect_preset(reference.lattice)
        if name is None:
            raise SystemExit("could not recognise the reference lattice for "
                             "an automatic k-path; pass --path (a preset from "
                             "'phunfold presets' or 'G 0 0 0, X 1/2 0 0, ...')")
    print(f"[phunfold] k-path: preset '{name}' (override with --path)")
    return name


def _run(a):
    """Shared by 'unfold' and 'check'. Returns (result, bulk bands, backend)."""
    from .structure import read_structure
    from .unfold import run_unfolding, Unfolder
    from .lattice import reference_from_supercell
    from .paths import resolve_path, make_band_path

    M = _parse_matrix(a.supercell_matrix)
    born_raw = eps_raw = None
    if a.backend == "native":
        from .dynmat import NativeBackend
        _ensure_force_constants(a.fc)
        cell = read_structure(a.unitcell)
    else:
        from .dynmat import PhonopyBackend
        backend = PhonopyBackend(a.phonopy_yaml, a.force_sets, a.fc_phonopy,
                                 born=a.born_phonopy)
        cell = backend.structure
    if a.reference == "auto":
        if M is None:
            raise SystemExit("--reference auto needs --supercell-matrix")
        reference = reference_from_supercell(cell, M)
        print(f"[phunfold] reference built from the supercell average: "
              f"{reference.natoms} sites {reference.symbols}")
    else:
        reference = read_structure(a.reference)
    nac = None
    if a.born:
        born_raw, eps_raw, z, e = _born_for_cell(a, cell, reference, M)
        nac = dict(born=z, epsilon=e, ref_volume=reference.volume)
    if a.backend == "native":
        backend = NativeBackend(cell, a.sposcar, a.fc,
                                masses=_parse_masses(a.mass),
                                fc_units=a.fc_units,
                                fc_threshold=a.fc_threshold, nac=nac)
        print(f"[phunfold] {backend.structure.natoms}-atom cell, "
              f"{backend.n_pairs} force-constant pairs")
        if backend.nac is not None:
            print(f"[phunfold] NAC (Gonze-Lee): {len(backend.nac.G)} G vectors, "
                  f"Lambda {backend.nac.Lambda:.3f}")
    elif nac is not None:
        backend.phonon.nac_params = dict(born=nac["born"],
                                         dielectric=nac["epsilon"])

    projections = {}
    if a.dw_info and os.path.exists(a.dw_info):
        info = json.load(open(a.dw_info))
        projections["wall"] = (f"slab:{info['normal_axis']}:"
                               f"{','.join(map(str, info['wall_positions']))}:"
                               f"{a.wall_width}")
    for item in a.project or []:
        name, spec = item.split("=", 1)
        projections[name.strip()] = spec.strip()

    path = _resolve_path_arg(a, reference)
    res = run_unfolding(backend, reference, path, path_frame=a.path_frame,
                        supercell_matrix=M, density=a.density,
                        distortion_phase=not a.no_distortion_phase,
                        projections=projections, resolve=a.resolve,
                        mapping_options=_mapping_options(a))
    res.save(a.out + ".npz")
    res.write_text(a.out + ".dat")
    print(f"[phunfold] sum rule {res.meta['sum_rule']:.4f} holds to "
          f"{res.meta['sum_rule_error']:.1e}; wrote {a.out}.npz, {a.out}.dat")
    ref_bands = None
    if a.bulk_dir:
        unf = Unfolder(backend.structure, reference, supercell_matrix=M,
                       **_mapping_options(a))
        segs = resolve_path(path, a.path_frame, reference.lattice,
                            unf.ref_lattice, backend.structure.lattice)
        bp = make_band_path(segs, unf.ref_recip, a.density)
        ref_bands = _bulk_bands(a.bulk_dir, bp, born_raw, eps_raw)
        np.save(a.out + "_bulk_bands.npy", ref_bands)
    return res, ref_bands, backend


def cmd_unfold(a):
    res, ref_bands, _ = _run(a)
    _plot(res, a, ref_bands)


def _compare(res, ref_bands, tol=1e-3):
    """Largest |summed weight of a degenerate group - number of reference
    branches at that frequency| over the path (0 for a perfect unfolding)."""
    worst = 0.0
    for f, w, b in zip(res.freqs, res.weights, ref_bands):
        o = np.argsort(f)
        f, w = f[o], w[o]
        for g in np.split(np.arange(len(f)), np.where(np.diff(f) > tol)[0] + 1):
            n = np.sum(np.abs(b - f[g].mean()) < tol + np.ptp(f[g]))
            worst = max(worst, abs(w[g].sum() - n))
    return worst


def cmd_check(a):
    a.unitcell, a.sposcar, a.reference, a.bulk_dir = \
        "SPOSCAR", "SPOSCAR", "POSCAR", "."
    a.fc = "FORCE_CONSTANTS"
    for f in ("POSCAR", "SPOSCAR"):
        if not os.path.exists(f):
            raise SystemExit(f"{f} not found: run 'phunfold check' inside the "
                             "folder where you ran 'phonopy -d'")
    res, ref_bands, backend = _run(a)
    dev = _compare(res, ref_bands)
    nsc, nref = backend.structure.natoms, res.meta["n_ref"]
    ok = dev < 1e-3
    print("")
    print(f"  supercell modes per k : {3 * nsc} ({nsc} atoms)")
    print(f"  reference branches    : {3 * nref} (unfolded weight per k = "
          f"{res.meta['sum_rule']:.1f})")
    print(f"  largest mismatch      : {dev:.1e}")
    print(f"  {'PASS' if ok else 'FAIL'}: the unfolded {nsc}-atom spectrum "
          f"{'reproduces' if ok else 'does NOT reproduce'} the "
          f"{nref}-atom dispersion")
    _plot_check(res, ref_bands, a, ok)
    if not ok:
        sys.exit(1)


# ------------------------------------------------------------------ plots
def _sigma(a):
    from .plot import UNITS
    return a.sigma if a.sigma is not None else 0.12 * UNITS[a.units]


def _plot(res, a, ref_bands=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plot import plot_unfolded

    ylim = tuple(a.ylim) if a.ylim else None
    sigma = _sigma(a)
    panels = [("total", None)]
    panels += [("proj", p) for p in res.projections()]
    panels += [("res", r) for r in res.resolved()]
    ncol = min(2, len(panels))
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 4.6 * nrow),
                             squeeze=False)
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    for ax, (kind, name) in zip(axes.ravel(), panels):
        if kind == "total":
            plot_unfolded(res, ax, style="spectral", units=a.units, ylim=ylim,
                          sigma=sigma, reference_bands=ref_bands,
                          title="Unfolded spectral function")
        elif kind == "proj":
            plot_unfolded(res, ax, style="scatter", units=a.units, ylim=ylim,
                          color_by=name, size=a.size,
                          title=f"Mode weight on '{name}' atoms")
        else:
            plot_unfolded(res, ax, style="spectral", units=a.units, ylim=ylim,
                          sigma=sigma, weights=res.extra[f"w_{name}"],
                          title=f"Unfolded spectral function: {name}")
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=a.dpi)
    print(f"[phunfold] wrote {a.out}.png ({len(panels)} panels)")


def _plot_check(res, ref_bands, a, ok):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plot import plot_unfolded, plot_folded, UNITS

    ylim = tuple(a.ylim) if a.ylim else None
    if ylim is None:
        f = res.freqs * UNITS[a.units]
        ylim = (min(0.0, f.min() * 1.05), f.max() * 1.05)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    plot_folded(res, a1, units=a.units, ylim=ylim, color="0.35")
    a1.set_title(f"Supercell bands, folded ({res.freqs.shape[1]} per k)")
    plot_unfolded(res, a2, style="spectral", units=a.units, ylim=ylim,
                  sigma=_sigma(a), reference_bands=ref_bands,
                  title=f"Unfolded (dashed: {3 * res.meta['n_ref']}-branch "
                        f"reference) — {'PASS' if ok else 'FAIL'}")
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=a.dpi)
    print(f"[phunfold] wrote {a.out}.png")


def cmd_plot(a):
    from .unfold import UnfoldResult

    res = UnfoldResult.load(a.npz)
    a.out = a.out or os.path.splitext(a.npz)[0]
    ref = np.load(a.bulk_bands) if a.bulk_bands else None
    _plot(res, a, ref)


# ---------------------------------------------------------------- options
def _plot_opts(p):
    g = p.add_argument_group("plotting")
    g.add_argument("--units", default="cm-1", choices=["THz", "cm-1", "meV"])
    g.add_argument("--ylim", type=float, nargs=2)
    g.add_argument("--sigma", type=float, default=None,
                   help="Lorentzian broadening in --units (default 0.12 THz "
                   "= 4 cm-1 = 0.5 meV)")
    g.add_argument("--size", type=float, default=10.0, help="marker size")
    g.add_argument("--dpi", type=int, default=200)


def _common_unfold_opts(p, check=False):
    g = p.add_argument_group("force constants")
    if not check:
        g.add_argument("--backend", default="native",
                       choices=["native", "phonopy"])
        g.add_argument("--unitcell", default="POSCAR",
                       help="the cell given to 'phonopy -d -c' (default "
                       "POSCAR)")
        g.add_argument("--sposcar", default="SPOSCAR")
        g.add_argument("--fc", default="FORCE_CONSTANTS",
                       help="written automatically from FORCE_SETS with "
                       "phonopy if missing")
    else:
        p.set_defaults(backend="native")
    g.add_argument("--fc-units", default="eV/A^2",
                   choices=["eV/A^2", "eV/bohr^2", "Ry/bohr^2", "Ha/bohr^2",
                            "kJ/mol/nm^2"])
    g.add_argument("--fc-threshold", type=float, default=0.0,
                   help="drop FC blocks below this (eV/A^2) for huge cells")
    g.add_argument("--mass", nargs="*", help="mass overrides, e.g. O=17.999")
    g.add_argument("--born", help="LO-TO correction: vasprun.xml or OUTCAR of "
                   "a LEPSILON run (for this cell, or for the reference cell "
                   "- then tiled by sublattice)")
    if not check:
        g.add_argument("--phonopy-yaml", default="phonopy_disp.yaml",
                       help="[--backend phonopy]")
        g.add_argument("--force-sets", default="FORCE_SETS",
                       help="[--backend phonopy]")
        g.add_argument("--fc-phonopy", default=None,
                       help="[--backend phonopy] FORCE_CONSTANTS instead")
        g.add_argument("--born-phonopy", help="[--backend phonopy] phonopy "
                       "BORN file of this exact cell")
    g = p.add_argument_group("reference cell and atom mapping")
    if not check:
        g.add_argument("--reference", required=True,
                       help="small cell to unfold onto (e.g. relaxed 5-atom "
                       "CONTCAR), or 'auto' = supercell average (needs "
                       "--supercell-matrix)")
    g.add_argument("--supercell-matrix", help="'2 2 2' (diagonal), 9 ints or "
                   "JSON; default: detected")
    g.add_argument("--species-map", help="substitutions, e.g. 'Sr=Ba,Zr=Ti'")
    g.add_argument("--any-species", action="store_true",
                   help="ignore species when matching atoms to sites")
    g.add_argument("--allow-vacancies", action="store_true")
    g.add_argument("--allow-interstitials", action="store_true")
    g.add_argument("--max-displacement", type=float, default=0.8,
                   help="max atom-to-site distance in A (default 0.8)")
    g.add_argument("--strain-tol", type=float, default=0.05)
    g.add_argument("--no-origin-search", action="store_true")
    g.add_argument("--no-rotation", action="store_true")
    g.add_argument("--no-distortion-phase", action="store_true",
                   help="phonopy.unfolding convention (ignore r_s - x_s)")
    g = p.add_argument_group("k path")
    g.add_argument("--path", default="auto",
                   help="'auto' (from the reference lattice; dw-tetragonal "
                   "with dw_info.json), a preset ('phunfold presets'), or "
                   "'G 0 0 0, X 0 1/2 0, M 1/2 1/2 0 | ...'")
    g.add_argument("--path-frame", default=None,
                   choices=["reference", "supercell", "cartesian"])
    g.add_argument("--density", type=float, default=100.0,
                   help="k-points per 1/A along the path (default 100)")
    if check:
        p.set_defaults(project=None, resolve=None, dw_info=None,
                       wall_width=2.5)
        return
    g = p.add_argument_group("projections and decomposition")
    g.add_argument("--project", nargs="*", metavar="NAME=SPEC",
                   help="atom groups, e.g. 'wall=slab:0:0,0.5:2.5' "
                   "'defect=sphere:.5,.5,.5:3' 'Ti=species:Ti' "
                   "'layer=atoms:0-19' 'Asite=sites:0' 'subs=substituted'")
    g.add_argument("--resolve", choices=["species", "sublattice"],
                   help="also plot species- or sublattice-resolved spectra")
    g.add_argument("--dw-info", default="dw_info.json",
                   help="adds a 'wall' projection if the file exists "
                   "(default ./dw_info.json)")
    g.add_argument("--wall-width", type=float, default=2.5)
    g.add_argument("--bulk-dir", help="bulk phonon folder (POSCAR, SPOSCAR, "
                   "FORCE_CONSTANTS) drawn as dashed reference bands")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="phunfold", description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    fmt = argparse.RawDescriptionHelpFormatter

    # ---- structures ----------------------------------------------------
    p = sub.add_parser("perovskite", help="cubic / polar ABX3 POSCAR",
                       formatter_class=fmt, description=(
                           "Ideal cubic ABX3 cell, or a tetragonal/polar "
                           "starting cell for a relaxation.\n\nexamples:\n"
                           "  phunfold perovskite --a 4.00 --out POSCAR_cubic\n"
                           "  phunfold perovskite --a 3.99 --c 4.03 "
                           "--polar-axis c \\\n      --disp B=0.10 X_ap=-0.08 "
                           "X_eq=-0.06 --out POSCAR_tet"))
    p.add_argument("--a", type=float, required=True, help="lattice constant (A)")
    p.add_argument("--c", type=float, help="length along the polar axis (A)")
    p.add_argument("--species", nargs=3, default=["Ba", "Ti", "O"],
                   metavar=("A", "B", "X"))
    p.add_argument("--polar-axis", choices=["a", "b", "c"],
                   help="axis of c and of the displacements")
    p.add_argument("--disp", nargs="*", metavar="SITE=A",
                   help="displacements along the polar axis in A: A=, B=, "
                   "X= (all X), X_ap= (apical), X_eq= (equatorial)")
    p.add_argument("--out", default="POSCAR")
    p.set_defaults(func=cmd_perovskite)

    p = sub.add_parser("build-dw", help="180-degree domain-wall supercell")
    p.add_argument("--bulk", required=True, help="relaxed polar bulk cell "
                   "(e.g. the tetragonal CONTCAR)")
    p.add_argument("--ref", help="centrosymmetric reference with the same "
                   "atom order (default: snap coordinates to 0 / 0.5)")
    p.add_argument("--ncells", type=int, default=4, help="cells per domain")
    p.add_argument("--normal", default="a", help="wall normal: a, b or c")
    p.add_argument("--lateral", type=int, nargs=2, default=[1, 1])
    p.add_argument("--offset", type=float, default=0.0,
                   help="0 = A-site (BaO) centred wall, 0.5 = B-site (TiO2)")
    p.add_argument("--width", type=float, default=0.0,
                   help="tanh wall width in cells (0 = abrupt)")
    p.add_argument("--allow-charged", action="store_true")
    p.add_argument("--out", default="POSCAR_DW")
    p.add_argument("--info", default="dw_info.json")
    p.set_defaults(func=cmd_build_dw)

    p = sub.add_parser("supercell", help="tile a cell with an integer matrix")
    p.add_argument("--reference", required=True)
    p.add_argument("--matrix", required=True,
                   help="'2 2 2' or '1 1 0 -1 1 0 0 0 2' or JSON")
    p.add_argument("--rattle", type=float, default=0.0,
                   help="random displacement amplitude (A)")
    p.add_argument("--seed", type=int)
    p.add_argument("--out", default="POSCAR_supercell")
    p.set_defaults(func=cmd_supercell)

    p = sub.add_parser("disp-dirs", help="folders for phonopy displacements",
                       formatter_class=fmt, description=(
                           "Run after 'phonopy -d'. Makes perfect/ (SPOSCAR) "
                           "and disp-001/, disp-002/, ... each with its "
                           "POSCAR, and copies your own input files in.\n\n"
                           "example:\n  phunfold disp-dirs --copy "
                           "../inputs/INCAR_static:INCAR ../inputs/KPOINTS "
                           "../inputs/POTCAR ../inputs/submit.sh"))
    p.add_argument("--copy", nargs="*", default=[],
                   help="files copied into every folder; SOURCE:NAME "
                   "renames (e.g. ../inputs/INCAR_static:INCAR)")
    p.add_argument("--no-perfect", action="store_true",
                   help="skip perfect/ (only needed for residual forces)")
    p.add_argument("--root", default=".")
    p.add_argument("--pattern", default="POSCAR-*")
    p.add_argument("--sposcar", default="SPOSCAR")
    p.set_defaults(func=cmd_disp_dirs)

    # ---- analysis ------------------------------------------------------
    p = sub.add_parser("check", help="verify unfolding in a bulk phonon folder",
                       formatter_class=fmt, description=(
                           "Run inside a bulk phonon folder (POSCAR, SPOSCAR, "
                           "FORCE_SETS or FORCE_CONSTANTS). Treats phonopy's "
                           "SPOSCAR as a supercell, unfolds it onto POSCAR and "
                           "compares with the POSCAR dispersion from the same "
                           "force constants: they must agree exactly.\n\n"
                           "example:\n  phunfold check --born ../born/"
                           "vasprun.xml"))
    _common_unfold_opts(p, check=True)
    p.add_argument("--out", default="check")
    _plot_opts(p)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("unfold", help="unfold and plot", formatter_class=fmt,
                       description="Unfold the phonons of any commensurate "
                       "supercell onto a reference cell.")
    _common_unfold_opts(p)
    p.add_argument("--out", default="unfolded")
    _plot_opts(p)
    p.set_defaults(func=cmd_unfold)

    p = sub.add_parser("plot", help="re-plot a saved .npz")
    p.add_argument("npz")
    p.add_argument("--bulk-bands", help="*_bulk_bands.npy from unfold")
    p.add_argument("--out")
    _plot_opts(p)
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("presets", help="list k-path presets")
    p.set_defaults(func=cmd_presets)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    sys.exit(main())
