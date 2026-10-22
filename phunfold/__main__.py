"""Command line interface:  phunfold <command> [options]   (or python -m phunfold)

Commands
  unfold      compute the unfolded phonon spectrum of any supercell and plot it
  plot        re-plot a saved unfold .npz
  supercell   tile a reference cell with any integer matrix (test cells, SQS seeds)
  build-dw    build a 180-degree ferroelectric domain-wall supercell
  setup       write POSCAR/INCAR/KPOINTS(/POTCAR) for one VASP stage
  setup-disp  one VASP directory per phonopy displacement (+ perfect/)
  presets     list the built-in k-path presets
Run any command with -h for its options.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k.strip().upper()] = v.strip()
    return out


def _variants(items):
    return dict(x.split("=", 1) for x in items) if items else None


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
    raise ValueError("matrix needs 3 (diagonal) or 9 integers")


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


# ----------------------------------------------------------------------
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
    print(f"wrote {a.info}  (use with: phunfold unfold --dw-info {a.info})")


def cmd_supercell(a):
    from .structure import read_structure, write_poscar, make_supercell, rattle

    prim = read_structure(a.reference)
    M = _parse_matrix(a.matrix)
    sc = make_supercell(prim, M)
    if a.rattle:
        sc = rattle(sc, a.rattle, a.seed)
    write_poscar(sc, a.out, comment=f"supercell {M.tolist()}")
    print(f"wrote {a.out}: {sc.natoms} atoms, M = {M.tolist()}")


def cmd_setup(a):
    from .vasp import setup_calculation

    mesh = setup_calculation(a.stage, a.poscar, a.dir, a.kspacing,
                             a.potcar_dir, _variants(a.potcar_variant),
                             _overrides(a.set))
    print(f"{a.dir}: stage {a.stage}, k-mesh {mesh}")


def cmd_setup_disp(a):
    from .vasp import setup_displacements

    made, mesh = setup_displacements(a.pattern, a.sposcar, a.root,
                                     a.kspacing, a.potcar_dir,
                                     _variants(a.potcar_variant),
                                     _overrides(a.set))
    print(f"created {len(made)} directories (perfect + {len(made) - 1} "
          f"displacements), k-mesh {mesh}")


def cmd_presets(a):
    from .paths import PRESETS

    for name, p in PRESETS.items():
        print(f"{name:14s} {p['path']}")
        for lab, q in p["points"].items():
            unit = " (2pi/a_conv, cubic axes)" if p["frame"] == "cubic_units" \
                else ""
            print(f"{'':16s}{lab:3s} {np.round(q, 4).tolist()}{unit}")


def _bulk_bands(bulk_dir, q_ref, fc_name="FORCE_CONSTANTS"):
    """Bands of the bulk reference on the same reduced-q path (native)."""
    from .dynmat import NativeBackend

    nb = NativeBackend(os.path.join(bulk_dir, "POSCAR"),
                       os.path.join(bulk_dir, "SPOSCAR"),
                       os.path.join(bulk_dir, fc_name))
    f, _ = nb.solve(q_ref @ nb.structure.recip)
    return f


def _mapping_options(a):
    return dict(species_map=_parse_species_map(a.species_map),
                match_species=not a.any_species,
                allow_vacancies=a.allow_vacancies,
                allow_interstitials=a.allow_interstitials,
                max_displacement=a.max_displacement,
                origin=None if a.no_origin_search else "auto",
                strain_tol=a.strain_tol,
                allow_rotation=not a.no_rotation)


def cmd_unfold(a):
    from .structure import read_structure
    from .unfold import run_unfolding, Unfolder
    from .lattice import reference_from_supercell

    if a.backend == "native":
        from .dynmat import NativeBackend
        backend = NativeBackend(a.unitcell, a.sposcar, a.fc,
                                masses=_parse_masses(a.mass),
                                fc_units=a.fc_units,
                                fc_threshold=a.fc_threshold)
        print(f"[phunfold] native backend: {backend.structure.natoms} atoms, "
              f"{backend.n_pairs} FC pairs; ASR correction "
              f"{backend.asr_violation:.2e} eV/A^2")
    else:
        from .dynmat import PhonopyBackend
        backend = PhonopyBackend(a.phonopy_yaml, a.force_sets, a.fc_phonopy,
                                 born=a.born)
    M = _parse_matrix(a.supercell_matrix)
    if a.reference == "auto":
        if M is None:
            raise SystemExit("--reference auto needs --supercell-matrix")
        reference = reference_from_supercell(backend.structure, M)
        print(f"[phunfold] reference built from the supercell average: "
              f"{reference.natoms} sites {reference.symbols}")
    else:
        reference = read_structure(a.reference)
    if a.backend == "phonopy" and a.born_outcar:
        from .vasp import parse_outcar_born, tile_born
        born_ref, eps = parse_outcar_born(a.born_outcar)
        unf = Unfolder(backend.structure, reference, supercell_matrix=M,
                       **_mapping_options(a))
        backend.phonon.nac_params = dict(born=tile_born(unf, born_ref),
                                         dielectric=eps)

    projections = {}
    if a.dw_info and os.path.exists(a.dw_info):
        info = json.load(open(a.dw_info))
        projections["wall"] = (f"slab:{info['normal_axis']}:"
                               f"{','.join(map(str, info['wall_positions']))}:"
                               f"{a.wall_width}")
    for item in a.project or []:
        name, spec = item.split("=", 1)
        projections[name.strip()] = spec.strip()

    res = run_unfolding(backend, reference, a.path, path_frame=a.path_frame,
                        supercell_matrix=M, density=a.density,
                        distortion_phase=not a.no_distortion_phase,
                        projections=projections, resolve=a.resolve,
                        mapping_options=_mapping_options(a))
    res.save(a.out + ".npz")
    res.write_text(a.out + ".dat")
    print(f"[phunfold] sum rule {res.meta['sum_rule']:.4f} holds to "
          f"{res.meta['sum_rule_error']:.1e}; wrote {a.out}.npz, {a.out}.dat")
    ref_bands = _bulk_bands(a.bulk_dir, res.path["q"]) if a.bulk_dir else None
    if ref_bands is not None:
        np.save(a.out + "_bulk_bands.npy", ref_bands)
    _plot(res, a, ref_bands)


def _plot(res, a, ref_bands=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plot import plot_unfolded

    ylim = tuple(a.ylim) if a.ylim else None
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
                          sigma=a.sigma, reference_bands=ref_bands,
                          title="Unfolded spectral function")
        elif kind == "proj":
            plot_unfolded(res, ax, style="scatter", units=a.units, ylim=ylim,
                          color_by=name, size=a.size,
                          title=f"Mode weight on '{name}' atoms")
        else:
            plot_unfolded(res, ax, style="spectral", units=a.units, ylim=ylim,
                          sigma=a.sigma, weights=res.extra[f"w_{name}"],
                          title=f"Unfolded spectral function: {name}")
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=a.dpi)
    print(f"[phunfold] wrote {a.out}.png ({len(panels)} panels)")


def cmd_plot(a):
    from .unfold import UnfoldResult

    res = UnfoldResult.load(a.npz)
    a.out = a.out or os.path.splitext(a.npz)[0]
    ref = np.load(a.bulk_bands) if a.bulk_bands else None
    _plot(res, a, ref)


# ----------------------------------------------------------------------
def _plot_opts(p):
    g = p.add_argument_group("plotting")
    g.add_argument("--units", default="THz", choices=["THz", "cm-1", "meV"])
    g.add_argument("--ylim", type=float, nargs=2)
    g.add_argument("--sigma", type=float, default=0.12,
                   help="Lorentzian broadening in --units")
    g.add_argument("--size", type=float, default=10.0, help="marker size")
    g.add_argument("--dpi", type=int, default=200)


def _vasp_opts(p):
    p.add_argument("--kspacing", type=float, default=0.2,
                   help="k spacing (1/A incl. 2pi, like VASP KSPACING)")
    p.add_argument("--potcar-dir", help="dir with <element>/POTCAR")
    p.add_argument("--potcar-variant", nargs="*",
                   help="e.g. Ba=Ba_sv Ti=Ti_sv O=O")
    p.add_argument("--set", nargs="*", help="override INCAR tags, "
                   "e.g. ENCUT=650 NCORE=8")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="phunfold", description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    fmt = argparse.RawDescriptionHelpFormatter

    # ---- unfold -------------------------------------------------------
    p = sub.add_parser("unfold", help="unfold and plot", formatter_class=fmt,
                       description="Unfold the phonons of any commensurate "
                       "supercell onto a reference cell.")
    g = p.add_argument_group("force constants / eigenvectors")
    g.add_argument("--backend", default="native", choices=["native", "phonopy"])
    g.add_argument("--unitcell", default="POSCAR",
                   help="[native] the cell given to phonopy -d -c (any "
                   "ASE-readable format if ase is installed)")
    g.add_argument("--sposcar", default="SPOSCAR", help="[native]")
    g.add_argument("--fc", default="FORCE_CONSTANTS",
                   help="[native] FORCE_CONSTANTS (text) or force_constants"
                   ".hdf5; --full-fc recommended")
    g.add_argument("--fc-units", default="eV/A^2",
                   choices=["eV/A^2", "eV/bohr^2", "Ry/bohr^2", "Ha/bohr^2",
                            "kJ/mol/nm^2"])
    g.add_argument("--fc-threshold", type=float, default=0.0,
                   help="[native] drop FC blocks below this (eV/A^2) to "
                   "save memory on very large cells")
    g.add_argument("--mass", nargs="*", help="[native] mass overrides, "
                   "e.g. O=17.999")
    g.add_argument("--phonopy-yaml", default="phonopy_disp.yaml",
                   help="[phonopy] any calculator interface phonopy supports")
    g.add_argument("--force-sets", default="FORCE_SETS", help="[phonopy]")
    g.add_argument("--fc-phonopy", default=None,
                   help="[phonopy] FORCE_CONSTANTS instead of FORCE_SETS")
    g.add_argument("--born", help="[phonopy] BORN file of the supercell")
    g.add_argument("--born-outcar", help="[phonopy] OUTCAR of a bulk "
                   "LEPSILON run; Z* tiled onto the supercell for NAC")
    g = p.add_argument_group("reference cell and atom mapping")
    g.add_argument("--reference", required=True,
                   help="small cell to unfold onto, or 'auto' = average of "
                   "the supercell (needs --supercell-matrix)")
    g.add_argument("--supercell-matrix", help="'8 1 1' (diagonal), 9 ints, "
                   "or JSON; default: search all matching orientations")
    g.add_argument("--species-map", help="substitutions, e.g. 'Sr=Ba,Zr=Ti'")
    g.add_argument("--any-species", action="store_true",
                   help="ignore species when matching atoms to sites")
    g.add_argument("--allow-vacancies", action="store_true")
    g.add_argument("--allow-interstitials", action="store_true")
    g.add_argument("--max-displacement", type=float, default=0.8,
                   help="max atom-to-site distance in A (default 0.8)")
    g.add_argument("--strain-tol", type=float, default=0.05,
                   help="max relative lattice mismatch (default 0.05)")
    g.add_argument("--no-origin-search", action="store_true")
    g.add_argument("--no-rotation", action="store_true",
                   help="only accept sc.lattice = M @ ref.lattice directly")
    g.add_argument("--no-distortion-phase", action="store_true",
                   help="phonopy.unfolding convention (ignore r_s - x_s)")
    g = p.add_argument_group("k path")
    g.add_argument("--path", default="dw-tetragonal",
                   help="preset name (see 'phunfold presets') or "
                   "'G 0 0 0, X 1/2 0 0 | ...'")
    g.add_argument("--path-frame", default=None,
                   choices=["reference", "supercell", "cartesian"],
                   help="frame of the --path coordinates (default reference)")
    g.add_argument("--density", type=float, default=60.0,
                   help="k-points per 1/A along the path")
    g = p.add_argument_group("projections and decomposition")
    g.add_argument("--project", nargs="*", metavar="NAME=SPEC",
                   help="atom groups, e.g. 'wall=slab:0:0,0.5:2.5' "
                   "'defect=sphere:.5,.5,.5:3' 'Ti=species:Ti' "
                   "'layer=atoms:0-19' 'Asite=sites:0' 'subs=substituted'")
    g.add_argument("--resolve", choices=["species", "sublattice"],
                   help="also store species- or sublattice-resolved weights")
    g.add_argument("--dw-info", default="dw_info.json",
                   help="adds a 'wall' projection if the file exists")
    g.add_argument("--wall-width", type=float, default=2.5)
    g.add_argument("--bulk-dir", help="dir with POSCAR, SPOSCAR, "
                   "FORCE_CONSTANTS of the reference cell (dashed overlay)")
    p.add_argument("--out", default="unfolded")
    _plot_opts(p)
    p.set_defaults(func=cmd_unfold)

    p = sub.add_parser("plot", help="re-plot a saved .npz")
    p.add_argument("npz")
    p.add_argument("--bulk-bands", help="*_bulk_bands.npy from unfold")
    p.add_argument("--out")
    _plot_opts(p)
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("supercell", help="tile a reference cell")
    p.add_argument("--reference", required=True)
    p.add_argument("--matrix", required=True,
                   help="'2 2 2' or '1 1 0 -1 1 0 0 0 2' or JSON")
    p.add_argument("--rattle", type=float, default=0.0,
                   help="random displacement amplitude (A)")
    p.add_argument("--seed", type=int)
    p.add_argument("--out", default="POSCAR_supercell")
    p.set_defaults(func=cmd_supercell)

    p = sub.add_parser("presets", help="list k-path presets")
    p.set_defaults(func=cmd_presets)

    p = sub.add_parser("build-dw", help="build 180-degree DW supercell")
    p.add_argument("--bulk", required=True, help="relaxed polar bulk POSCAR")
    p.add_argument("--ref", help="centrosymmetric reference POSCAR (same "
                   "atom order); default: snap coordinates to 0/0.5")
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

    p = sub.add_parser("setup", help="write inputs for one VASP stage")
    p.add_argument("--stage", required=True,
                   choices=["relax_bulk", "born", "relax_dw", "single_point"])
    p.add_argument("--poscar", required=True)
    p.add_argument("--dir", required=True)
    _vasp_opts(p)
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("setup-disp", help="dirs for phonopy displacements")
    p.add_argument("--root", default=".")
    p.add_argument("--pattern", default="POSCAR-*")
    p.add_argument("--sposcar", default="SPOSCAR")
    _vasp_opts(p)
    p.set_defaults(func=cmd_setup_disp)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    sys.exit(main())
