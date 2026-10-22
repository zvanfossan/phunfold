"""Command line interface:  python -m phunfold <command> [options]

Commands
  build-dw    build a 180-degree domain-wall supercell from a polar bulk cell
  setup       write POSCAR/INCAR/KPOINTS(/POTCAR) for one VASP stage
  setup-disp  one VASP directory per phonopy displacement (+ perfect/)
  unfold      compute the unfolded phonon spectrum and plot it
  plot        re-plot a saved unfold .npz
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


# ----------------------------------------------------------------------
def cmd_build_dw(a):
    from .structure import read_poscar, write_poscar
    from .domainwall import build_180_wall, write_info

    polar = read_poscar(a.bulk)
    ref = read_poscar(a.ref) if a.ref else None
    sc, info = build_180_wall(polar, ncells=a.ncells, normal=a.normal,
                              lateral=tuple(a.lateral), offset=a.offset,
                              width=a.width, ref=ref,
                              allow_charged=a.allow_charged)
    write_poscar(sc, a.out, comment=f"180deg DW {a.ncells}+{a.ncells} cells")
    write_info(info, a.info)
    print(f"wrote {a.out}: {sc.natoms} atoms, supercell matrix "
          f"{info['supercell_matrix']}; walls at fractional "
          f"{info['wall_positions']} along lattice vector {info['normal_axis']}")
    print(f"wrote {a.info}")


def cmd_setup(a):
    from .vasp import setup_calculation

    mesh = setup_calculation(a.stage, a.poscar, a.dir, a.kspacing,
                             a.potcar_dir, _variants(a.potcar_variant),
                             _overrides(a.set))
    print(f"{a.dir}: stage {a.stage}, k-mesh {mesh}")


def _variants(items):
    return dict(x.split("=", 1) for x in items) if items else None


def cmd_setup_disp(a):
    from .vasp import setup_displacements

    made, mesh = setup_displacements(a.pattern, a.sposcar, a.root,
                                     a.kspacing, a.potcar_dir,
                                     _variants(a.potcar_variant),
                                     _overrides(a.set))
    print(f"created {len(made)} directories (perfect + {len(made) - 1} "
          f"displacements), k-mesh {mesh}")


def _bulk_bands(bulk_dir, q_ref):
    """Bands of the bulk reference on the same reduced-q path (native)."""
    from .dynmat import NativeBackend

    nb = NativeBackend(os.path.join(bulk_dir, "POSCAR"),
                       os.path.join(bulk_dir, "SPOSCAR"),
                       os.path.join(bulk_dir, "FORCE_CONSTANTS"))
    f, _ = nb.solve(q_ref @ nb.structure.recip)
    return f


def cmd_unfold(a):
    from .structure import read_poscar
    from .unfold import run_unfolding, Unfolder, DEFAULT_PATH

    if a.backend == "native":
        from .dynmat import NativeBackend
        backend = NativeBackend(a.unitcell, a.sposcar, a.fc)
        print(f"[phunfold] native backend; ASR correction applied "
              f"(max violation {backend.asr_violation:.2e} eV/A^2)")
    else:
        from .dynmat import PhonopyBackend
        nac = None
        if a.born_outcar:
            from .vasp import parse_outcar_born, tile_born
            born_ref, eps = parse_outcar_born(a.born_outcar)
            tmp = PhonopyBackend(a.phonopy_yaml, a.force_sets, a.fc_phonopy)
            unf = Unfolder(tmp.structure, read_poscar(a.reference))
            nac = dict(born=tile_born(unf, born_ref), dielectric=eps)
            backend = tmp
            backend.phonon.nac_params = nac
        else:
            backend = PhonopyBackend(a.phonopy_yaml, a.force_sets,
                                     a.fc_phonopy, born=a.born)
    reference = read_poscar(a.reference)
    wall_axis = wall_pos = None
    if a.dw_info and os.path.exists(a.dw_info):
        info = json.load(open(a.dw_info))
        wall_axis, wall_pos = info["normal_axis"], info["wall_positions"]
    M = json.loads(a.supercell_matrix) if a.supercell_matrix else None
    res = run_unfolding(backend, reference, a.path or DEFAULT_PATH,
                        supercell_matrix=M, density=a.density,
                        distortion_phase=not a.no_distortion_phase,
                        wall_axis=wall_axis, wall_positions=wall_pos,
                        wall_width=a.wall_width)
    res.save(a.out + ".npz")
    res.write_text(a.out + ".dat")
    print(f"[phunfold] sum-rule error {res.meta['sum_rule_error']:.1e}; "
          f"wrote {a.out}.npz and {a.out}.dat")
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
    panels = ["spectral"] + (["wall"] if "wall_participation" in res.extra
                             else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 4.8),
                             squeeze=False)
    plot_unfolded(res, axes[0, 0], style="spectral", units=a.units, ylim=ylim,
                  sigma=a.sigma, reference_bands=ref_bands,
                  title="Unfolded spectral function")
    if "wall" in panels:
        plot_unfolded(res, axes[0, 1], style="scatter", units=a.units,
                      ylim=ylim, color_by="wall_participation", size=a.size,
                      title=f"Weight on atoms within {a.wall_width} Å of walls")
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=a.dpi)
    print(f"[phunfold] wrote {a.out}.png")


def cmd_plot(a):
    from .unfold import UnfoldResult

    res = UnfoldResult.load(a.npz)
    a.out = a.out or os.path.splitext(a.npz)[0]
    ref = np.load(a.bulk_bands) if a.bulk_bands else None
    _plot(res, a, ref)


# ----------------------------------------------------------------------
def _plot_opts(p):
    p.add_argument("--units", default="THz", choices=["THz", "cm-1", "meV"])
    p.add_argument("--ylim", type=float, nargs=2)
    p.add_argument("--sigma", type=float, default=0.12,
                   help="Lorentzian broadening in --units")
    p.add_argument("--size", type=float, default=10.0, help="marker size")
    p.add_argument("--wall-width", type=float, default=2.5,
                   help="atoms closer than this (A) to a wall count as wall")
    p.add_argument("--dpi", type=int, default=200)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="phunfold", description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

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

    def vasp_opts(p):
        p.add_argument("--kspacing", type=float, default=0.2,
                       help="k spacing (1/A incl. 2pi, like VASP KSPACING)")
        p.add_argument("--potcar-dir", help="dir with <element>/POTCAR")
        p.add_argument("--potcar-variant", nargs="*",
                       help="e.g. Ba=Ba_sv Ti=Ti_sv O=O")
        p.add_argument("--set", nargs="*", help="override INCAR tags, "
                       "e.g. ENCUT=650 NCORE=8")

    p = sub.add_parser("setup", help="write inputs for one VASP stage")
    p.add_argument("--stage", required=True,
                   choices=["relax_bulk", "born", "relax_dw", "single_point"])
    p.add_argument("--poscar", required=True)
    p.add_argument("--dir", required=True)
    vasp_opts(p)
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("setup-disp", help="dirs for phonopy displacements")
    p.add_argument("--root", default=".")
    p.add_argument("--pattern", default="POSCAR-*")
    p.add_argument("--sposcar", default="SPOSCAR")
    vasp_opts(p)
    p.set_defaults(func=cmd_setup_disp)

    p = sub.add_parser("unfold", help="unfold and plot")
    p.add_argument("--backend", default="native", choices=["native", "phonopy"])
    p.add_argument("--unitcell", default="POSCAR",
                   help="[native] the cell given to 'phonopy -d -c'")
    p.add_argument("--sposcar", default="SPOSCAR", help="[native]")
    p.add_argument("--fc", default="FORCE_CONSTANTS",
                   help="[native] phonopy FORCE_CONSTANTS (use --full-fc)")
    p.add_argument("--phonopy-yaml", default="phonopy_disp.yaml",
                   help="[phonopy]")
    p.add_argument("--force-sets", default="FORCE_SETS", help="[phonopy]")
    p.add_argument("--fc-phonopy", default=None,
                   help="[phonopy] FORCE_CONSTANTS instead of FORCE_SETS")
    p.add_argument("--born", help="[phonopy] BORN file of the DW cell itself")
    p.add_argument("--born-outcar", help="[phonopy] OUTCAR of the bulk "
                   "LEPSILON run; Z* tiled onto the DW cell for NAC")
    p.add_argument("--reference", required=True,
                   help="5-atom reference POSCAR to unfold onto")
    p.add_argument("--supercell-matrix", help="JSON, e.g. '[[8,0,0],[0,1,0],"
                   "[0,0,1]]' (auto-detected if omitted)")
    p.add_argument("--path", help="'G 0 0 0, X 1/2 0 0, M 1/2 1/2 0 | ...' "
                   "in reference reduced coords (default: tetragonal path)")
    p.add_argument("--density", type=float, default=60.0,
                   help="k-points per 1/A along the path")
    p.add_argument("--dw-info", default="dw_info.json")
    p.add_argument("--bulk-dir", help="dir with POSCAR, SPOSCAR, "
                   "FORCE_CONSTANTS of the bulk 5-atom phonon run (overlay)")
    p.add_argument("--no-distortion-phase", action="store_true",
                   help="phonopy.unfolding convention (ignore r_s - x_s)")
    p.add_argument("--out", default="unfolded")
    _plot_opts(p)
    p.set_defaults(func=cmd_unfold)

    p = sub.add_parser("plot", help="re-plot a saved .npz")
    p.add_argument("npz")
    p.add_argument("--bulk-bands", help="*_bulk_bands.npy from unfold")
    p.add_argument("--out")
    _plot_opts(p)
    p.set_defaults(func=cmd_plot)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    sys.exit(main())
