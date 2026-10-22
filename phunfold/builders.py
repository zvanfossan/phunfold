"""POSCAR builders for the steps of the workflow that need new structures.

* ``perovskite``: ideal cubic ABX3, or a tetragonal/polar starting cell for
  a relaxation (displacements along one axis).
* domain walls: see ``domainwall.build_180_wall``.
* supercells / rattled test cells: ``structure.make_supercell``, ``rattle``.
* displacement folders for VASP: ``make_disp_dirs``.
"""

from __future__ import annotations

import glob
import os
import re
import shutil

import numpy as np

from .structure import Structure, write_poscar

AXES = {"a": 0, "b": 1, "c": 2, "x": 0, "y": 1, "z": 2}


def perovskite(a, c=None, species=("Ba", "Ti", "O"), polar_axis=None,
               disp=None):
    """ABX3 perovskite, A at the origin, B at the body centre.

    a          : lattice constant (A).
    c          : length along ``polar_axis`` (default a) -> tetragonal cell.
    polar_axis : 'a' | 'b' | 'c' | None. Axis of c and of the displacements.
    disp       : Cartesian displacements (A) along polar_axis, keys
                 'A', 'B', 'X_ap' (apical X, on the B-X bond along the polar
                 axis), 'X_eq' (the two equatorial X), or 'X' for all three.
                 Typical BaTiO3 start: B=+0.10, X_ap=-0.08, X_eq=-0.06.
    Atom order: A, B, X_ax0, X_ax1, X_ax2, where X_axk has coordinate 0 along
    axis k (X_ax2 = (1/2,1/2,0) is apical for a polarization along c).
    """
    ax = None if polar_axis in (None, "none") else AXES[polar_axis]
    lengths = [a, a, a]
    if c is not None:
        if ax is None:
            ax = 2
        lengths[ax] = c
    lat = np.diag(lengths).astype(float)
    frac = np.array([[0, 0, 0], [.5, .5, .5],
                     [0, .5, .5], [.5, 0, .5], [.5, .5, 0]], float)
    sym = [species[0], species[1], species[2], species[2], species[2]]
    if disp:
        if ax is None:
            raise ValueError("displacements need a polar axis")
        d = dict(disp)
        dx = np.zeros(5)
        dx[0] = d.get("A", 0.0)
        dx[1] = d.get("B", 0.0)
        for k in range(3):
            key = "X_ap" if k == ax else "X_eq"
            dx[2 + k] = d.get(key, d.get("X", 0.0))
        frac[:, ax] += dx / lengths[ax]
    return Structure(lat, frac % 1.0, sym)


def parse_disp(items):
    """['B=0.1', 'X_ap=-0.08'] -> dict."""
    out = {}
    for it in items or []:
        k, v = it.split("=")
        out[k.strip()] = float(v)
    return out


def make_disp_dirs(root=".", pattern="POSCAR-*", sposcar="SPOSCAR",
                   copy=None, perfect=True, prefix="disp-"):
    """Create one folder per phonopy displacement with its POSCAR, plus
    ``perfect/`` with the undisplaced SPOSCAR (for residual forces).

    copy : list of your own files (INCAR, KPOINTS, POTCAR, submit.sh, ...)
           copied unchanged into every folder. 'SOURCE:NAME' renames on the
           way in, e.g. '../inputs/INCAR_static:INCAR'.
    Returns the list of folders created.
    """
    files = sorted(glob.glob(os.path.join(root, pattern)),
                   key=lambda p: int(re.findall(r"(\d+)$", p)[0]))
    if not files:
        raise FileNotFoundError(
            f"no {pattern} files in {os.path.abspath(root)}; run "
            "'phonopy -d ...' (phonopy-init -d for phonopy >= 4) first")
    pairs = []
    for c in copy or []:
        src, _, dest = c.partition(":")
        pairs.append((src, dest or os.path.basename(src)))
    missing = [s for s, _ in pairs if not os.path.isfile(s)]
    if missing:
        raise FileNotFoundError(f"cannot copy {missing}: not found")
    jobs = [(os.path.join(root, sposcar), "perfect")] if perfect else []
    jobs += [(f, prefix + re.findall(r"(\d+)$", f)[0]) for f in files]
    made = []
    for src, tag in jobs:
        if not os.path.isfile(src):
            raise FileNotFoundError(src)
        d = os.path.join(root, tag)
        os.makedirs(d, exist_ok=True)
        shutil.copy(src, os.path.join(d, "POSCAR"))
        for cs, dest in pairs:
            shutil.copy(cs, os.path.join(d, dest))
        made.append(d)
    return made


def write_structure(structure, filename, comment):
    """Write POSCAR with atoms grouped by species (VASP order)."""
    return write_poscar(structure, filename, comment=comment, sort=True)
