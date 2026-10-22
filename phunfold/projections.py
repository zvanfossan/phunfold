"""Atom groups for projecting supercell modes.

A projection is an atom mask; for every mode the code stores the fraction of
|e|^2 on that group (0..1). Groups can be given as masks/index lists in
Python or as compact strings (CLI ``--project NAME=SPEC``):

  species:Ti,O               atoms of these species
  atoms:0-9,15               supercell atom indices (0-based, backend order)
  sites:1,2                  atoms sitting on these reference sublattices
  slab:AXIS:F1,F2:W          within W Angstrom of planes at fractional
                             coordinates F1, F2, ... along supercell lattice
                             vector AXIS (0/1/2) -- e.g. domain walls,
                             interfaces, stacking faults, surfaces
  sphere:F1,F2,F3:R          within R Angstrom (periodic) of a point given in
                             supercell fractional coordinates -- point defects
  interstitials              atoms that were mapped as interstitials
  substituted                atoms whose species differs from their site's

Geometric selections use ideal site positions, so they are insensitive to the
relaxation.
"""

from __future__ import annotations

import numpy as np


def _index_list(text):
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def slab_distance(sc, ideal_cart, axis, positions):
    """Distance (A) from each ideal site to the nearest of the planes."""
    f = ideal_cart @ np.linalg.inv(sc.lattice)
    spacing = 1.0 / np.linalg.norm(sc.recip[axis])
    d = np.abs(f[:, axis][:, None] - np.asarray(positions, float)[None, :])
    d = np.minimum(d % 1.0, 1.0 - d % 1.0)
    return d.min(axis=1) * spacing


def sphere_distance(sc, ideal_cart, center_frac):
    f = ideal_cart @ np.linalg.inv(sc.lattice) - np.asarray(center_frac, float)
    f -= np.rint(f)
    return np.linalg.norm(f @ sc.lattice, axis=1)


def parse_projection(spec, unfolder):
    """Turn a spec string (see module doc) into a boolean atom mask."""
    sc, mp = unfolder.sc, unfolder.mapping
    kind, _, rest = spec.partition(":")
    kind = kind.strip().lower()
    n = sc.natoms
    if kind == "species":
        sp = {s.strip() for s in rest.split(",")}
        return np.array([s in sp for s in sc.symbols])
    if kind == "atoms":
        m = np.zeros(n, bool)
        m[_index_list(rest)] = True
        return m
    if kind == "sites":
        return np.isin(mp.kappa, _index_list(rest))
    if kind == "slab":
        axis, pos, width = rest.split(":")
        pos = [float(p) for p in pos.split(",")]
        return slab_distance(sc, mp.ideal_cart, int(axis), pos) < float(width)
    if kind == "sphere":
        c, r = rest.split(":")
        c = [float(x) for x in c.split(",")]
        return sphere_distance(sc, mp.ideal_cart, c) < float(r)
    if kind == "interstitials":
        m = np.zeros(n, bool)
        m[mp.interstitial_atoms] = True
        return m
    if kind == "substituted":
        ref = unfolder.ref.symbols
        return np.array([k < mp.n_ref and s != ref[k]
                         for s, k in zip(sc.symbols, mp.kappa)])
    raise ValueError(f"unknown projection type {kind!r} in {spec!r}")


def participation(eigvecs, atom_mask):
    """Fraction of each mode's |e|^2 on the masked atoms, (nq, 3N)."""
    m = np.repeat(np.asarray(atom_mask, bool), 3)
    return (np.abs(eigvecs[:, m, :]) ** 2).sum(axis=1)


def inverse_participation_ratio(eigvecs):
    """Per-mode IPR = sum_s |e_s|^4 (|e_s|^2 summed over x, y, z).
    1/N for a fully delocalised mode, 1 for a single-atom mode."""
    n = eigvecs.shape[1] // 3
    a = (np.abs(eigvecs.reshape(eigvecs.shape[0], n, 3, -1)) ** 2).sum(axis=2)
    return (a ** 2).sum(axis=1)
