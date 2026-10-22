"""Relating a supercell lattice to a reference lattice.

The supercell may be strained, rotated, or built on a different but
equivalent choice of reference lattice vectors. We look for integer
matrices M with  sc.lattice ~= M @ ref.lattice @ R  (R a proper rotation
plus small strain). Because the unfolding reference lattice is always
rebuilt as inv(M) @ sc.lattice, any such M yields an exactly commensurate
reference frame; the mapping step then picks the M whose basis matches the
atoms best (this matters for low-symmetry bases, e.g. polar cells).
"""

from __future__ import annotations

import itertools

import numpy as np

from .structure import Structure


def _metric_error(M, ref_lat, sc_lat):
    """Relative mismatch between the reference metric and inv(M) sc metric."""
    eff = np.linalg.inv(M) @ sc_lat
    g_eff, g_ref = eff @ eff.T, ref_lat @ ref_lat.T
    return np.abs(g_eff - g_ref).max() / np.abs(g_ref).max()


def supercell_matrix_candidates(ref: Structure, sc: Structure, tol=0.05,
                                allow_rotation=True, max_candidates=96):
    """List of integer M (best metric match first).

    tol : allowed relative strain between reference and supercell metrics.
    Only orientation-preserving matches (proper rotations) are returned so
    the reference basis is never mirrored.
    """
    ref_lat, sc_lat = ref.lattice, sc.lattice
    proper_sign = np.sign(np.linalg.det(sc_lat) * np.linalg.det(ref_lat))
    out = []
    m = sc_lat @ np.linalg.inv(ref_lat)
    mr = np.rint(m)
    if np.abs(m - mr).max() < 2 * tol * max(1, np.abs(m).max()) and \
            round(np.linalg.det(mr)) != 0:
        out.append(mr.astype(int))
    if allow_rotation:
        ref_recip = np.linalg.inv(ref_lat).T
        lens = np.linalg.norm(sc_lat, axis=1)
        per_vec = []
        for a, la in zip(sc_lat, lens):
            nmax = [int(np.ceil(la * np.linalg.norm(b) * (1 + tol))) for b in
                    ref_recip]
            grid = np.array(list(itertools.product(
                *[range(-n, n + 1) for n in nmax])))
            v = grid @ ref_lat
            lv = np.linalg.norm(v, axis=1)
            ok = np.abs(lv - la) < tol * la
            per_vec.append(grid[ok])
        g_sc = sc_lat @ sc_lat.T
        scale = lens.max() ** 2
        c0, c1, c2 = per_vec
        for n0 in c0:
            v0 = n0 @ ref_lat
            # prune pairs first
            d01 = (c1 @ ref_lat) @ v0
            good1 = c1[np.abs(d01 - g_sc[0, 1]) < 2 * tol * scale]
            for n1 in good1:
                v1 = n1 @ ref_lat
                v2s = c2 @ ref_lat
                good2 = c2[(np.abs(v2s @ v0 - g_sc[0, 2]) < 2 * tol * scale)
                           & (np.abs(v2s @ v1 - g_sc[1, 2]) < 2 * tol * scale)]
                for n2 in good2:
                    M = np.array([n0, n1, n2])
                    det = round(np.linalg.det(M))
                    if det == 0 or np.sign(det) != proper_sign:
                        continue
                    out.append(M)
    # unique, sorted by metric error
    uniq, seen = [], set()
    for M in out:
        key = tuple(M.ravel())
        if key in seen:
            continue
        seen.add(key)
        err = _metric_error(M, ref_lat, sc_lat)
        if err < 3 * tol:
            uniq.append((err, len(uniq), M))
    uniq.sort(key=lambda t: (round(t[0], 6), t[1]))
    if not uniq:
        raise ValueError(
            "Could not express the supercell lattice as an integer "
            "combination of the reference lattice (within strain tolerance "
            f"{tol}). Check the reference cell, or pass supercell_matrix.")
    return [M for _, _, M in uniq[:max_candidates]]


def find_supercell_matrix(ref: Structure, sc: Structure, tol=0.05,
                          allow_rotation=True):
    """Best-metric integer M (kept for backward compatibility)."""
    return supercell_matrix_candidates(ref, sc, tol, allow_rotation)[0]


def reference_from_supercell(sc: Structure, M, tol=0.5):
    """Average 'virtual' reference cell obtained by folding every supercell
    atom into the reference cell inv(M) @ sc.lattice and averaging the atoms
    that land on the same site (periodic clustering, ``tol`` in Angstrom).

    Useful for alloys / special quasirandom structures or relaxed cells with
    no clean reference: species of a site is the majority species there.
    """
    M = np.array(M, dtype=int)
    lat = np.linalg.inv(M) @ sc.lattice
    f = (sc.cart @ np.linalg.inv(lat)) % 1.0
    centers, members = [], []
    for i, fi in enumerate(f):
        placed = False
        for c, (cf, mem) in enumerate(zip(centers, members)):
            d = fi - cf
            d -= np.rint(d)
            if np.linalg.norm(d @ lat) < tol:
                mem.append(i)
                # running periodic mean
                ref0 = f[mem[0]]
                dd = f[mem] - ref0
                dd -= np.rint(dd)
                centers[c] = (ref0 + dd.mean(axis=0)) % 1.0
                placed = True
                break
        if not placed:
            centers.append(fi.copy())
            members.append([i])
    n_cells = int(round(abs(np.linalg.det(M))))
    sizes = [len(m) for m in members]
    if any(s > n_cells for s in sizes) or sum(sizes) != sc.natoms:
        raise ValueError("site clustering failed; reduce tol")
    symbols, masses = [], []
    for mem in members:
        syms = [sc.symbols[i] for i in mem]
        maj = max(set(syms), key=syms.count)
        symbols.append(maj)
        masses.append(np.mean([sc.masses[i] for i in mem if sc.symbols[i] == maj]))
    return Structure(lat, np.array(centers), symbols, masses)
