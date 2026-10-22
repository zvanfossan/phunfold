"""Construct 180-degree ferroelectric domain-wall supercells from a relaxed
polar bulk cell (any ABO3 perovskite, or any small polar cell in general).

Model: r_s = r_ref + f(x_s) * u_kappa, where u_kappa = r_polar - r_ref is the
bulk polar displacement of sublattice kappa, x_s is the position along the
wall normal (in reference-cell units) and f is a periodic profile that is +1
in domain 1, -1 in domain 2 and 0 on the two wall planes. Two walls are
created (periodic boundary conditions), separated by ``ncells`` cells.
"""

from __future__ import annotations

import json

import numpy as np

from .structure import Structure

AXES = {"a": 0, "b": 1, "c": 2, "x": 0, "y": 1, "z": 2}


def snap_reference(polar: Structure, grid=0.5) -> Structure:
    """Centrosymmetric reference by snapping fractional coordinates to the
    nearest multiple of ``grid`` (0.5 works for the standard perovskite
    setting). The uniform shift that minimises the total displacement is
    removed first so an arbitrary origin is harmless."""
    f = polar.frac.copy()
    snapped = np.rint(f / grid) * grid
    shift = (f - snapped).mean(axis=0)            # remove rigid translation
    snapped = np.rint((f - shift) / grid) * grid
    return Structure(polar.lattice, snapped, polar.symbols, polar.masses)


def polar_displacements(polar: Structure, ref: Structure):
    """u_kappa (Cartesian, minimum image), with the acoustic (mean, mass
    weighted) part removed so that domains do not drift."""
    if polar.symbols != ref.symbols:
        raise ValueError("reference and polar cell must have identical atom "
                         "order")
    d = polar.frac - ref.frac
    d -= np.rint(d)
    u = d @ polar.lattice
    m = polar.masses[:, None]
    u -= (m * u).sum(0) / m.sum()
    return u


def wall_profile(x, ncells, offset=0.0, width=0.0):
    """Periodic profile on x in [0, 2*ncells): +1 / -1 domains, 0 at walls
    located at x = offset and x = offset + ncells."""
    L = 2 * ncells
    xp = np.mod(np.asarray(x) - offset, L)
    if width <= 0:
        s = np.sign(np.where(xp < ncells, 1.0, -1.0))
        on_wall = (np.abs(xp) < 1e-6) | (np.abs(xp - ncells) < 1e-6) | (
            np.abs(xp - L) < 1e-6)
        return np.where(on_wall, 0.0, s)
    return (np.tanh(xp / width) * np.tanh((ncells - xp) / width)
            * np.tanh((L - xp) / width))


def build_180_wall(polar: Structure, ncells=4, normal="a", lateral=(1, 1),
                   offset=0.0, width=0.0, ref: Structure | None = None,
                   allow_charged=False):
    """Return (supercell Structure, info dict).

    ncells  : reference cells per domain along the wall normal (wall
              separation = ncells * a). Total cells along normal = 2*ncells.
    normal  : which bulk lattice vector the walls are perpendicular to.
    lateral : repetitions along the two in-plane lattice vectors (in a, b, c
              order with the normal removed).
    offset  : wall position in cell units; 0 puts the wall on the plane of
              the atom at the origin (A-site / BaO-centred for standard
              settings), 0.5 on the B-site (TiO2) plane.
    width   : tanh width (cells); 0 = abrupt wall (relaxation sorts it out).
    """
    ax = AXES[normal] if isinstance(normal, str) else int(normal)
    if ref is None:
        ref = snap_reference(polar)
    u = polar_displacements(polar, ref)
    pdir = np.abs(u).sum(axis=0)
    polar_axis = int(np.argmax(pdir))
    pol_lat = int(np.argmax(np.abs(polar.lattice @ np.eye(3)[polar_axis])))
    if pol_lat == ax and not allow_charged:
        raise ValueError(
            "The polarisation is along the chosen wall normal: this makes "
            "charged head-to-head/tail-to-tail walls. Choose another normal "
            "(neutral 180-degree walls need P parallel to the wall plane) or "
            "pass allow_charged=True.")
    reps = [0, 0, 0]
    reps[ax] = 2 * ncells
    others = [i for i in range(3) if i != ax]
    reps[others[0]], reps[others[1]] = lateral
    M = np.diag(reps)
    lat = M @ polar.lattice
    inv = np.linalg.inv(lat)
    frac, syms, masses = [], [], []
    for n in np.ndindex(*reps):
        for k in range(polar.natoms):
            xid = ref.frac[k] + np.array(n)          # ideal, in bulk units
            x_norm = xid[ax]
            f = wall_profile(x_norm, ncells, offset, width)
            r = xid @ polar.lattice + f * u[k]
            frac.append(r @ inv)
            syms.append(polar.symbols[k])
            masses.append(polar.masses[k])
    sc = Structure(lat, np.array(frac) % 1.0, syms, masses)
    walls = [(offset % (2 * ncells)) / (2 * ncells),
             ((offset + ncells) % (2 * ncells)) / (2 * ncells)]
    info = dict(type="180deg", normal_axis=ax, wall_positions=walls,
                supercell_matrix=M.tolist(), ncells_per_domain=ncells,
                offset=offset, width=width, polar_axis_cartesian=polar_axis,
                displacements_A=np.round(u, 5).tolist(),
                symbols_ref=polar.symbols)
    return sc, info


def write_info(info, filename="dw_info.json"):
    with open(filename, "w") as fh:
        json.dump(info, fh, indent=2)


def read_info(filename="dw_info.json"):
    with open(filename) as fh:
        return json.load(fh)
