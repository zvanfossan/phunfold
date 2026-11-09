"""Ferroelectric domain-wall supercells with arbitrary polarization directions.

Both domains are built from ONE relaxed polar bulk cell (a 5-atom
pseudocubic perovskite cell, axes along x, y, z: tetragonal, orthorhombic,
rhombohedral...). Each domain i is the bulk structure transformed by a cubic
point operation g_i chosen so that g_i maps the bulk polarization direction
onto the requested direction p_i ([001], [00-1], [111], [-111], [110], ...).
Applying g_i to the whole bulk pattern, not just rotating a vector, keeps
non-collinear displacements correct: e.g. which oxygen is apical, and the
spontaneous strain, which is rotated as g_i U g_i^T.

Model, with x the coordinate along the wall normal in stacking repeats:
    r_s = a_pc * p_s @ U_avg^T + w1(x) u^(1)_kappa + w2(x) u^(2)_kappa
p_s is the ideal (centrosymmetric) site in cubic units, U_avg the average of
the two domains' stretch tensors (one lattice must host both domains),
u^(i) the domain-i displacement pattern, and w1 = (1+f)/2, w2 = (1-f)/2
with f the periodic wall profile (+1 domain 1, -1 domain 2, 0 on the
walls). For p2 = -p1 this reduces exactly to the original 180-degree
builder. Two walls are created (periodic boundary conditions).
"""

from __future__ import annotations

import itertools
import json
import warnings
from functools import reduce
from math import gcd

import numpy as np

from .structure import Structure, make_supercell

AXES = {"a": 0, "b": 1, "c": 2, "x": 0, "y": 1, "z": 2}


# ----------------------------------------------------------------- basics
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


# ----------------------------------------------------- crystallography
def primitive_int(v):
    """Integer direction with gcd 1, e.g. [0,0,2] -> [0,0,1]."""
    v = np.rint(np.asarray(v, float)).astype(int)
    if not v.any():
        raise ValueError("direction [0 0 0] is not allowed")
    g = reduce(gcd, [abs(int(x)) for x in v if x])
    return v // g


def fmt_dir(v):
    return "[" + "".join(str(int(x)) for x in v) + "]"


def cubic_point_group():
    """The 48 operations of m-3m as integer matrices. Order: identity,
    inversion, other proper rotations, other improper operations (a stable,
    simple-first preference when several operations do the same job)."""
    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            g = np.zeros((3, 3), int)
            for i, p in enumerate(perm):
                g[i, p] = signs[i]
            ops.append(g)
    eye = np.eye(3, dtype=int)

    def key(g):
        if np.array_equal(g, eye):
            return (0,)
        if np.array_equal(g, -eye):
            return (1,)
        return (2 if round(np.linalg.det(g)) > 0 else 3,)
    return sorted(ops, key=key)


def find_operation(p_from, p_to):
    """First cubic point operation g with g @ p_from == p_to."""
    for g in cubic_point_group():
        if np.array_equal(g @ p_from, p_to):
            return g
    return None


def sublattice_permutation(g, frac, symbols, tol=1e-3):
    """perm[k] = k' such that g @ tau_k == tau_k' (mod 1), same species."""
    perm = []
    for k, t in enumerate(frac):
        gt = g @ t
        hit = [j for j, t2 in enumerate(frac)
               if symbols[j] == symbols[k]
               and np.abs((gt - t2) - np.rint(gt - t2)).max() < tol]
        if len(hit) != 1:
            raise ValueError(
                "the centrosymmetric parent of the bulk cell is not invariant "
                f"under {g.tolist()}: is the bulk a 5-atom pseudocubic "
                "perovskite cell with axes along x, y, z?")
        perm.append(hit[0])
    return np.array(perm)


def stretch_and_rotation(lattice, tol=0.15):
    """a_pc, symmetric stretch U and rotation R with lattice = a_pc (R U)^T
    (rows = lattice vectors), for a cell whose vectors are close to a_pc*x,
    a_pc*y, a_pc*z."""
    a_pc = abs(np.linalg.det(lattice)) ** (1 / 3)
    F = (lattice / a_pc).T                      # columns = lattice vectors
    if np.abs(F - np.eye(3)).max() > tol:
        raise ValueError(
            "the bulk cell must be a 5-atom pseudocubic cell with lattice "
            "vectors close to a*x, a*y, a*z (as written by 'phunfold "
            "perovskite'); this one deviates by "
            f"{np.abs(F - np.eye(3)).max():.3f}")
    w, s, vt = np.linalg.svd(F)
    R = w @ vt
    U = vt.T @ np.diag(s) @ vt
    return a_pc, U, R


def infer_polar_direction(u):
    """Closest <100>/<110>/<111>-type direction to the largest atomic
    displacement (sign: the direction that atom moves). Returns
    (integer direction, cosine)."""
    v = u[np.argmax(np.linalg.norm(u, axis=1))]
    cands = [np.array(c) for c in itertools.product((-1, 0, 1), repeat=3)
             if any(c)]
    cos = [c @ v / np.linalg.norm(c) / np.linalg.norm(v) for c in cands]
    i = int(np.argmax(cos))
    return primitive_int(cands[i]), float(cos[i])


def inplane_basis(n, nmax=4):
    """Shortest integer vectors v1, v2 perpendicular to n that span the
    primitive 2D lattice of the (hkl) plane, with det[v1, v2, n] > 0."""
    n = np.asarray(n)
    cand = [np.array(c) for c in itertools.product(range(-nmax, nmax + 1),
                                                   repeat=3)
            if any(c) and np.dot(c, n) == 0]
    cand.sort(key=lambda c: (np.dot(c, c), [-x for x in c]))
    v1 = cand[0]
    area = np.linalg.norm(n)
    for v2 in cand[1:]:
        cr = np.cross(v1, v2)
        if np.linalg.norm(cr) > 1e-9 and abs(np.linalg.norm(cr) - area) < 1e-9:
            if np.dot(cr, n) < 0:
                v2 = -v2
            return v1, v2
    raise ValueError(f"no in-plane basis found for normal {fmt_dir(n)}")


def parse_normal(normal):
    """'a'/'b'/'c', an integer triple, or a string like '0 1 1' -> int vector."""
    if isinstance(normal, str):
        tok = normal.replace(",", " ").split()
        if len(tok) == 1 and tok[0].lower() in AXES:
            v = np.zeros(3, int)
            v[AXES[tok[0].lower()]] = 1
            return v
        return primitive_int([int(t) for t in tok])
    if isinstance(normal, (int, np.integer)):
        v = np.zeros(3, int)
        v[int(normal)] = 1
        return v
    return primitive_int(normal)


# ----------------------------------------------------------------- builder
def build_wall(bulk: Structure, p1=None, p2=None, normal="a", ncells=4,
               lateral=(1, 1), offset=0.0, width=0.0,
               ref: Structure | None = None, allow_charged=False,
               bulk_direction=None, strain="average"):
    """Domain-wall supercell between domains polarized along p1 and p2.

    bulk     : relaxed polar 5-atom pseudocubic cell (axes along x, y, z).
    p1, p2   : polarization directions [uvw] of the two domains (cubic
               axes). Must be symmetry-equivalent to the bulk polar
               direction. Defaults: p1 = bulk direction, p2 = -p1 (180 deg).
    normal   : wall normal: 'a'/'b'/'c' or a direction [hkl] such as
               (0, 1, 1). Axis-aligned normals keep the stacking along that
               lattice vector (diagonal supercell matrix, as before); other
               normals use lattice vectors (v1, v2, n), stacking along the
               third.
    ncells   : stacking repeats per domain. One repeat is the vector n
               (length |n| a_pc; |n|^2 cells of volume).
    lateral  : repetitions along the two in-plane vectors.
    offset   : wall position in stacking repeats (0: plane through the
               origin atom, 0.5 for [100] normals: the B-site plane).
    width    : tanh width (repeats); 0 = abrupt wall.
    bulk_direction : override the inferred bulk polar direction.
    strain   : 'average' (lattice = mean of the two domain strains) or
               'domain1'.

    Returns (supercell, info, (domain1_cell, domain2_cell)).
    """
    if ref is None:
        ref = snap_reference(bulk)
    a_pc, U, R = stretch_and_rotation(bulk.lattice)
    u = polar_displacements(bulk, ref) @ R          # into the parent frame
    if bulk_direction is None:
        p_b, cos = infer_polar_direction(u)
        if cos < 0.95:
            warnings.warn(
                f"bulk displacements are {np.degrees(np.arccos(cos)):.1f} deg "
                f"from {fmt_dir(p_b)}; pass bulk_direction if that is not the "
                "polar axis")
    else:
        p_b = primitive_int(bulk_direction)
    p1 = p_b.copy() if p1 is None else primitive_int(p1)
    p2 = -p1 if p2 is None else primitive_int(p2)
    pattern = sorted(abs(x) for x in p_b)
    for name, p in (("p1", p1), ("p2", p2)):
        if sorted(abs(x) for x in p) != pattern:
            raise ValueError(
                f"{name} = {fmt_dir(p)} is not equivalent to the bulk polar "
                f"direction {fmt_dir(p_b)}: a domain must be a symmetry copy "
                "of the bulk phase (use a bulk cell of the right phase, e.g. "
                "rhombohedral for <111> domains)")
    g1, g2 = find_operation(p_b, p1), find_operation(p_b, p2)
    perm1 = sublattice_permutation(g1, ref.frac, ref.symbols)
    perm2 = sublattice_permutation(g2, ref.frac, ref.symbols)
    nb = bulk.natoms
    u1 = np.zeros((nb, 3))
    u2 = np.zeros((nb, 3))
    u1[perm1] = u @ g1.T                            # u1[g tau_k] = g u_k
    u2[perm2] = u @ g2.T
    U1, U2 = g1 @ U @ g1.T, g2 @ U @ g2.T
    if strain == "average":
        Ua = 0.5 * (U1 + U2)
    elif strain == "domain1":
        Ua = U1
    else:
        raise ValueError("strain must be 'average' or 'domain1'")

    # ---- supercell matrix and stacking
    n = parse_normal(normal)
    axis_aligned = int(np.count_nonzero(n)) == 1
    if axis_aligned:
        ax = int(np.flatnonzero(n)[0])
        reps = [0, 0, 0]
        reps[ax] = 2 * ncells
        others = [i for i in range(3) if i != ax]
        reps[others[0]], reps[others[1]] = lateral
        M = np.diag(reps)
        n = np.abs(n)
        v1, v2 = np.eye(3, dtype=int)[others[0]], np.eye(3, dtype=int)[others[1]]
    else:
        ax = 2
        v1, v2 = inplane_basis(n)
        M = np.array([lateral[0] * v1, lateral[1] * v2, 2 * ncells * n])

    # ---- neutrality and mechanical compatibility
    ph1, ph2 = p1 / np.linalg.norm(p1), p2 / np.linalg.norm(p2)
    nh = n / np.linalg.norm(n)
    bound = float((ph1 - ph2) @ nh)
    if abs(bound) > 1e-8 and not allow_charged:
        dp = p1 * np.linalg.norm(p2) - p2 * np.linalg.norm(p1)
        raise ValueError(
            f"wall normal {fmt_dir(n)} gives a CHARGED wall: (P1 - P2) . n != "
            f"0 for P1 {fmt_dir(p1)}, P2 {fmt_dir(p2)}. Neutral walls need "
            f"the normal perpendicular to {np.round(dp, 3).tolist()}. Pass "
            "allow_charged=True (--allow-charged) to build it anyway.")
    dU = U1 - U2
    t = [v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)]
    inplane = max(abs(ti @ dU @ tj) for ti in t for tj in t)
    if np.abs(dU).max() > 1e-5 and inplane > 0.05 * np.abs(dU).max():
        warnings.warn(
            f"the spontaneous strains of the two domains differ within the "
            f"wall plane (mismatch {inplane:.2e}): this wall orientation is "
            "not mechanically compatible, and both domains are clamped to the "
            "average lattice")

    # ---- positions
    parent = Structure(np.eye(3), ref.frac, ref.symbols, ref.masses)
    sc0 = make_supercell(parent, M)
    p = sc0.frac @ M                                # ideal sites, cubic units
    kappa = np.tile(np.arange(nb), len(p) // nb)
    x = p[:, ax] if axis_aligned else p @ n / float(n @ n)
    f = wall_profile(x, ncells, offset, width)
    w1, w2 = (1 + f) / 2, (1 - f) / 2
    disp = w1[:, None] * u1[kappa] + w2[:, None] * u2[kappa]
    lat = a_pc * M @ Ua.T
    cart = a_pc * p @ Ua.T + disp
    sc = Structure(lat, (cart @ np.linalg.inv(lat)) % 1.0,
                   [bulk.symbols[k] for k in kappa], bulk.masses[kappa])

    # ---- single-domain 5-atom cells (bulk references, overlays)
    doms = []
    for Ui, ui in ((U1, u1), (U2, u2)):
        li = a_pc * Ui.T
        ci = a_pc * ref.frac @ Ui.T + ui
        doms.append(Structure(li, (ci @ np.linalg.inv(li)) % 1.0,
                              bulk.symbols, bulk.masses))

    # ---- info
    L = 2 * ncells
    walls = [(offset % L) / L, ((offset + ncells) % L) / L]
    angle = float(np.degrees(np.arccos(np.clip(ph1 @ ph2, -1, 1))))
    legacy = (axis_aligned and ax == 0 and sorted(abs(x) for x in p1) == [0, 0, 1]
              and abs(p1[2]) == 1 and np.array_equal(p2, -p1))
    N, T1, T2 = n / 2, v1 / 2, v2 / 2
    fm = lambda v: " ".join(f"{c:g}" for c in v)  # noqa: E731
    path = ("dw-tetragonal" if legacy else
            f"G 0 0 0, N {fm(N)} | G 0 0 0, T1 {fm(T1)}, T1+N {fm(T1 + N)} | "
            f"G 0 0 0, T2 {fm(T2)}")
    info = dict(
        type=f"{angle:.0f}deg", wall_angle_deg=round(angle, 3),
        p1=p1.tolist(), p2=p2.tolist(), bulk_direction=p_b.tolist(),
        normal_hkl=n.tolist(), inplane_vectors=[v1.tolist(), v2.tolist()],
        normal_axis=ax, wall_positions=walls, supercell_matrix=M.tolist(),
        ncells_per_domain=ncells, repeat_length_A=float(np.linalg.norm(a_pc * n @ Ua.T)),
        offset=offset, width=width, strain=strain,
        lattice_constant_pc=float(a_pc), rotations=[g1.tolist(), g2.tolist()],
        bulk_rotation=R.tolist(), basis_frac=np.asarray(ref.frac).tolist(),
        symbols_ref=list(bulk.symbols), displacements_A=np.round(u, 5).tolist(),
        polar_axis_cartesian=int(np.argmax(np.abs(p_b))), path=path,
        charged=bool(abs(bound) > 1e-8))
    return sc, info, tuple(doms)


def build_180_wall(polar: Structure, ncells=4, normal="a", lateral=(1, 1),
                   offset=0.0, width=0.0, ref: Structure | None = None,
                   allow_charged=False):
    """180-degree wall (p2 = -p1, p1 = bulk direction). Kept for backward
    compatibility; returns (supercell, info)."""
    sc, info, _ = build_wall(polar, normal=normal, ncells=ncells,
                             lateral=lateral, offset=offset, width=width,
                             ref=ref, allow_charged=allow_charged)
    return sc, info


# ---------------------------------------------------- Born charges per domain
def domain_born_charges(cell: Structure, info, born_bulk, eps_bulk,
                        on_wall_tol=0.15):
    """Born charges for every atom of a wall cell built by ``build_wall``
    (relaxed or not, any atom order), from the bulk LEPSILON tensors.

    Atoms in domain i get g_i Z_k g_i^T for the bulk sublattice k they come
    from. Atoms on a wall plane get the average of both domains. eps_inf is
    the average of the two rotated tensors. ``born_bulk`` must be in the atom
    order of the bulk cell given to build_wall.
    Returns (born (n, 3, 3), eps (3, 3), domain index per atom: 1, 2, 0=wall).
    """
    M = np.array(info["supercell_matrix"])
    g = [np.array(r) for r in info["rotations"]]
    R = np.array(info.get("bulk_rotation", np.eye(3)))
    basis = np.array(info["basis_frac"])
    syms = info["symbols_ref"]
    Zb = np.einsum("ji,njk,kl->nil", R, np.asarray(born_bulk, float), R)  # R^T Z R
    Eb = R.T @ np.asarray(eps_bulk, float) @ R
    if len(Zb) != len(basis):
        raise ValueError(f"Born charges for {len(Zb)} atoms, but the bulk cell "
                         f"of this wall has {len(basis)}")
    perms = [sublattice_permutation(gi, basis, syms) for gi in g]
    inv = [np.argsort(pm) for pm in perms]          # k' -> bulk k
    Zd = [np.einsum("ij,njk,lk->nil", gi, Zb, gi)[iv] for gi, iv in zip(g, inv)]
    p = cell.frac @ M
    n = np.array(info["normal_hkl"])
    ax = info["normal_axis"]
    x = p[:, ax] if int(np.count_nonzero(n)) == 1 else p @ n / float(n @ n)
    ncl = info["ncells_per_domain"]
    xp = np.mod(x - info["offset"], 2 * ncl)
    dwall = np.minimum.reduce([np.abs(xp), np.abs(xp - ncl), np.abs(xp - 2 * ncl)])
    dom = np.where(dwall < on_wall_tol, 0, np.where(xp < ncl, 1, 2))
    born = np.zeros((cell.natoms, 3, 3))
    for s in range(cell.natoms):
        d = (p[s] % 1.0) - basis
        d -= np.rint(d)
        cand = [k for k in range(len(basis)) if syms[k] == cell.symbols[s]]
        k = min(cand, key=lambda j: np.linalg.norm(d[j]))
        born[s] = (Zd[0][k] if dom[s] == 1 else Zd[1][k] if dom[s] == 2
                   else 0.5 * (Zd[0][k] + Zd[1][k]))
    born -= born.mean(axis=0)[None]
    eps = 0.5 * (g[0] @ Eb @ g[0].T + g[1] @ Eb @ g[1].T)
    return born, eps, dom


def write_info(info, filename="dw_info.json"):
    with open(filename, "w") as fh:
        json.dump(info, fh, indent=2)


def read_info(filename="dw_info.json"):
    with open(filename) as fh:
        return json.load(fh)
