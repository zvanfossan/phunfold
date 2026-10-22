"""Assign supercell atoms to sites of a reference lattice.

Handles, in general:
* arbitrary origin shift between supercell and reference (``origin='auto'``)
* rotated / re-based supercells (several candidate M are tried; the one with
  the smallest atom-to-site residual wins)
* substitutions: ``species_map={'Sr': 'Ba'}`` lets Sr occupy Ba sites, or
  ``match_species=False`` ignores species altogether (any atom, any site)
* vacancies (``allow_vacancies``): empty reference sites are simply absent
* interstitials (``allow_interstitials``): atoms farther than
  ``max_displacement`` from every site get their own one-atom "ghost"
  sublattice; their weight is spread uniformly over k (as in
  phonopy.unfolding, where an interstitial lives in one virtual cell).
"""

from __future__ import annotations

import numpy as np

from .structure import Structure


class SiteMapping:
    """Result of mapping; attributes documented in ``map_sites``."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def summary(self):
        s = (f"{self.n_mapped} atoms on {self.n_sites} sites "
             f"({self.N} cells x {self.n_ref} sites)")
        if self.n_vacancies:
            s += f", {self.n_vacancies} vacancies"
        if self.n_interstitials:
            s += f", {self.n_interstitials} interstitials"
        if self.n_substitutions:
            s += f", {self.n_substitutions} substituted atoms"
        return s


def _compatible(sc_symbols, ref_symbols, species_map, match_species):
    """Boolean (n_sc, n_ref): may supercell atom s sit on reference site k?"""
    n_sc, n_ref = len(sc_symbols), len(ref_symbols)
    if not match_species:
        return np.ones((n_sc, n_ref), bool)
    species_map = species_map or {}
    allowed = {}
    for s in set(sc_symbols):
        opts = species_map.get(s, s)
        opts = [opts] if isinstance(opts, str) else list(opts)
        if s not in opts:
            opts = opts + [s]
        allowed[s] = set(opts)
    return np.array([[r in allowed[s] for r in ref_symbols]
                     for s in sc_symbols])


def _assign(cart, ref_lat, ref_frac, comp, shift):
    """Nearest compatible site for every atom. Returns kappa, R, residual,
    dist."""
    f = (cart - shift) @ np.linalg.inv(ref_lat)
    n = len(cart)
    kappa = np.full(n, -1)
    lat_pt = np.zeros((n, 3), int)
    best = np.full(n, np.inf)
    resid = np.zeros((n, 3))
    for k, tau in enumerate(ref_frac):
        d = f - tau
        r = np.rint(d)
        res = (d - r) @ ref_lat
        dist = np.linalg.norm(res, axis=1)
        take = comp[:, k] & (dist < best)
        kappa[take], lat_pt[take] = k, r[take].astype(int)
        best[take], resid[take] = dist[take], res[take]
    return kappa, lat_pt, resid, best


def map_sites(sc: Structure, ref: Structure, M, origin="auto",
              species_map=None, match_species=True, max_displacement=0.8,
              allow_vacancies=False, allow_interstitials=False):
    """Map ``sc`` atoms onto the lattice inv(M) @ sc.lattice decorated with
    ``ref``'s basis.

    Returns SiteMapping with
      kappa (n_sc,)       sublattice index; >= n_ref for interstitials
      n_sub               number of sublattices incl. ghost ones
      lattice_points      integer reference-lattice point of each atom
      displacements       r_s - x_s (Cartesian, 0 for interstitials)
      ideal_cart          x_s
      origin              Cartesian shift applied to the reference
      max_residual, rms_residual
    """
    M = np.array(M, dtype=int)
    N = int(round(abs(np.linalg.det(M))))
    ref_lat = np.linalg.inv(M) @ sc.lattice
    comp = _compatible(sc.symbols, ref.symbols, species_map, match_species)
    if not comp.any(axis=1).all() and not allow_interstitials:
        bad = sorted({sc.symbols[i] for i in np.where(~comp.any(1))[0]})
        raise ValueError(f"species {bad} have no compatible reference site; "
                         "use species_map (e.g. {'Sr': 'Ba'}), "
                         "match_species=False or allow_interstitials")
    cart = sc.cart

    # candidate origin shifts: align one atom of the rarest mappable species
    shifts = [np.zeros(3)]
    if isinstance(origin, str) and origin == "auto":
        counts = {}
        for i, s in enumerate(sc.symbols):
            if comp[i].any():
                counts.setdefault(s, []).append(i)
        if counts:
            sp = min(counts, key=lambda s: len(counts[s]))
            s0 = counts[sp][0]
            for k in np.where(comp[s0])[0]:
                shifts.append(cart[s0] - ref.frac[k] @ ref_lat)
    elif origin is not None:
        shifts = [np.asarray(origin, float)]

    best = None
    for t in shifts:
        kappa, R, resid, dist = _assign(cart, ref_lat, ref.frac, comp, t)
        on_site = dist <= max_displacement
        cost = (dist[on_site].max() if on_site.any() else np.inf,
                (~on_site).sum())
        key = (cost[1], round(cost[0], 4))
        if best is None or key < best[0]:
            best = (key, t, kappa, R, resid, dist)
    _, shift, kappa, R, resid, dist = best

    inter = dist > max_displacement
    if inter.any() and not allow_interstitials:
        i = int(np.argmax(dist))
        raise ValueError(
            f"atom {i} ({sc.symbols[i]}) is {dist[i]:.2f} A from the nearest "
            "compatible site. Wrong reference cell? If it is a genuine "
            "interstitial use allow_interstitials=True; if the distortion is "
            "genuinely that large increase max_displacement.")
    n_ref = ref.natoms
    kappa = kappa.copy()
    resid = resid.copy()
    ghost = np.where(inter)[0]
    for j, s in enumerate(ghost):
        kappa[s] = n_ref + j
        resid[s] = 0.0
        R[s] = np.floor((cart[s] - shift) @ np.linalg.inv(ref_lat)).astype(int)

    # one atom per site
    g = R[~inter] @ np.linalg.inv(M)
    g -= np.floor(g + 1e-6)
    keys = [(int(k), *np.rint(gi * 1e4).astype(int)) for k, gi in
            zip(kappa[~inter], g)]
    if len(set(keys)) != len(keys):
        raise ValueError("two atoms were assigned to the same reference site; "
                         "check the reference cell / max_displacement")
    n_sites = N * n_ref
    n_vac = n_sites - len(keys)
    if n_vac and not allow_vacancies:
        raise ValueError(f"{n_vac} reference sites are empty; pass "
                         "allow_vacancies=True if the supercell has vacancies")
    subst = sum(1 for s, k in zip(np.where(~inter)[0], kappa[~inter])
                if sc.symbols[s] != ref.symbols[k])
    on = dist[~inter]
    return SiteMapping(
        kappa=kappa, n_ref=n_ref, n_sub=n_ref + len(ghost), N=N, M=M,
        ref_lattice=ref_lat, lattice_points=R, displacements=resid,
        ideal_cart=cart - resid, origin=shift, n_sites=n_sites,
        n_mapped=int((~inter).sum()), n_vacancies=int(n_vac),
        n_interstitials=int(len(ghost)), n_substitutions=int(subst),
        interstitial_atoms=ghost,
        max_residual=float(on.max()) if len(on) else 0.0,
        rms_residual=float(np.sqrt((on ** 2).mean())) if len(on) else 0.0)
