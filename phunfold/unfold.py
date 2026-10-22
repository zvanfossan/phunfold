"""Phonon band unfolding for arbitrary commensurate supercells.

Method: Allen, Berlijn, Casavant & Soler, PRB 87, 085322 (2013); phonon form
as in Ikeda, Carreras, Seko, Togo & Tanaka, PRB 95, 024305 (2017).

Every supercell atom s is assigned to a reference sublattice kappa(s) and an
ideal site x_s = R_s + tau_kappa (see ``mapping.map_sites``). For a
supercell mode J with r-gauge eigenvector e_{s a}(K, J) the weight on
reference wave vector k = K + G is

    W_J(k) = (1/N) sum_{kappa, a} | sum_{s in kappa} e_{s a}(K, J)
                                   exp(2 pi i [K . r_s - k . x_s]) |^2 ,

N = |det M|. We diagonalise at K = k itself, so the phase is
exp(2 pi i k . (r_s - x_s)) (``distortion_phase=True``; False reproduces
phonopy.unfolding, which drops it). The sum over kappa can be kept resolved
(``per_sublattice``): W_J = sum_kappa W_{J kappa}, giving sublattice- or
species-projected unfolded spectra.

Sum rule: sum_J W_J(k) = 3 n_atoms / N for any k (= 3 n_ref without
vacancies or interstitials). For a perfect supercell W is 0 or 1 (summed over
degenerate modes).
"""

from __future__ import annotations

import json
import warnings

import numpy as np

from .structure import Structure
from .lattice import supercell_matrix_candidates, find_supercell_matrix  # noqa
from .mapping import map_sites
from .paths import (DEFAULT_PATH, make_band_path, parse_path_string,  # noqa
                    resolve_path)
from .projections import (participation, inverse_participation_ratio,  # noqa
                          parse_projection, slab_distance)


class Unfolder:
    """Map a supercell onto a reference cell and compute unfolding weights.

    Parameters
    ----------
    supercell : Structure
        Must be the backend's own structure (``backend.structure``) so that
        positions match the gauge of the eigenvectors.
    reference : Structure
        The small cell to unfold onto. Only its basis (species + fractional
        positions) and lattice *shape* are used; orientation and origin may
        differ from the supercell.
    supercell_matrix : (3, 3) int, optional
        sc.lattice = M @ ref.lattice (rows). If omitted, all integer M that
        match the lattice metrics (any proper rotation, strain <
        ``strain_tol``) are tried and the best atom-to-site fit is kept.
    origin : 'auto', None (no shift) or Cartesian shift vector.
    species_map : dict, e.g. {'Sr': 'Ba', 'Zr': ['Ti', 'Hf']}; supercell
        species -> reference species whose sites it may occupy.
    match_species : False lets any atom occupy any site (nearest wins).
    allow_vacancies, allow_interstitials : see ``mapping``.
    max_displacement : largest allowed atom-to-site distance (A).
    distortion_phase : include exp(2 pi i k.(r_s - x_s)).
    """

    def __init__(self, supercell: Structure, reference: Structure,
                 supercell_matrix=None, max_displacement=0.8,
                 distortion_phase=True, origin="auto", species_map=None,
                 match_species=True, allow_vacancies=False,
                 allow_interstitials=False, strain_tol=0.05,
                 allow_rotation=True):
        self.sc = supercell
        self.ref = reference
        self.distortion_phase = distortion_phase
        opts = dict(origin=origin, species_map=species_map,
                    match_species=match_species,
                    max_displacement=max_displacement,
                    allow_vacancies=allow_vacancies,
                    allow_interstitials=allow_interstitials)
        self.used_rotation_search = False
        if supercell_matrix is not None:
            cands = [np.array(supercell_matrix, dtype=int)]
        else:
            # 1) the orientation as written in the files, if it works
            cands = []
            m = supercell.lattice @ np.linalg.inv(reference.lattice)
            mr = np.rint(m)
            if (np.abs(m - mr).max() < 2 * strain_tol * max(1, np.abs(m).max())
                    and round(np.linalg.det(mr)) != 0):
                cands = [mr.astype(int)]
        best, errors = None, []
        for stage in ("direct", "search"):
            if stage == "search":
                if best is not None or supercell_matrix is not None \
                        or not allow_rotation:
                    break
                # 2) any proper rotation / re-basing of the reference lattice
                cands = supercell_matrix_candidates(
                    reference, supercell, tol=strain_tol, allow_rotation=True)
                self.used_rotation_search = True
            keys = []
            for idx, M in enumerate(cands):
                try:
                    mp = map_sites(supercell, reference, M, **opts)
                except ValueError as err:
                    errors.append(str(err))
                    continue
                # ties (symmetry-equivalent choices) keep the earliest
                quality = (mp.n_interstitials, mp.n_vacancies,
                           round(mp.max_residual, 3), round(mp.rms_residual, 4))
                keys.append(quality)
                if best is None or (quality, idx) < best[0]:
                    best = ((quality, idx), mp)
            self.n_candidates_tried = len(cands)
            self.n_equivalent = sum(1 for k in keys if best and k == best[0][0])
        if best is None:
            raise ValueError("no supercell matrix gave a valid atom mapping. "
                             f"First error: {errors[0] if errors else '-'}")
        self.mapping = mp = best[1]
        self.M, self.N = mp.M, mp.N
        self.ref_lattice = mp.ref_lattice
        self.ref_recip = np.linalg.inv(self.ref_lattice).T
        self.kappa = mp.kappa
        self.n_sub = mp.n_sub
        self.lattice_points = mp.lattice_points
        self.displacements = mp.displacements
        self.ideal_cart = mp.ideal_cart
        self.max_site_displacement = mp.max_residual
        self.sum_rule = 3.0 * supercell.natoms / self.N

    def orientation_report(self):
        """How the reference axes were placed in the supercell (Cartesian
        unit vectors), to check which path direction is which."""
        lines = []
        for lab, v in zip("abc", self.ref_lattice):
            u = v / np.linalg.norm(v)
            lines.append(f"ref {lab} -> ({u[0]:+.3f}, {u[1]:+.3f}, {u[2]:+.3f})"
                         f", |{lab}| = {np.linalg.norm(v):.4f} A")
        return lines

    # ------------------------------------------------------------------
    def sublattice_labels(self):
        labs = [f"{i}:{s}" for i, s in enumerate(self.ref.symbols)]
        labs += [f"int{j}:{self.sc.symbols[s]}" for j, s in
                 enumerate(self.mapping.interstitial_atoms)]
        return labs

    def sublattice_species(self):
        """Reference species of each sublattice (interstitials: own)."""
        return list(self.ref.symbols) + [self.sc.symbols[s] for s in
                                         self.mapping.interstitial_atoms]

    def k_cart(self, q_ref):
        """Reference reduced coordinates -> Cartesian (1/A, no 2 pi)."""
        return np.atleast_2d(q_ref) @ self.ref_recip

    def weights(self, k_cart, eigvecs, gauge="r", per_sublattice=False):
        """Unfolding weights.

        k_cart : (nq, 3), the same k used to obtain ``eigvecs`` (K = k).
        eigvecs: (nq, 3 n_sc, 3 n_sc), columns = modes.
        gauge  : 'r' (phonopy / phunfold backends: phase uses atomic
                 positions) or 'R' (phase uses only lattice translations of
                 the supercell, as in some other codes).
        Returns (nq, 3 n_sc), or (nq, 3 n_sc, n_sub) if per_sublattice.
        """
        k_cart = np.atleast_2d(k_cart)
        n_sc = self.sc.natoms
        shape = (len(k_cart), 3 * n_sc) + ((self.n_sub,) if per_sublattice
                                           else ())
        out = np.empty(shape)
        r = self.sc.cart
        for iq, (k, ev) in enumerate(zip(k_cart, eigvecs)):
            e = ev.reshape(n_sc, 3, -1)
            phase = np.zeros(n_sc)
            if self.distortion_phase:
                phase += self.displacements @ k
            if gauge == "R":
                phase -= r @ k
            elif gauge != "r":
                raise ValueError("gauge must be 'r' or 'R'")
            if phase.any():
                e = e * np.exp(2j * np.pi * phase)[:, None, None]
            b = np.zeros((self.n_sub, 3, e.shape[2]), dtype=complex)
            np.add.at(b, self.kappa, e)
            p = (np.abs(b) ** 2).sum(axis=1) / self.N      # (n_sub, nb)
            out[iq] = p.T if per_sublattice else p.sum(axis=0)
        return out

    # backwards compatible helper
    def plane_distance(self, axis, positions):
        return slab_distance(self.sc, self.ideal_cart, axis, positions)


# ----------------------------------------------------------------------
def spectral_function(freqs, weights, omega, sigma=0.15, kind="lorentzian"):
    """A(k, w) = sum_J W_J(k) L(w - w_J). Returns (len(omega), nq)."""
    d = omega[:, None, None] - freqs[None, :, :]
    if kind == "lorentzian":
        kern = (sigma / np.pi) / (d ** 2 + sigma ** 2)
    elif kind == "gaussian":
        kern = np.exp(-0.5 * (d / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    else:
        raise ValueError(kind)
    return (kern * weights[None, :, :]).sum(axis=2)


class UnfoldResult:
    """Container that can be saved/loaded (.npz) and plotted.

    extra keys: 'proj_<name>' (participation of a projection), 'ipr',
    'w_<label>' (species/sublattice-resolved weights; they sum to weights).
    """

    def __init__(self, path, freqs, weights, extra=None, meta=None):
        self.path = path
        self.freqs = freqs
        self.weights = weights
        self.extra = extra or {}
        self.meta = meta or {}

    def projections(self):
        return [k[5:] for k in self.extra if k.startswith("proj_")]

    def resolved(self):
        return [k[2:] for k in self.extra if k.startswith("w_")]

    def save(self, filename):
        np.savez_compressed(
            filename, q=self.path["q"], k_cart=self.path["k_cart"],
            x=self.path["x"], ticks=self.path["ticks"],
            labels=np.array(self.path["labels"]), freqs=self.freqs,
            weights=self.weights, meta=json.dumps(self.meta),
            **{f"extra_{k}": v for k, v in self.extra.items()})

    @classmethod
    def load(cls, filename):
        z = np.load(filename, allow_pickle=False)
        path = dict(q=z["q"], k_cart=z["k_cart"], x=z["x"], ticks=z["ticks"],
                    labels=[str(s) for s in z["labels"]])
        extra = {k[6:]: z[k] for k in z.files if k.startswith("extra_")}
        return cls(path, z["freqs"], z["weights"], extra,
                   json.loads(str(z["meta"])))

    def write_text(self, filename, wmin=1e-3):
        """One line per mode with weight > wmin:
        x  q1 q2 q3  freq(THz)  weight  [extras...]"""
        keys = list(self.extra)
        with open(filename, "w") as fh:
            fh.write("# x(1/A) q1 q2 q3 freq(THz) weight " + " ".join(keys)
                     + "\n")
            for i, x in enumerate(self.path["x"]):
                q = self.path["q"][i]
                for j in np.where(self.weights[i] > wmin)[0]:
                    ex = " ".join("%.5f" % self.extra[k][i, j] for k in keys)
                    fh.write("%.6f %.5f %.5f %.5f %.6f %.6f %s\n" % (
                        x, *q, self.freqs[i, j], self.weights[i, j], ex))


def run_unfolding(backend, reference, path=DEFAULT_PATH, path_frame=None,
                  supercell_matrix=None, density=100.0, npts=None,
                  distortion_phase=True, projections=None, resolve=None,
                  gauge="r", batch=20, verbose=True, mapping_options=None,
                  # backwards-compatible domain-wall shortcut
                  wall_axis=None, wall_positions=None, wall_width=2.5,
                  path_segments=None):
    """Full pipeline: map atoms, build path, diagonalise, unfold.

    path        : preset name (see paths.PRESETS), path string, or segments.
    path_frame  : 'reference' | 'supercell' | 'cartesian' for path strings.
    projections : dict name -> spec string or boolean atom mask.
    resolve     : None | 'species' | 'sublattice' -> 'w_<label>' arrays.
    mapping_options : extra Unfolder kwargs (species_map, allow_vacancies,
                  allow_interstitials, origin, match_species, ...).
    """
    if path_segments is not None:
        path = path_segments
    unf = Unfolder(backend.structure, reference,
                   supercell_matrix=supercell_matrix,
                   distortion_phase=distortion_phase,
                   **(mapping_options or {}))
    segs = resolve_path(path, path_frame, reference.lattice, unf.ref_lattice,
                        backend.structure.lattice)
    bp = make_band_path(segs, unf.ref_recip, density, npts)
    projections = dict(projections or {})
    if wall_axis is not None and wall_positions is not None:
        projections.setdefault(
            "wall", f"slab:{wall_axis}:{','.join(map(str, wall_positions))}:"
                    f"{wall_width}")
    masks = {name: (parse_projection(spec, unf) if isinstance(spec, str)
                    else np.asarray(spec, bool))
             for name, spec in projections.items()}
    if verbose:
        how = ("orientation search" if unf.used_rotation_search
               else "as given" if supercell_matrix is not None
               else "files' own orientation")
        print(f"[phunfold] supercell: {backend.structure.natoms} atoms, "
              f"M = {unf.M.tolist()} (N = {unf.N}; {how})")
        if unf.used_rotation_search:
            for line in unf.orientation_report():
                print(f"[phunfold]   {line}")
            if unf.n_equivalent > 1:
                warnings.warn(
                    f"{unf.n_equivalent} orientations of the reference fit "
                    "the supercell equally well; the first was used. If the "
                    "supercell has lower symmetry than the reference (walls, "
                    "interfaces, strain), path directions such as X vs Y "
                    "depend on this choice: pass --supercell-matrix or give "
                    "both cells in the same orientation.")
        print(f"[phunfold] mapping: {unf.mapping.summary()}; largest "
              f"displacement from ideal site {unf.max_site_displacement:.3f} A")
        print(f"[phunfold] {len(bp['q'])} k-points along "
              f"{' '.join(bp['labels'])}")
        for name, m in masks.items():
            print(f"[phunfold] projection '{name}': {int(m.sum())} atoms")

    nq, nb = len(bp["q"]), 3 * backend.structure.natoms
    freqs, weights = np.empty((nq, nb)), np.empty((nq, nb))
    extra = {f"proj_{n}": np.empty((nq, nb)) for n in masks}
    extra["ipr"] = np.empty((nq, nb))
    groups = None
    if resolve in ("species", "sublattice"):
        if resolve == "sublattice":
            labels = unf.sublattice_labels()
            groups = {lab: [i] for i, lab in enumerate(labels)}
        else:
            groups = {}
            for i, sp in enumerate(unf.sublattice_species()):
                groups.setdefault(sp, []).append(i)
        for lab in groups:
            extra[f"w_{lab}"] = np.empty((nq, nb))
    elif resolve not in (None, "none"):
        raise ValueError("resolve must be None, 'species' or 'sublattice'")

    for start in range(0, nq, batch):
        sl = slice(start, min(start + batch, nq))
        kc = bp["k_cart"][sl]
        if getattr(backend, "accepts_q_direction", False):
            # at Gamma, approach along the path leg (LO-TO splitting with NAC)
            # direction of the path leg: used whenever some q+G vanishes
            # (Gamma, and Gamma folded into a supercell at e.g. X, M, R)
            qd = list(bp["leg_direction"][sl])
            f, ev = backend.solve(kc, q_directions=qd)
        else:
            f, ev = backend.solve(kc)
        freqs[sl] = f
        if groups is None:
            weights[sl] = unf.weights(kc, ev, gauge=gauge)
        else:
            wk = unf.weights(kc, ev, gauge=gauge, per_sublattice=True)
            weights[sl] = wk.sum(axis=2)
            for lab, idx in groups.items():
                extra[f"w_{lab}"][sl] = wk[:, :, idx].sum(axis=2)
        for n, m in masks.items():
            extra[f"proj_{n}"][sl] = participation(ev, m)
        extra["ipr"][sl] = inverse_participation_ratio(ev)
        if verbose:
            print(f"[phunfold]   {sl.stop}/{nq} k-points done", flush=True)
    err = float(np.abs(weights.sum(axis=1) - unf.sum_rule).max())
    if err > 1e-6:
        warnings.warn(f"weight sum rule off by {err:.2e} (non-orthonormal "
                      "eigenvectors?)")
    mp = unf.mapping
    meta = dict(M=unf.M.tolist(), N=unf.N, n_ref=reference.natoms,
                rotation_search=unf.used_rotation_search,
                reference_axes=unf.orientation_report(),
                distortion_phase=distortion_phase, gauge=gauge,
                max_site_displacement=unf.max_site_displacement,
                sum_rule=unf.sum_rule, sum_rule_error=err,
                vacancies=mp.n_vacancies, interstitials=mp.n_interstitials,
                substitutions=mp.n_substitutions,
                origin=np.round(mp.origin, 6).tolist())
    meta["nac"] = bool(getattr(backend, "nac", None) is not None)
    return UnfoldResult(bp, freqs, weights, extra, meta)
