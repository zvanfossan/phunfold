"""Phonon band unfolding (Allen, Berlijn, Casavant & Soler, PRB 87, 085322
(2013); phonon form as in Ikeda, Carreras, Seko, Togo & Tanaka, PRB 95,
024305 (2017)).

Theory in one paragraph
-----------------------
Let the large cell (e.g. a domain-wall supercell, N = |det M| reference
cells) have r-gauge eigenvector e_{s a}(K, J) for atom s, Cartesian
direction a, and mode J. Assign every supercell atom to a reference-cell
sublattice kappa(s) and a reference lattice point R_s, with ideal position
x_s = R_s + tau_kappa. The weight with which supercell mode (K, J) carries
reference-cell wave vector k (k = K + G, G a supercell reciprocal vector) is

    W_J(k) = (1/N) sum_{kappa, a} | sum_{s in kappa} e_{s a}(K, J)
                                   exp(2 pi i [K . r_s - k . x_s]) |^2 .

We always diagonalise at K = k itself (D(K) is defined for any K), so the
phase reduces to exp(2 pi i k . (r_s - x_s)): only the small displacement of
each atom from its ideal site enters (``distortion_phase=True``). Setting it
False reproduces the convention of phonopy.unfolding, which ignores it.
Properties: 0 <= W <= 1, and sum_J W_J(k) = 3 n_ref exactly (the number of
reference-cell branches). For a perfect (unrelaxed) supercell W is 0 or 1.
"""

from __future__ import annotations

import json
import warnings

import numpy as np

from .structure import Structure


# ----------------------------------------------------------------------
def find_supercell_matrix(prim: Structure, sc: Structure, tol=0.08):
    """Integer M with sc.lattice ~= M @ prim.lattice (rows convention).

    ``tol`` is the allowed deviation of M from integers; relaxed DW cells
    differ slightly from bulk, which is fine because the unfolding reference
    lattice is rebuilt as inv(M) @ sc.lattice.
    """
    m = sc.lattice @ np.linalg.inv(prim.lattice)
    mr = np.rint(m)
    if np.abs(m - mr).max() > tol:
        raise ValueError(
            "Supercell lattice is not an integer combination of the reference "
            f"cell lattice (max deviation {np.abs(m - mr).max():.3f}). Check "
            "that both cells share the same orientation, or pass "
            "supercell_matrix explicitly.\nM =\n" + np.array2string(m))
    mr = mr.astype(int)
    if round(abs(np.linalg.det(mr))) == 0:
        raise ValueError("singular supercell matrix")
    return mr


class Unfolder:
    """Map a supercell onto a reference cell and compute unfolding weights.

    Parameters
    ----------
    supercell : Structure
        Must be the backend's own structure (``backend.structure``) so that
        positions match the gauge of the eigenvectors.
    reference : Structure
        The small cell to unfold onto (5-atom perovskite). Only its basis
        (species + fractional positions) and orientation are used.
    supercell_matrix : (3, 3) int, optional
        sc.lattice = M @ ref.lattice. Auto-detected when omitted.
    max_displacement : float
        Largest allowed distance (Angstrom) between a supercell atom and its
        ideal site. Ferroelectric displacements are ~0.1-0.2 A.
    distortion_phase : bool
        Include exp(2 pi i k.(r_s - x_s)); see module docstring.
    """

    def __init__(self, supercell: Structure, reference: Structure,
                 supercell_matrix=None, max_displacement=0.8,
                 distortion_phase=True):
        self.sc = supercell
        self.ref = reference
        self.M = (find_supercell_matrix(reference, supercell)
                  if supercell_matrix is None
                  else np.array(supercell_matrix, dtype=int))
        self.N = int(round(abs(np.linalg.det(self.M))))
        # effective reference lattice exactly commensurate with the supercell
        self.ref_lattice = np.linalg.inv(self.M) @ supercell.lattice
        self.ref_recip = np.linalg.inv(self.ref_lattice).T
        self.distortion_phase = distortion_phase
        if supercell.natoms != self.N * reference.natoms:
            raise ValueError(
                f"supercell has {supercell.natoms} atoms but det(M) x "
                f"{reference.natoms} = {self.N * reference.natoms}. Vacancies/"
                "interstitials need a custom mapping.")
        self._map_atoms(max_displacement)

    # ------------------------------------------------------------------
    def _map_atoms(self, max_disp):
        sc, ref = self.sc, self.ref
        f = sc.cart @ np.linalg.inv(self.ref_lattice)    # in ref coordinates
        kappa = np.full(sc.natoms, -1, dtype=int)
        lat_pt = np.zeros((sc.natoms, 3), dtype=int)
        best = np.full(sc.natoms, np.inf)
        resid = np.zeros((sc.natoms, 3))
        for k in range(ref.natoms):
            same = np.array([s == ref.symbols[k] for s in sc.symbols])
            d = f - ref.frac[k]
            r = np.rint(d)
            res = (d - r) @ self.ref_lattice
            dist = np.linalg.norm(res, axis=1)
            take = same & (dist < best)
            kappa[take], lat_pt[take] = k, r[take].astype(int)
            best[take], resid[take] = dist[take], res[take]
        if np.any(kappa < 0):
            raise ValueError("some supercell atoms have no matching species in "
                             "the reference cell")
        if best.max() > max_disp:
            i = int(best.argmax())
            raise ValueError(
                f"atom {i} ({sc.symbols[i]}) is {best.max():.2f} A from the "
                "nearest ideal site. Wrong reference orientation/origin? "
                "Increase max_displacement only if this is genuine.")
        # bijection check: every (kappa, R mod supercell) exactly once
        g = lat_pt @ np.linalg.inv(self.M)        # supercell fractional coords
        g -= np.floor(g + 1e-6)
        keys = {(int(k), *np.rint(gi * 1e4).astype(int)) for k, gi in
                zip(kappa, g)}
        if len(keys) != sc.natoms:
            raise ValueError("atom mapping is not one-to-one: two atoms were "
                             "assigned to the same ideal site")
        self.kappa = kappa
        self.lattice_points = lat_pt
        self.displacements = resid                # r_s - x_s (Cartesian)
        self.ideal_cart = sc.cart - resid
        self.max_site_displacement = float(best.max())

    # ------------------------------------------------------------------
    def k_cart(self, q_ref):
        """Reference-cell reduced coordinates -> Cartesian (1/A, no 2 pi)."""
        return np.atleast_2d(q_ref) @ self.ref_recip

    def weights(self, k_cart, eigvecs):
        """Unfolding weights, shape (nq, 3 n_sc).

        k_cart : (nq, 3) — the same k used to obtain ``eigvecs`` (K = k).
        eigvecs: (nq, 3 n_sc, 3 n_sc) r-gauge eigenvectors, columns = modes.
        """
        k_cart = np.atleast_2d(k_cart)
        n_sc, n_ref = self.sc.natoms, self.ref.natoms
        out = np.empty((len(k_cart), 3 * n_sc))
        for iq, (k, ev) in enumerate(zip(k_cart, eigvecs)):
            e = ev.reshape(n_sc, 3, -1)
            if self.distortion_phase:
                ph = np.exp(2j * np.pi * self.displacements @ k)
                e = e * ph[:, None, None]
            b = np.zeros((n_ref, 3, e.shape[2]), dtype=complex)
            np.add.at(b, self.kappa, e)
            out[iq] = (np.abs(b) ** 2).sum(axis=(0, 1)) / self.N
        return out

    # ------------------------------------------------------------------
    def plane_distance(self, axis, wall_positions):
        """Distance (A) of every atom's ideal site from the nearest wall.

        axis : 0/1/2, supercell lattice vector the walls cut (wall normal is
               along the reciprocal vector b_axis).
        wall_positions : fractional coordinates of the wall planes along it.
        """
        f = self.ideal_cart @ np.linalg.inv(self.sc.lattice)
        spacing = 1.0 / np.linalg.norm(self.sc.recip[axis])
        d = np.abs(f[:, axis][:, None] - np.asarray(wall_positions)[None, :])
        d = np.minimum(d % 1.0, 1.0 - d % 1.0)
        return d.min(axis=1) * spacing


def participation(eigvecs, atom_mask):
    """Fraction of each mode's |e|^2 on the masked atoms. (nq, 3N)."""
    m = np.repeat(np.asarray(atom_mask, bool), 3)
    return (np.abs(eigvecs[:, m, :]) ** 2).sum(axis=1)


def inverse_participation_ratio(eigvecs):
    """Per-mode IPR = sum_s |e_s|^4 (with |e_s|^2 summed over x,y,z).
    1/N for a fully delocalised mode, 1 for a single-atom mode."""
    n = eigvecs.shape[1] // 3
    a = (np.abs(eigvecs.reshape(eigvecs.shape[0], n, 3, -1)) ** 2).sum(axis=2)
    return (a ** 2).sum(axis=1)


# ----------------------------------------------------------------------
def make_band_path(segments, recip, density=60.0, npts=None):
    """Build a piecewise-linear k path.

    segments : list of lists of (label, [q1, q2, q3]) in reduced coordinates
               of ``recip``; each inner list is a continuous path, and
               consecutive inner lists are joined by a break ("X|R").
    recip    : reciprocal lattice rows (1/A, no 2 pi).
    density  : points per (1/A) of path length; or fixed ``npts`` per leg.

    Returns dict(q, k_cart, x, ticks, labels).
    """
    qs, xs, ticks, labels = [], [], [], []
    x0 = 0.0
    for iseg, seg in enumerate(segments):
        if iseg == 0:
            ticks.append(x0)
            labels.append(seg[0][0])
        else:  # break: same x position, combined label "X|R"
            labels[-1] = f"{labels[-1]}|{seg[0][0]}"
        for (_, qa), (lb, qb) in zip(seg[:-1], seg[1:]):
            qa, qb = np.asarray(qa, float), np.asarray(qb, float)
            length = np.linalg.norm((qb - qa) @ recip)
            n = npts or max(int(np.ceil(length * density)), 2)
            t = np.linspace(0.0, 1.0, n + 1)
            qs.append(qa + t[:, None] * (qb - qa))
            xs.append(x0 + t * length)
            x0 += length
            ticks.append(x0)
            labels.append(lb)
    q = np.vstack(qs)
    return dict(q=q, k_cart=q @ recip, x=np.concatenate(xs),
                ticks=np.array(ticks), labels=labels)


def parse_path_string(text):
    """'G 0 0 0, X .5 0 0, M .5 .5 0 | Z 0 0 .5, R .5 .5 .5' -> segments."""
    segments = []
    for part in text.split("|"):
        seg = []
        for item in part.split(","):
            tok = item.split()
            if not tok:
                continue
            if len(tok) != 4:
                raise ValueError(f"bad path entry {item!r}: need 'LABEL q1 q2 q3'")
            lab = "Γ" if tok[0].upper() in ("G", "GAMMA") else tok[0]
            seg.append((lab, [float(eval(t, {}, {})) for t in tok[1:]]))
        segments.append(seg)
    return segments


# Default path for a (pseudo)tetragonal 5-atom perovskite with the polar axis
# along c. X is along a (the default DW normal), Y along b (in the wall plane).
DEFAULT_PATH = ("G 0 0 0, X 1/2 0 0, M 1/2 1/2 0, G 0 0 0, Y 0 1/2 0 | "
                "G 0 0 0, Z 0 0 1/2, R 1/2 0 1/2, A 1/2 1/2 1/2, Z 0 0 1/2")


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
    """Container that can be saved/loaded (.npz) and plotted."""

    def __init__(self, path, freqs, weights, extra=None, meta=None):
        self.path = path
        self.freqs = freqs
        self.weights = weights
        self.extra = extra or {}
        self.meta = meta or {}

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
        """Columns: x  q1 q2 q3  freq(THz)  weight [extras]; one line per
        mode with weight > wmin (gnuplot/origin friendly)."""
        keys = list(self.extra)
        with open(filename, "w") as fh:
            fh.write("# x(1/A) q1 q2 q3 freq(THz) weight " + " ".join(keys) + "\n")
            for i, x in enumerate(self.path["x"]):
                q = self.path["q"][i]
                for j in np.where(self.weights[i] > wmin)[0]:
                    ex = " ".join("%.5f" % self.extra[k][i, j] for k in keys)
                    fh.write("%.6f %.5f %.5f %.5f %.6f %.6f %s\n" % (
                        x, *q, self.freqs[i, j], self.weights[i, j], ex))


def run_unfolding(backend, reference, path_segments=DEFAULT_PATH,
                  supercell_matrix=None, density=60.0, npts=None,
                  distortion_phase=True, wall_axis=None, wall_positions=None,
                  wall_width=2.5, batch=20, verbose=True):
    """Full pipeline: map atoms, build path, diagonalise, unfold.

    Returns UnfoldResult. If wall_axis/wall_positions are given, extra
    arrays 'wall_participation' (fraction of |e|^2 within ``wall_width`` A of
    a wall) and 'ipr' are stored per mode.
    """
    if isinstance(path_segments, str):
        path_segments = parse_path_string(path_segments)
    unf = Unfolder(backend.structure, reference,
                   supercell_matrix=supercell_matrix,
                   distortion_phase=distortion_phase)
    path = make_band_path(path_segments, unf.ref_recip, density, npts)
    if verbose:
        print(f"[phunfold] supercell: {backend.structure.natoms} atoms = "
              f"{unf.N} x {reference.natoms}-atom reference; "
              f"M = {unf.M.tolist()}")
        print(f"[phunfold] largest displacement from ideal site: "
              f"{unf.max_site_displacement:.3f} A")
        print(f"[phunfold] {len(path['q'])} k-points along "
              f"{' '.join(path['labels'])}")
    mask = None
    if wall_axis is not None and wall_positions is not None:
        mask = unf.plane_distance(wall_axis, wall_positions) < wall_width
        if verbose:
            print(f"[phunfold] {mask.sum()} atoms within {wall_width} A of "
                  "a wall")
    nq = len(path["q"])
    nb = 3 * backend.structure.natoms
    freqs = np.empty((nq, nb))
    weights = np.empty((nq, nb))
    extra = {}
    if mask is not None:
        extra["wall_participation"] = np.empty((nq, nb))
        extra["ipr"] = np.empty((nq, nb))
    for start in range(0, nq, batch):
        sl = slice(start, min(start + batch, nq))
        kc = path["k_cart"][sl]
        f, ev = backend.solve(kc)
        freqs[sl] = f
        weights[sl] = unf.weights(kc, ev)
        if mask is not None:
            extra["wall_participation"][sl] = participation(ev, mask)
            extra["ipr"][sl] = inverse_participation_ratio(ev)
        if verbose:
            print(f"[phunfold]   {sl.stop}/{nq} k-points done", flush=True)
    s = weights.sum(axis=1)
    target = 3 * reference.natoms
    if np.abs(s - target).max() > 1e-6:
        warnings.warn(f"weight sum rule off by {np.abs(s - target).max():.2e}")
    meta = dict(M=unf.M.tolist(), N=unf.N, n_ref=reference.natoms,
                distortion_phase=distortion_phase,
                max_site_displacement=unf.max_site_displacement,
                sum_rule_error=float(np.abs(s - target).max()))
    return UnfoldResult(path, freqs, weights, extra, meta)
