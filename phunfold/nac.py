"""Non-analytical term correction (NAC) for polar materials: Gonze-Lee
dipole-dipole method, as implemented in phonopy (its default 'gonze' NAC).

Scheme (Gonze & Lee, PRB 55, 10355 (1997); phonopy documentation):
  1. D_dd(q): reciprocal-space Ewald sum of the dipole-dipole interaction of
     Born charges Z* screened by eps_inf,
        D_dd_st(q) = 4 pi e^2 / V  sum_G  (K.Z_s)(K.Z_t)^T / (K.eps.K)
                     * exp(-K.eps.K / 4 Lambda^2) * exp(2 pi i G.(r_s - r_t))
     with K = q + G (r-gauge, identical to phonopy's), minus the q = 0
     term on the diagonal (acoustic sum rule).
  2. The dipole-dipole part of the supercell force constants (D_dd at the
     commensurate q of the force-constant supercell, Fourier transformed) is
     subtracted, leaving short-range force constants.
  3. D(q) = D_short(q) (minimum-image interpolation) + D_dd(q). At q -> 0
     the G = 0 term is kept along the approach direction, giving LO-TO
     splitting.

Lambda and the G cutoff follow phonopy's defaults (300 G points for the
volume ``ref_volume``). Passing the *reference* (small) cell volume, as the
phunfold CLI does, makes the dipole sum identical for a cell and any of its
supercells, so unfolded spectra stay exactly consistent.

Born-charge convention (same as VASP output and phonopy):
Z[atom, a, b] = dF_b / dE_a = dP_a / du_b, so the force direction couples
through (K . Z)_b = sum_a K_a Z[a, b].
"""

from __future__ import annotations

import itertools
import xml.etree.ElementTree as ET

import numpy as np

E2 = 14.399645  # e^2 / (4 pi eps0) in eV * Angstrom


# ----------------------------------------------------------------------
def read_born(filename):
    """Born charges (n, 3, 3) and eps_inf (3, 3) from vasprun.xml or OUTCAR
    of a VASP LEPSILON run. Charge neutrality is imposed."""
    name = str(filename)
    if name.endswith(".xml"):
        born, eps = _read_vasprun_born(name)
    else:
        from .vasp import parse_outcar_born
        born, eps = parse_outcar_born(name)
    born = np.asarray(born, float)
    born -= born.mean(axis=0)[None]
    return born, np.asarray(eps, float)


def _read_vasprun_born(filename):
    born, eps = None, None
    for _, el in ET.iterparse(filename, events=("end",)):
        if el.tag == "array" and el.get("name") == "born_charges":
            born = [[[float(x) for x in v.text.split()] for v in s.findall("v")]
                    for s in el.findall("set")]
        elif el.tag == "varray" and el.get("name") in ("epsilon", "epsilon_scf"):
            eps = [[float(x) for x in v.text.split()] for v in el.findall("v")]
    if born is None or eps is None:
        raise ValueError(f"{filename}: no born_charges/epsilon found (was it "
                         "a LEPSILON = .TRUE. run?)")
    return np.array(born), np.array(eps)


def rotation_between(ref_lattice, eff_lattice):
    """Orthogonal R with eff ~= ref @ R.T (rows = lattice vectors), i.e. the
    rotation that carries reference-frame vectors into the supercell frame."""
    t = np.linalg.solve(ref_lattice, eff_lattice)       # eff = ref @ t
    u, _, vt = np.linalg.svd(t.T)
    return u @ vt


def tile_born(unfolder, born_ref, eps_ref):
    """Assign reference-cell Born charges to every supercell atom by its
    sublattice, rotating tensors if the supercell is rotated."""
    mp = unfolder.mapping
    if mp.n_interstitials:
        raise ValueError("cannot tile Born charges onto interstitial atoms; "
                         "provide Born charges for the full cell instead")
    R = rotation_between(unfolder.ref.lattice, unfolder.ref_lattice)
    zr = np.einsum("ij,njk,lk->nil", R, np.asarray(born_ref), R)
    born = zr[unfolder.kappa]
    born -= born.mean(axis=0)[None]
    return born, R @ np.asarray(eps_ref) @ R.T


# ----------------------------------------------------------------------
class GonzeNAC:
    """Dipole-dipole dynamical matrix of ``structure`` (Born charges ``born``
    in its atom order)."""

    def __init__(self, structure, born, epsilon, ref_volume=None,
                 num_G_points=300, g_cutoff=None, Lambda=None):
        self.structure = structure
        self.born = np.asarray(born, float)
        self.eps = np.asarray(epsilon, float)
        if self.born.shape != (structure.natoms, 3, 3):
            raise ValueError(f"need Born charges for {structure.natoms} atoms, "
                             f"got {self.born.shape[0]}")
        v = ref_volume or structure.volume
        self.g_cutoff = g_cutoff or (3 * num_G_points / (4 * np.pi) / v) ** (1 / 3)
        gec = self.g_cutoff ** 2 * np.trace(self.eps) / 3
        self.Lambda = Lambda or np.sqrt(-gec / 4 / np.log(1e-10))
        self.factor = E2 * 4 * np.pi / structure.volume
        self.G = self._g_list()
        self.pos = structure.cart
        self._q0 = None
        self._q0 = self._asr_term()

    def _g_list(self):
        rec = self.structure.recip
        nmax = [int(np.ceil(self.g_cutoff * np.linalg.norm(a))) + 1
                for a in self.structure.lattice]
        grid = np.array(list(itertools.product(
            *[range(-n, n + 1) for n in nmax])), float)
        g = grid @ rec
        return g[np.linalg.norm(g, axis=1) < self.g_cutoff]

    def _sum(self, q, direction=None):
        """(n,3,n,3) complex, before the ASR term and the prefactor."""
        n = self.structure.natoms
        K = q[None, :] + self.G
        keps = np.einsum("ga,ab,gb->g", K, self.eps, K)
        norm = np.linalg.norm(K, axis=1)
        small = norm < 1e-10
        w = np.zeros(len(K))
        w[~small] = np.exp(-keps[~small] / (4 * self.Lambda ** 2)) / keps[~small]
        Ku = K.copy()
        if small.any():
            if direction is not None:
                d = np.asarray(direction, float)
                Ku[small] = d
                w[small] = 1.0 / (d @ self.eps @ d)
        kz = np.einsum("ga,nab->gnb", Ku, self.born)        # (nG, n, 3)
        ph = np.exp(2j * np.pi * self.G @ self.pos.T)        # (nG, n): e^{2pi i G.r_s}
        v = (kz * ph.conj()[:, :, None]).reshape(len(K), 3 * n)
        u = (kz * ph[:, :, None]).reshape(len(K), 3 * n)
        # D_st = sum_G w (K.Z_s)(K.Z_t) e^{2 pi i G.(r_s - r_t)}
        d = (u * w[:, None]).T @ v
        return d.reshape(n, 3, n, 3)

    def _asr_term(self):
        d0 = self._sum(np.zeros(3))
        return d0.sum(axis=2)                                # (n, 3, 3)

    def dynamical_matrix(self, q, direction=None):
        """Unweighted D_dd(q) in eV/A^2, shape (n, 3, n, 3)."""
        d = self._sum(np.asarray(q, float), direction)
        n = self.structure.natoms
        d[np.arange(n), :, np.arange(n), :] -= self._q0
        return self.factor * d

    def commensurate_points(self, supercell_lattice):
        """Cartesian q (no 2 pi) of the force-constant supercell that are
        distinct modulo the unit cell reciprocal lattice."""
        P = np.rint(supercell_lattice @ np.linalg.inv(self.structure.lattice))
        Pt = P.T.astype(int)
        n = int(round(abs(np.linalg.det(Pt))))
        inv = np.linalg.inv(Pt)
        corners = np.array(list(itertools.product([0, 1], repeat=3))) @ Pt
        lo, hi = corners.min(0), corners.max(0)
        grid = np.array(list(itertools.product(
            *[range(a, b + 1) for a, b in zip(lo, hi)])))
        g = grid @ inv
        pts = grid[np.all((g > -1e-8) & (g < 1 - 1e-8), axis=1)]
        assert len(pts) == n
        rec_s = np.linalg.inv(supercell_lattice).T
        return pts @ rec_s
