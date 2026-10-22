"""Toy force-constant model of a cubic ABO3 perovskite used to validate the
unfolding machinery without DFT. Pair springs with longitudinal (kL) and
transverse (kT) stiffness between neighbours within a cutoff; satisfies the
acoustic sum rule exactly. Spring constants may be modulated per bond to
mimic a domain wall."""

from __future__ import annotations

import itertools

import numpy as np

from phunfold.structure import Structure
from phunfold.dynmat import VASP_TO_THZ

A0 = 4.0
BASIS = (["Ba", "Ti", "O", "O", "O"],
         [[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])

# (kL, kT) in eV/A^2 keyed by sorted species pair and rounded distance
SPRINGS = {
    ("O", "Ti", 2.0): (9.0, 1.2),
    ("Ba", "O", 2.83): (2.5, 0.4),
    ("O", "O", 2.83): (1.2, 0.3),
    ("Ba", "Ti", 3.46): (1.0, 0.2),
    ("Ba", "Ba", 4.0): (0.6, 0.15),
    ("Ti", "Ti", 4.0): (0.8, 0.2),
    ("O", "O", 4.0): (0.3, 0.1),
}
CUTOFF = 4.05


def perovskite(a=A0):
    return Structure(np.eye(3) * a, BASIS[1], BASIS[0])


def tile(prim: Structure, M):
    """Supercell with lattice M @ prim.lattice (M integer, any shape)."""
    M = np.array(M, dtype=int)
    lat = M @ prim.lattice
    inv = np.linalg.inv(lat)
    n = int(round(abs(np.linalg.det(M))))
    # enumerate lattice points inside the supercell
    corners = np.array(list(itertools.product([0, 1], repeat=3))) @ M
    lo, hi = corners.min(0), corners.max(0)
    pts = []
    for r in itertools.product(*[range(l, h + 1) for l, h in zip(lo, hi)]):
        g = np.array(r) @ prim.lattice @ inv
        if np.all(g > -1e-8) and np.all(g < 1 - 1e-8):
            pts.append(r)
    assert len(pts) == n, (len(pts), n)
    frac, sym = [], []
    for r in pts:
        for f, s in zip(prim.frac, prim.symbols):
            frac.append((np.array(r) + f) @ prim.lattice @ inv)
            sym.append(s)
    frac = np.array(frac) % 1.0
    return Structure(lat, frac, sym)


def bonds(ideal: Structure, modulate=None, cutoff=CUTOFF):
    """List of (i, j, v_ideal, K 3x3) over all periodic images; K is the
    spring tensor so that Phi_ij(v) = -K, Phi_ii += K. ``modulate(i, j, v)``
    can rescale the spring (e.g. near a wall)."""
    cart = ideal.cart
    lat = ideal.lattice
    nmax = [int(np.ceil(cutoff * np.linalg.norm(b))) + 1 for b in ideal.recip]
    shifts = np.array(list(itertools.product(*[range(-m, m + 1) for m in nmax])))
    out = []
    for i in range(ideal.natoms):
        for j in range(ideal.natoms):
            v = cart[j] - cart[i] + shifts @ lat
            d = np.linalg.norm(v, axis=1)
            for vv, dd in zip(v[(d > 1e-6) & (d < cutoff)],
                              d[(d > 1e-6) & (d < cutoff)]):
                key = tuple(sorted([ideal.symbols[i], ideal.symbols[j]])) + (
                    round(dd, 2),)
                kl, kt = SPRINGS[key]
                s = 1.0 if modulate is None else modulate(i, j, vv)
                n = vv / dd
                K = s * (kl * np.outer(n, n) + kt * (np.eye(3) - np.outer(n, n)))
                out.append((i, j, vv, K))
    return out


class ToyBackend:
    """Exact dynamical matrix from the bond list; positions may be distorted
    (``structure`` holds the distorted positions; FCs from the ideal cell)."""

    def __init__(self, ideal: Structure, displaced: Structure = None,
                 modulate=None):
        self.ideal = ideal
        self.structure = displaced if displaced is not None else ideal
        self.delta = self.structure.cart - ideal.cart
        b = bonds(ideal, modulate)
        self._i = np.array([x[0] for x in b])
        self._j = np.array([x[1] for x in b])
        self._v = np.array([x[2] for x in b]) + self.delta[self._j] - self.delta[self._i]
        self._K = np.array([x[3] for x in b])
        m = self.structure.masses
        self._ism = 1 / np.sqrt(np.outer(m, m))

    def dynamical_matrix(self, k):
        n = self.structure.natoms
        d = np.zeros((n, n, 3, 3), dtype=complex)
        ph = np.exp(2j * np.pi * self._v @ k)
        np.add.at(d, (self._i, self._j), -self._K * ph[:, None, None])
        np.add.at(d, (self._i, self._i), self._K)
        d *= self._ism[:, :, None, None]
        d = d.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
        return 0.5 * (d + d.conj().T)

    def solve(self, k_cart):
        k_cart = np.atleast_2d(k_cart)
        n = 3 * self.structure.natoms
        f = np.empty((len(k_cart), n))
        e = np.empty((len(k_cart), n, n), complex)
        for i, k in enumerate(k_cart):
            w2, v = np.linalg.eigh(self.dynamical_matrix(k))
            f[i] = np.sign(w2) * np.sqrt(np.abs(w2)) * VASP_TO_THZ
            e[i] = v
        return f, e

    def real_space_fc(self):
        """Full Phi (n, n, 3, 3) for THIS cell treated as a phonopy SPOSCAR
        (all periodic images summed, as finite displacements would)."""
        n = self.structure.natoms
        fc = np.zeros((n, n, 3, 3))
        np.add.at(fc, (self._i, self._j), -self._K)
        np.add.at(fc, (self._i, self._i), self._K)
        return fc


def write_fc(fc, filename):
    n = fc.shape[0]
    lines = ["%4d %4d" % (n, fc.shape[1])]
    for i in range(n):
        for j in range(fc.shape[1]):
            lines.append("%d %d" % (i + 1, j + 1))
            for row in fc[i, j]:
                lines.append(("%22.15f" * 3) % tuple(row))
    open(filename, "w").write("\n".join(lines))
