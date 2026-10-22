"""Supercell dynamical matrices: native NumPy engine (phonopy FORCE_CONSTANTS
from any calculator), phonopy wrapper, or any user-supplied solver.

Both backends expose the same interface::

    backend.structure          # Structure: the cell whose modes are computed
                               # (here: the relaxed domain-wall supercell)
    freqs, eigvecs = backend.solve(k_cart)
        # k_cart : (nq, 3) Cartesian wave vectors, 1/Angstrom, no 2*pi
        # freqs  : (nq, 3N) THz, imaginary modes returned as negative numbers
        # eigvecs: (nq, 3N, 3N) complex, eigvecs[q][:, j] is mode j

Phase convention ("r-gauge", identical to phonopy): the dynamical matrix is

    D_{s a, t b}(k) = sum_L Phi_{s a, t b}(0, L) / sqrt(m_s m_t)
                      * exp(2 pi i k . (r_t + L - r_s))

with r_s the atomic positions *as stored in* ``backend.structure``. The
physical displacement pattern of mode j is then
u_s(L) ~ e_s / sqrt(m_s) * exp(2 pi i k . (r_s + L)). The unfolding code
relies on this convention, so positions must be taken from the backend's own
structure, never from a separately wrapped copy of the file.
"""

from __future__ import annotations

import itertools

import numpy as np

from .structure import Structure, read_poscar

VASP_TO_THZ = 15.633302  # sqrt(eV / (Angstrom^2 amu)) / (2 pi) in THz
THZ_TO_CM = 33.35641
THZ_TO_MEV = 4.135667


# ----------------------------------------------------------------------
def read_force_constants(filename, n_supercell=None):
    """Read phonopy's text FORCE_CONSTANTS (full or compact).

    Returns (fc, rows) where fc has shape (n_rows, n_sc, 3, 3) in eV/A^2 and
    ``rows[i]`` is the 0-based supercell index of row i.
    """
    with open(filename) as fh:
        tokens = fh.read().split()
    n_rows = int(tokens[0])
    n_sc = int(tokens[1])
    if n_supercell is not None and n_sc != n_supercell:
        raise ValueError(f"{filename}: {n_sc} columns but supercell has "
                         f"{n_supercell} atoms")
    body = np.array(tokens[2:], dtype=float).reshape(n_rows * n_sc, 11)
    rows_idx = body[:, 0].astype(int).reshape(n_rows, n_sc)[:, 0] - 1
    cols_idx = body[:, 1].astype(int).reshape(n_rows, n_sc) - 1
    fc = body[:, 2:].reshape(n_rows, n_sc, 3, 3)
    # put columns in supercell order (they normally already are)
    order = np.argsort(cols_idx, axis=1)
    fc = np.take_along_axis(fc, order[:, :, None, None], axis=1)
    return fc, rows_idx


def read_force_constants_hdf5(filename):
    """Read phonopy's force_constants.hdf5 (needs h5py). Returns (fc, rows)."""
    import h5py  # optional dependency

    with h5py.File(filename, "r") as h:
        fc = h["force_constants"][:]
        rows = (h["p2s_map"][:] if "p2s_map" in h and len(h["p2s_map"]) ==
                fc.shape[0] else np.arange(fc.shape[0]))
    return fc, np.asarray(rows, int)


def load_force_constants(filename, n_supercell=None):
    """Text FORCE_CONSTANTS or .hdf5, chosen by extension."""
    if str(filename).endswith((".hdf5", ".h5")):
        return read_force_constants_hdf5(filename)
    return read_force_constants(filename, n_supercell)


# Conversion of force constants to eV/A^2 (positions must be in Angstrom).
BOHR = 0.529177210903
FC_UNITS = {
    "eV/A^2": 1.0,
    "eV/bohr^2": 1.0 / BOHR ** 2,
    "Ry/bohr^2": 13.605693122994 / BOHR ** 2,
    "Ha/bohr^2": 27.211386245988 / BOHR ** 2,
    "kJ/mol/nm^2": 1.0 / 96.485332 / 100.0,
}


def map_to_unitcell(unitcell: Structure, supercell: Structure, tol=1e-3):
    """For every supercell atom return the index of the unit-cell atom it is a
    lattice translate of (``s2p``). Tolerance in Angstrom."""
    inv_u = np.linalg.inv(unitcell.lattice)
    fu = supercell.cart @ inv_u                     # (n_sc, 3)
    s2p = np.full(supercell.natoms, -1, dtype=int)
    for p in range(unitcell.natoms):
        d = fu - unitcell.frac[p]
        d -= np.rint(d)
        dist = np.linalg.norm(d @ unitcell.lattice, axis=1)
        hit = dist < tol
        if np.any(hit & (s2p >= 0)):
            raise ValueError("two unit-cell atoms map onto the same "
                             "supercell atom; tolerance too large?")
        s2p[hit] = p
    if np.any(s2p < 0):
        bad = np.where(s2p < 0)[0][:5]
        raise ValueError(f"supercell atoms {bad} are not lattice translates of "
                         "any unit-cell atom (wrong POSCAR/SPOSCAR pair?)")
    n_l = supercell.natoms / unitcell.natoms
    counts = np.bincount(s2p, minlength=unitcell.natoms)
    if not np.all(counts == round(n_l)):
        raise ValueError("inconsistent unit-cell -> supercell mapping")
    return s2p


def _minimum_image_vectors(vec, lattice, tol=1e-3, shells=2):
    """For each difference vector in ``vec`` (n, 3) return all equivalent
    shortest images under ``lattice`` (rows). Returns (index, vectors)."""
    # reduce into the cell first so a small shell range is enough
    inv = np.linalg.inv(lattice)
    f = vec @ inv
    f -= np.rint(f)
    shifts = np.array(list(itertools.product(range(-shells, shells + 1),
                                             repeat=3)), dtype=float)
    cand = (f[:, None, :] + shifts[None, :, :]) @ lattice   # (n, ns, 3)
    lengths = np.linalg.norm(cand, axis=2)
    lmin = lengths.min(axis=1, keepdims=True)
    mask = lengths < lmin + tol
    idx, sh = np.nonzero(mask)
    return idx, cand[idx, sh]


class NativeBackend:
    """Dynamical matrix of ``unitcell`` from real-space force constants.

    Parameters
    ----------
    unitcell : Structure or path
        The cell whose phonons you want (the relaxed DW supercell, i.e. the
        file you passed to ``phonopy -d -c``).
    supercell : Structure or path
        Phonopy's SPOSCAR (unitcell x --dim) in which forces were computed.
    force_constants : path or (fc, rows)
        phonopy FORCE_CONSTANTS (text). Full or compact both work.
    acoustic_sum_rule : bool
        Re-impose sum_j Phi_ij = 0 on the self terms (cheap, recommended).
    masses : array (n_unitcell,) or dict {'O': 17.999} to override masses
        (isotopes, mass-disorder studies).
    fc_units : key of FC_UNITS; structure files must be in Angstrom.
    fc_threshold : drop force-constant blocks with max |Phi| below this
        (eV/A^2) after the ASR correction; speeds up and shrinks very large
        cells. 0 keeps everything (exact).
    nac : None, or dict(born=(n_unitcell, 3, 3), epsilon=(3, 3)
        [, ref_volume, num_G_points, g_cutoff, Lambda]) to apply the
        Gonze-Lee non-analytical correction (see phunfold.nac).
    """

    accepts_q_direction = True

    def __init__(self, unitcell, supercell, force_constants,
                 acoustic_sum_rule=True, masses=None, tol=1e-3,
                 fc_units="eV/A^2", fc_threshold=0.0, chunk=200000, nac=None):
        from .structure import read_structure

        self.structure = (read_structure(unitcell) if not isinstance(
            unitcell, Structure) else unitcell.copy())
        sc = (read_structure(supercell) if not isinstance(
            supercell, Structure) else supercell)
        if isinstance(masses, dict):
            self.structure.masses = np.array(
                [masses.get(s, m) for s, m in zip(self.structure.symbols,
                                                  self.structure.masses)])
        elif masses is not None:
            self.structure.masses = np.asarray(masses, float)
        if isinstance(force_constants, (tuple, list)):
            fc, rows = force_constants
        else:
            fc, rows = load_force_constants(force_constants, sc.natoms)
        fc = np.array(fc, dtype=float) * FC_UNITS[fc_units]
        s2p = map_to_unitcell(self.structure, sc, tol=tol)
        nu = self.structure.natoms

        # one representative row per unit-cell atom
        rep = np.full(nu, -1, dtype=int)
        for i, s in enumerate(rows):
            p = s2p[s]
            if rep[p] < 0:
                rep[p] = i
        if np.any(rep < 0):
            raise ValueError("FORCE_CONSTANTS lacks rows for some unit-cell "
                             "atoms")
        fc_rep = fc[rep]                               # (nu, n_sc, 3, 3)
        rep_sc = np.asarray(rows)[rep]                 # supercell indices
        if acoustic_sum_rule:
            drift = fc_rep.sum(axis=1)                 # (nu, 3, 3)
            fc_rep[np.arange(nu), rep_sc] -= drift
            self.asr_violation = np.abs(drift).max()
        else:
            self.asr_violation = np.abs(fc_rep.sum(axis=1)).max()

        # Gonze-Lee NAC: remove the dipole-dipole part of the supercell force
        # constants (its Fourier transform at the commensurate q points);
        # it is added back analytically in dynamical_matrix()
        self.nac = None
        if nac is not None:
            from .nac import GonzeNAC
            opts = dict(nac)
            self.nac = GonzeNAC(self.structure, opts.pop("born"),
                                opts.pop("epsilon"), **opts)
            r_sc = sc.cart
            qs = self.nac.commensurate_points(sc.lattice)
            phi_dd = np.zeros_like(fc_rep)
            dvec = r_sc[None, :, :] - r_sc[rep_sc][:, None, :]   # (nu, n_sc, 3)
            for q in qs:
                ddq = self.nac.dynamical_matrix(q)               # (nu,3,nu,3)
                blk = ddq[:, :, s2p, :].transpose(0, 2, 1, 3)    # (nu,n_sc,3,3)
                ph = np.exp(-2j * np.pi * dvec @ q)
                phi_dd += (blk * ph[:, :, None, None]).real
            phi_dd /= len(qs)
            fc_rep = fc_rep - phi_dd
            self.n_commensurate = len(qs)

        # difference vectors r_j - r_rep(p) with minimum-image multiplicity,
        # skipping negligible blocks and working in chunks to bound memory
        r_sc = sc.cart
        p_all = np.repeat(np.arange(nu), sc.natoms)
        j_all = np.tile(np.arange(sc.natoms), nu)
        if fc_threshold > 0:
            keep = (np.abs(fc_rep).max(axis=(2, 3)) >= fc_threshold).ravel()
            p_all, j_all = p_all[keep], j_all[keep]
        self.n_pairs = len(p_all)
        P, PP, V, FC = [], [], [], []
        mult_max = 1
        for a in range(0, len(p_all), chunk):
            p_idx, j_idx = p_all[a:a + chunk], j_all[a:a + chunk]
            vec = r_sc[j_idx] - r_sc[rep_sc[p_idx]]
            ent, vecs = _minimum_image_vectors(vec, sc.lattice, tol=tol)
            mult = np.bincount(ent, minlength=len(vec))
            mult_max = max(mult_max, int(mult.max()))
            P.append(p_idx[ent])
            PP.append(s2p[j_idx[ent]])
            V.append(vecs)
            FC.append(fc_rep[p_idx[ent], j_idx[ent]] / mult[ent][:, None, None])
        self._p, self._pp = np.concatenate(P), np.concatenate(PP)
        self._vec, self._fc = np.concatenate(V), np.concatenate(FC)
        self.max_multiplicity = mult_max
        m = self.structure.masses
        self._inv_sqrt_mm = 1.0 / np.sqrt(np.outer(m, m))

    def dynamical_matrix(self, k_cart, q_direction=None):
        """Mass-weighted D(k). With NAC, ``q_direction`` (Cartesian) sets the
        direction of approach to Gamma (LO-TO splitting); at k = 0 without
        a direction only TO modes are obtained, as in phonopy."""
        nu = self.structure.natoms
        k_cart = np.asarray(k_cart, float)
        phase = np.exp(2j * np.pi * self._vec @ k_cart)
        d = np.zeros((nu, nu, 3, 3), dtype=complex)
        np.add.at(d, (self._p, self._pp), self._fc * phase[:, None, None])
        if self.nac is not None:
            d += self.nac.dynamical_matrix(k_cart, q_direction).transpose(
                0, 2, 1, 3)
        d *= self._inv_sqrt_mm[:, :, None, None]
        d = d.transpose(0, 2, 1, 3).reshape(3 * nu, 3 * nu)
        return 0.5 * (d + d.conj().T)

    def solve(self, k_cart, q_directions=None):
        k_cart = np.atleast_2d(k_cart)
        n = 3 * self.structure.natoms
        freqs = np.empty((len(k_cart), n))
        vecs = np.empty((len(k_cart), n, n), dtype=complex)
        for i, k in enumerate(k_cart):
            qd = None if q_directions is None else q_directions[i]
            w2, ev = np.linalg.eigh(self.dynamical_matrix(k, qd))
            freqs[i] = np.sign(w2) * np.sqrt(np.abs(w2)) * VASP_TO_THZ
            vecs[i] = ev
        return freqs, vecs


class CallableBackend:
    """Wrap any eigen-solver: ``fn(k_cart) -> (freqs, eigvecs)`` for ONE k
    (Cartesian, 1/A, no 2 pi; eigvecs columns = modes, r-gauge unless you
    pass gauge='R' to the unfolding). Use it for eigenvectors from other
    codes, ML potentials with analytic Hessians, tight-binding models, ..."""

    def __init__(self, structure: Structure, fn):
        self.structure = structure
        self._fn = fn

    def solve(self, k_cart):
        k_cart = np.atleast_2d(k_cart)
        out = [self._fn(k) for k in k_cart]
        return (np.array([o[0] for o in out]),
                np.array([o[1] for o in out]))


class DynamicalMatrixBackend(CallableBackend):
    """``fn(k_cart) -> D (3N x 3N complex, mass-weighted, eV/(A^2 amu))``;
    diagonalised here and converted to THz (times ``factor``)."""

    def __init__(self, structure: Structure, fn, factor=VASP_TO_THZ):
        def solve_one(k):
            w2, ev = np.linalg.eigh(fn(k))
            return np.sign(w2) * np.sqrt(np.abs(w2)) * factor, ev
        super().__init__(structure, solve_one)


class PhonopyBackend:
    """Use phonopy (on your cluster) to build and diagonalise D(k).

    Use this when you need non-analytical (LO-TO) corrections via a BORN file.
    ``primitive_matrix`` is forced to the identity so phonopy's "primitive"
    cell is exactly your DW supercell and atom order is preserved.
    """

    def __init__(self, phonopy_yaml="phonopy_disp.yaml",
                 force_sets="FORCE_SETS", force_constants=None, born=None,
                 nac_params=None, symmetrize_fc=True):
        import phonopy  # noqa: WPS433  (optional dependency)

        kwargs = dict(primitive_matrix=np.eye(3),
                      is_nac=born is not None and nac_params is None,
                      symmetrize_fc=symmetrize_fc)
        if force_constants is not None:
            kwargs["force_constants_filename"] = force_constants
        else:
            kwargs["force_sets_filename"] = force_sets
        if born is not None and nac_params is None:
            kwargs["born_filename"] = born  # phonopy BORN file for THIS cell
        self.phonon = phonopy.load(phonopy_yaml, **kwargs)
        if nac_params is not None:
            # dict(born=(n_atoms, 3, 3), dielectric=(3, 3)); see nac.tile_born
            self.phonon.nac_params = nac_params
        prim = self.phonon.primitive
        self.structure = Structure(prim.cell, prim.scaled_positions,
                                   prim.symbols, prim.masses)

    def solve(self, k_cart, nac_q_direction=None):
        k_cart = np.atleast_2d(k_cart)
        q_red = k_cart @ self.structure.lattice.T      # reduced coordinates
        self.phonon.run_qpoints(q_red, with_eigenvectors=True,
                                nac_q_direction=nac_q_direction)
        out = self.phonon.get_qpoints_dict()
        return np.asarray(out["frequencies"]), np.asarray(out["eigenvectors"])
