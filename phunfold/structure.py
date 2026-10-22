"""Minimal crystal-structure container and VASP POSCAR I/O.

Conventions used everywhere in phunfold
---------------------------------------
* Lattice matrices hold lattice vectors as ROWS (same as POSCAR / phonopy):
  ``lattice[i]`` is the i-th lattice vector a_i in Angstrom.
* Cartesian position r = f @ lattice for fractional coordinates f (row vector).
* Reciprocal lattice WITHOUT the 2*pi factor: ``recip = inv(lattice).T``, so
  a_i . b_j = delta_ij. A wave vector in reduced coordinates q has Cartesian
  form k = q @ recip (units 1/Angstrom, no 2*pi), and plane-wave phases are
  exp(2*pi*i k . r).
"""

from __future__ import annotations

import numpy as np

# Standard atomic weights (amu). Extend as needed or pass masses explicitly.
ATOMIC_MASSES = {
    "H": 1.00794, "Li": 6.941, "Be": 9.012182, "B": 10.811, "C": 12.0107,
    "N": 14.0067, "O": 15.9994, "F": 18.9984032, "Na": 22.98976928,
    "Mg": 24.305, "Al": 26.9815386, "Si": 28.0855, "P": 30.973762,
    "S": 32.065, "Cl": 35.453, "K": 39.0983, "Ca": 40.078, "Sc": 44.955912,
    "Ti": 47.867, "V": 50.9415, "Cr": 51.9961, "Mn": 54.938045,
    "Fe": 55.845, "Co": 58.933195, "Ni": 58.6934, "Cu": 63.546,
    "Zn": 65.38, "Ga": 69.723, "Ge": 72.64, "As": 74.9216, "Se": 78.96,
    "Br": 79.904, "Rb": 85.4678, "Sr": 87.62, "Y": 88.90585, "Zr": 91.224,
    "Nb": 92.90638, "Mo": 95.96, "Ru": 101.07, "Rh": 102.9055,
    "Pd": 106.42, "Ag": 107.8682, "Cd": 112.411, "In": 114.818,
    "Sn": 118.71, "Sb": 121.76, "Te": 127.6, "I": 126.90447,
    "Cs": 132.9054519, "Ba": 137.327, "La": 138.90547, "Ce": 140.116,
    "Pr": 140.90765, "Nd": 144.242, "Sm": 150.36, "Eu": 151.964,
    "Gd": 157.25, "Hf": 178.49, "Ta": 180.94788, "W": 183.84,
    "Ir": 192.217, "Pt": 195.084, "Au": 196.966569, "Hg": 200.59,
    "Pb": 207.2, "Bi": 208.9804,
}


class Structure:
    """Periodic structure: lattice (rows), fractional positions, symbols."""

    def __init__(self, lattice, frac, symbols, masses=None):
        self.lattice = np.array(lattice, dtype=float).reshape(3, 3)
        self.frac = np.array(frac, dtype=float).reshape(-1, 3)
        self.symbols = list(symbols)
        if len(self.symbols) != len(self.frac):
            raise ValueError("symbols and positions have different lengths")
        if masses is None:
            try:
                masses = [ATOMIC_MASSES[s] for s in self.symbols]
            except KeyError as err:
                raise KeyError(
                    f"No default mass for element {err}; pass masses explicitly"
                ) from None
        self.masses = np.array(masses, dtype=float)

    # ------------------------------------------------------------------
    @property
    def natoms(self) -> int:
        return len(self.symbols)

    @property
    def cart(self) -> np.ndarray:
        return self.frac @ self.lattice

    @property
    def recip(self) -> np.ndarray:
        """Reciprocal lattice vectors as rows, no 2*pi."""
        return np.linalg.inv(self.lattice).T

    @property
    def volume(self) -> float:
        return abs(np.linalg.det(self.lattice))

    def copy(self) -> "Structure":
        return Structure(self.lattice.copy(), self.frac.copy(),
                         list(self.symbols), self.masses.copy())

    def species_order(self):
        """Unique species in order of first appearance."""
        seen = []
        for s in self.symbols:
            if s not in seen:
                seen.append(s)
        return seen

    def sorted_by_species(self) -> "Structure":
        """Return a copy with atoms grouped by species (VASP needs this)."""
        order = []
        for sp in self.species_order():
            order += [i for i, s in enumerate(self.symbols) if s == sp]
        return Structure(self.lattice, self.frac[order],
                         [self.symbols[i] for i in order], self.masses[order])

    def __repr__(self):
        return f"Structure({self.natoms} atoms, {''.join(self.species_order())})"


# ----------------------------------------------------------------------
def read_poscar(filename) -> Structure:
    """Read a VASP 5+ POSCAR/CONTCAR (species line required)."""
    with open(filename) as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    scale = float(lines[1].split()[0])
    lat = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    if scale < 0:  # negative scale = target volume
        lat *= (-scale / abs(np.linalg.det(lat))) ** (1.0 / 3.0)
    else:
        lat *= scale
    tokens = lines[5].split()
    if not all(t.replace("_", "").replace("/", "").isalpha() for t in tokens):
        raise ValueError(
            f"{filename}: line 6 must list element symbols (VASP 5 format)")
    species = [t.split("_")[0].split("/")[0] for t in tokens]
    counts = [int(x) for x in lines[6].split()]
    idx = 7
    if lines[idx].strip()[0] in "sS":  # Selective dynamics
        idx += 1
    cartesian = lines[idx].strip()[0] in "cCkK"
    idx += 1
    n = sum(counts)
    pos = np.array([[float(x) for x in lines[idx + i].split()[:3]]
                    for i in range(n)])
    if cartesian:
        pos = pos * (scale if scale > 0 else 1.0)
        frac = pos @ np.linalg.inv(lat)
    else:
        frac = pos
    symbols = [sp for sp, c in zip(species, counts) for _ in range(c)]
    return Structure(lat, frac, symbols)


def write_poscar(structure: Structure, filename, comment="phunfold",
                 sort=True) -> Structure:
    """Write a VASP 5 POSCAR (Direct). Returns the structure actually written
    (atoms grouped by species when ``sort`` is True)."""
    s = structure.sorted_by_species() if sort else structure
    species = s.species_order() if sort else None
    if not sort:
        # consecutive runs of identical symbols
        species, counts = [], []
        for sym in s.symbols:
            if species and species[-1] == sym:
                counts[-1] += 1
            else:
                species.append(sym)
                counts.append(1)
    else:
        counts = [s.symbols.count(sp) for sp in species]
    out = [comment, "1.0"]
    out += ["  %21.16f %21.16f %21.16f" % tuple(v) for v in s.lattice]
    out.append("  " + " ".join("%4s" % sp for sp in species))
    out.append("  " + " ".join("%4d" % c for c in counts))
    out.append("Direct")
    out += ["  %20.16f %20.16f %20.16f" % tuple(f) for f in s.frac]
    with open(filename, "w") as fh:
        fh.write("\n".join(out) + "\n")
    return s


def read_structure(filename) -> Structure:
    """POSCAR/CONTCAR natively; any other format (cif, xyz, QE/ABINIT input,
    ...) through ASE if it is installed. Units must be Angstrom."""
    try:
        return read_poscar(filename)
    except Exception as err_poscar:  # noqa: BLE001
        try:
            import ase.io
        except ImportError:
            raise ValueError(
                f"could not read {filename} as POSCAR ({err_poscar}); install "
                "ase to read other formats") from None
        atoms = ase.io.read(filename)
        return Structure(atoms.cell[:], atoms.get_scaled_positions(),
                         atoms.get_chemical_symbols(), atoms.get_masses())


def make_supercell(prim: Structure, M) -> Structure:
    """Supercell with lattice M @ prim.lattice (any integer M, det > 0).
    Atom order: for each lattice point, the full primitive basis."""
    import itertools

    M = np.array(M, dtype=int).reshape(3, 3)
    n = int(round(abs(np.linalg.det(M))))
    if n == 0:
        raise ValueError("singular supercell matrix")
    lat = M @ prim.lattice
    # lattice points inside the supercell: all R with R inv(M) in [0, 1)
    inv_m = np.linalg.inv(M)
    corners = np.array(list(itertools.product([0, 1], repeat=3))) @ M
    lo, hi = corners.min(0), corners.max(0)
    grid = np.array(list(itertools.product(
        *[range(a, b + 1) for a, b in zip(lo, hi)])))
    g = grid @ inv_m
    inside = np.all((g > -1e-8) & (g < 1 - 1e-8), axis=1)
    pts = grid[inside]
    assert len(pts) == n, (len(pts), n)
    frac = ((pts[:, None, :] + prim.frac[None, :, :]) @ inv_m).reshape(-1, 3)
    syms = list(prim.symbols) * n
    masses = np.tile(prim.masses, n)
    return Structure(lat, frac % 1.0, syms, masses)


def rattle(structure: Structure, amplitude=0.01, seed=None) -> Structure:
    """Copy with Gaussian random displacements (Angstrom)."""
    rng = np.random.default_rng(seed)
    s = structure.copy()
    s.frac = s.frac + rng.normal(scale=amplitude, size=s.frac.shape) @ \
        np.linalg.inv(s.lattice)
    return s
