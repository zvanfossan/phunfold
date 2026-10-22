"""Second dry run, for a non-domain-wall case: a 3x3x3 toy-perovskite
supercell containing one Sr-on-Ba substitution and one O vacancy, rotated
and shifted relative to the reference file (as cells from other tools often
are). Exercises: orientation/origin search, species_map, vacancies, sphere
projection around the defect, and species-resolved weights.

    python examples/toy_defect_demo.py   -> examples/toy_defect/unfolded_defect.png
"""

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]

from phunfold import Structure, write_poscar, make_supercell  # noqa: E402
from toymodel import perovskite, ToyBackend, write_fc  # noqa: E402

run = os.path.join(HERE, "toy_defect")
os.makedirs(run, exist_ok=True)
os.chdir(run)


def sh(cmd):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True,
                   env=dict(os.environ, PYTHONPATH=ROOT))


prim = perovskite()
write_poscar(prim, "POSCAR_ref", sort=False)

# defect supercell in the reference orientation ...
sc = make_supercell(prim, np.diag([3, 3, 3]))
center = np.array([0.5, 0.5, 0.5])
d = sc.frac - center
d -= np.rint(d)
dist = np.linalg.norm(d @ sc.lattice, axis=1)
ba = min((i for i, s in enumerate(sc.symbols) if s == "Ba"), key=lambda i: dist[i])
sc.symbols[ba], sc.masses[ba] = "Sr", 87.62
o = min((i for i, s in enumerate(sc.symbols) if s == "O"), key=lambda i: dist[i])
keep = [i for i in range(sc.natoms) if i != o]
sc = Structure(sc.lattice, sc.frac[keep], [sc.symbols[i] for i in keep],
               sc.masses[keep])

# ... then rotated 30 deg about z and shifted, like a cell from another code
t = np.radians(30)
rot = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
shift = np.array([0.37, 0.11, 0.05])
sc = Structure(sc.lattice @ rot.T, (sc.frac + shift) % 1, sc.symbols, sc.masses)

# "phonopy" step: supercell for forces = the defect cell itself (dim 1 1 1,
# 27 cells is already 12 A); toy FCs stand in for DFT
write_poscar(sc, "POSCAR", sort=False)
write_poscar(sc, "SPOSCAR", sort=False)
write_fc(ToyBackend(sc).real_space_fc(), "FORCE_CONSTANTS")

defect_center = ",".join("%.4f" % x for x in (center + shift) % 1)
sh(f"{sys.executable} -m phunfold unfold --unitcell POSCAR --sposcar SPOSCAR "
   "--fc FORCE_CONSTANTS --reference POSCAR_ref --path cubic "
   "--species-map Sr=Ba --allow-vacancies "
   f"--project defect=sphere:{defect_center}:3.0 Sr=species:Sr "
   "--resolve species --density 40 --out unfolded_defect")
