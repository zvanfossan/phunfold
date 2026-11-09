"""End-to-end dry run of the whole file-based workflow on a toy perovskite
force-field, standing in for VASP. Exercises exactly the files you will have
on the cluster (POSCAR_DW, SPOSCAR, FORCE_CONSTANTS, dw_info.json) and the
same CLI commands.

    python examples/toy_dw_demo.py   -> examples/toy_run/unfolded_toy.png
"""

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]

from phunfold import read_poscar, write_poscar, Unfolder, Structure  # noqa
from phunfold.domainwall import snap_reference  # noqa
from toymodel import perovskite, tile, ToyBackend, write_fc  # noqa

run = os.path.join(HERE, "toy_run")
os.makedirs(run, exist_ok=True)
os.chdir(run)


def sh(cmd):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True,
                   env=dict(os.environ, PYTHONPATH=ROOT))


# --- "relaxed polar bulk" 5-atom cell (stand-in for VASP step 1) ----------
bulk = perovskite()
polar = bulk.copy()
polar.frac[1, 2] += 0.08 / 4.0          # Ti off-centre along c
polar.frac[2:, 2] -= 0.04 / 4.0         # O counter-displacement
write_poscar(polar, "POSCAR_bulk", sort=False)

# --- step 3: build the 180-degree DW supercell (4 + 4 cells along a) ------
sh(f"{sys.executable} -m phunfold build-dw --bulk POSCAR_bulk --ncells 4 "
   "--normal a --offset 0")

# --- step 4 stand-in: "phonopy -d --dim 1 2 2" and force constants --------
dw = read_poscar("POSCAR_DW")
dim = np.diag([1, 2, 2])
sposcar = tile(dw, dim)
# Ideal (centrosymmetric) site of every SPOSCAR atom, NOT re-wrapped into the
# cell, so force-constant index j refers to exactly SPOSCAR atom j. (Tiling a
# separately wrapped ideal cell can pick a different periodic copy for atoms
# sitting next to a cell boundary.)
unf = Unfolder(sposcar, snap_reference(polar))
sposcar_ideal = Structure(sposcar.lattice,
                          unf.ideal_cart @ np.linalg.inv(sposcar.lattice),
                          sposcar.symbols)
Lx = dw.lattice[0, 0]
cart = sposcar_ideal.cart


def soften_wall(i, j, v):
    """Ti-O springs within 1.5 A of the wall planes (x = 0, Lx/2) are 45 %
    softer: a crude stand-in for the different bonding at the wall."""
    if {sposcar_ideal.symbols[i], sposcar_ideal.symbols[j]} != {"Ti", "O"}:
        return 1.0
    x = (cart[i, 0] + v[0] / 2) % Lx
    d = min(abs(x), abs(x - Lx / 2), abs(x - Lx))
    return 0.55 if d < 1.5 else 1.0


fc = ToyBackend(sposcar_ideal, modulate=soften_wall).real_space_fc()
write_poscar(sposcar, "SPOSCAR", sort=False)
write_fc(fc, "FORCE_CONSTANTS")

# bulk reference phonons (3x3x3) for the dashed overlay
os.makedirs("bulk", exist_ok=True)
write_poscar(polar, "bulk/POSCAR", sort=False)
write_poscar(tile(polar, np.diag([3, 3, 3])), "bulk/SPOSCAR", sort=False)
write_fc(ToyBackend(tile(bulk, np.diag([3, 3, 3]))).real_space_fc(),
         "bulk/FORCE_CONSTANTS")

# --- step 6: unfold ----------------------------------------------------------
sh(f"{sys.executable} -m phunfold unfold --unitcell POSCAR_DW --sposcar SPOSCAR "
   "--fc FORCE_CONSTANTS --reference POSCAR_bulk --bulk-dir bulk "
   "--out unfolded_toy --density 50 --wall-width 2.1")
