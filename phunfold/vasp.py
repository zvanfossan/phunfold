"""VASP input generation for every stage of the workflow, plus parsers for
Born charges / dielectric tensor from OUTCAR.

The INCAR templates are sensible, conservative defaults for phonons of an
insulating oxide like BaTiO3 with PBEsol. ENCUT, the k-point density, and
the functional MUST be converged/chosen for your system (step 0 of the
README); override any tag with ``overrides={...}``.
"""

from __future__ import annotations

import glob
import os
import re
import shutil

import numpy as np

from .structure import Structure, read_poscar

COMMON = dict(
    PREC="Accurate", ENCUT=600, EDIFF="1E-8", ISMEAR=0, SIGMA=0.02,
    GGA="PS", LREAL=".FALSE.", ADDGRID=".TRUE.", LASPH=".TRUE.",
    LWAVE=".FALSE.", LCHARG=".FALSE.", ALGO="Normal", NELM=200,
)

STAGES = {
    # 1. bulk 5-atom cell: full relaxation (run twice to kill Pulay stress)
    "relax_bulk": dict(COMMON, SYSTEM="bulk relax", IBRION=2, ISIF=3,
                       NSW=100, EDIFFG="-1E-3"),
    # 2. bulk Born charges + eps_inf (DFPT). Do not set NCORE/NPAR here.
    "born": dict(COMMON, SYSTEM="bulk Born/eps", IBRION=8, LEPSILON=".TRUE.",
                 NSW=1),
    # 3. domain-wall supercell: ions only, bulk lattice kept fixed
    "relax_dw": dict(COMMON, SYSTEM="DW relax", IBRION=1, ISIF=2, NSW=300,
                     EDIFFG="-1E-3", POTIM=0.3),
    # 4a. perfect (undisplaced) supercell: residual forces for --fz
    # 4b. one single point per displaced supercell
    "single_point": dict(COMMON, SYSTEM="phonopy displacement", IBRION=-1,
                         NSW=0, ISYM=0),
}


def write_incar(stage, filename="INCAR", overrides=None, extra_comment=""):
    tags = dict(STAGES[stage])
    tags.update(overrides or {})
    lines = [f"# phunfold stage: {stage} {extra_comment}".rstrip()]
    lines += [f"{k:8s} = {v}" for k, v in tags.items() if v is not None]
    with open(filename, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def kmesh(structure: Structure, kspacing=0.2):
    """Gamma-centred mesh with spacing ``kspacing`` (1/A, incl. 2 pi, same
    meaning as VASP's KSPACING)."""
    b = 2 * np.pi * np.linalg.norm(structure.recip, axis=1)
    return [max(1, int(np.ceil(x / kspacing - 1e-6))) for x in b]


def write_kpoints(structure: Structure, filename="KPOINTS", kspacing=0.2,
                  mesh=None):
    mesh = mesh or kmesh(structure, kspacing)
    with open(filename, "w") as fh:
        fh.write(f"Gamma-centred, kspacing={kspacing}\n0\nGamma\n"
                 f"{mesh[0]} {mesh[1]} {mesh[2]}\n0 0 0\n")
    return mesh


def write_potcar(structure: Structure, potcar_dir, filename="POTCAR",
                 variants=None):
    """Concatenate POTCARs in POSCAR species order. ``variants`` maps an
    element to the PAW flavour, e.g. {'Ba': 'Ba_sv', 'Ti': 'Ti_sv'}."""
    variants = variants or {}
    species = structure.sorted_by_species().species_order()
    with open(filename, "wb") as out:
        for sp in species:
            path = os.path.join(potcar_dir, variants.get(sp, sp), "POTCAR")
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            with open(path, "rb") as fh:
                out.write(fh.read())
    return species


def setup_calculation(stage, structure_file, workdir, kspacing=0.2,
                      potcar_dir=None, potcar_variants=None, overrides=None):
    """Create ``workdir`` with POSCAR, INCAR, KPOINTS (and POTCAR)."""
    os.makedirs(workdir, exist_ok=True)
    s = read_poscar(structure_file)
    shutil.copy(structure_file, os.path.join(workdir, "POSCAR"))
    write_incar(stage, os.path.join(workdir, "INCAR"), overrides)
    mesh = write_kpoints(s, os.path.join(workdir, "KPOINTS"), kspacing)
    if potcar_dir:
        write_potcar(s, potcar_dir, os.path.join(workdir, "POTCAR"),
                     potcar_variants)
    return mesh


def setup_displacements(pattern="POSCAR-*", sposcar="SPOSCAR", root=".",
                        kspacing=0.2, potcar_dir=None, potcar_variants=None,
                        overrides=None):
    """One directory per phonopy displacement (disp-001, ...) plus
    ``perfect/`` (the undisplaced SPOSCAR, used with ``phonopy --fz``)."""
    files = sorted(glob.glob(os.path.join(root, pattern)),
                   key=lambda p: int(re.findall(r"(\d+)$", p)[0]))
    if not files:
        raise FileNotFoundError(f"no {pattern} in {root}; run phonopy -d first")
    made = []
    for path in [os.path.join(root, sposcar)] + files:
        tag = ("perfect" if path.endswith(sposcar)
               else "disp-" + re.findall(r"(\d+)$", path)[0])
        d = os.path.join(root, tag)
        mesh = setup_calculation("single_point", path, d, kspacing,
                                 potcar_dir, potcar_variants, overrides)
        made.append(d)
    return made, mesh


# ----------------------------------------------------------------------
def parse_outcar_born(outcar="OUTCAR"):
    """Return (born (natoms,3,3), eps_inf (3,3)) from a LEPSILON run.

    Uses the LAST 'BORN EFFECTIVE CHARGES' block and the last
    'MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects'
    block (electronic eps_inf, not the ionic contribution).
    Rows of each 3x3 Z* are the field direction as printed by VASP
    (phonopy-vasp-born uses the same convention).
    """
    with open(outcar) as fh:
        lines = fh.readlines()
    born_start = eps_start = None
    for i, ln in enumerate(lines):
        if "BORN EFFECTIVE CHARGES" in ln:
            born_start = i
        if ("MACROSCOPIC STATIC DIELECTRIC TENSOR" in ln
                and "including local field effects" in ln
                and "IONIC" not in ln):
            eps_start = i
    if born_start is None or eps_start is None:
        raise ValueError(f"{outcar}: Born charges / dielectric tensor not found")
    eps = np.array([[float(x) for x in lines[eps_start + 2 + r].split()[:3]]
                    for r in range(3)])
    born, i = [], born_start + 2
    while i < len(lines) and lines[i].split()[:1] == ["ion"]:
        z = [[float(x) for x in lines[i + 1 + r].split()[1:4]] for r in range(3)]
        born.append(z)
        i += 4
    born = np.array(born)
    # enforce charge neutrality (small DFPT residuals)
    born -= born.mean(axis=0)[None]
    return born, eps


def tile_born(unfolder, born_ref):
    """Born charges for every supercell atom, copied from its sublattice in
    the reference cell (``unfolder.kappa``). Good approximation away from the
    walls; compute Z* of the DW cell directly if you need more."""
    born_ref = np.asarray(born_ref)
    return born_ref[unfolder.kappa]
