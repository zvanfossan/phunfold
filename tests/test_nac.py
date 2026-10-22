"""Tests for the Gonze-Lee non-analytical correction (phunfold.nac).

Run:  python tests/test_nac.py      (or pytest tests/)
"""

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

from phunfold import (NativeBackend, Unfolder, make_supercell,  # noqa: E402
                      write_poscar, run_unfolding, rattle)
from phunfold.nac import GonzeNAC, tile_born, E2, read_born  # noqa: E402
from toymodel import perovskite, ToyBackend, write_fc  # noqa: E402

rng = np.random.default_rng(3)
# cubic-BaTiO3-like Born charges; O anisotropic along its Ti-O bond
ZB = np.array([np.eye(3) * 2.7, np.eye(3) * 7.2,
               np.diag([-2.1, -2.1, -5.7]),     # O at (.5,.5,0): bond along z
               np.diag([-2.1, -5.7, -2.1]),     # O at (.5,0,.5): along y
               np.diag([-5.7, -2.1, -2.1])])    # O at (0,.5,.5): along x
EPS = np.eye(3) * 6.0


def _backend(unit, dim, nac=None, rattle_amp=0.0):
    """NativeBackend from toy spring FCs computed on unit x dim."""
    spos = make_supercell(unit, np.diag(dim))
    fc = ToyBackend(spos).real_space_fc()
    with tempfile.TemporaryDirectory() as tmp:
        write_poscar(unit, f"{tmp}/POSCAR", sort=False)
        write_poscar(spos, f"{tmp}/SPOSCAR", sort=False)
        write_fc(fc, f"{tmp}/FC")
        return NativeBackend(f"{tmp}/POSCAR", f"{tmp}/SPOSCAR", f"{tmp}/FC",
                             nac=nac)


def test_commensurate_points_unchanged():
    """At the force-constant supercell's commensurate q, Gonze NAC must give
    back exactly the raw force-constant D(q): the dipole part removed from
    the FCs and the one added analytically cancel (tests phase conventions)."""
    prim = perovskite()
    dim = (3, 3, 3)
    b0 = _backend(prim, dim)
    b1 = _backend(prim, dim, nac=dict(born=ZB, epsilon=EPS))
    qs = b1.nac.commensurate_points(make_supercell(prim, np.diag(dim)).lattice)
    err = max(np.abs(b0.dynamical_matrix(q) - b1.dynamical_matrix(q)).max()
              for q in qs[1:])
    # ~1e-7: the G sum is truncated on a sphere of |G| (phonopy's choice),
    # not |q+G|, so terms damped to ~1e-7 at the edge differ between q and -q
    assert err < 1e-5, err
    # and away from them the correction matters
    q = np.array([0.07, 0.03, 0.11])
    assert np.abs(b0.dynamical_matrix(q) - b1.dynamical_matrix(q)).max() > 1e-3
    print(f"  {len(qs)} commensurate points reproduced to {err:.1e}; "
          "non-commensurate q changed as expected")


def test_lo_to_term():
    """q -> 0 limit equals the explicit 4 pi e^2/V (d.Z)(d.Z)/(d.eps.d) term."""
    prim = perovskite()
    nac = GonzeNAC(prim, ZB, EPS)
    d = np.array([0.0, 0.0, 1.0])
    jump = nac.dynamical_matrix(np.zeros(3), d) - nac.dynamical_matrix(np.zeros(3))
    dz = np.einsum("a,nab->nb", d, ZB)
    expect = 4 * np.pi * E2 / prim.volume * np.einsum(
        "sa,tb->satb", dz, dz) / (d @ EPS @ d)
    assert np.allclose(jump, expect, atol=1e-10)
    small = nac.dynamical_matrix(1e-7 * d)
    assert np.allclose(small, nac.dynamical_matrix(np.zeros(3), d), atol=1e-4)
    # LO-TO splitting appears in the full spectrum
    b = _backend(prim, (3, 3, 3), nac=dict(born=ZB, epsilon=EPS))
    f_to, _ = b.solve(np.zeros((1, 3)))
    f_lo, _ = b.solve(np.zeros((1, 3)), q_directions=[d])
    assert f_lo[0].max() > f_to[0].max() + 0.5
    print(f"  LO-TO term = 4 pi e^2/V (dZ)(dZ)/(d eps d) exactly; highest mode "
          f"{f_to[0].max():.2f} -> {f_lo[0].max():.2f} THz with NAC")


def test_nac_unfolding_consistent():
    """A supercell (tiled Born charges, same dipole sum) unfolds exactly onto
    the NAC primitive bands, at general k and at Gamma along a direction."""
    prim = perovskite()
    p_nac = dict(born=ZB, epsilon=EPS, ref_volume=prim.volume)
    bp = _backend(prim, (2, 2, 2), nac=p_nac)
    unit = make_supercell(prim, np.diag([2, 2, 2]))       # 40 atoms, dim 1
    unf0 = Unfolder(unit, prim)
    zt, et = tile_born(unf0, ZB, EPS)
    bs = _backend(unit, (1, 1, 1), nac=dict(born=zt, epsilon=et,
                                           ref_volume=prim.volume))
    q = np.vstack([rng.uniform(-.5, .5, size=(4, 3)), np.zeros(3)])
    k = q @ prim.recip
    dirs = [None] * 4 + [np.array([1.0, 1.0, 0.0]) / np.sqrt(2)]
    fp, _ = bp.solve(k, q_directions=dirs)
    fs, es = bs.solve(k, q_directions=dirs)
    w = Unfolder(bs.structure, prim).weights(k, es)
    worst = 0.0
    for i in range(len(k)):
        order = np.argsort(fs[i])
        f, ww = fs[i][order], w[i][order]
        groups = np.split(np.arange(len(f)), np.where(np.diff(f) > 1e-5)[0] + 1)
        for g in groups:
            n = np.sum(np.abs(fp[i] - f[g].mean()) < 1e-5)
            worst = max(worst, abs(ww[g].sum() - n))
    assert worst < 1e-6, worst
    print(f"  40-atom NAC supercell unfolds onto 5-atom NAC bands to {worst:.1e} "
          "(incl. LO-TO at Gamma)")


def test_run_unfolding_with_nac_and_rattle():
    prim = perovskite()
    unit = rattle(make_supercell(prim, np.diag([2, 1, 1])), 0.02, seed=2)
    b = _backend(unit, (2, 3, 3), nac=dict(
        born=np.tile(ZB, (2, 1, 1)), epsilon=EPS, ref_volume=prim.volume))
    res = run_unfolding(b, prim, path="cubic", density=15, verbose=False)
    assert res.meta["nac"] and res.meta["sum_rule_error"] < 1e-9
    print("  run_unfolding with NAC: sum rule holds, Gamma directions passed")


def test_read_born_vasprun():
    xml = """<?xml version="1.0"?><modeling><calculation>
 <varray name="epsilon"><v> 6.1 0 0 </v><v> 0 6.1 0 </v><v> 0 0 5.9 </v></varray>
 <array name="born_charges">
  <set><v> 2.7 0 0 </v><v> 0 2.7 0 </v><v> 0 0 2.6 </v></set>
  <set><v> -2.7 0 0 </v><v> 0 -2.7 0 </v><v> 0 0 -2.6 </v></set>
 </array></calculation></modeling>"""
    with tempfile.TemporaryDirectory() as tmp:
        open(f"{tmp}/vasprun.xml", "w").write(xml)
        z, e = read_born(f"{tmp}/vasprun.xml")
    assert z.shape == (2, 3, 3) and np.isclose(e[2, 2], 5.9) and np.isclose(z[0, 2, 2], 2.6)
    print("  vasprun.xml Born charges / eps_inf parsed")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("ALL TESTS PASSED")
