"""Tests for the general domain-wall builder (crystallographic polarization
directions, any wall normal) and for unfolding onto a rhombohedral cell.

Run:  python tests/test_domainwall.py      (or pytest tests/)
"""

import json
import os
import sys
import tempfile
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

from phunfold import (Structure, make_supercell, write_poscar,  # noqa: E402
                      NativeBackend, Unfolder, run_unfolding, read_poscar)
from phunfold.builders import perovskite  # noqa: E402
from phunfold.domainwall import (build_wall, build_180_wall,  # noqa: E402
                                 snap_reference, polar_displacements,
                                 wall_profile, domain_born_charges)
from toymodel import ToyBackend, write_fc  # noqa: E402

TAU = np.array([[0, 0, 0], [.5, .5, .5], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])


def rhombohedral_bulk(a=4.0, shear=0.0025, ti=0.12, o=-0.06, o_perp=0.02):
    """Toy R3m cell, pseudocubic setting, P along [111]; the O atoms also move
    perpendicular to [111] so a wrong symmetry mapping would be visible."""
    F = np.eye(3) + shear * (np.ones((3, 3)) - np.eye(3))
    lat = a * F.T
    P = np.ones(3) / np.sqrt(3)
    perp = {2: [-2, 1, 1], 3: [1, -2, 1], 4: [1, 1, -2]}
    u = np.zeros((5, 3))
    u[1] = ti * P
    for k in (2, 3, 4):
        u[k] = o * P + o_perp * np.array(perp[k]) / np.sqrt(6)
    return Structure(lat, (TAU @ lat + u) @ np.linalg.inv(lat),
                     ["Ba", "Ti", "O", "O", "O"])


def _old_build_180(polar, ncells=4, normal="a", lateral=(1, 1), offset=0.0,
                   width=0.0):
    """Frozen copy of the original 180-degree builder (regression target)."""
    ax = {"a": 0, "b": 1, "c": 2}[normal]
    ref = snap_reference(polar)
    u = polar_displacements(polar, ref)
    reps = [0, 0, 0]
    reps[ax] = 2 * ncells
    others = [i for i in range(3) if i != ax]
    reps[others[0]], reps[others[1]] = lateral
    lat = np.diag(reps) @ polar.lattice
    cart, syms = [], []
    for n in np.ndindex(*reps):
        for k in range(polar.natoms):
            xid = ref.frac[k] + np.array(n)
            cart.append(xid @ polar.lattice
                        + wall_profile(xid[ax], ncells, offset, width) * u[k])
            syms.append(polar.symbols[k])
    return lat, np.array(cart), syms


def _canon(lat, cart, syms):
    f = (np.asarray(cart) @ np.linalg.inv(lat)) % 1.0
    f[np.isclose(f, 1.0)] = 0.0
    return sorted((s, *np.round(x, 8)) for s, x in zip(syms, f))


def test_180_regression():
    tet = perovskite(3.99, c=4.03, polar_axis="c",
                     disp={"B": 0.10, "X_ap": -0.08, "X_eq": -0.06})
    for kw in (dict(ncells=4, normal="a"),
               dict(ncells=3, normal="b", offset=0.5, lateral=(2, 1)),
               dict(ncells=4, normal="a", width=1.0)):
        lat, cart, syms = _old_build_180(tet, **kw)
        for sc in (build_180_wall(tet, **kw)[0],
                   build_wall(tet, p1=[0, 0, 1], p2=[0, 0, -1], **kw)[0]):
            assert np.allclose(sc.lattice, lat, atol=1e-12)
            assert _canon(sc.lattice, sc.cart, sc.symbols) == \
                _canon(lat, cart, syms)
    print("  [001]/[00-1] reproduces the original 180-degree builder exactly "
          "(3 geometries, wrapper and explicit directions)")


def test_rhombohedral_walls():
    rh = rhombohedral_bulk()
    ub = polar_displacements(rh, snap_reference(rh))
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # no compatibility warning
        sc, info, (d1, d2) = build_wall(rh, p1=[1, 1, 1], p2=[-1, 1, 1],
                                        normal=[0, 1, 1], ncells=2)
    assert sc.natoms == 40 and info["type"] == "71deg"
    assert info["supercell_matrix"] == [[1, 0, 0], [0, 1, -1], [0, 4, 4]]
    mx = np.diag([-1, 1, 1])
    for d, g in ((d1, np.eye(3)), (d2, mx)):
        ud = polar_displacements(d, snap_reference(d))
        assert np.allclose(ud, ub @ g.T, atol=1e-10)      # incl. O perp parts
    # every atom off the wall planes carries exactly its domain's pattern
    from phunfold.domainwall import stretch_and_rotation
    M = np.array(info["supercell_matrix"])
    a_pc, U, _ = stretch_and_rotation(rh.lattice)
    Ua = 0.5 * (U + mx @ U @ mx)
    p = np.round(sc.frac @ M * 2) / 2                    # ideal sites (cells)
    disp = sc.cart - a_pc * p @ Ua.T
    disp -= np.rint(disp @ np.linalg.inv(sc.lattice)) @ sc.lattice
    x = (p @ np.array([0, 1, 1]) / 2.0) % 4               # repeats along [011]
    checked = 0
    for s in range(sc.natoms):
        if min(x[s], abs(x[s] - 2), 4 - x[s]) < 1e-6:
            continue                                      # on a wall plane
        k = int(np.argmin([np.abs(((p[s] % 1) - t) - np.rint((p[s] % 1) - t)).max()
                           for t in TAU]))
        # m_x leaves every perovskite site in place (coordinates 0 or 1/2)
        want = ub[k] if x[s] < 2 else mx @ ub[k]
        assert np.allclose(disp[s], want, atol=1e-10), (s, disp[s], want)
        checked += 1
    assert checked >= 30
    # 109 deg on (001): fine; on (1-10): neutral but strain-incompatible
    _, i109, _ = build_wall(rh, p1=[1, 1, 1], p2=[-1, -1, 1], normal=[0, 0, 1],
                            ncells=2)
    assert i109["type"] == "109deg"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        build_wall(rh, p1=[1, 1, 1], p2=[-1, -1, 1], normal=[1, -1, 0],
                   ncells=2)
        assert any("not mechanically compatible" in str(x.message) for x in w)
    for bad in (dict(p2=[-1, 1, 1], normal=[1, 0, 0]),       # charged
                dict(p1=[0, 0, 1], p2=[0, 0, -1])):         # wrong family
        try:
            build_wall(rh, **{"p1": [1, 1, 1], **bad})
            raise AssertionError(f"should have failed: {bad}")
        except ValueError:
            pass
    print("  71 deg (011): domain patterns = bulk and its x-mirror (incl. "
          "off-axis O); 109 deg (001) OK; (1-10) warned; charged and "
          "non-equivalent directions refused")


def test_domain_born_charges():
    rh = rhombohedral_bulk()
    sc, info, _ = build_wall(rh, p1=[1, 1, 1], p2=[-1, 1, 1], normal=[0, 1, 1],
                             ncells=2)
    with tempfile.TemporaryDirectory() as tmp:          # species-sorted order
        write_poscar(sc, f"{tmp}/P")
        cell = read_poscar(f"{tmp}/P")
    J = np.ones((3, 3)) - np.eye(3)
    Z = np.array([2.7 * np.eye(3), 7.0 * np.eye(3) + 0.4 * J,
                  -2.1 * np.eye(3) - 0.1 * J, -2.1 * np.eye(3) - 0.1 * J,
                  -2.1 * np.eye(3) - 0.1 * J])
    for k in (2, 3, 4):
        Z[k][k - 2, k - 2] = -5.5
    Z -= Z.mean(axis=0)
    born, eps, dom = domain_born_charges(cell, info, Z, 6 * np.eye(3) + 0.1 * J)
    ti = [s for s in range(cell.natoms) if cell.symbols[s] == "Ti"]
    mx = np.diag([-1, 1, 1])
    for s in ti:
        expect = {1: Z[1], 2: mx @ Z[1] @ mx, 0: 0.5 * (Z[1] + mx @ Z[1] @ mx)}
        assert np.allclose(born[s], expect[dom[s]], atol=1e-12)
    assert set(dom) == {0, 1, 2} and np.allclose(born.sum(axis=0), 0)
    assert np.allclose(eps, 0.5 * ((6 * np.eye(3) + 0.1 * J)
                                   + mx @ (6 * np.eye(3) + 0.1 * J) @ mx))
    print("  Born charges: domain 2 gets the mirrored tensors, wall atoms the "
          "average, eps_inf averaged; works on a species-sorted POSCAR")


def _fc_from_sposcar(sp, M_total):
    """Toy FCs on the undistorted cubic geometry of exactly these atoms."""
    parent = np.round(sp.frac @ M_total * 2) / 2
    ideal = Structure(4.0 * M_total.astype(float),
                      parent @ np.linalg.inv(M_total), sp.symbols)
    return ToyBackend(ideal).real_space_fc()


def test_unfold_onto_rhombohedral_cell():
    rh = rhombohedral_bulk()
    sc, info, (d1, _) = build_wall(rh, p1=[1, 1, 1], p2=[-1, 1, 1],
                                   normal=[0, 1, 1], ncells=2)
    M = np.array(info["supercell_matrix"])
    dim = np.diag([3, 2, 1])
    sp = make_supercell(sc, dim)
    bsp = make_supercell(rh, np.diag([3, 3, 3]))
    with tempfile.TemporaryDirectory() as tmp:
        write_fc(_fc_from_sposcar(sp, dim @ M), f"{tmp}/FC")
        write_fc(_fc_from_sposcar(bsp, 3 * np.eye(3, dtype=int)), f"{tmp}/FCb")
        wall = NativeBackend(sc, sp, f"{tmp}/FC")
        bulk = NativeBackend(rh, bsp, f"{tmp}/FCb")
    path = "G 0 0 0, N 0 1/2 1/2 | G 0 0 0, Q 1/4 1/2 0, F 1/2 1/2 0"
    r_rh = run_unfolding(wall, d1, path, density=30, verbose=False)
    r_cub = run_unfolding(wall, perovskite(4.0), path, density=30,
                          verbose=False)
    # 1) same translation lattice -> identical weights for both references
    assert np.abs(r_rh.weights - r_cub.weights).max() < 1e-10
    # 2) springs that ignore P: the wall cell is the bulk crystal again, so
    #    every weighted mode sits on a rhombohedral bulk band
    q = r_rh.path["q"]
    fb, _ = bulk.solve(q @ bulk.structure.recip)
    dev = max(np.min(np.abs(fb[i] - f)) for i in range(len(q))
              for f, w in zip(r_rh.freqs[i], r_rh.weights[i]) if w > 1e-3)
    assert dev < 1e-6, dev
    assert r_rh.meta["sum_rule_error"] < 1e-10
    print(f"  unfolding onto the rhombohedral cell: identical weights to the "
          f"cubic reference; control spectrum on the R3m bulk bands to "
          f"{dev:.0e} THz")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("ALL TESTS PASSED")
