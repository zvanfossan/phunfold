"""Tests for the general-supercell features: rotated/shifted cells,
substitutions, vacancies, interstitials, auto reference, gauges, resolved
weights, projections, path presets, FC pruning.

Run:  python tests/test_general.py      (or pytest tests/)
"""

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

from phunfold import (Structure, Unfolder, run_unfolding,  # noqa: E402
                      make_supercell, rattle, reference_from_supercell,
                      resolve_path, NativeBackend, write_poscar,
                      CallableBackend)
from toymodel import perovskite, ToyBackend, write_fc  # noqa: E402

rng = np.random.default_rng(11)


def random_rotation():
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def transform(s: Structure, rot=np.eye(3), shift_frac=(0, 0, 0)):
    """Rigidly rotate (Cartesian) and shift (fractional) a structure."""
    return Structure(s.lattice @ rot.T, (s.frac + shift_frac) % 1.0,
                     s.symbols, s.masses)


def degenerate_groups(f, tol=1e-5):
    order = np.argsort(f)
    groups, cur = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if f[b] - f[a] < tol:
            cur.append(b)
        else:
            groups.append(cur)
            cur = [b]
    groups.append(cur)
    return groups


def assert_exact_unfolding(f, w, fp, tol=1e-6):
    for iq in range(len(f)):
        for g in degenerate_groups(f[iq]):
            n_prim = np.sum(np.abs(fp[iq] - f[iq, g].mean()) < 1e-5)
            assert abs(w[iq, g].sum() - n_prim) < tol, (iq, w[iq, g].sum(),
                                                         n_prim)


# anisotropic springs (bonds along the unrotated z are 60 % stiffer) so a
# wrong orientation choice would give wrong frequencies
def aniso(rot):
    def mod(i, j, v):
        n = (v @ rot) / np.linalg.norm(v)   # back to the unrotated frame
        return 1.0 + 0.6 * n[2] ** 2
    return mod


def polar(prim):
    p = prim.copy()
    p.frac[1, 2] += 0.02
    p.frac[2:, 2] -= 0.01
    return p


def test_rotated_shifted_supercell():
    prim = perovskite()
    ref = polar(prim)
    for M in ([[3, 0, 0], [0, 1, 0], [0, 0, 2]],
              [[1, 1, 0], [-1, 1, 0], [0, 0, 2]]):
        rot = random_rotation()
        shift = rng.uniform(size=3)
        ideal = transform(make_supercell(prim, M), rot, shift)
        actual = rattle(transform(make_supercell(ref, M), rot, shift), 0.03,
                        seed=1)
        sc = ToyBackend(ideal, actual, modulate=aniso(rot))
        unf = Unfolder(sc.structure, ref)
        q = rng.uniform(-.5, .5, size=(6, 3))
        k = unf.k_cart(q)
        f, e = sc.solve(k)
        w = unf.weights(k, e)
        fp, _ = ToyBackend(prim, modulate=aniso(np.eye(3))).solve(
            q @ prim.recip)
        assert_exact_unfolding(f, w, fp)
        print(f"  M={M}, random rotation + origin shift: found M="
              f"{unf.M.tolist()} after {unf.n_candidates_tried} candidates; "
              "exact")


def test_substitution():
    prim = perovskite()
    ideal = make_supercell(prim, np.diag([2, 2, 2]))
    ideal.symbols[0] = "Sr"
    ideal.masses[0] = 87.62
    sc = ToyBackend(ideal)
    try:
        Unfolder(sc.structure, prim)
        raise AssertionError("should have failed without species_map")
    except ValueError:
        pass
    for kw in (dict(species_map={"Sr": "Ba"}), dict(match_species=False)):
        unf = Unfolder(sc.structure, prim, **kw)
        k = unf.k_cart(rng.uniform(-.5, .5, size=(4, 3)))
        f, e = sc.solve(k)
        w = unf.weights(k, e)
        assert np.allclose(w.sum(1), 15) and w.min() > -1e-12 and w.max() < 1 + 1e-9
    assert unf.mapping.n_substitutions == 1
    print("  Sr on a Ba site: mapped via species_map / match_species=False; "
          "sum rule 15")


def test_vacancy():
    prim = perovskite()
    full = make_supercell(prim, np.diag([2, 2, 2]))
    keep = [i for i in range(full.natoms) if i != 7]           # drop one O
    ideal = Structure(full.lattice, full.frac[keep],
                      [full.symbols[i] for i in keep])
    sc = ToyBackend(ideal)
    unf = Unfolder(sc.structure, prim, allow_vacancies=True)
    k = unf.k_cart(rng.uniform(-.5, .5, size=(4, 3)))
    f, e = sc.solve(k)
    w = unf.weights(k, e)
    assert unf.mapping.n_vacancies == 1
    assert np.allclose(w.sum(1), 3 * 39 / 8), w.sum(1)
    print(f"  O vacancy: sum rule = 3 x 39 / 8 = {unf.sum_rule:.3f} holds")


def test_interstitial():
    prim = perovskite()
    host = make_supercell(prim, np.diag([2, 2, 2]))
    ideal = Structure(host.lattice, np.vstack([host.frac, [.13, .12, .11]]),
                      host.symbols + ["H"])
    sc = ToyBackend(ideal)            # H has no springs: 3 free modes
    unf = Unfolder(sc.structure, prim, allow_interstitials=True,
                   match_species=False, max_displacement=0.5)
    k = unf.k_cart(rng.uniform(-.5, .5, size=(3, 3)))
    f, e = sc.solve(k)
    w = unf.weights(k, e)
    assert unf.mapping.n_interstitials == 1
    assert np.allclose(w.sum(1), 3 * 41 / 8)
    print("  interstitial H: own ghost sublattice, sum rule 3 x 41 / 8 holds")


def test_auto_reference_and_gauges():
    prim = perovskite()
    M = [[2, 0, 0], [0, 2, 0], [0, 0, 3]]
    ideal = make_supercell(prim, M)
    actual = rattle(ideal, 0.05, seed=3)
    sc = ToyBackend(ideal, actual)
    ref = reference_from_supercell(actual, M)       # no reference file needed
    assert ref.natoms == 5 and sorted(ref.symbols) == sorted(prim.symbols)
    unf = Unfolder(sc.structure, ref, supercell_matrix=M)
    q = rng.uniform(-.5, .5, size=(5, 3))
    k = unf.k_cart(q)
    f, e = sc.solve(k)
    w = unf.weights(k, e)
    fp, _ = ToyBackend(prim).solve(q @ prim.recip)
    assert_exact_unfolding(f, w, fp)
    # R-gauge eigenvectors: e_R = e_r * exp(+2 pi i k . r_s)
    r = sc.structure.cart
    eR = np.array([ev * np.repeat(np.exp(2j * np.pi * r @ kk), 3)[:, None]
                   for kk, ev in zip(k, e)])
    assert np.allclose(unf.weights(k, eR, gauge="R"), w, atol=1e-10)
    # sublattice-resolved weights add up to the total
    ws = unf.weights(k, e, per_sublattice=True)
    assert np.allclose(ws.sum(2), w, atol=1e-12)
    print("  reference built from the supercell average: exact; R-gauge "
          "input and sublattice decomposition consistent")


def test_run_unfolding_projections_resolve():
    prim = perovskite()
    M = np.diag([4, 1, 1])
    ideal = make_supercell(prim, M)
    sc = ToyBackend(ideal, rattle(ideal, 0.02, seed=5))
    backend = CallableBackend(sc.structure,
                              lambda k: tuple(a[0] for a in sc.solve(k)))
    res = run_unfolding(backend, prim, path="cubic", density=20,
                        resolve="species", verbose=False,
                        projections={"Ti": "species:Ti",
                                     "plane": "slab:0:0.0:1.0",
                                     "point": "sphere:0.125,0.5,0.5:2.1"})
    tot = sum(res.extra[f"w_{s}"] for s in ("Ba", "Ti", "O"))
    assert np.allclose(tot, res.weights)
    assert res.meta["sum_rule_error"] < 1e-9
    assert set(res.projections()) == {"Ti", "plane", "point"}
    p = res.extra["proj_Ti"]
    assert p.min() > -1e-12 and p.max() < 1 + 1e-12
    print(f"  run_unfolding on preset 'cubic' path ({len(res.path['q'])} k): "
          "species-resolved weights sum to total; 3 projections OK")


def test_path_frames_and_presets():
    a = 4.0
    fcc = Structure(np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]) * a,
                    [[0, 0, 0]], ["Cu"])
    segs = resolve_path("fcc", None, fcc.lattice, fcc.lattice, fcc.lattice)
    pts = {lab: q for seg in segs for lab, q in seg}
    assert np.allclose(pts["X"], [.5, 0, .5]) and np.allclose(pts["L"], .5)
    # supercell frame: (1/2,0,0) of a 2x1x1 supercell == (1/4,0,0) of ref
    prim = perovskite()
    sc = make_supercell(prim, np.diag([2, 1, 1]))
    segs = resolve_path("G 0 0 0, X 1/2 0 0", "supercell", prim.lattice,
                        prim.lattice, sc.lattice)
    assert np.allclose(segs[0][1][1], [.25, 0, 0])
    print("  fcc preset gives X=(1/2,0,1/2), L=(1/2,1/2,1/2) in primitive "
          "coords; supercell frame converts correctly")


def test_native_pruning_units():
    prim = perovskite()
    unit = make_supercell(prim, np.diag([2, 1, 1]))
    spos = make_supercell(unit, np.diag([2, 3, 3]))
    fc = ToyBackend(make_supercell(make_supercell(prim, np.diag([2, 1, 1])),
                                   np.diag([2, 3, 3]))).real_space_fc()
    with tempfile.TemporaryDirectory() as tmp:
        write_poscar(unit, f"{tmp}/POSCAR", sort=False)
        write_poscar(spos, f"{tmp}/SPOSCAR", sort=False)
        write_fc(fc, f"{tmp}/FC")
        b0 = NativeBackend(f"{tmp}/POSCAR", f"{tmp}/SPOSCAR", f"{tmp}/FC")
        b1 = NativeBackend(f"{tmp}/POSCAR", f"{tmp}/SPOSCAR", f"{tmp}/FC",
                           fc_threshold=1e-8, chunk=777)
        b2 = NativeBackend(f"{tmp}/POSCAR", f"{tmp}/SPOSCAR", f"{tmp}/FC",
                           masses={"O": 17.999})
    k = rng.uniform(-.5, .5, size=(4, 3)) @ unit.recip
    f0, _ = b0.solve(k)
    f1, _ = b1.solve(k)
    f2, _ = b2.solve(k)
    assert np.abs(f0 - f1).max() < 1e-10 and b1.n_pairs < b0.n_pairs
    assert f2.max() < f0.max()              # heavier O isotope softens modes
    print(f"  FC pruning keeps {b1.n_pairs}/{b0.n_pairs} pairs, identical "
          "frequencies; mass override (18O) works")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("ALL TESTS PASSED")


def test_builders_and_disp_dirs():
    from phunfold.builders import perovskite, make_disp_dirs
    from phunfold.paths import detect_preset
    from phunfold.domainwall import build_180_wall
    cub = perovskite(4.0)
    assert detect_preset(cub.lattice) == "cubic" and cub.natoms == 5
    tet = perovskite(3.99, c=4.03, polar_axis="c",
                     disp={"B": 0.1, "X_ap": -0.08, "X_eq": -0.06})
    assert detect_preset(tet.lattice) == "tetragonal"
    z = tet.cart[:, 2]
    assert np.isclose(z[1], 4.03 / 2 + 0.1)              # Ti up
    assert np.isclose(z[4] % 4.03, 4.03 - 0.08)           # apical O down
    # the polar cell feeds straight into the domain-wall builder
    dw, info = build_180_wall(tet, ncells=3)
    assert dw.natoms == 30 and info["supercell_matrix"][0][0] == 6
    with tempfile.TemporaryDirectory() as tmp:
        sp = make_supercell(cub, np.diag([2, 2, 2]))
        write_poscar(sp, f"{tmp}/SPOSCAR")
        for i in (1, 2, 10):
            write_poscar(sp, f"{tmp}/POSCAR-{i:03d}")
        for name in ("INCAR_static", "submit.sh"):
            open(f"{tmp}/{name}", "w").write("x\n")
        made = make_disp_dirs(tmp, copy=[f"{tmp}/INCAR_static:INCAR",
                                         f"{tmp}/submit.sh"])
        names = sorted(os.path.basename(m) for m in made)
        assert names == ["disp-001", "disp-002", "disp-010", "perfect"]
        assert sorted(os.listdir(f"{tmp}/disp-010")) == ["INCAR", "POSCAR",
                                                         "submit.sh"]
    print("  perovskite (cubic/polar) -> build-dw chain OK; disp-dirs makes "
          "perfect + disp-NNN with your files")
