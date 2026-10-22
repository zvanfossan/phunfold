"""Validation of phunfold against an exactly solvable toy perovskite.

Run:  python tests/test_unfold.py        (or pytest tests/)
"""

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from phunfold import (NativeBackend, Unfolder, run_unfolding,  # noqa: E402
                      write_poscar, read_poscar, make_band_path,
                      parse_path_string, DEFAULT_PATH)
from toymodel import perovskite, tile, ToyBackend, write_fc  # noqa: E402

rng = np.random.default_rng(7)


def _displaced(s, amp):
    d = s.copy()
    d.frac = d.frac + rng.normal(scale=amp, size=d.frac.shape) @ np.linalg.inv(d.lattice)
    return d


def _degenerate_groups(freqs, tol=1e-5):
    order = np.argsort(freqs)
    groups, cur = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if freqs[b] - freqs[a] < tol:
            cur.append(b)
        else:
            groups.append(cur)
            cur = [b]
    groups.append(cur)
    return groups


def _check_perfect(M, amp, distortion_phase=True, tol=1e-8):
    prim = perovskite()
    ideal = tile(prim, M)
    disp = _displaced(ideal, amp)
    sc_backend = ToyBackend(ideal, disp)
    prim_backend = ToyBackend(prim)
    unf = Unfolder(sc_backend.structure, prim, distortion_phase=distortion_phase)
    q = rng.uniform(-0.5, 0.5, size=(12, 3))
    q[0] = [0.5, 0.25, 0.0]
    q[1] = [0, 0, 0]
    k = unf.k_cart(q)
    f, e = sc_backend.solve(k)
    w = unf.weights(k, e)
    fp, _ = prim_backend.solve(k)
    assert np.allclose(w.sum(1), 15, atol=1e-9), w.sum(1)
    # Degenerate supercell modes may mix folded k's, so per-mode weights are
    # only 0/1 for non-degenerate modes; the gauge-invariant statement is
    # that the summed weight of every degenerate group equals the number of
    # primitive branches at that frequency.
    binary = 0.0
    for iq in range(len(q)):
        groups = _degenerate_groups(f[iq])
        for g in groups:
            wsum = w[iq, g].sum()
            n_prim = np.sum(np.abs(fp[iq] - f[iq, g].mean()) < 1e-5)
            binary = max(binary, abs(wsum - round(wsum)))
            if distortion_phase:
                assert abs(wsum - n_prim) < 1e-6, (iq, f[iq, g], wsum, n_prim)
    if distortion_phase:
        assert binary < tol, f"weights not 0/1 (dev {binary:.2e})"
    return binary


def test_perfect_supercells():
    for M in ([[4, 0, 0], [0, 1, 1], [0, 1, -1]],      # 90deg-rotated cell
              [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
              [[6, 0, 0], [0, 1, 0], [0, 0, 1]],       # DW-like stacking
              [[1, 1, 0], [-1, 1, 0], [0, 0, 3]]):
        dev = _check_perfect(M, amp=0.08)
        print(f"  M={M}: perfect-supercell weights are 0/1 to {dev:.1e}; "
              "unfolded bands = primitive bands")
    # Without the distortion phase (phonopy.unfolding convention) weights are
    # only approximately 0/1 when atoms are displaced from ideal sites:
    dev = _check_perfect([[6, 0, 0], [0, 1, 0], [0, 0, 1]], 0.08,
                         distortion_phase=False)
    print(f"  without distortion phase: max deviation from 0/1 = {dev:.3f} "
          "(expected small but non-zero)")
    assert 1e-4 < dev < 0.3


def test_native_backend_roundtrip():
    """FORCE_CONSTANTS + SPOSCAR -> NativeBackend reproduces exact D(k)."""
    prim = perovskite()
    for dim, amp in (((2, 3, 3), 0.06), ((2, 2, 2), 0.0)):   # 2nd: ties
        ideal_u = tile(prim, [[4, 0, 0], [0, 1, 0], [0, 0, 1]])
        unit = _displaced(ideal_u, amp) if amp else ideal_u
        exact = ToyBackend(ideal_u, unit)
        ideal_s = tile(ideal_u, np.diag(dim))
        # SPOSCAR = tile of the *displaced* unit cell, same atom order as
        # tile(ideal_u) -> displacement copied cell by cell
        sposcar = tile(unit, np.diag(dim))
        fc = ToyBackend(ideal_s).real_space_fc()
        with tempfile.TemporaryDirectory() as tmp:
            write_poscar(unit, f"{tmp}/POSCAR", sort=False)
            write_poscar(sposcar, f"{tmp}/SPOSCAR", sort=False)
            write_fc(fc, f"{tmp}/FORCE_CONSTANTS")
            nb = NativeBackend(f"{tmp}/POSCAR", f"{tmp}/SPOSCAR",
                               f"{tmp}/FORCE_CONSTANTS")
        q = rng.uniform(-0.5, 0.5, size=(6, 3))
        k = q @ unit.recip
        f1, _ = exact.solve(k)
        f2, _ = nb.solve(k)
        err = np.abs(f1 - f2).max()
        print(f"  dim={dim}, disp={amp}: max |freq error| = {err:.2e} THz "
              f"(max multiplicity {nb.max_multiplicity})")
        assert err < 1e-6
        # same eigenvectors gauge => same unfolding weights
        unf = Unfolder(nb.structure, prim)
        _, e1 = exact.solve(k)
        _, e2 = nb.solve(k)
        w1, w2 = unf.weights(k, e1), unf.weights(k, e2)
        assert np.allclose(np.sort(w1, 1), np.sort(w2, 1), atol=1e-6)


def test_path():
    prim = perovskite()
    p = make_band_path(parse_path_string(DEFAULT_PATH), prim.recip, 40)
    assert p["labels"] == ["Γ", "X", "M", "Γ", "Y|Γ", "Z", "R", "A", "Z"]
    assert np.allclose(p["q"][0], 0) and np.allclose(p["q"][-1], [0, 0, .5])


def test_poscar_roundtrip():
    s = _displaced(tile(perovskite(), np.diag([3, 1, 2])), 0.05)
    with tempfile.TemporaryDirectory() as tmp:
        w = write_poscar(s, f"{tmp}/POSCAR")
        r = read_poscar(f"{tmp}/POSCAR")
    assert r.symbols == w.symbols and np.allclose(r.frac, w.frac)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("ALL TESTS PASSED")
