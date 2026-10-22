"""k-point paths: parsing, presets and coordinate frames.

A path is a list of continuous segments; each segment is a list of
(label, q) points. Points can be given in three frames:

* ``reference`` (default): reduced coordinates of the reference cell as
  written in its POSCAR.
* ``supercell``: reduced coordinates of the supercell reciprocal lattice.
* ``cartesian``: Cartesian 1/Angstrom (no 2 pi) in the reference POSCAR's
  own Cartesian frame. Presets for fcc/bcc use this frame internally so they
  work whatever primitive vectors your reference file uses, as long as its
  cubic axes are along x, y, z.

All frames are converted to reduced coordinates of the *effective*
reference lattice inv(M) @ supercell.lattice, which is what the unfolding
uses (this also takes care of rotated supercells).
"""

from __future__ import annotations

import numpy as np

F = 1.0  # readability


def _pts(**kw):
    return {k: np.array(v, float) for k, v in kw.items()}


# Setyawan & Curtarolo, Comp. Mat. Sci. 49, 299 (2010) conventions.
PRESETS = {
    "cubic": dict(frame="reference",
                  points=_pts(G=[0, 0, 0], X=[0, .5, 0], M=[.5, .5, 0],
                              R=[.5, .5, .5]),
                  path="G X M G R X | M R"),
    "tetragonal": dict(frame="reference",
                       points=_pts(G=[0, 0, 0], X=[0, .5, 0], M=[.5, .5, 0],
                                   Z=[0, 0, .5], R=[0, .5, .5],
                                   A=[.5, .5, .5]),
                       path="G X M G Z R A Z | X R | M A"),
    "orthorhombic": dict(frame="reference",
                         points=_pts(G=[0, 0, 0], X=[.5, 0, 0], Y=[0, .5, 0],
                                     Z=[0, 0, .5], S=[.5, .5, 0],
                                     U=[.5, 0, .5], T=[0, .5, .5],
                                     R=[.5, .5, .5]),
                         path="G X S Y G Z U R T Z | Y T | U X | S R"),
    "hexagonal": dict(frame="reference",
                      points=_pts(G=[0, 0, 0], M=[.5, 0, 0],
                                  K=[1 / 3, 1 / 3, 0], A=[0, 0, .5],
                                  L=[.5, 0, .5], H=[1 / 3, 1 / 3, .5]),
                      path="G M K G A L H A | L M | K H"),
    # cubic-axis Cartesian coordinates in units of 2 pi / a_conv
    "fcc": dict(frame="cubic_units", conv_factor=4,
                points=_pts(G=[0, 0, 0], X=[0, 1, 0], W=[.5, 1, 0],
                            K=[.75, .75, 0], L=[.5, .5, .5], U=[.25, 1, .25]),
                path="G X W K G L U W L K | U X"),
    "bcc": dict(frame="cubic_units", conv_factor=2,
                points=_pts(G=[0, 0, 0], H=[0, 0, 1], N=[0, .5, .5],
                            P=[.25, .25, .25]),
                path="G H N G P H | P N"),
}
# Domain-wall default: in-plane (Y) and out-of-plane (X) directions kept
# separate, polar axis along c.
PRESETS["dw-tetragonal"] = dict(
    frame="reference",
    points=_pts(G=[0, 0, 0], X=[.5, 0, 0], Y=[0, .5, 0], M=[.5, .5, 0],
                Z=[0, 0, .5], R=[.5, 0, .5], A=[.5, .5, .5]),
    path="G X M G Y | G Z R A Z")
DEFAULT_PATH = "dw-tetragonal"


def _label(t):
    return "Γ" if t.upper() in ("G", "GAMMA", "Γ") else t


def parse_path_string(text):
    """'G 0 0 0, X 1/2 0 0, M .5 .5 0 | Z 0 0 1/2, R .5 .5 .5' -> segments."""
    segments = []
    for part in text.split("|"):
        seg = []
        for item in part.split(","):
            tok = item.split()
            if not tok:
                continue
            if len(tok) != 4:
                raise ValueError(f"bad path entry {item!r}: need 'LABEL q1 q2 q3'")
            vals = []
            for t in tok[1:]:
                if "/" in t:
                    a, b = t.split("/")
                    vals.append(float(a) / float(b))
                else:
                    vals.append(float(t))
            seg.append((_label(tok[0]), vals))
        if seg:
            segments.append(seg)
    return segments


def preset_segments(name, ref_lattice=None):
    """Segments for a preset; fcc/bcc returned in Cartesian (1/A) frame."""
    p = PRESETS[name]
    pts = p["points"]
    if p["frame"] == "cubic_units":
        vol = abs(np.linalg.det(ref_lattice))
        a = (p["conv_factor"] * vol) ** (1 / 3)
        pts = {k: v / a for k, v in pts.items()}
        frame = "cartesian"
    else:
        frame = p["frame"]
    if name == "hexagonal" and ref_lattice is not None:
        a, b = ref_lattice[0], ref_lattice[1]
        gamma = np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))
        if abs(gamma - 120) > 2:
            raise ValueError("hexagonal preset assumes gamma = 120 deg "
                             f"(found {gamma:.1f}); give the path explicitly")
    segs = [[(_label(l), pts[l]) for l in part.split()]
            for part in p["path"].split("|")]
    return segs, frame


def resolve_path(spec, frame, ref_lattice_file, ref_lattice_eff, sc_lattice):
    """Return segments in reduced coords of the effective reference lattice.

    spec : preset name, path string, or list of segments.
    frame: frame of the given coordinates (ignored for presets).
    ref_lattice_file : lattice of the reference as read from its file.
    ref_lattice_eff  : inv(M) @ sc.lattice.
    """
    if isinstance(spec, str) and spec.strip() in PRESETS:
        segs, frame = preset_segments(spec.strip(), ref_lattice_file)
    elif isinstance(spec, str):
        segs = parse_path_string(spec)
    else:
        segs = spec
    frame = frame or "reference"
    if frame == "reference":
        # reduced coordinates of the file lattice. If the supercell is
        # rotated, the effective lattice is the rotated version of the same
        # basis choice M picked, so the reduced coordinates carry over.
        conv = lambda q: np.asarray(q, float)  # noqa: E731
    elif frame == "supercell":
        conv = lambda q: (np.asarray(q, float) @ np.linalg.inv(sc_lattice).T  # noqa
                          @ ref_lattice_eff.T)
    elif frame == "cartesian":
        # Cartesian in the reference FILE frame -> its reduced coordinates
        conv = lambda q: np.asarray(q, float) @ ref_lattice_file.T  # noqa
    else:
        raise ValueError(f"unknown frame {frame!r}")
    return [[(lab, conv(q)) for lab, q in seg] for seg in segs]


def make_band_path(segments, recip, density=60.0, npts=None):
    """Piecewise-linear path through ``segments`` (reduced coords of
    ``recip``). Returns dict(q, k_cart, x, ticks, labels)."""
    qs, xs, ticks, labels = [], [], [], []
    x0 = 0.0
    for iseg, seg in enumerate(segments):
        if iseg == 0:
            ticks.append(x0)
            labels.append(seg[0][0])
        else:
            labels[-1] = f"{labels[-1]}|{seg[0][0]}"
        for (_, qa), (lb, qb) in zip(seg[:-1], seg[1:]):
            qa, qb = np.asarray(qa, float), np.asarray(qb, float)
            length = np.linalg.norm((qb - qa) @ recip)
            n = npts or max(int(np.ceil(length * density)), 2)
            t = np.linspace(0.0, 1.0, n + 1)
            qs.append(qa + t[:, None] * (qb - qa))
            xs.append(x0 + t * length)
            x0 += length
            ticks.append(x0)
            labels.append(lb)
    q = np.vstack(qs)
    return dict(q=q, k_cart=q @ recip, x=np.concatenate(xs),
                ticks=np.array(ticks), labels=labels)
