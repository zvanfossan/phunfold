"""Plotting helpers for unfolded phonon spectra (matplotlib)."""

from __future__ import annotations

import numpy as np

from .dynmat import THZ_TO_CM, THZ_TO_MEV
from .unfold import spectral_function

UNITS = {"THz": 1.0, "cm-1": THZ_TO_CM, "meV": THZ_TO_MEV}
UNIT_LABEL = {"THz": "Frequency (THz)", "cm-1": r"Frequency (cm$^{-1}$)",
              "meV": "Energy (meV)"}


def _decorate(ax, path, ylim, units):
    for t in path["ticks"]:
        ax.axvline(t, color="0.75", lw=0.6, zorder=0)
    ax.axhline(0, color="0.75", lw=0.6, zorder=0)
    ax.set_xticks(path["ticks"])
    ax.set_xticklabels(path["labels"])
    ax.set_xlim(path["x"][0], path["x"][-1])
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(UNIT_LABEL[units])


def _match(fa, fb, max_jump, fa_pred=None):
    """Greedy pairing of modes at two neighbouring k-points, closest first.
    ``fa_pred`` (fa extrapolated along each branch's slope) keeps nearby
    branches from swapping. Returns pairs and the unmatched indices."""
    if len(fa) == 0 or len(fb) == 0:
        return [], list(range(len(fa))), list(range(len(fb)))
    ref = fa if fa_pred is None else fa_pred
    d = np.abs(ref[:, None] - fb[None, :])
    order = np.argsort(d, axis=None)
    used_a, used_b, pairs = set(), set(), []
    for flat in order:
        i, j = divmod(int(flat), len(fb))
        if d[i, j] > max_jump:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j))
    return (pairs, [i for i in range(len(fa)) if i not in used_a],
            [j for j in range(len(fb)) if j not in used_b])


def _upsample(x, f, w, factor, wmin=1e-3, max_jump=None):
    """Display-only refinement of the spectral map between computed k-points.

    Only modes that carry weight are interpolated, and each is paired with the
    nearest-frequency weighted mode at the next k-point. A mode without a
    partner fades out (or in) at constant frequency. Returns a list of
    (x, freqs, weights) "pseudo k-points" with variable numbers of modes,
    flattened into padded arrays (zero weight padding)."""
    if factor <= 1:
        return x, f, w
    span = np.ptp(f) if f.size else 1.0
    max_jump = max_jump or 0.08 * span
    xs, cols_f, cols_w = [], [], []

    def add(xv, fv, wv):
        xs.append(xv)
        cols_f.append(np.asarray(fv, float))
        cols_w.append(np.asarray(wv, float))

    sel0 = w[0] > wmin
    add(x[0], f[0][sel0], w[0][sel0])
    slope = np.zeros(int(sel0.sum()))
    for i in range(len(x) - 1):
        sa, sb = w[i] > wmin, w[i + 1] > wmin
        fa, wa, fb, wb = f[i][sa], w[i][sa], f[i + 1][sb], w[i + 1][sb]
        dx = x[i + 1] - x[i]
        if dx <= 1e-12:                        # path break: no interpolation
            add(x[i + 1], fb, wb)
            slope = np.zeros(len(fb))
            continue
        pairs, ua, ub = _match(fa, fb, max_jump, fa + slope * dx)
        new_slope = np.zeros(len(fb))
        for a_, b_ in pairs:
            new_slope[b_] = (fb[b_] - fa[a_]) / dx
        slope = new_slope
        pa = np.array([p[0] for p in pairs], int)
        pb = np.array([p[1] for p in pairs], int)
        for t in np.linspace(0, 1, factor + 1)[1:]:
            ff = [fa[pa] + t * (fb[pb] - fa[pa]) if len(pairs) else []]
            ww = [wa[pa] + t * (wb[pb] - wa[pa]) if len(pairs) else []]
            # unmatched modes fade over the first / last half of the interval
            fade_out = max(0.0, 1 - 2 * t)
            fade_in = max(0.0, 2 * t - 1)
            ff += [fa[ua], fb[ub]]
            ww += [wa[ua] * fade_out, wb[ub] * fade_in]
            add(x[i] + t * (x[i + 1] - x[i]), np.concatenate(ff),
                np.concatenate(ww))
    n = max(len(c) for c in cols_f)
    F = np.zeros((len(xs), n))
    W = np.zeros((len(xs), n))
    for k, (cf, cw) in enumerate(zip(cols_f, cols_w)):
        F[k, :len(cf)] = cf
        W[k, :len(cw)] = cw
    return np.array(xs), F, W


def plot_unfolded(result, ax=None, style="spectral", units="THz", ylim=None,
                  sigma=None, n_omega=600, wmin=0.02, size=14.0,
                  color_by=None, cmap=None, reference_bands=None,
                  title=None, colorbar=True, weights=None, vmax=None,
                  upsample=6):
    """Plot an UnfoldResult.

    style : 'spectral' (broadened A(k,w) map), 'scatter' (fat bands, marker
            area ~ weight), or 'both'.
    color_by : None, or the name of an extra array (e.g.
            'wall_participation') used to colour scatter markers.
    reference_bands : optional (freqs (nq, nb)) from a bulk reference cell
            calculation on the same path, drawn as thin dashed lines.
    weights : plot these weights instead of result.weights (e.g. a
            species-resolved array result.extra['w_Ti']).
    vmax : colour-scale maximum of the spectral map (share between panels).
    upsample : interpolation factor between k-points for the spectral map
            (display only; 1 shows exactly the computed k-points).
    sigma : broadening in ``units`` (default 0.12 THz converted to them).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    c = UNITS[units]
    if sigma is None:
        sigma = 0.12 * c
    f = result.freqs * c
    w = result.weights if weights is None else weights
    x = result.path["x"]
    if ylim is None:
        ylim = (min(0.0, f.min() * 1.05), f.max() * 1.05)
    mappable = None
    if style in ("spectral", "both"):
        omega = np.linspace(ylim[0], ylim[1], n_omega)
        xu, fu, wu = _upsample(x, f, w, upsample)
        a = spectral_function(fu, wu, omega, sigma=sigma)
        mappable = ax.pcolormesh(xu, omega, a, shading="gouraud",
                                 cmap=cmap or "Greys",
                                 vmin=0, vmax=vmax or np.percentile(a, 99.5))
        if colorbar and color_by is None:
            plt.colorbar(mappable, ax=ax, pad=0.01).set_label("A(k, ω) (arb.)")
    if style in ("scatter", "both"):
        sel = w > wmin
        xx = np.broadcast_to(x[:, None], f.shape)[sel]
        if color_by is not None:
            key = color_by if color_by in result.extra else f"proj_{color_by}"
            col = result.extra[key][sel]
            order = np.argsort(col)   # draw strongly localised modes on top
            sc = ax.scatter(xx[order], f[sel][order], s=size * w[sel][order],
                            c=col[order], cmap=cmap or "viridis", vmin=0,
                            vmax=max(col.max(), 1e-9), lw=0, zorder=3)
            if colorbar:
                plt.colorbar(sc, ax=ax, pad=0.01).set_label(
                    f"fraction on '{color_by}'")
        else:
            ax.scatter(xx, f[sel], s=size * w[sel],
                       c="C3" if style == "both" else "k", lw=0, zorder=3)
    if reference_bands is not None:
        ax.plot(x, reference_bands * c, color="C0", lw=0.7, ls="--",
                alpha=0.8, zorder=2)
    _decorate(ax, result.path, ylim, units)
    if title:
        ax.set_title(title)
    return ax


def plot_folded(result, ax=None, units="THz", ylim=None, color="k"):
    """Raw supercell bands (no weights) on the same path, for comparison."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(result.path["x"], result.freqs * UNITS[units], color=color,
            lw=0.4)
    _decorate(ax, result.path, ylim, units)
    return ax
