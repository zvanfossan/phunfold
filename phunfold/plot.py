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


def _upsample(x, f, w, factor):
    """Linearly interpolate (sorted) frequencies and weights between
    neighbouring k-points so steep branches render as continuous lines in the
    spectral map. Path breaks (repeated x) are not interpolated across."""
    if factor <= 1:
        return x, f, w
    xs, fs, ws = [x[:1]], [f[:1]], [w[:1]]
    for i in range(len(x) - 1):
        if x[i + 1] - x[i] <= 1e-12:
            xs.append(x[i + 1:i + 2])
            fs.append(f[i + 1:i + 2])
            ws.append(w[i + 1:i + 2])
            continue
        t = np.linspace(0, 1, factor + 1)[1:, None]
        xs.append(x[i] + t[:, 0] * (x[i + 1] - x[i]))
        fs.append(f[i] + t * (f[i + 1] - f[i]))
        ws.append(w[i] + t * (w[i + 1] - w[i]))
    return np.concatenate(xs), np.vstack(fs), np.vstack(ws)


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
        xu, fu, wu = _upsample(x, np.sort(f, axis=1),
                               np.take_along_axis(w, np.argsort(f, axis=1), 1),
                               upsample)
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
