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


def plot_unfolded(result, ax=None, style="spectral", units="THz", ylim=None,
                  sigma=0.12, n_omega=600, wmin=0.02, size=14.0,
                  color_by=None, cmap=None, reference_bands=None,
                  title=None, colorbar=True):
    """Plot an UnfoldResult.

    style : 'spectral' (broadened A(k,w) map), 'scatter' (fat bands, marker
            area ~ weight), or 'both'.
    color_by : None, or the name of an extra array (e.g.
            'wall_participation') used to colour scatter markers.
    reference_bands : optional (freqs (nq, nb)) from a bulk reference cell
            calculation on the same path, drawn as thin dashed lines.
    sigma : broadening in ``units``.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    c = UNITS[units]
    f = result.freqs * c
    w = result.weights
    x = result.path["x"]
    if ylim is None:
        ylim = (min(0.0, f.min() * 1.05), f.max() * 1.05)
    mappable = None
    if style in ("spectral", "both"):
        omega = np.linspace(ylim[0], ylim[1], n_omega)
        a = spectral_function(f, w, omega, sigma=sigma)
        mappable = ax.pcolormesh(x, omega, a, shading="gouraud",
                                 cmap=cmap or "Greys",
                                 vmin=0, vmax=np.percentile(a, 99.5))
        if colorbar and color_by is None:
            plt.colorbar(mappable, ax=ax, pad=0.01).set_label("A(k, ω) (arb.)")
    if style in ("scatter", "both"):
        sel = w > wmin
        xx = np.broadcast_to(x[:, None], f.shape)[sel]
        if color_by is not None:
            col = result.extra[color_by][sel]
            order = np.argsort(col)   # draw strongly localised modes on top
            sc = ax.scatter(xx[order], f[sel][order], s=size * w[sel][order],
                            c=col[order], cmap=cmap or "viridis", vmin=0,
                            vmax=max(col.max(), 1e-9), lw=0, zorder=3)
            if colorbar:
                plt.colorbar(sc, ax=ax, pad=0.01).set_label(
                    color_by.replace("_", " "))
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
