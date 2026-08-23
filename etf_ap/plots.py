"""Publication-style figures for the AP HJB-QVI solution."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .analytics import free_boundary_x, slice_at
from .simulate import PathBatch
from .solver import Solution


def _style(plt) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )


def make_all_figures(sol: Solution, batches: dict[str, PathBatch], outdir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    sl = slice_at(sol, t_frac=0.0, qi=0)
    x, qe = sl["x"], sl["qe"]
    X, Q = np.meshgrid(x, qe, indexing="ij")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    im = ax.pcolormesh(Q, X, sl["V"], shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label=r"$V(0,x,q^E,0)$")
    ax.set_xlabel(r"ETF inventory $q^E$")
    ax.set_ylabel(r"mispricing $X = S^E - I$")
    ax.set_title("Value function at t = 0, q^I = 0")
    fig.tight_layout()
    p = outdir / "value_function.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    written.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    for ax, key, title in zip(
        axes,
        ["delta_b", "delta_a"],
        [r"optimal bid half-spread $\delta^{b*}$", r"optimal ask half-spread $\delta^{a*}$"],
    ):
        im = ax.pcolormesh(Q, X, sl[key], shading="auto", cmap="magma")
        fig.colorbar(im, ax=ax, label="spread")
        ax.set_xlabel(r"$q^E$")
        ax.set_title(title)
    axes[0].set_ylabel(r"$X$")
    fig.tight_layout()
    p = outdir / "optimal_spreads.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    written.append(p)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    # region: -1 redeem, 0 cont, +1 create
    im = ax.pcolormesh(Q, X, sl["region"], shading="auto", cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, ticks=[-1, 0, 1], label="redeem / continuation / create")
    ax.set_xlabel(r"$q^E$")
    ax.set_ylabel(r"$X$")
    ax.set_title("QVI partition at t = 0, q^I = 0")
    fig.tight_layout()
    p = outdir / "qvi_regions.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    written.append(p)

    fb = free_boundary_x(sol, t_index=0, qi=0)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(fb["qe"], fb["create_x"], "o-", label="create interface")
    ax.plot(fb["qe"], fb["redeem_x"], "s-", label="redeem interface")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"$q^E$")
    ax.set_ylabel(r"impulse threshold in $X$")
    ax.set_title("Free boundary (q^I = 0, t = 0)")
    ax.legend()
    fig.tight_layout()
    p = outdir / "free_boundary.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    written.append(p)

    # inventory-plane region at x slices
    g = sol.grid
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), sharey=True)
    QE, QI = np.meshgrid(g.qe, g.qi, indexing="ij")
    for ax, xv, title in zip(axes, [-0.2, 0.0, 0.2], ["X = -0.20 (cheap ETF)", "X = 0", "X = +0.20 (rich ETF)"]):
        i = int(np.argmin(np.abs(g.x - xv)))
        im = ax.pcolormesh(QE, QI, sol.region[i, :, :, 0], shading="auto", cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title(title)
        ax.set_xlabel(r"$q^E$")
    axes[0].set_ylabel(r"$q^I$")
    fig.colorbar(im, ax=axes.ravel().tolist(), ticks=[-1, 0, 1], fraction=0.02, pad=0.02)
    fig.suptitle("Impulse regions in inventory, t = 0")
    fig.tight_layout()
    p = outdir / "impulse_inventory.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    written.append(p)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    i0, j0, k0 = g.i0, g.qe_index(0), g.qi_index(0)
    ax.plot(g.x, sol.V[:, j0, k0, 0], label=r"$q^E=q^I=0$")
    ax.plot(g.x, sol.V[:, g.qe_index(min(2, g.params.q_e_max)), k0, 0], label=r"$q^E=2,q^I=0$")
    ax.plot(g.x, sol.V[:, g.qe_index(max(-2, -g.params.q_e_max)), k0, 0], label=r"$q^E=-2,q^I=0$")
    ax.set_xlabel(r"$X$")
    ax.set_ylabel(r"$V(0,X,q^E,0)$")
    ax.set_title("Value vs mispricing")
    ax.legend()
    fig.tight_layout()
    p = outdir / "value_vs_x.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    written.append(p)

    if batches:
        fig, ax = plt.subplots(figsize=(7.0, 4.3))
        for name, batch in batches.items():
            w = batch.wealth[-1]
            ax.hist(w, bins=40, density=True, alpha=0.45, label=name)
        ax.set_xlabel("terminal marked wealth")
        ax.set_ylabel("density")
        ax.set_title("Monte Carlo policy comparison")
        ax.legend()
        fig.tight_layout()
        p = outdir / "mc_wealth.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        written.append(p)

        fig, ax = plt.subplots(figsize=(7.0, 4.3))
        for name, batch in batches.items():
            ax.plot(batch.t, np.mean(batch.wealth, axis=1), label=name)
        ax.set_xlabel("t")
        ax.set_ylabel("mean marked wealth")
        ax.set_title("Mean wealth along the horizon")
        ax.legend()
        fig.tight_layout()
        p = outdir / "mc_mean_wealth.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        written.append(p)

        opt = batches.get("optimal")
        if opt is not None:
            fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
            axes[0].plot(opt.t, np.mean(opt.qe, axis=1), label=r"mean $q^E$")
            axes[0].plot(opt.t, np.mean(opt.qi, axis=1), label=r"mean $q^I$")
            axes[0].legend()
            axes[0].set_ylabel("inventory")
            axes[1].plot(opt.t, np.mean(np.abs(opt.x), axis=1))
            axes[1].set_ylabel(r"mean $|X_t|$")
            axes[1].set_xlabel("t")
            fig.suptitle("Optimal policy: inventory and spread tightness")
            fig.tight_layout()
            p = outdir / "mc_inventory.png"
            fig.savefig(p, dpi=160)
            plt.close(fig)
            written.append(p)

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.plot(sol.grid.t[:-1], sol.residual)
    ax.set_xlabel("t")
    ax.set_ylabel("max QVI projection gap")
    ax.set_title("Impulse complementarity residual")
    fig.tight_layout()
    p = outdir / "qvi_residual.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    written.append(p)

    return written


make_all_figures = make_all_figures
