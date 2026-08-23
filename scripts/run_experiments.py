"""Run the paper-scale solver, Monte Carlo horse-race, and write figures + metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from etf_ap.analytics import (
    quoting_identity_error,
    smooth_pasting_error,
    asymmetry_residual,
    value_at_origin,
)
from etf_ap.params import ModelParams
from etf_ap.plots import make_all_figures
from etf_ap.simulate import simulate, summarize
from etf_ap.solver import solve


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", dest="profile", choices=["paper", "default", "fine"], default="paper")
    ap.add_argument("--paths", dest="paths", type=int, default=4000)
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    return ap.parse_args()


def build_params(profile: str) -> ModelParams:
    base = ModelParams()
    if profile == "paper":
        return base.paper()
    if profile == "fine":
        return base.fine()
    return base


def main() -> None:
    args = parse_args()
    profile = args.profile
    n_paths = args.paths
    params = build_params(profile)
    outdir = args.outdir
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== HJB-QVI solve ===")
    print(params)
    sol = solve(params, verbose=True)

    print("=== Monte Carlo ===")
    batches = {}
    for name in ("optimal", "mm_only", "naive_arb", "hold"):
        batches[name] = simulate(sol, n_paths=n_paths, seed=11, policy=name)
        print(name, summarize(batches[name]))

    print("=== Writing figures ===")
    paths = make_all_figures(sol, batches, figdir)
    for p in paths:
        print(" wrote", p)

    metrics = {
        "profile": profile,
        "params": {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in params.__dict__.items()},
        "V0": value_at_origin(sol),
        "quoting_identity": quoting_identity_error(sol, 0),
        "smooth_pasting": smooth_pasting_error(sol, 0, 0),
        "parity_residual": asymmetry_residual(sol, 0),
        "qvi_gap_mean": float(np.mean(sol.residual)),
        "policies": {k: summarize(v) for k, v in batches.items()},
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("metrics ->", outdir / "metrics.json")

    lines = [
        "# Experiment metrics",
        "",
        f"Profile `{args.profile}`. Value at the origin V(0,0,0,0) = **{metrics['V0']:.6f}**.",
        "",
        "| policy | mean wealth | std | 5% | 95% | creates | redeems |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in metrics["policies"].items():
        lines.append(
            f"| {name} | {s['mean_wealth']:.4f} | {s['std_wealth']:.4f} | "
            f"{s['p05']:.4f} | {s['p95']:.4f} | {s['mean_creates']:.3f} | {s['mean_redeems']:.3f} |"
        )
    lines += [
        "",
        f"Quoting identity MAE (bid/ask): {metrics['quoting_identity']['bid_mae']:.2e} / "
        f"{metrics['quoting_identity']['ask_mae']:.2e}.",
        f"Parity residual: {metrics['parity_residual']:.3e}.",
        f"Smooth-pasting mean |ΔV_x| at interfaces: {metrics['smooth_pasting']['mean_abs_jump_Vx']}.",
        "",
    ]
    (outdir / "METRICS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
