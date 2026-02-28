#!/usr/bin/env python3
"""
Plot the posterior power-law lifetime relation tau(M) using samples from the
radial completeness + lifetime PyMC fit.

Model:
    tau(M) = tau0_Myr * (M / M0)^alpha_tau

We:
  - load the ArviZ netcdf trace,
  - draw samples of (tau0_Myr, alpha_tau) from the posterior,
  - evaluate tau(M) on a mass grid for each draw,
  - plot:
      * thin grey lines = individual posterior realisations,
      * orange line      = posterior median,
      * orange band      = 16–84% credible interval.

Usage example:
  python plot_tau_vs_mass_posterior.py \
      --trace herbig_simple_radial_fit.nc \
      --Mmin 1.0 --Mmax 12.0 \
      --nM 200 \
      --n_draws 300 \
      --out tau_vs_mass_posterior.png
"""

import argparse
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from mpl_setup import *  # your usual plotting style


def flatten_draws(idata, name):
    """Return posterior draws for variable `name` as a 1D array (chain*draw)."""
    if "posterior" not in idata.groups() or name not in idata.posterior:
        return None
    arr = idata.posterior[name].values  # (chain, draw[, ...])
    return arr.reshape(-1)


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--trace", required=True,
                    help="ArviZ netcdf file from the radial fit "
                         "(e.g. herbig_simple_radial_fit.nc)")
    ap.add_argument("--tau0_name", default="tau0_Myr",
                    help="Name of tau0 variable in the trace "
                         "(if missing, will try log_tau0 and exponentiate)")
    ap.add_argument("--alpha_name", default="alpha_tau",
                    help="Name of alpha_tau variable in the trace")
    ap.add_argument("--M0", type=float, default=1.0,
                    help="Reference mass M0 in tau(M) = tau0 (M/M0)^alpha")
    ap.add_argument("--Mmin", type=float, default=1.0,
                    help="Minimum mass [Msun] for the plot")
    ap.add_argument("--Mmax", type=float, default=12.0,
                    help="Maximum mass [Msun] for the plot")
    ap.add_argument("--nM", type=int, default=200,
                    help="Number of mass grid points")
    ap.add_argument("--n_draws", type=int, default=300,
                    help="Number of posterior draws to plot")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for subsampling posterior draws")
    ap.add_argument("--out", default="tau_vs_mass_posterior.png",
                    help="Output figure filename")
    args = ap.parse_args()

    # --- Load posterior ---
    idata = az.from_netcdf(args.trace)

    tau0_draws = flatten_draws(idata, args.tau0_name)
    if tau0_draws is None:
        # fall back: assume we have log_tau0 and define tau0_Myr = exp(log_tau0)
        log_tau0 = flatten_draws(idata, "log_tau0")
        if log_tau0 is None:
            raise SystemExit(
                f"Could not find '{args.tau0_name}' or 'log_tau0' in posterior."
            )
        tau0_draws = np.exp(log_tau0)

    alpha_draws = flatten_draws(idata, args.alpha_name)
    if alpha_draws is None:
        raise SystemExit(f"Could not find '{args.alpha_name}' in posterior.")

    # Make sure the two arrays are the same length
    n_total = min(tau0_draws.size, alpha_draws.size)
    tau0_draws = tau0_draws[:n_total]
    alpha_draws = alpha_draws[:n_total]

    # Subsample draws for plotting
    rng = np.random.default_rng(args.seed)
    n_plot = min(args.n_draws, n_total)
    idx = rng.choice(n_total, size=n_plot, replace=False)
    tau0_plot = tau0_draws[idx]
    alpha_plot = alpha_draws[idx]

    # --- Mass grid ---
    M_grid = np.logspace(np.log10(args.Mmin), np.log10(args.Mmax), args.nM)
    M0 = float(args.M0)

    # --- Evaluate tau(M) for each draw ---
    tau_samples = np.empty((n_plot, args.nM), dtype=float)
    for k in range(n_plot):
        tau_samples[k, :] = tau0_plot[k] * (M_grid / M0) ** alpha_plot[k]

    # Summary curves
    tau_med = np.median(tau_samples, axis=0)
    tau_p16 = np.percentile(tau_samples, 16, axis=0)
    tau_p84 = np.percentile(tau_samples, 84, axis=0)

    tau0_med = float(np.median(tau0_draws))
    alpha_med = float(np.median(alpha_draws))

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    # individual posterior realisations
    for k in range(n_plot):
        ax.loglog(
            M_grid,
            tau_samples[k, :],
            color="0.7",
            alpha=0.15,
            linewidth=0.7,
            zorder=1,
        )

    # 68% credible band
    ax.fill_between(
        M_grid,
        tau_p16,
        tau_p84,
        color="tab:orange",
        alpha=0.4,
        label="$16$–$84\\%$ credible interval",
        zorder=2,
    )

    # median curve
    ax.loglog(
        M_grid,
        tau_med,
        color="tab:orange",
        linewidth=2.0,
        label="Posterior median",
        zorder=3,
    )

    # Reference 3 Myr line (optional but nice visual cue)
    ax.axhline(
        3.0,
        color="k",
        linestyle=":",
        linewidth=1.0,
        label="3 Myr",
        zorder=0,
    )

    ax.set_xlabel(r"Stellar mass $M_\star\ [{\rm M}_\odot]$")
    ax.set_ylabel(r"Herbig lifetime $\tau(M_\star)\ [{\rm Myr}]$")
    ax.set_xlim(args.Mmin, args.Mmax)

    # Y-limits: auto from data, but keep within a sensible dynamic range
    ymin = np.nanmin(tau_p16)
    ymax = np.nanmax(tau_p84)
    ax.set_ylim(0.5 * ymin, 2.0 * ymax)

    #ax.grid(True, which="both", alpha=0.3)

    # Annotate the median relation
    text = (
        rf"$\tau(M_\star) = {tau0_med:.1f}\,\mathrm{{Myr}}"
        rf"\,\left(\frac{{M_\star}}{{{M0:.1f}\,M_\odot}}\right)^{{{alpha_med:.2f}}}$"
    )
    ax.text(
        0.05, 0.95,
        text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5),
    )

    ax.legend(loc="lower left", fontsize=8, frameon=True)

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"[OK] saved {args.out}")


if __name__ == "__main__":
    main()
