#!/usr/bin/env python3
"""
Plot completeness p_det(R, M) as a 2D map from the simplified radial+mass model:

    logit p_det(R,M) = a0 + a_R * ln(R/R0) + a_M * ln(M/M0)

Inputs:
  --trace    : ArviZ netcdf file with posterior for a0, a_R, a_M
               (e.g. herbig_simple_radial_fit.nc)
  --catalog  : TSV/CSV (e.g. herbig_data.tsv) with RA, Dec, distance, and Mstar

The catalog stars are overplotted as points in the (R, M) plane, where
  R = sqrt(x^2 + y^2) from Sun-centred Galactic Cartesian coordinates.

Example
-------
  python plot_completeness_R_M.py \
      --trace herbig_simple_radial_fit.nc \
      --catalog herbig_data.tsv \
      --col_ra ra --col_dec dec --col_dist distance --col_M Mstar \
      --Rmin 50 --Rmax 1200 --nR 300 \
      --Mmin 1.5 --Mmax 8.0 --nM 250 \
      --R0 500 --M0 1.0 \
      --out completeness_R_M.png
"""

import argparse
import os
import sys
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from mpl_setup import *  # your usual plotting style

from astropy.coordinates import SkyCoord
import astropy.units as u

try:
    import pandas as pd
except Exception:
    pd = None


# ---------- small helpers ----------

def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def flatten_draws(idata, name):
    """
    Extract posterior draws for parameter `name` as a 1D array (chain*draw).
    Returns None if not present.
    """
    if "posterior" not in idata.groups() or name not in idata.posterior:
        return None
    arr = idata.posterior[name].values   # (chain, draw[, ...])
    return arr.reshape(-1)


def load_catalog(path, sep=None):
    """
    Load TSV/CSV into a dict of numpy arrays. Uses pandas if available,
    otherwise falls back to numpy.genfromtxt.
    """
    if pd is not None:
        if sep is None:
            sep = "\t" if str(path).lower().endswith(".tsv") else ","
        df = pd.read_csv(path, sep=sep)
        return {c: df[c].to_numpy() for c in df.columns}

    data = np.genfromtxt(
        path, names=True, delimiter=sep if sep else ",",
        dtype=None, encoding=None
    )
    return {c: np.asarray(data[c]) for c in data.dtype.names}


def get_column(cat_dict, name):
    """
    Fetch column `name` from catalog dict (case-insensitive fallback).
    Returns None if not found.
    """
    if not name:
        return None
    if name in cat_dict:
        return np.asarray(cat_dict[name], dtype=float)
    lname = name.lower()
    for k in cat_dict:
        if k.lower() == lname:
            return np.asarray(cat_dict[k], dtype=float)
    return None


def icrs_to_gal_lbd(ra_deg, dec_deg):
    """ICRS (deg) -> Galactic (l,b) deg via astropy."""
    sky = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    gal = sky.galactic
    return gal.l.deg, gal.b.deg


def stars_to_xy(ra_deg, dec_deg, d_pc):
    """
    Convert RA,Dec,Distance to Sun-centred Cartesian (x,y,z) [pc],
    in the same convention as your SFR map code.
    """
    l_deg, b_deg = icrs_to_gal_lbd(ra_deg, dec_deg)
    l = np.deg2rad(l_deg)
    b = np.deg2rad(b_deg)
    x = d_pc * np.cos(b) * np.cos(l)
    y = d_pc * np.cos(b) * np.sin(l)
    z = d_pc * np.sin(b)
    return x, y, z


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Posterior / model
    ap.add_argument("--trace", required=True,
                    help="ArviZ netcdf with posterior for a0, a_R, a_M "
                         "(e.g. herbig_simple_radial_fit.nc)")
    ap.add_argument("--a0_name",   default="a0",
                    help="Name of intercept parameter in trace")
    ap.add_argument("--aR_name",   default="a_R",
                    help="Name of radial slope parameter in trace")
    ap.add_argument("--aM_name",   default="a_M",
                    help="Name of mass slope parameter in trace")
    ap.add_argument("--R0", type=float, default=500.0,
                    help="Reference radius R0 [pc] in logit p_det (must match fit)")
    ap.add_argument("--M0", type=float, default=1.0,
                    help="Reference mass M0 [Msun] in logit p_det (must match fit)")
    ap.add_argument("--draws_max", type=int, default=2000,
                    help="(Unused in current version; kept for backward compatibility)")

    # Grids
    ap.add_argument("--Rmin", type=float, default=50.0,
                    help="Minimum radius [pc] for map")
    ap.add_argument("--Rmax", type=float, default=1250.0,
                    help="Maximum radius [pc] for map")
    ap.add_argument("--nR",   type=int,   default=300,
                    help="Number of R grid points")
    ap.add_argument("--Mmin", type=float, default=1.0,
                    help="Minimum stellar mass [Msun] for map")
    ap.add_argument("--Mmax", type=float, default=10.0,
                    help="Maximum stellar mass [Msun] for map")
    ap.add_argument("--nM",   type=int,   default=250,
                    help="Number of mass grid points")

    # Catalog (herbig_data.tsv style)
    ap.add_argument("--catalog", required=True,
                    help="TSV/CSV with RA, Dec, distance, and mass (e.g. herbig_data.tsv)")
    ap.add_argument("--sep", default=None,
                    help="Column separator for catalog (auto if omitted)")
    ap.add_argument("--col_ra",   default="ra",
                    help="Right ascension column [deg]")
    ap.add_argument("--col_dec",  default="dec",
                    help="Declination column [deg]")
    ap.add_argument("--col_dist", default="distance",
                    help="Distance column [pc] (or distance modulus if --catalog_has_mu)")
    ap.add_argument("--catalog_has_mu", action="store_true",
                    help="Interpret --col_dist as distance modulus mu instead of pc")
    ap.add_argument("--col_M",    default="Mstar",
                    help="Mass column in catalog [Msun]")

    # Marker appearance
    ap.add_argument("--marker", default="o",
                    help="Marker style for stars")
    ap.add_argument("--ms", type=float, default=24.0,
                    help="Marker size in points^2")
    ap.add_argument("--mfc", default="w",
                    help="Marker face colour")
    ap.add_argument("--mec", default="k",
                    help="Marker edge colour")
    ap.add_argument("--mew", type=float, default=0.8,
                    help="Marker edge width")

    # Plot appearance
    ap.add_argument("--out", default="completeness_R_M.png",
                    help="Output figure name")
    ap.add_argument("--cmap", default="viridis",
                    help="Colormap for p_det")
    ap.add_argument("--levels", nargs="+", type=float, default=[0.1, 0.5, 0.9],
                    help="p_det contour levels")
    ap.add_argument("--figwidth", type=float, default=6.0,
                    help="Figure width [inches]")
    ap.add_argument("--figheight", type=float, default=4.5,
                    help="Figure height [inches]")

    args = ap.parse_args()

    # --- Load posterior and extract parameter medians ---
    idata = az.from_netcdf(args.trace)
    a0_draws = flatten_draws(idata, args.a0_name)
    aR_draws = flatten_draws(idata, args.aR_name)
    aM_draws = flatten_draws(idata, args.aM_name)

    miss = [nm for nm, arr in [
        (args.a0_name, a0_draws),
        (args.aR_name, aR_draws),
        (args.aM_name, aM_draws),
    ] if arr is None]
    if miss:
        raise SystemExit(f"Missing parameters in posterior: {miss}")

    a0_med = np.median(a0_draws)
    aR_med = np.median(aR_draws)
    aM_med = np.median(aM_draws)

    print(f"[info] Median parameters: a0={a0_med:.3f}, "
          f"a_R={aR_med:.3f}, a_M={aM_med:.3f}", file=sys.stderr)

    # --- Grids in R and M ---
    R_grid = np.linspace(args.Rmin, args.Rmax, args.nR)
    M_grid = np.linspace(args.Mmin, args.Mmax, args.nM)

    # 2D meshes: RR (x-axis = radius), MM (y-axis = mass)
    RR, MM = np.meshgrid(R_grid, M_grid, indexing="xy")  # shape (nM, nR)

    RR_safe = np.clip(RR, 1e-6, None)
    MM_safe = np.clip(MM, 1e-6, None)

    LOGR = np.log(RR_safe / args.R0)
    LOGM = np.log(MM_safe / args.M0)

    # --- Evaluate p_det(R,M) at median parameters ---
    eta_med = a0_med + aR_med * LOGR + aM_med * LOGM
    P_med = logistic(eta_med)  # shape (nM, nR)

    # --- Make plot ---
    fig, ax = plt.subplots(
        1, 1, figsize=(args.figwidth, args.figheight),
        constrained_layout=True
    )

    im = ax.imshow(
        P_med,
        origin="lower",
        extent=[args.Rmin, args.Rmax, args.Mmin, args.Mmax],
        aspect="auto",
        vmin=0.0, vmax=1.0,
        cmap=args.cmap,
        interpolation="nearest",
    )

    # Contours of completeness
    if args.levels:
        CS = ax.contour(
            R_grid, M_grid, P_med,
            levels=args.levels,
            colors="k", linewidths=1.0, alpha=0.9,
        )
        ax.clabel(
            CS,
            fmt=lambda v: f"{int(round(100*v))}%",
            fontsize=9,
            inline=True,
        )

    ax.set_xlabel(r"Radial coordinate $R$ [pc]")
    ax.set_ylabel(r"Stellar mass $M_\star\ [{\rm M}_\odot]$")
    ax.set_xlim(args.Rmin, args.Rmax)
    ax.set_ylim(args.Mmin, args.Mmax)

    # --- Overlay catalog stars (herbig_data.tsv) ---
    cat = load_catalog(args.catalog, sep=args.sep)
    ra_cat = get_column(cat, args.col_ra)
    dec_cat = get_column(cat, args.col_dec)
    d_cat = get_column(cat, args.col_dist)
    M_cat = get_column(cat, args.col_M)

    if ra_cat is None or dec_cat is None or d_cat is None or M_cat is None:
        raise SystemExit(
            "Catalog must contain RA, Dec, distance, and mass columns.\n"
            f"Got columns: {list(cat.keys())}"
        )

    # if distance is actually distance modulus, convert μ -> pc
    if args.catalog_has_mu:
        mu = d_cat
        d_cat = 10.0 * (10.0 ** (mu / 5.0))

    # finite mask
    mask = np.isfinite(ra_cat) & np.isfinite(dec_cat) & np.isfinite(d_cat) & np.isfinite(M_cat)
    ra_cat = ra_cat[mask]
    dec_cat = dec_cat[mask]
    d_cat = d_cat[mask]
    M_cat = M_cat[mask]

    # convert to cylindrical radius R
    x_star, y_star, z_star = stars_to_xy(ra_cat, dec_cat, d_cat)
    R_cat = np.sqrt(x_star**2 + y_star**2)

    ax.scatter(
        R_cat, M_cat,
        s=args.ms,
        marker=args.marker,
        facecolors=args.mfc,
        edgecolors=args.mec,
        linewidths=args.mew,
        zorder=5,
    )
    ax.text(
        0.02, 0.98, f"N={mask.sum()}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox=dict(
            facecolor="white",
            alpha=0.7,
            edgecolor="none",
            pad=1.5,
        ),
        zorder=6,
    )

    # Colourbar on the right
    cbar = fig.colorbar(im, ax=ax, location="right", fraction=0.05, pad=0.04)
    cbar.set_label(r"$f_{\rm det}(R,M_\star)$")

    plt.savefig(args.out, dpi=220, bbox_inches="tight")
    print(f"[OK] saved {os.path.abspath(args.out)}")

    # Uncomment for interactive use:
    # plt.show()


if __name__ == "__main__":
    main()
