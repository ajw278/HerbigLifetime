#!/usr/bin/env python3
"""
Completeness panels p_det(d, A_V) with overplotted catalog stars (with error bars),
binned by nearest mass panel (or by user-specified mass edges).

Panels: one per mass in --masses (default: 2, 3, 6 Msun) OR per interval in --mass_edges.

Model:
  logit p_det = a0 + a_mu*(mu - mu_ref) + a_Av*(k_lambda*A_V) + a_logM*log10(M)

Inputs:
  --trace : ArviZ netcdf from your PPP run (must contain a0,a_mu,a_Av,a_logM)
  --catalog : TSV/CSV with columns for distance [pc], A_V [mag], mass [Msun] (and optional errors)

Outputs:
  --out : PNG/PDF multi-panel figure

Requires: numpy, matplotlib, arviz
Optional: pandas (for convenient catalog reading)
"""

import argparse, os, sys
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from mpl_setup import *

try:
    import pandas as pd
except Exception:
    pd = None

def flatten_draws(idata, name):
    if "posterior" not in idata.groups() or name not in idata.posterior:
        return None
    arr = idata.posterior[name].values  # (chain, draw[, ...])
    return arr.reshape(-1)

def logistic(x):
    return 1.0/(1.0 + np.exp(-x))

def load_catalog(path, sep=None):
    """
    Load TSV/CSV into dict of columns (numpy arrays).
    Falls back to numpy if pandas is unavailable.
    """
    if pd is not None:
        if sep is None:
            # guess by extension
            sep = "\t" if str(path).lower().endswith(".tsv") else ","
        df = pd.read_csv(path, sep=sep)
        return {c: df[c].to_numpy() for c in df.columns}
    # numpy fallback (expects header row)
    data = np.genfromtxt(path, names=True, delimiter=sep if sep else ",", dtype=None, encoding=None)
    return {c: np.asarray(data[c]) for c in data.dtype.names}

def assign_panels_by_centers(M_star, centers):
    """Return panel index for each star by nearest center mass."""
    centers = np.asarray(centers, float)
    # shape (N, K) of absolute deltas
    idx = np.argmin(np.abs(M_star[:, None] - centers[None, :]), axis=1)
    return idx

def assign_panels_by_edges(M_star, edges):
    """Return panel index via np.digitize against edges (len(edges)-1 panels)."""
    edges = np.asarray(edges, float)
    # bins: [edges[i], edges[i+1])
    idx = np.digitize(M_star, edges) - 1
    # clamp outside to nearest valid bin
    idx = np.clip(idx, 0, len(edges)-2)
    return idx

def load_pack_Av_stats(pack_path):
    P = np.load(pack_path, allow_pickle=True)
    # find A_V nodes
    av_keys = ["Av_nodes", "A_V_nodes", "AV_nodes", "av_nodes"]
    w_keys  = ["d_weights", "weights"]
    name_keys = ["Name", "name", "names"]

    Av_nodes = next((P[k] for k in av_keys if k in P), None)
    W_nodes  = next((P[k] for k in w_keys  if k in P), None)
    names    = next((P[k] for k in name_keys if k in P), None)

    if Av_nodes is None or W_nodes is None:
        raise ValueError(f"Could not find Av_nodes and weights in pack: keys={list(P.keys())}")

    Av_nodes = np.asarray(Av_nodes, float)   # (N, K)
    W_nodes  = np.asarray(W_nodes,  float)   # (N, K)

    # normalize weights per star
    W = np.clip(W_nodes, 1e-30, None)
    W = W / W.sum(axis=1, keepdims=True)

    Av_mean = (W * Av_nodes).sum(axis=1)
    Av_var  = (W * (Av_nodes - Av_mean[:, None])**2).sum(axis=1)
    Av_std  = np.sqrt(np.maximum(Av_var, 0.0))

    # optional name array -> as flat list of strings
    if names is not None:
        try:
            names = np.asarray(names).astype(str).tolist()
        except Exception:
            names = None

    return {"names": names, "Av_mean": Av_mean, "Av_std": Av_std}


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Posterior / model
    ap.add_argument("--trace", required=True, help="ArviZ netcdf (e.g., ppp_trace.nc)")
    ap.add_argument("--mu_ref", type=float, default=10.0, help="Reference distance modulus")
    ap.add_argument("--k_lambda", type=float, default=1.0, help="Band-to-V factor (A_band = k_lambda*A_V)")
    ap.add_argument("--draws_max", type=int, default=1000, help="Max posterior draws to use")

    # Grids for panels
    ap.add_argument("--dmin", type=float, default=20.0, help="Min distance [pc]")
    ap.add_argument("--dmax", type=float, default=1250.0, help="Max distance [pc]")
    ap.add_argument("--nd", type=int, default=300, help="# distance grid points")
    ap.add_argument("--Av_min", type=float, default=0.0, help="Min A_V [mag]")
    ap.add_argument("--Av_max", type=float, default=5.0, help="Max A_V [mag]")
    ap.add_argument("--nAv", type=int, default=250, help="# A_V grid points")

    # Mass panels (choose one of the two)
    ap.add_argument("--masses", type=float, nargs="+", default=[2.0, 3.0, 6.0],
                    help="Mass centers [Msun] for panels (nearest-center assignment)")
    ap.add_argument("--mass_edges", type=float, nargs="+", default=None,
                    help="Optional mass edges [Msun] defining bins; overrides --masses")

    # Catalog + columns
    ap.add_argument("--catalog", required=True, help="Path to TSV/CSV catalog")
    ap.add_argument("--sep", default=None, help="Column separator (auto by extension if omitted)")
    ap.add_argument("--col_dist", default="distance", help="Distance column [pc]")
    ap.add_argument("--col_Av",   default="A_V",      help="Extinction column [mag]")
    ap.add_argument("--col_edist", default="", help="Optional distance error column [pc]")
    ap.add_argument("--col_eAv",   default="", help="Optional A_V error column [mag]")
    ap.add_argument("--col_emass", default="", help="Optional mass error column [Msun]")
    ap.add_argument("--catalog_has_mu", action="store_true",
                    help="Interpret --col_dist as distance modulus mu and convert to pc")

    # Plotting
    ap.add_argument("--out", default="completeness_with_data.png", help="Output figure")
    ap.add_argument("--cmap", default="viridis", help="Colormap for p_det")
    ap.add_argument("--levels", type=float, nargs="+", default=[0.1, 0.5, 0.9], help="Contour levels")
    ap.add_argument("--figwidth_per_panel", type=float, default=4.0, help="Width per panel [inches]")
    ap.add_argument("--figheight", type=float, default=4.0, help="Figure height [inches]")
    ap.add_argument("--xlog", action="store_true", help="Log scale for distance axis")
    ap.add_argument("--marker", default="o", help="Marker for data points")
    ap.add_argument("--ms", type=float, default=28, help="Marker size (points^2)")
    ap.add_argument("--mec", default="k", help="Marker edge color")
    ap.add_argument("--mew", type=float, default=0.85, help="Marker edge width")
    ap.add_argument("--mfc", default="w", help="Marker face color")
    ap.add_argument("--ealpha", type=float, default=0.9, help="Errorbar alpha")
    ap.add_argument("--elinew", type=float, default=1.0, help="Errorbar line width")
    ap.add_argument("--cap", type=float, default=0.0, help="Errorbar cap size (points)")
    ap.add_argument("--col_mass", default="Mstar", help="Mass column [Msun] (e.g. Mstar)")
    ap.add_argument("--star_marg_pack", default="", help="NPZ with star-node marginals")
    ap.add_argument("--use_pack_Av", action="store_true",
                    help="Use A_V mean/std from star_marg_pack instead of catalog columns")

    args = ap.parse_args()

    # --- Load posterior + coefficients ---
    idata = az.from_netcdf(args.trace)
    need = ["a0", "a_mu", "a_Av", "a_logM"]
    coeffs = {nm: flatten_draws(idata, nm) for nm in need}
    miss = [nm for nm, v in coeffs.items() if v is None]
    if miss:
        raise SystemExit(f"Missing coefficients in posterior: {miss}")

    # Thin draws uniformly
    n_draws = min(args.draws_max, *[v.size for v in coeffs.values()])
    sel = np.linspace(0, coeffs["a0"].size - 1, n_draws).astype(int)
    for k in coeffs:
        coeffs[k] = coeffs[k][sel]

    # --- Grids ---
    d_grid  = np.linspace(args.dmin, args.dmax, args.nd)
    mu_grid = 5.0 * np.log10(np.clip(d_grid, 1e-9, None) / 10.0)
    Av_grid = np.linspace(args.Av_min, args.Av_max, args.nAv)

    MU = mu_grid[None, :, None]  # (1, nd, 1)
    AV = Av_grid[None, None, :]  # (1, 1, nAv)

    a0     = coeffs["a0"][:, None, None]
    a_mu   = coeffs["a_mu"][:, None, None]
    a_Av   = coeffs["a_Av"][:, None, None]
    a_logM = coeffs["a_logM"][:, None, None]

    # --- Panels: centers or edges ---
    if args.mass_edges is not None:
        edges = np.asarray(args.mass_edges, float)
        if edges.size < 2:
            raise SystemExit("--mass_edges must have at least two values")
        mass_panels = 0.5*(edges[:-1] + edges[1:])
        use_edges = True
    else:
        mass_panels = np.asarray(args.masses, float)
        use_edges = False

    # --- Prepare figure ---
    n_pan = mass_panels.size
    fig, axs = plt.subplots(1, n_pan, figsize=(args.figwidth_per_panel*n_pan, args.figheight),
                            sharey=True, constrained_layout=True)
    if n_pan == 1:
        axs = [axs]

    # --- Compute median p_det(d, A_V | M) per panel ---
    vmin, vmax = 0.0, 1.0
    last_im = None
    for j, (ax, M) in enumerate(zip(axs, mass_panels)):
        logM = np.log10(float(M))
        LOGM = a_logM * logM
        eta  = a0 + a_mu*(MU - args.mu_ref) + a_Av*(args.k_lambda*AV) + LOGM   # (draw, nd, nAv)
        P    = logistic(eta)
        P50  = np.percentile(P, 50, axis=0)                                     # (nd, nAv)

        last_im = ax.imshow(
            P50.T, origin="lower",
            extent=[args.dmin, args.dmax, args.Av_min, args.Av_max],
            aspect="auto", vmin=vmin, vmax=vmax, cmap=args.cmap, interpolation="nearest"
        )
        CS = ax.contour(d_grid, Av_grid, P50.T, levels=args.levels,
                        colors="k", linewidths=1.0, alpha=0.9)
        ax.clabel(CS, fmt=lambda v: f"{int(round(100*v))}%", fontsize=10, inline=True)

        ax.set_title(fr"$M={M:.2g}\,M_\odot$")
        ax.set_xlabel("Distance $d$ [pc]")
        if args.xlog:
            ax.set_xscale("log")
        if j == 0:
            ax.set_ylabel("$A_V$ [mag]")
        ax.set_xlim(args.dmin, args.dmax)
        ax.set_ylim(args.Av_min, args.Av_max)

    # --- Load catalog and overlay points ---
    cat = load_catalog(args.catalog, sep=args.sep)

    def get_col(name):
        if not name:
            return None
        if name in cat:
            return np.asarray(cat[name], dtype=float)
        # case-insensitive fallback
        for k in cat:
            if k.lower() == name.lower():
                return np.asarray(cat[k], dtype=float)
        return None

    d  = get_col(args.col_dist)
    Av = get_col(args.col_Av) if hasattr(args, "col_Av") else None  # may be missing
    M  = get_col(args.col_mass)

    if d is None or M is None:
        raise SystemExit(
            "Catalog must contain distance and mass columns "
            "(and A_V unless --use_pack_Av). "
            f"Got keys: {list(cat.keys())}"
        )

    if args.catalog_has_mu:
        # convert mu -> distance [pc]
        d = 10.0**(0.2*d + 1.0)

    ed = get_col(args.col_edist) if args.col_edist else None
    eA = get_col(args.col_eAv)   if args.col_eAv   else None
    eM = get_col(args.col_emass) if args.col_emass else None


    # If using pack A_V, compute and override Av/eA
    if args.use_pack_Av:
        if not args.star_marg_pack:
            raise SystemExit("--use_pack_Av set but --star_marg_pack not provided.")
        pack = load_pack_Av_stats(args.star_marg_pack)
        Av_mean, Av_std = pack["Av_mean"], pack["Av_std"]
        if pack["names"] is not None and "Name" in cat:
            # map by Name
            idx_by_name = {nm: i for i, nm in enumerate(pack["names"])}
            Av_new  = np.full_like(d, np.nan, dtype=float)
            eAv_new = np.full_like(d, np.nan, dtype=float)
            for i, nm in enumerate(np.asarray(cat["Name"]).astype(str)):
                j = idx_by_name.get(nm, None)
                if j is not None:
                    Av_new[i]  = Av_mean[j]
                    eAv_new[i] = Av_std[j]
            # keep existing Av where pack missing, else replace
            if Av is None:
                Av = Av_new
            else:
                Av = np.where(np.isfinite(Av_new), Av_new, Av)
            eA = eAv_new if eA is None else np.where(np.isfinite(eAv_new), eAv_new, eA)
        else:
            # assume same order and length
            if Av is None:
                Av = Av_mean
            else:
                Av = np.where(np.isfinite(Av), Av, Av_mean)
            eA = Av_std if eA is None else np.where(np.isfinite(eA), eA, Av_std)


    # sanitize finite
    print(d, Av, M)
    mask = np.isfinite(d) & np.isfinite(Av) & np.isfinite(M)
    if ed is not None: mask &= np.isfinite(ed)
    if eA is not None: mask &= np.isfinite(eA)
    if eM is not None: mask &= np.isfinite(eM)
    d, Av, M = d[mask], Av[mask], M[mask]
    if ed is not None: ed = ed[mask]
    if eA is not None: eA = eA[mask]
    if eM is not None: eM = eM[mask]

    # assign each star to a panel
    if use_edges:
        idx = assign_panels_by_edges(M, edges)
    else:
        idx = assign_panels_by_centers(M, mass_panels)

    # overlay per panel
    for j, ax in enumerate(axs):
        sel = (idx == j)
        if not np.any(sel):
            continue
        dd, AA = d[sel], Av[sel]
        # error bars (optional)
        xerr = ed[sel] if ed is not None else None
        yerr = eA[sel] if eA is not None else None

        ax.errorbar(
            dd, AA, xerr=xerr, yerr=yerr,
            fmt=args.marker, ms=np.sqrt(args.ms),  # ms in points^2 -> convert
            mfc=args.mfc, mec=args.mec, mew=args.mew,
            ecolor=args.mec, elinewidth=args.elinew, alpha=args.ealpha, capsize=args.cap,
            zorder=9, linestyle="none"
        )

        # small count label in corner
        ax.text(0.02, 0.98, f"N={sel.sum()}", transform=ax.transAxes,
                ha="left", va="top", fontsize=10,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5), zorder=10)

    # shared colorbar
    cbar = fig.colorbar(last_im, ax=axs, fraction=0.03, pad=0.02)
    cbar.set_label("$p_\\mathrm{det}$")

    plt.savefig(args.out, dpi=220)
    print(f"[OK] saved {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
