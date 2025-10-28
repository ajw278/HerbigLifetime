#!/usr/bin/env python3
# build_ppp_all_packs.py
"""
Build all PPP precompute packs in one shot:

- grid_pack   : HEALPix×distance cells (x,y,z,d, μ, ΔV, Σ_SFR, Σ_birth, footprint, optional A_V)
- mass_pack   : Chabrier(2003) IMF on [0.08,120] Msun, mean mass, and Herbig-range quadrature
- star_pack   : per-star (l,b,d,x,y,z, μ, Σ_SFR, Σ_birth)
- star_marg_pack: per-star marginalization nodes/weights in distance and mass,
                  with per-distance-node A_V(d) and Σ_birth(x,y).

Inputs:
  --sfr_npz : NPZ with x_grid, y_grid and either Sigma_SFR_smoothed or Sigma_SFR
  --tsv     : Herbig TSV (cols: ra, dec, distance, optional: min_dist, max_dist, Mstar)

Optional dust:
  Adds A_V to grid and per-star distance nodes via Edenhofer 2023 if available.

Dependencies:
  numpy pandas astropy scipy healpy dustmaps (for A_V)
"""

import argparse
import json
import os
import sys
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm
from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp

# ---- Optional dust map (Edenhofer 2023) ----
_HAVE_DUST = True
try:
    from dustmaps.config import config as dustmaps_config
    from dustmaps.edenhofer2023 import Edenhofer2023Query
except Exception as e:
    _HAVE_DUST = False
    DUST_ERR = str(e)

# ---------- IMF: Chabrier (2003, single-star) ----------
def chabrier2003_unnorm_pdf(M):
    """Unnormalized dN/dM (linear mass)."""
    M = np.asarray(M, float)
    out = np.zeros_like(M)
    ok = M > 0
    if not np.any(ok):
        return out
    

    lo = ok & (M <= 1.0)
    hi = ok & (M > 1.0)
    # lognormal (base-10) below 1 Msun
    m_c, sigma, A = 0.079, 0.69, 0.158

    if np.any(lo):
        log10M = np.log10(M[lo])
        out[lo] = (A / (M[lo] * np.log(10.0))) * np.exp(-0.5 * ((log10M - np.log10(m_c))/sigma)**2)
    # power-law above 1 Msun
    m_c, sigma, A = 0.22, 0.57, 0.086
    if np.any(hi):
        log10M = np.log10(M[hi])
        out[hi] = (A / (M[hi] * np.log(10.0))) * np.exp(-0.5 * ((log10M - np.log10(m_c))/sigma)**2)
    return out

def normalize_imf_and_mean(Mmin=0.08, Mmax=120.0, n=20000):
    x = np.logspace(np.log10(Mmin), np.log10(Mmax), n)
    y = chabrier2003_unnorm_pdf(x)
    Z = np.trapz(y, x)
    mean_mass = float(np.trapz(x*y, x) / Z)
    def xi_norm(M):
        return chabrier2003_unnorm_pdf(M) / Z
    return xi_norm, mean_mass

def gauss_legendre_nodes_weights(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    nodes = 0.5 * ((b - a) * x + (b + a))
    weights = 0.5 * (b - a) * w
    return nodes.astype(float), weights.astype(float)

# ---------- helpers ----------
def distance_modulus(d_pc):
    d_pc = np.asarray(d_pc, float)
    return 5.0 * np.log10(np.maximum(d_pc, 1.0) / 10.0)

def sfr_interpolator(x_grid, y_grid, Sigma):
    # arrays are (Ny,Nx) with indexing='xy' (imshow convention)
    return RegularGridInterpolator((y_grid, x_grid), Sigma, bounds_error=False, fill_value=np.nan)

def healpix_lb(nside):
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))  # theta=colat, phi=lon
    l_deg = np.degrees(phi)             # [0,360)
    b_deg = 90.0 - np.degrees(theta)    # [-90,90]
    return l_deg, b_deg

def query_av_integrated(l_rad, b_rad, d_pc, chunk=200_000):
    """
    A_V integrated to distance (mean sample), using Edenhofer2023Query(integrated=True).
    l_rad, b_rad, d_pc are 1D arrays of equal length.
    Returns A_V (float array) with NaNs replaced by 0 outside map support.
    """
    if not _HAVE_DUST:
        raise RuntimeError(f"dustmaps not available: {DUST_ERR}")
    q = Edenhofer2023Query(integrated=True, load_samples=False)
    N = d_pc.size
    out = np.zeros(N, float)
    i = 0
    while i < N:
        j = min(i + chunk, N)
        coords = SkyCoord(l_rad[i:j]*u.rad, b_rad[i:j]*u.rad, d_pc[i:j]*u.pc, frame="galactic")
        E_int = q.query(coords, mode="mean")
        Av = 2.8 * np.where(np.isfinite(E_int), E_int, 0.0)
        out[i:j] = Av
        i = j
    return out

def truncated_normal_nodes_weights(mu, sigma, a=None, b=None, K=7):
    if (not np.isfinite(sigma)) or (sigma <= 0):
        return np.array([mu], float), np.array([1.0], float)
    Fa = norm.cdf(((a - mu)/sigma)) if a is not None else 0.0
    Fb = norm.cdf(((b - mu)/sigma)) if b is not None else 1.0
    if Fb <= Fa + 1e-12:
        return np.array([mu], float), np.array([1.0], float)
    edges = np.linspace(Fa, Fb, K+1)
    mids  = 0.5*(edges[:-1] + edges[1:])
    nodes = mu + sigma * norm.ppf(mids)
    w     = np.diff(edges)
    w    /= np.sum(w)
    return nodes.astype(float), w.astype(float)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--sfr_npz", required=True, help="NPZ with x_grid, y_grid, and Sigma_SFR_smoothed or Sigma_SFR")
    ap.add_argument("--tsv", required=True, help="Herbig TSV (ra, dec, distance[,min_dist,max_dist,Mstar])")
    ap.add_argument("--out_prefix", default="ppp", help="Output prefix for pack files")
    # Grid (sky × distance)
    ap.add_argument("--nside", type=int, default=32, help="HEALPix NSIDE")
    ap.add_argument("--dmin", type=float, default=20.0, help="Min distance (pc)")
    ap.add_argument("--dmax", type=float, default=1250.0, help="Max distance (pc)")
    ap.add_argument("--dr",   type=float, default=25.0, help="Shell width (pc)")
    # IMF & mass ranges
    ap.add_argument("--Mmin_all", type=float, default=0.08)
    ap.add_argument("--Mmax_all", type=float, default=120.0)
    ap.add_argument("--Mmin_herbig", type=float, default=1.5)
    ap.add_argument("--Mmax_herbig", type=float, default=8.0)
    ap.add_argument("--Mq", type=int, default=48, help="Gauss–Legendre nodes in Herbig mass integral")
    # Marginalization nodes
    ap.add_argument("--k_d", type=int, default=7, help="# distance nodes per star")
    ap.add_argument("--k_m", type=int, default=5, help="# mass nodes per star")
    ap.add_argument("--sigma_logM", type=float, default=0.20, help="dex scatter in log10(M) if no errors")
    ap.add_argument("--d_sigma_frac", type=float, default=0.10, help="fallback fractional distance σ")
    # Dust/A_V
    ap.add_argument("--add_av_grid", action="store_true", help="Also build A_V grid for (sky×distance) cells")
    ap.add_argument("--dust_dir", default=None, help="dustmaps data dir (default /dustmaps_data)")
    ap.add_argument("--fetch_dust", action="store_true", help="Fetch Edenhofer 2023 map if missing")
    # Combined file (optional)
    ap.add_argument("--combined_npz", default="", help="If set, also write a single combined NPZ here")
    ap.add_argument("--mass_err_col", default="e_Mstar", help="Column name for per-star mass errors")
    ap.add_argument("--mass_err_kind", choices=["abs","frac","dex"], default="abs",
                    help="Interpretation of mass errors: 'abs' (Msun), 'frac' (σ_M / M), or 'dex' (σ in log10 M).")
    ap.add_argument("--min_sigma_logM", type=float, default=0.05, help="Floor for σ_log10M (dex).")
    ap.add_argument("--max_sigma_logM", type=float, default=0.60, help="Cap for σ_log10M (dex).")
    args = ap.parse_args()

    # Load SFR grid
    D = np.load(args.sfr_npz, allow_pickle=True)
    xg, yg = D["x_grid"], D["y_grid"]
    if "Sigma_SFR_smoothed" in D.files:
        S = D["Sigma_SFR_smoothed"]
    else:
        S = D["Sigma_SFR"]
    Ny, Nx = S.shape
    if Nx != xg.size or Ny != yg.size:
        raise SystemExit("Σ_SFR grid shape mismatch with x_grid/y_grid.")
    S_interp = sfr_interpolator(xg, yg, S)

    # IMF: normalized PDF on [Mmin_all, Mmax_all] + mean mass
    xi_norm, Mbar = normalize_imf_and_mean(args.Mmin_all, args.Mmax_all)
    # Herbig quadrature
    Mq, Wq = gauss_legendre_nodes_weights(args.Mmin_herbig, args.Mmax_herbig, args.Mq)
    xi_q_unnorm = xi_norm(Mq)
    Z_herbig = float(np.sum(Wq * xi_q_unnorm))
    xi_q = (xi_q_unnorm / Z_herbig).astype(np.float64)

    # Sky grid
    l_deg, b_deg = healpix_lb(args.nside)
    npix = l_deg.size
    l_rad = np.radians(l_deg)[:, None]  # (npix,1)
    b_rad = np.radians(b_deg)[:, None]

    # Distance shells
    # Distance shells
    edges   = np.arange(args.dmin, args.dmax + args.dr, args.dr)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dd      = np.diff(edges)                  # shape (Nshell,)
    d_shell = centers[None, :]                # shape (1, Nshell)

    # Geometry broadcast (unchanged)
    cosb, sinb = np.cos(b_rad), np.sin(b_rad)
    cosl, sinl = np.cos(l_rad), np.sin(l_rad)
    X = d_shell * cosb * cosl                 # (npix, Nshell)
    Y = d_shell * cosb * sinl                 # (npix, Nshell)
    Z = d_shell * sinb                        # (npix, Nshell)

    # --- NEW: tile distance- and shell-only arrays across pixels ---
    d_full  = np.repeat(d_shell, npix, axis=0)        # (npix, Nshell)
    mu_full = distance_modulus(d_full)                # (npix, Nshell)

    # Volume element per cell: ΔV = d^2 Δd ΔΩ ; same for every pixel at a given shell,
    # so compute the shell value then repeat across pixels.
    delta_omega = 4.0 * np.pi / npix
    dV_shell = (d_shell**2) * dd[None, :] * delta_omega   # (1, Nshell)
    dV       = np.repeat(dV_shell, npix, axis=0)          # (npix, Nshell)

    # Evaluate Σ_SFR on grid
    pts = np.stack([Y.ravel(), X.ravel()], axis=1)
    S_vals = S_interp(pts).reshape(X.shape)  # Msun/yr/kpc^2
    footprint = np.isfinite(S_vals) & (S_vals > 0)

    # Σ_birth per pc^2
    Sigma_birth_pc2 = (S_vals / max(Mbar, 1e-9)) / 1e6  # stars/yr/pc^2


    # Optional A_V grid
    Av_grid = None
    if args.add_av_grid:
        if not _HAVE_DUST:
            print(f"[warn] dustmaps not available; skipping A_V grid: {DUST_ERR}", file=sys.stderr)
        else:
            if args.dust_dir:
                dustmaps_config["data_dir"] = args.dust_dir
            if args.fetch_dust:
                # fetch lazily only if needed
                try:
                    import dustmaps.edenhofer2023 as eden
                    eden.fetch()
                except Exception as e:
                    print(f"[warn] dust fetch failed: {e}", file=sys.stderr)
            print("[info] querying A_V grid...", file=sys.stderr)
            # Flatten (l,b,d) for query
            l_flat = np.repeat(l_rad, centers.size, axis=1).ravel()
            b_flat = np.repeat(b_rad, centers.size, axis=1).ravel()
            d_flat = np.repeat(d_shell,     npix,          axis=0).ravel()
            Av_flat = query_av_integrated(l_flat, b_flat, d_flat, chunk=200_000)
            Av_grid = Av_flat.reshape(X.shape)  # (npix, Nshell)

    # ---- Save grid_pack ----
    G = npix * centers.size
    print(G, npix, centers.size, file=sys.stderr)
    flat = lambda A: A.reshape(G)
    grid_pack = dict(
        nside=np.int32(args.nside),
        npix=np.int32(npix),
        dmin=args.dmin, dmax=args.dmax, dr=args.dr,
        ell=flat(np.repeat(l_rad, centers.size, axis=1)).astype(np.float32),
        bee=flat(np.repeat(b_rad, centers.size, axis=1)).astype(np.float32),
        x_pc=flat(X).astype(np.float32),
        y_pc=flat(Y).astype(np.float32),
        z_pc=flat(Z).astype(np.float32),
        d_pc=flat(d_full).astype(np.float32),           # was: np.repeat(d, npix, axis=0)
        mu=flat(mu_full).astype(np.float32),            # was: flat(mu)
        dV_pc3=flat(dV).astype(np.float32),             # was: flat(dV) with shape (1, Nshell)
        delta_d_pc=np.repeat(dd, npix).astype(np.float32),
        delta_omega_sr=np.full(G, delta_omega, dtype=np.float32),
        Sigma_SFR_Msun_yr_kpc2=flat(S_vals).astype(np.float32),
        Sigma_birth_yr_pc2=flat(Sigma_birth_pc2).astype(np.float32),
        footprint=flat(footprint).astype(np.bool_)
    )
    if Av_grid is not None:
        grid_pack["A_V"] = flat(Av_grid).astype(np.float32)

    np.savez_compressed(f"{args.out_prefix}_grid_pack.npz", **grid_pack)
    print(f"[OK] wrote {args.out_prefix}_grid_pack.npz", file=sys.stderr)

    # ---- Save mass_pack ----
    mass_pack = dict(
        Mmin_all=args.Mmin_all, Mmax_all=args.Mmax_all,
        Mmin_herbig=args.Mmin_herbig, Mmax_herbig=args.Mmax_herbig,
        Mq=Mq.astype(np.float32), Wq=Wq.astype(np.float32), xi_q=xi_q.astype(np.float32),
        mean_mass=Mbar
    )
    np.savez_compressed(f"{args.out_prefix}_mass_pack.npz", **mass_pack)
    print(f"[OK] mean IMF mass <M> = {Mbar:.3f} Msun -> wrote {args.out_prefix}_mass_pack.npz", file=sys.stderr)

    # ---- Per-star packs ----
    cat = pd.read_csv(args.tsv, sep="\t")
    if not {"ra","dec","distance"}.issubset(cat.columns):
        raise SystemExit("TSV must include: ra, dec, distance (pc)")
    ra = cat["ra"].to_numpy(float); dec = cat["dec"].to_numpy(float)
    d_obs = cat["distance"].to_numpy(float)
    d_min = cat["min_dist"].to_numpy(float) if "min_dist" in cat.columns else np.full_like(d_obs, np.nan)
    d_max = cat["max_dist"].to_numpy(float) if "max_dist" in cat.columns else np.full_like(d_obs, np.nan)
    M_obs = cat["Mstar"].to_numpy(float) if "Mstar" in cat.columns else np.full_like(d_obs, np.nan)
    # Mass-error column: use user-specified name or default to the last column in the file
    if args.mass_err_col and args.mass_err_col in cat.columns:
        M_err_raw = cat[args.mass_err_col].to_numpy(float)
    else:
        M_err_raw = cat.iloc[:, -1].to_numpy(float)  # last column

    # Helper to convert errors to σ_log10M (dex)
    LOG10_E = 1.0 / np.log(10.0)   # ≈ 0.4343

    def to_sigma_log10M(M_val, err_val):
        # Returns σ_log10M for one star given raw error value and interpretation.
        if not np.isfinite(err_val) or err_val <= 0:
            return np.nan
        if args.mass_err_kind == "dex":
            return float(err_val)
        if args.mass_err_kind == "frac":
            return float(LOG10_E * err_val)
        # "abs": absolute Msun
        if not np.isfinite(M_val) or M_val <= 0:
            return np.nan
        return float(LOG10_E * (err_val / M_val))

    gal = SkyCoord(ra*u.deg, dec*u.deg, frame="icrs").galactic
    l_s, b_s = gal.l.rad, gal.b.rad
    cosb_s, sinb_s = np.cos(b_s), np.sin(b_s)
    cosl_s, sinl_s = np.cos(l_s), np.sin(l_s)

    # star_pack (single point per star)
    x_star = d_obs * cosb_s * cosl_s
    y_star = d_obs * cosb_s * sinl_s
    z_star = d_obs * sinb_s
    S_star = S_interp(np.column_stack([y_star, x_star]))
    Sigma_birth_star_pc2 = (S_star / max(Mbar,1e-9)) / 1e6

    star_pack = dict(
        l_rad=l_s.astype(np.float32),
        b_rad=b_s.astype(np.float32),
        d_pc=d_obs.astype(np.float32),
        x_pc=x_star.astype(np.float32),
        y_pc=y_star.astype(np.float32),
        z_pc=z_star.astype(np.float32),
        mu=distance_modulus(d_obs).astype(np.float32),
        Mstar=np.asarray(M_obs, dtype=np.float32),
        Sigma_SFR_Msun_yr_kpc2=np.asarray(S_star, dtype=np.float32),
        Sigma_birth_yr_pc2=np.asarray(Sigma_birth_star_pc2, dtype=np.float32),
    )
    np.savez_compressed(f"{args.out_prefix}_star_pack.npz", **star_pack)
    print(f"[OK] wrote {args.out_prefix}_star_pack.npz", file=sys.stderr)

    # star_marg_pack (distance + mass nodes)
    N = d_obs.size; Kd, Km = args.k_d, args.k_m
    d_nodes = np.full((N, Kd), np.nan, float)
    d_weights = np.full((N, Kd), np.nan, float)
    mu_nodes = np.full((N, Kd), np.nan, float)
    x_nodes  = np.full((N, Kd), np.nan, float)
    y_nodes  = np.full((N, Kd), np.nan, float)
    z_nodes  = np.full((N, Kd), np.nan, float)
    Av_nodes = np.full((N, Kd), np.nan, float)
    birth_nodes = np.full((N, Kd), np.nan, float)  # stars/yr/pc^2

    M_nodes = np.full((N, Km), np.nan, float)
    M_weights = np.full((N, Km), np.nan, float)
    logM_nodes = np.full((N, Km), np.nan, float)
    # quick IMF evaluator (normalized on [Mmin_all, Mmax_all])
    def xi_of_M(M):
        return xi_norm(M)

    # Precompute A_V at distance nodes if dust available
    do_av_nodes = _HAVE_DUST
    if _HAVE_DUST and args.dust_dir:
        dustmaps_config["data_dir"] = args.dust_dir

    for i in range(N):
        print('[info] processing star {}/{}...'.format(i+1, N), file=sys.stderr)
        mu_d = d_obs[i]
        if np.isfinite(d_min[i]) and np.isfinite(d_max[i]) and (d_max[i] > d_min[i]):
            sigma_d = 0.25 * (d_max[i] - d_min[i])
            a,b = d_min[i], d_max[i]
        else:
            sigma_d = args.d_sigma_frac * max(mu_d, 1.0)
            a,b = None, None

        dn, dw = truncated_normal_nodes_weights(mu_d, sigma_d, a, b, K=Kd)
        ddmu = distance_modulus(dn)
        cb, sb = cosb_s[i], sinb_s[i]
        cl, sl = cosl_s[i], sinl_s[i]
        x = dn * cb * cl
        y = dn * cb * sl
        z = dn * sb

        d_nodes[i, :dn.size] = dn
        d_weights[i, :dw.size] = dw
        mu_nodes[i, :dn.size] = ddmu
        x_nodes[i, :dn.size] = x
        y_nodes[i, :dn.size] = y
        z_nodes[i, :dn.size] = z

        # Σ_birth at nodes
        S_here = S_interp(np.column_stack([y, x]))
        birth_nodes[i, :dn.size] = (S_here / max(Mbar, 1e-9)) / 1e6

        # A_V at nodes
        if do_av_nodes:
            try:
                Av_nodes[i, :dn.size] = query_av_integrated(
                    np.full_like(dn, l_s[i]), np.full_like(dn, b_s[i]), dn
                )
            except Exception as e:
                if i == 0:
                    print(f"[warn] A_V(node) query failed, filling zeros: {e}", file=sys.stderr)
                Av_nodes[i, :dn.size] = 0.0

        # Mass nodes: Normal in log10 M, truncated to [0.08,120]
        Mi = M_obs[i] if np.isfinite(M_obs[i]) and (M_obs[i] > 0) else 2.0
        mu_logM = np.log10(Mi)
        # Convert the provided mass error to σ_log10M (dex)
        sigma_logM = to_sigma_log10M(Mi, M_err_raw[i])

        # Fallback if conversion fails
        if (not np.isfinite(sigma_logM)) or (sigma_logM <= 0):
            sigma_logM = 0.20  # sensible default if the row has no usable error

        # Enforce floor/cap for stability
        sigma_logM = float(np.clip(sigma_logM, args.min_sigma_logM, args.max_sigma_logM))

        # Truncate to global mass limits
        aL, bL = np.log10(args.Mmin_all), np.log10(args.Mmax_all)

        # Quantile nodes for truncated Normal in log10 M
        cdf_a = norm.cdf((aL - mu_logM) / sigma_logM)
        cdf_b = norm.cdf((bL - mu_logM) / sigma_logM)
        if cdf_b <= cdf_a + 1e-12:
            # Degenerate case: collapse to a single node at the closest bound
            logMn = np.array([np.clip(mu_logM, aL, bL)], float)
            wM    = np.array([1.0], float)
        else:
            edges = np.linspace(cdf_a, cdf_b, Km + 1)
            mids  = 0.5 * (edges[:-1] + edges[1:])
            logMn = mu_logM + sigma_logM * norm.ppf(mids)
            wM    = np.diff(edges)
            wM   /= np.sum(wM)

        Mn = np.clip(10.0 ** logMn, args.Mmin_all, args.Mmax_all)

        M_nodes[i, :Mn.size]    = Mn
        logM_nodes[i, :Mn.size] = logMn
        M_weights[i, :wM.size]  = wM

    star_marg_pack = dict(
        l_rad=l_s.astype(np.float32),
        b_rad=b_s.astype(np.float32),
        d_nodes=d_nodes.astype(np.float32),
        d_weights=d_weights.astype(np.float32),
        mu_nodes=mu_nodes.astype(np.float32),
        x_nodes=x_nodes.astype(np.float32),
        y_nodes=y_nodes.astype(np.float32),
        z_nodes=z_nodes.astype(np.float32),
        Av_nodes=Av_nodes.astype(np.float32),
        birth_nodes=birth_nodes.astype(np.float32),
        M_nodes=M_nodes.astype(np.float32),
        logM_nodes=logM_nodes.astype(np.float32),
        M_weights=M_weights.astype(np.float32),
        mean_mass_IMF=np.float32(Mbar)
    )
    np.savez_compressed(f"{args.out_prefix}_star_marg_pack.npz", **star_marg_pack)
    print(f"[OK] wrote {args.out_prefix}_star_marg_pack.npz", file=sys.stderr)

    # Optional combined bundle:
    if args.combined_npz:
        combo = {}
        # tag to avoid collisions
        combo.update({f"grid::{k}": v for k,v in grid_pack.items()})
        combo.update({f"mass::{k}": v for k,v in mass_pack.items()})
        combo.update({f"star::{k}": v for k,v in star_pack.items()})
        combo.update({f"starmarg::{k}": v for k,v in star_marg_pack.items()})
        meta = dict(
            note="Combined PPP precompute bundle",
            sfr_npz=os.path.abspath(args.sfr_npz),
            tsv=os.path.abspath(args.tsv),
            nside=args.nside, dmin=args.dmin, dmax=args.dmax, dr=args.dr,
            Mmin_all=args.Mmin_all, Mmax_all=args.Mmax_all,
            Mmin_herbig=args.Mmin_herbig, Mmax_herbig=args.Mmax_herbig, Mq=args.Mq,
            kd=args.k_d, km=args.k_m, sigma_logM=args.sigma_logM, d_sigma_frac=args.d_sigma_frac,
            add_av_grid=args.add_av_grid
        )
        combo["meta_json"] = np.array(json.dumps(meta))
        np.savez_compressed(args.combined_npz, **combo)
        print(f"[OK] wrote combined bundle: {args.combined_npz}", file=sys.stderr)

    # Final summary
    n_cells = G
    n_valid = int(np.count_nonzero(grid_pack["footprint"]))
    print(f"[summary] cells={n_cells}, with Σ_SFR support={n_valid} ({100*n_valid/n_cells:.1f}%)", file=sys.stderr)
    print(f"[summary] IMF <M>={Mbar:.3f} Msun; Herbig quadrature nodes={args.Mq}", file=sys.stderr)
    if args.add_av_grid:
        print(f"[summary] A_V grid included: {Av_grid is not None}", file=sys.stderr)

if __name__ == "__main__":
    main()

