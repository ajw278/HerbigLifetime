#!/usr/bin/env python3
"""
Build a Sun-centered x–y map of Σ_SFR(x,y) from a 3D dust map (Edenhofer+2023 via dustmaps),
overlay Herbig locations, and **cache** the full 3D A_V-per-pc cube for re-use.

- Σ_gas ≈ 21.3 Msun/pc^2 per mag(A_V) (tunable via --av_to_gas)
- Σ_SFR = A * Σ_gas^N  (default A=2.5e-4, N=1.4)
- A_V = 2.8 * E (per-pc extinction-density units from Edenhofer map; tunable via --av_per_E)
- Caching: NPZ file with x_grid, y_grid, z_vals, AV_per_pc(float32), meta_json
"""

import argparse
import json
import os
import sys
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u

from dustmaps.config import config as dustmaps_config
import dustmaps.edenhofer2023 as eden
from dustmaps.edenhofer2023 import Edenhofer2023Query
from scipy.ndimage import gaussian_filter 


def icrs_to_gal_lbd(ra_deg: np.ndarray, dec_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ICRS (deg) -> Galactic (l,b) deg via astropy."""
    sky = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    gal = sky.galactic
    return gal.l.deg, gal.b.deg


def xy_to_lbd(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sun-centered Cartesian (pc) -> Galactic (l,b,d)."""
    d = np.sqrt(x**2 + y**2 + z**2).astype(np.float64)
    d = np.where(d < 1e-6, 1e-6, d)
    l = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    b = np.degrees(np.arcsin(z / d))
    return l, b, d


def _grids_match(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> bool:
    return a.shape == b.shape and np.allclose(a, b, atol=tol, rtol=0)


def _try_load_av_cache(path: str,
                       x_grid: np.ndarray,
                       y_grid: np.ndarray,
                       z_vals: np.ndarray):
    if not (path and os.path.exists(path)):
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            xg = data["x_grid"]
            yg = data["y_grid"]
            zg = data["z_vals"]
            if not (_grids_match(xg, x_grid) and _grids_match(yg, y_grid) and _grids_match(zg, z_vals)):
                print("AV cache found but grid differs; ignoring.", file=sys.stderr)
                return None
            AV_per_pc = data["AV_per_pc"].astype(np.float64)  # (Ny, Nx, Nz)
            meta_json = str(data["meta_json"].item()) if "meta_json" in data.files else "{}"
            meta = json.loads(meta_json)
            print(f"Loaded AV_per_pc from cache '{path}' (shape={AV_per_pc.shape}).", file=sys.stderr)
            return {"AV_per_pc": AV_per_pc, "meta": meta}
    except Exception as e:
        print(f"Failed to load AV cache '{path}': {e}", file=sys.stderr)
        return None


def _save_av_cache(path: str,
                   x_grid: np.ndarray,
                   y_grid: np.ndarray,
                   z_vals: np.ndarray,
                   AV_per_pc: np.ndarray,
                   meta: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    meta_json = json.dumps(meta)
    np.savez_compressed(
        path,
        x_grid=x_grid.astype(np.float64),
        y_grid=y_grid.astype(np.float64),
        z_vals=z_vals.astype(np.float64),
        AV_per_pc=AV_per_pc.astype(np.float32),  # smaller on disk
        meta_json=np.array(meta_json),
    )
    print(f"Saved AV_per_pc cache to '{path}' (shape={AV_per_pc.shape}).", file=sys.stderr)

def local_sfr(npz_path, radii_pc, treat_zeros_as_nan=False):
    data = np.load(npz_path, allow_pickle=False)
    x = data["x_grid"]          # pc
    y = data["y_grid"]          # pc
    S = data["Sigma_SFR"]       # Msun / yr / kpc^2
    # Optional: ignore cells that are exactly 0 (helps if dust map masked d<~69 pc)
    if treat_zeros_as_nan:
        S = np.where(S==0.0, np.nan, S)

    XX, YY = np.meshgrid(x, y, indexing="xy")
    dx_pc = float(x[1]-x[0]); dy_pc = float(y[1]-y[0])
    A_pix_kpc2 = (dx_pc/1000.0)*(dy_pc/1000.0)

    out = []
    for R_pc in radii_pc:
        R = float(R_pc)
        mask = (XX**2 + YY**2) <= R*R
        S_mask = S[mask]
        # mean Σ_SFR (area-weighted == plain mean on a uniform grid)
        mean_sig = np.nanmean(S_mask)
        # total area and total SFR inside aperture
        area_kpc2 = np.count_nonzero(mask) * A_pix_kpc2
        sfr_total = mean_sig * area_kpc2
        # simple spread (16–84th percentiles of pixel values)
        p16, p50, p84 = np.nanpercentile(S_mask, [16, 50, 84])
        out.append(dict(
            R_pc=R,
            mean_Sigma_SFR=mean_sig,      # Msun/yr/kpc^2
            median_Sigma_SFR=p50,
            pct16=p16, pct84=p84,
            area_kpc2=area_kpc2,
            SFR_total=sfr_total           # Msun/yr
        ))
    return out

def compute_AV_slab_from_dust(
    x_grid: np.ndarray, y_grid: np.ndarray, z_vals: np.ndarray,
    dmin: float = 69.0, dmax: float = 1250.0,
    AV_per_E: float = 2.8,
    chunk_size: int = 500_000,
    verbose: bool = True,
    av_cache: Optional[str] = None,
    cache_mode: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute (and optionally cache) the 3D A_V-per-pc cube and its z-integrated slab.

    Returns (AV_slab, AV_per_pc) where:
      - AV_slab: (Ny, Nx) with indexing='xy'
      - AV_per_pc: (Ny, Nx, Nz)

    cache_mode:
      - 'auto': load if cache exists & matches grids; else compute and save
      - 'load': require cache; error if missing/mismatch
      - 'save': compute then save (overwrite)
      - 'off' : compute fresh; do not save
    """
    NX, NY, NZ = x_grid.size, y_grid.size, z_vals.size

    if av_cache and cache_mode in ("auto", "load"):
        loaded = _try_load_av_cache(av_cache, x_grid, y_grid, z_vals)
        if loaded is not None:
            AV_per_pc = loaded["AV_per_pc"]
            if NZ < 2:
                raise ValueError("z_vals must have at least 2 points.")
            dZ = float(z_vals[1] - z_vals[0])
            return (AV_per_pc * dZ).sum(axis=2), AV_per_pc
        if cache_mode == "load":
            raise FileNotFoundError(f"Cache mode 'load' but cache not found/compatible: {av_cache}")

    # Build flattened 3D grid
    XX, YY, ZZ = np.meshgrid(x_grid, y_grid, z_vals, indexing="xy")
    Xf = XX.ravel().astype(np.float64)
    Yf = YY.ravel().astype(np.float64)
    Zf = ZZ.ravel().astype(np.float64)

    # Convert to Galactic (l,b,d)
    l_deg, b_deg, d_pc = xy_to_lbd(Xf, Yf, Zf)

    q_den = Edenhofer2023Query(integrated=False, load_samples=False)
    AV_per_pc_flat = np.zeros_like(Xf, dtype=np.float64)

    N = Xf.size
    i = 0
    while i < N:
        j = min(i + chunk_size, N)
        if verbose:
            print(f"Dust query: {i:,}–{j:,}/{N:,}...", file=sys.stderr)
        coords = SkyCoord(l_deg[i:j] * u.deg, b_deg[i:j] * u.deg, d_pc[i:j] * u.pc, frame="galactic")
        E_per_pc = q_den.query(coords, mode="mean")
        mask = (d_pc[i:j] < dmin) | (d_pc[i:j] > dmax) | ~np.isfinite(E_per_pc)
        E_per_pc = np.where(mask, 0.0, E_per_pc)
        AV_per_pc_flat[i:j] = AV_per_E * E_per_pc
        i = j

    AV_per_pc = AV_per_pc_flat.reshape((NY, NX, NZ))

    # Save cache
    if av_cache and cache_mode in ("auto", "save"):
        meta = dict(dmin=dmin, dmax=dmax, AV_per_E=AV_per_E, chunk_size=chunk_size)
        _save_av_cache(av_cache, x_grid, y_grid, z_vals, AV_per_pc, meta)

    if NZ < 2:
        raise ValueError("z_vals must have at least 2 points.")
    dZ = float(z_vals[1] - z_vals[0])
    AV_slab = (AV_per_pc * dZ).sum(axis=2)
    return AV_slab, AV_per_pc


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--tsv", required=True, help="Path to Herbig catalog TSV with columns: ra, dec, distance (pc)")
    ap.add_argument("--out", default="sfr_map_xy_from_dust.png", help="Output PNG path")
    ap.add_argument("--npz", default="sfr_grid_xy_from_dust.npz", help="Output NPZ (grid + Σ_SFR)")
    ap.add_argument("--zmax", type=float, default=200.0, help="Vertical half-thickness for integration (pc)")
    ap.add_argument("--dz", type=float, default=10.0, help="Vertical step for integration (pc)")
    ap.add_argument("--ks_A", type=float, default=2.5e-4, help="Kennicutt–Schmidt A (Msun yr^-1 kpc^-2)")
    ap.add_argument("--ks_N", type=float, default=1.4, help="Kennicutt–Schmidt N")
    ap.add_argument("--av_to_gas", type=float, default=21.3, help="Σ_gas [Msun/pc^2] per mag(A_V)")
    ap.add_argument("--dust_to_gas", type=float, default=1e-2,
                help="Dust-to-gas mass ratio (Σ_dust = Σ_gas * DGR)")
    ap.add_argument("--kernel_pc", type=float, default=200.0,
                help="Gaussian smoothing kernel FWHM (pc) for gas before K–S")
    ap.add_argument("--av_per_E", type=float, default=2.8, help="A_V per E unit from Edenhofer map")
    ap.add_argument("--grid", type=int, default=201, help="Grid size per axis (Nx=Ny)")
    ap.add_argument("--margin", type=float, default=200.0, help="Margin (pc) beyond star extent")
    ap.add_argument("--dmin", type=float, default=69.0, help="Dust map valid inner radius (pc)")
    ap.add_argument("--dmax", type=float, default=1250.0, help="Dust map valid outer radius (pc)")
    ap.add_argument("--chunk", type=int, default=500_000, help="Dust query chunk size")
    ap.add_argument("--fetch", action="store_true", help="Fetch Edenhofer dust map if not already present")
    ap.add_argument("--dust_dir", default=None, help="dustmaps data directory; defaults to ./dustmaps_data")
    # caching
    ap.add_argument("--av_cache", default="av_per_pc_cache.npz", help="Path to cache the A_V-per-pc 3D cube")
    ap.add_argument("--cache_mode", choices=["auto", "load", "save", "off"], default="auto", help="Cache behavior")
    ap.add_argument("--radii_pc", nargs="+", type=float, default=[200, 500, 1000])
    ap.add_argument("--ignore_masked_zeros", action="store_true",
                    help="Treat zeros as NaN (avoids bias from dust-map invalid radii).")
    args = ap.parse_args()
    

    # dustmaps location + optional fetch
    dustmaps_config["data_dir"] = args.dust_dir or os.path.expanduser("./dustmaps_data")
    if args.fetch:
        print("Fetching Edenhofer 2023 map (only needed once)...", file=sys.stderr)
        eden.fetch()

    # Load catalog
    cat = pd.read_csv(args.tsv, sep="\t")
    for col in ("ra", "dec", "distance"):
        if col not in cat.columns:
            raise SystemExit("TSV must include columns: ra, dec, distance (pc)")
    ra_deg = cat["ra"].to_numpy(float)
    dec_deg = cat["dec"].to_numpy(float)
    d_pc = cat["distance"].to_numpy(float)

    # Herbig x–y positions
    l_deg, b_deg = icrs_to_gal_lbd(ra_deg, dec_deg)
    l = np.deg2rad(l_deg)
    b = np.deg2rad(b_deg)
    x_star = d_pc * np.cos(b) * np.cos(l)
    y_star = d_pc * np.cos(b) * np.sin(l)

    # x–y grid bounds
    margin = float(args.margin)
    xmin = float(np.floor(min(x_star.min(), -x_star.max()) - margin))
    xmax = float(np.ceil(max(x_star.max(), -x_star.min()) + margin))
    ymin = float(np.floor(min(y_star.min(), -y_star.max()) - margin))
    ymax = float(np.ceil(max(y_star.max(), -y_star.min()) + margin))
    rmax = float(np.ceil(max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))))
    xmin, xmax, ymin, ymax = -rmax, rmax, -rmax, rmax

    NX = NY = int(args.grid)
    x_grid = np.linspace(xmin, xmax, NX)
    y_grid = np.linspace(ymin, ymax, NY)
    z_vals = np.arange(-args.zmax, args.zmax + args.dz, args.dz)

    # A_V slab (with caching)
    AV_slab, AV_per_pc = compute_AV_slab_from_dust(
        x_grid, y_grid, z_vals,
        dmin=args.dmin, dmax=args.dmax,
        AV_per_E=args.av_per_E,
        chunk_size=args.chunk,
        verbose=True,
        av_cache=args.av_cache,
        cache_mode=args.cache_mode,
    )


    # --- pixel scale & Gaussian σ in pc/pixels ---
    dx_pc = float(x_grid[1] - x_grid[0])
    dy_pc = float(y_grid[1] - y_grid[0])
    sigma_pc    = args.kernel_pc / 2.355                     # smoothing length (Gaussian σ)
    sigma_x_pix = sigma_pc / max(dx_pc, 1e-9)
    sigma_y_pix = sigma_pc / max(dy_pc, 1e-9)

    # --- base fields (native resolution) ---
    dust_per_mag = args.av_to_gas * args.dust_to_gas         # e.g. 21.3 * 0.01 = 0.213
    Sigma_dust   = dust_per_mag * AV_slab                    # [Msun/pc^2]
    Sigma_gas    = Sigma_dust / max(args.dust_to_gas, 1e-12) # 100× dust by default

    # --- define domain by "vertical half-height at d = dmax" >= smoothing length ---
    XX, YY = np.meshgrid(x_grid, y_grid, indexing='xy')
    R_xy   = np.sqrt(XX**2 + YY**2)                          # cylindrical radius in plane
    # height of the sphere surface above the mid-plane at this R (outer boundary only)
    z_surf = np.sqrt(np.maximum(0.0, args.dmax**2 - R_xy**2))  # [pc]
    # also cap by the slab (|z| <= zmax)
    H_eff  = np.minimum(z_surf, args.zmax)                   # effective half-height available
    domain = (H_eff >= sigma_pc*2.)                             # require at least one σ of vertical support

    # (FYI: the resulting truncation radius is R_trunc ≈ sqrt(dmax^2 - σ^2).)

    # --- geometric-mean fill outside + normalized Gaussian smoothing in log space ---
    # Ensure strictly positive values before log:
    inside = Sigma_gas[domain]
    # pick a tiny floor that won’t matter but avoids log(0); use a very low percentile of inside
    floor  = float(max(1e-12, (np.nanpercentile(inside, 0.1) * 1e-6) if inside.size else 1e-12))
    Sigma_gas_safe = np.where(domain, np.maximum(Sigma_gas, floor), floor)

    logSigma = np.log(Sigma_gas_safe)
    mu       = float(np.nanmean(logSigma[domain]))           # mean(log Σ) inside → geometric mean outside

    # de-mean within domain; zero outside so cval=0 matches
    arr = np.where(domain, logSigma - mu, 0.0)
    w   = np.where(domain, 1.0, 0.0)

    num = gaussian_filter(arr, sigma=(sigma_y_pix, sigma_x_pix), mode="constant", cval=0.0)
    den = gaussian_filter(w,   sigma=(sigma_y_pix, sigma_x_pix), mode="constant", cval=0.0)

    eps = 1e-6  # linear-space threshold on weights
    logSigma_sm  = np.where(den > eps, mu + num/den, mu)
    Sigma_gas_sm = np.exp(logSigma_sm)

    # --- K–S on the smoothed gas ---
    Sigma_SFR_sm = args.ks_A * np.power(np.maximum(Sigma_gas_sm, 0.0), args.ks_N)

    # Percentile thresholds **inside the dust domain** only (robust color scaling)
    dust_domain = np.isfinite(Sigma_dust) & (Sigma_dust > 0)
    v1 = np.nanpercentile(Sigma_SFR_sm[dust_domain], 1) if np.any(dust_domain) else 0.0

    Sigma_SFR_sm[~dust_domain] = np.nan

 
    # ---------- plotting: two panels ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Left: dust surface density (log10)
    im0 = axes[0].imshow(
        np.log10(np.maximum(Sigma_dust, np.percentile(Sigma_dust, 1)*1e-3)),
        extent=[xmin, xmax, ymin, ymax],
        origin="lower", aspect="equal"
    )
    axes[0].scatter(x_star, y_star, s=10)
    axes[0].set_xlabel("X [pc]"); axes[0].set_ylabel("Y [pc]")
    axes[0].set_title(r"$\Sigma_{\rm dust}$  [M$_\odot$ pc$^{-2}$] (native)")
    cb0 = plt.colorbar(im0, ax=axes[0]); cb0.set_label(r"log$_{10}$ $\Sigma_{\rm dust}$")

    # Right: smoothed SFR (log10)
    im1 = axes[1].imshow(
        np.log10(Sigma_SFR_sm*1e6),
        extent=[xmin, xmax, ymin, ymax],
        origin="lower", aspect="equal",
        vmin=-1, vmax=4.
    )
    axes[1].scatter(x_star, y_star, s=10)
    axes[1].set_xlabel("X [pc]"); axes[1].set_ylabel("Y [pc]")
    axes[1].set_title(r"Smoothed $\dot{\Sigma}_{\rm SFR}$  [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]")
    cb1 = plt.colorbar(im1, ax=axes[1]); cb1.set_label(r"log$_{10}$ $\dot{\Sigma}_{\rm SFR}$")

    plt.savefig(args.out, dpi=150)
    print(f"Saved figure to {args.out}", file=sys.stderr)

    # ---------- save all the useful arrays ----------
    np.savez_compressed(
        args.npz,
        x_grid=x_grid,
        y_grid=y_grid,
        AV_slab=AV_slab,
        Sigma_dust=Sigma_dust,
        Sigma_gas=Sigma_gas,
        Sigma_gas_smoothed=Sigma_gas_sm,
        Sigma_SFR_smoothed=Sigma_SFR_sm,
        meta=dict(
            zmax=args.zmax, dz=args.dz,
            ks_A=args.ks_A, ks_N=args.ks_N,
            av_to_gas=args.av_to_gas,
            dust_to_gas=args.dust_to_gas,
            dust_per_mag=dust_per_mag,
            kernel_pc=args.kernel_pc
        ),
    )

    plt.show()
if __name__ == "__main__":
    main()
