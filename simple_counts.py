#!/usr/bin/env python3
"""
Compare expected vs observed cumulative counts of Herbig stars as a function of
cylindrical radius R, under:

  (1) Constant lifetime t = 3 Myr
  (2) IMF-weighted PMS lifetime t_PMS(M)
  (3) Fitted τ(M) & completeness from the simple radial PyMC model
  (4) Same fitted model but with τ(M) -> τ(M) + τ_add (default τ_add = 0.01 Myr)

All four panels use the *same* axisymmetric radial star-formation profile Σ_SFR(R),
constructed by averaging Σ_SFR(x,y) in cylindrical rings of width --radial_ring_dR_pc.

Inputs
------
--sfr_npz      : NPZ from build_sfr_map_xy_from_dust.py
                 (must contain x_grid, y_grid, and Sigma_SFR_smoothed or Sigma_SFR)
--tsv          : Herbig TSV (cols: ra [deg], dec [deg], distance [pc], Mstar [Msun])
--radial_trace : netCDF trace from the simple radial PyMC fit
                 (e.g. herbig_simple_radial_fit.nc with log_k_geom, a0, a_R, a_M, alpha_tau)

Output
------
PNG figure with 4 panels: 3 Myr, t_PMS(M), fitted τ(M), fitted τ(M)+τ_add.
"""

import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
from mpl_setup import *

from astropy.coordinates import SkyCoord
import astropy.units as u


# =====================
#  IMF + PMS lifetime
# =====================

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
        out[lo] = (A / (M[lo] * np.log(10.0))) * np.exp(
            -0.5 * ((log10M - np.log10(m_c)) / sigma) ** 2
        )
    # lognormal-like high-mass tail (your adopted variant)
    m_c, sigma, A = 0.086, 0.57, 0.22
    if np.any(hi):
        log10M = np.log10(M[hi])
        out[hi] = (A / (M[hi] * np.log(10.0))) * np.exp(
            -0.5 * ((log10M - np.log10(m_c)) / sigma) ** 2
        )
    return out


def xi_norm_on_interval(M, Mmin=0.08, Mmax=120.0):
    """Normalized Chabrier pdf on [Mmin,Mmax] evaluated at M."""
    x = np.logspace(np.log10(Mmin), np.log10(Mmax), 20000)
    y = chabrier2003_unnorm_pdf(x)
    Z = np.trapz(y, x)
    return chabrier2003_unnorm_pdf(M) / Z


def imf_mean_mass(Mmin=0.08, Mmax=120.0, n=20000):
    """Mean stellar mass <M> for the normalized Chabrier IMF on [Mmin,Mmax]."""
    x = np.logspace(np.log10(Mmin), np.log10(Mmax), n)
    y = xi_norm_on_interval(x, Mmin=Mmin, Mmax=Mmax)
    num = np.trapz(x * y, x)
    den = np.trapz(y, x)
    return float(num / den)


def t_pms_Myr(M):
    """
    Approximate pre-main-sequence lifetime in Myr as a function of mass (Msun):

        t_PMS(M) ≈ 10 Myr * (M / Msun)^(-2.5)

    motivated by the KH timescale scaling (Pols 2011).
    """
    M = np.asarray(M, float)
    return 10.0 * np.power(np.maximum(M, 1e-3), -2.5)


# =====================
#  Coordinates
# =====================

def icrs_to_gal_lbd(ra_deg, dec_deg):
    """ICRS (deg) -> Galactic (l,b) deg via astropy."""
    sky = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    gal = sky.galactic
    return gal.l.deg, gal.b.deg


def stars_to_xy(ra_deg, dec_deg, d_pc):
    """
    Convert catalog RA,Dec,Distance to Sun-centered Cartesian (x,y,z) [pc],
    in the same convention as build_sfr_map_xy_from_dust.py.
    """
    l_deg, b_deg = icrs_to_gal_lbd(ra_deg, dec_deg)
    l = np.deg2rad(l_deg)
    b = np.deg2rad(b_deg)
    x = d_pc * np.cos(b) * np.cos(l)
    y = d_pc * np.cos(b) * np.sin(l)
    z = d_pc * np.sin(b)
    return x, y, z


# =====================
#  Radial SFR profile helper
# =====================

def build_sfr_rings(x_grid, y_grid, Sigma_SFR, ring_dR_pc):
    """
    Build an axisymmetric Σ_SFR(R) profile by averaging Σ_SFR(x,y) in cylindrical rings.

    Returns
    -------
    R_edges      : array, shape (n_ring+1,)
        Radial edges of rings [pc], starting at 0.
    Sigma_SFR_R  : array, shape (n_ring,)
        Mean Σ_SFR in each ring [Msun yr^-1 kpc^-2].
    """
    XX, YY = np.meshgrid(x_grid, y_grid, indexing="xy")
    R_grid = np.sqrt(XX**2 + YY**2)

    valid = np.isfinite(Sigma_SFR) & (Sigma_SFR > 0)
    if not np.any(valid):
        raise RuntimeError("No valid Σ_SFR cells to build radial profile from.")

    R_valid = R_grid[valid]
    R_max = float(np.nanmax(R_valid))
    dR = float(ring_dR_pc)

    R_edges = np.arange(0.0, R_max + dR, dR)
    if R_edges.size < 2:
        R_edges = np.array([0.0, R_max], float)

    n_ring = R_edges.size - 1
    Sigma_SFR_R = np.zeros(n_ring, float)

    R_flat = R_grid[valid]
    S_flat = Sigma_SFR[valid]

    for i in range(n_ring):
        m = (R_flat >= R_edges[i]) & (R_flat < R_edges[i + 1])
        if np.any(m):
            Sigma_SFR_R[i] = float(np.mean(S_flat[m]))
        else:
            Sigma_SFR_R[i] = 0.0

    return R_edges, Sigma_SFR_R


def cumulative_counts_from_rings(R_edges, lambda_ring_bins, R_eval):
    """
    Convert ring-level counts λ_ring (per *full* ring) into cumulative N(<R_eval).

    Parameters
    ----------
    R_edges         : array, (n_ring+1,)
        Ring edges [pc].
    lambda_ring_bins: array, (n_bins, n_ring)
        Expected counts per *full* ring, per mass bin.
    R_eval          : array, (n_R,)
        Radii at which to evaluate cumulative counts.

    Returns
    -------
    N_bins_R : array, shape (n_bins, n_R)
        Cumulative counts N(<R) for each mass bin.
    """
    lambda_ring_bins = np.asarray(lambda_ring_bins, float)
    R_eval = np.asarray(R_eval, float)

    if not np.all(np.diff(R_eval) >= 0):
        raise ValueError("R_eval must be non-decreasing.")

    n_bins, n_ring = lambda_ring_bins.shape
    n_R = R_eval.size

    Rin = R_edges[:-1]
    Rout = R_edges[1:]

    N_all = np.zeros((n_bins, n_R), float)

    for j in range(n_bins):
        lam_ring = lambda_ring_bins[j, :]
        N = np.zeros(n_R, float)
        for i in range(n_ring):
            lam = lam_ring[i]
            if lam <= 0.0:
                continue
            R0, R1 = Rin[i], Rout[i]
            if R1 <= R0 + 1e-6:
                continue

            # Full contribution where R_eval >= R1
            full = R_eval >= R1
            if np.any(full):
                N[full] += lam

            # Partial contribution where R0 < R_eval < R1
            part = (R_eval > R0) & (R_eval < R1)
            if np.any(part):
                num = R_eval[part]**2 - R0**2
                den = R1**2 - R0**2
                frac = np.clip(num / max(den, 1e-12), 0.0, 1.0)
                N[part] += lam * frac

        N_all[j, :] = N

    return N_all


# =====================
#  Radial-fit helper
# =====================

def compute_radialfit_counts_vs_R(
    trace_path,
    R_edges,
    Sigma_SFR_R,
    area_ring_kpc2,
    bin_frac,
    Mbar,
    mass_bins,
    R_eval,
    tau0_Myr=3.0,
    R0_pc=500.0,
    M0_Msun=1.0,
    tau_min_Myr=0.0,
):
    """
    Compute expected cumulative counts N(<R) per mass bin from the simplified
    radial completeness + τ(M) model, using posterior median parameters:

        λ(R,M) = exp(log_k_geom) * SFR_ring(R) * τ(M)
                 * (bin_frac / Mbar) * f_det(R,M)

    with
        SFR_ring(R)    = Σ_SFR(R) * A_ring * 1e6   [Msun/Myr]
        τ(M) [Myr]     = τ0 * (M / M0)^alpha_tau + τ_add
        logit f_det    = a0 + a_R * ln(R/R0) + a_M * ln(M/M0)

    All panels share the same Σ_SFR(R) profile; only τ(M) and f_det differ.
    Returns
    -------
    N_fit : array, shape (n_bins, n_R)
        Cumulative counts N(<R) for each mass bin.
    alpha_tau : float
        Posterior-median slope of the τ(M) relation.
    log_k_geom : float
        Posterior-median geometric / global SFR scaling factor.

    """
    import arviz as az

    idata = az.from_netcdf(trace_path)
    post = idata.posterior

    def med(name):
        if name not in post:
            raise KeyError(f"Parameter '{name}' not found in posterior.")
        arr = post[name].values  # (chain, draw, ...)
        return float(np.median(arr))

    log_k_geom = med("log_k_geom")
    a0 = med("a0")
    a_R = med("a_R")
    a_M = med("a_M")
    alpha_tau = med("alpha_tau")

    print(f"log_k_geom = {log_k_geom:.3f}", file=sys.stderr)
    print(f"a0 = {a0:.3f}, a_R = {a_R:.3f}, a_M = {a_M:.3f}, alpha_tau = {alpha_tau:.3f}", file=sys.stderr)

    n_ring = Sigma_SFR_R.size
    if R_edges.size != n_ring + 1:
        raise ValueError("R_edges length inconsistent with Sigma_SFR_R.")

    R_centers = 0.5 * (R_edges[:-1] + R_edges[1:])

    # SFR per ring in Msun/Myr
    SFR_ring_myr = Sigma_SFR_R * area_ring_kpc2 * 1e6  # Σ [Msun/yr/kpc^2] * A [kpc^2] * 1e6 [yr/Myr]

    # Mass-bin centres for evaluating τ(M) and completeness
    mass_bins = list(mass_bins)
    n_bins = len(mass_bins)
    M_centers = np.array([0.5 * (Mlo + Mhi) for (Mlo, Mhi) in mass_bins], float)

    # τ(M) in Myr
    tau_M = np.maximum(tau0_Myr * (M_centers / M0_Msun) ** alpha_tau, np.absolute(tau_min_Myr))
    #tau_M = np.maximum(tau_M, 0.0)

    # Shapes
    logR = np.log(np.clip(R_centers / R0_pc, 1e-6, None))    # (n_ring,)
    logM = np.log(np.clip(M_centers / M0_Msun, 1e-6, None))  # (n_bins,)

    # Completeness
    eta = a0 + a_R * logR[:, None] + a_M * logM[None, :]     # (n_ring, n_bins)
    eta_clip = np.clip(eta, -50.0, 50.0)
    f_det = 1.0 / (1.0 + np.exp(-eta_clip))                  # (n_ring, n_bins)

    # IMF bin fractions
    bin_frac = np.asarray(bin_frac, float)
    if bin_frac.size != n_bins:
        raise ValueError("bin_frac must have same length as mass_bins.")
    frac_2d = (bin_frac / Mbar)[None, :]                     # (1, n_bins)

    # λ_ring per (ring, mass bin)
    lambda_ring = (
        np.exp(log_k_geom) *
        SFR_ring_myr[:, None] *          # (n_ring, 1)
        tau_M[None, :] *                 # (1, n_bins)
        frac_2d *                        # (1, n_bins)
        f_det                            # (n_ring, n_bins)
    )  # -> (n_ring, n_bins)

    # Clean up any numerical issues
    lambda_ring = np.where(
        np.isfinite(lambda_ring) & (lambda_ring > 0.0),
        lambda_ring,
        0.0,
    )

    # Convert to N(<R_eval). cumulative_counts_from_rings expects (n_bins, n_ring)
    N_fit = cumulative_counts_from_rings(R_edges, lambda_ring.T, R_eval)

    return N_fit, alpha_tau, log_k_geom

# =====================
#  Main
# =====================

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--sfr_npz", required=True,
                    help="NPZ with x_grid, y_grid, and Sigma_SFR_smoothed or Sigma_SFR")
    ap.add_argument("--tsv", required=True,
                    help="Herbig TSV (cols: ra, dec, distance, Mstar)")
    ap.add_argument("--radial_trace", required=True,
                    help="NetCDF from simple radial completeness+τ(M) fit "
                         "(e.g. herbig_simple_radial_fit.nc with log_k_geom)")
    ap.add_argument("--out", default="herbig_counts_vs_radius_4panels.png",
                    help="Output PNG figure")

    # SFR scaling (e.g. 0.55 from Quintana+ inside 1 kpc)
    ap.add_argument("--sfr_scale", type=float, default=None,
                    help="Global scale factor to apply to Σ_SFR")

    # Radius grid for plotting
    ap.add_argument("--Rmax", type=float, default=1000.0,
                    help="Max cylindrical radius (pc) for cumulative profiles")
    ap.add_argument("--nR", type=int, default=200,
                    help="# of radius points between 0 and Rmax")

    # Mass bins
    ap.add_argument("--mass_edges", nargs="+", type=float,
                    default=[1.5, 2.5, 4.0, 6.0, 10.0],
                    help="Mass-bin edges in Msun, e.g. 1.5 2.5 4 8 gives 3 bins")

    # Radial-fit parameters (should match the fit you ran)
    ap.add_argument("--radial_ring_dR_pc", type=float, default=50.0,
                    help="Ring width [pc] used to build Σ_SFR(R) for all panels "
                         "(should match dR_pc used in the radial fit)")
    ap.add_argument("--tau0_Myr_fit", type=float, default=3.0,
                    help="τ0 at 1 Msun [Myr] used in the radial fit")
    ap.add_argument("--R0_pc_fit", type=float, default=500.0,
                    help="Reference radius R0 [pc] used in the radial fit")
    ap.add_argument("--M0_Msun_fit", type=float, default=1.0,
                    help="Reference mass M0 [Msun] used in the radial fit")
    ap.add_argument("--tau_min", type=float, default=0.2,
                    help="Additive lifetime offset (Myr) for 4th panel (e.g. 0.01 ≈ 1e4 yr)")

    args = ap.parse_args()

    # ------------- Load SFR grid -------------
    D = np.load(args.sfr_npz, allow_pickle=True)
    xg = D["x_grid"]
    yg = D["y_grid"]

    if "Sigma_SFR_smoothed" in D.files:
        Sigma_SFR = D["Sigma_SFR_smoothed"]
    elif "Sigma_SFR" in D.files:
        Sigma_SFR = D["Sigma_SFR"]
    else:
        raise SystemExit("SFR NPZ must contain Sigma_SFR_smoothed or Sigma_SFR")

    if Sigma_SFR.shape != (yg.size, xg.size):
        raise SystemExit("Σ_SFR grid shape mismatch with x_grid/y_grid")
    
    if args.sfr_scale is None:
        sfr_scale = 1.0
    else:
        sfr_scale = args.sfr_scale

    # Apply global scaling
    Sigma_SFR = sfr_scale * Sigma_SFR   # Msun / yr / kpc^2

    # Build a common Σ_SFR(R) profile for ALL panels
    R_edges, Sigma_SFR_R = build_sfr_rings(
        xg, yg, Sigma_SFR, ring_dR_pc=args.radial_ring_dR_pc
    )
    Rin = R_edges[:-1]
    Rout = R_edges[1:]
    area_ring_kpc2 = np.pi * (Rout**2 - Rin**2) * 1e-6      # full ring area [kpc^2]

    # ------------- Herbig catalog -------------
    cat = pd.read_csv(args.tsv, sep="\t")
    for col in ("ra", "dec", "distance", "Mstar"):
        if col not in cat.columns:
            raise SystemExit("TSV must include columns: ra, dec, distance, Mstar")

    ra_deg = cat["ra"].to_numpy(float)
    dec_deg = cat["dec"].to_numpy(float)
    d_pc = cat["distance"].to_numpy(float)
    M_star = cat["Mstar"].to_numpy(float)

    x_star, y_star, z_star = stars_to_xy(ra_deg, dec_deg, d_pc)
    R_star = np.sqrt(x_star**2 + y_star**2)

    # ------------- Radius grid -------------
    R_eval = np.linspace(0.0, args.Rmax, args.nR)

    # ------------- IMF & mass bins -------------
    edges = np.asarray(args.mass_edges, float)
    if edges.size < 2:
        raise SystemExit("Need at least two mass_edges values.")
    mass_bins = [(edges[i], edges[i + 1]) for i in range(edges.size - 1)]
    n_bins = len(mass_bins)

    Mmin_all, Mmax_all = 0.08, 120.0
    Mgrid = np.logspace(np.log10(Mmin_all), np.log10(Mmax_all), 20000)
    xi_vals = xi_norm_on_interval(Mgrid, Mmin=Mmin_all, Mmax=Mmax_all)
    t_pms_vals = t_pms_Myr(Mgrid)

    # Mean stellar mass <M> over full IMF
    Mbar = imf_mean_mass(Mmin=Mmin_all, Mmax=Mmax_all)
    print(f"[info] IMF mean mass <M> ≈ {Mbar:.3f} Msun", file=sys.stderr)

    # For each mass bin: IMF fraction, IMF-weighted PMS lifetime
    bin_frac = np.zeros(n_bins, float)
    bin_t_pms = np.zeros(n_bins, float)
    for j, (Mlo, Mhi) in enumerate(mass_bins):
        mask = (Mgrid >= Mlo) & (Mgrid < Mhi)
        if not np.any(mask):
            raise RuntimeError(f"No IMF grid points in mass bin [{Mlo},{Mhi}] Msun")

        frac = np.trapz(xi_vals[mask], Mgrid[mask])
        num_t = np.trapz(xi_vals[mask] * t_pms_vals[mask], Mgrid[mask])

        bin_frac[j] = frac
        bin_t_pms[j] = num_t / max(frac, 1e-20)
        print(
            f"[info] bin {j}: {Mlo:.2f}–{Mhi:.2f} Msun "
            f"-> IMF fraction ≈ {frac:.3e}, <t_PMS> ≈ {bin_t_pms[j]:.2f} Myr",
            file=sys.stderr
        )

    # ------------- Observed cumulative counts N_obs(j,R) -------------
    N_obs = np.zeros((n_bins, args.nR), float)
    for j, (Mlo, Mhi) in enumerate(mass_bins):
        m_bin = np.isfinite(M_star) & (M_star >= Mlo) & (M_star < Mhi)
        R_bin = np.sort(R_star[m_bin])
        if R_bin.size == 0:
            continue
        N_obs[j, :] = np.searchsorted(R_bin, R_eval, side="right")

    # ------------- tau0: posterior median from radial_trace -------------
    import arviz as az
    idata = az.from_netcdf(args.radial_trace)

    if "tau0_Myr" in idata.posterior:
        tau0_post = idata.posterior["tau0_Myr"].values  # (chain, draw)
        tau0_Myr_med = float(np.median(tau0_post))
        print(f"[info] Using posterior median tau0_Myr = {tau0_Myr_med:.3f} Myr", file=sys.stderr)
    else:
        tau0_Myr_med = args.tau0_Myr_fit
        print(
            f"[warn] 'tau0_Myr' not found in trace; "
            f"falling back to CLI tau0_Myr_fit={tau0_Myr_med:.3f} Myr",
            file=sys.stderr,
        )

    # ------------- Radial-fit predictions -------------
    # Panel 3: τ(M) as fitted (no additive offset)
    N_fit_nominal, alpha_tau_fit, log_k_geom_fit = compute_radialfit_counts_vs_R(
        trace_path=args.radial_trace,
        R_edges=R_edges,
        Sigma_SFR_R=Sigma_SFR_R,
        area_ring_kpc2=area_ring_kpc2,
        bin_frac=bin_frac,
        Mbar=Mbar,
        mass_bins=mass_bins,
        R_eval=R_eval,
        tau0_Myr=tau0_Myr_med,
        R0_pc=args.R0_pc_fit,
        M0_Msun=args.M0_Msun_fit,
        tau_min_Myr=0.0,
    )
    print("[info] Radial-fit prediction (no offset) computed from posterior median.", file=sys.stderr)

    # Panel 4: τ(M) + τ_add
    N_fit_plus, _, _ = compute_radialfit_counts_vs_R(
        trace_path=args.radial_trace,
        R_edges=R_edges,
        Sigma_SFR_R=Sigma_SFR_R,
        area_ring_kpc2=area_ring_kpc2,
        bin_frac=bin_frac,
        Mbar=Mbar,
        mass_bins=mass_bins,
        R_eval=R_eval,
        tau0_Myr=tau0_Myr_med,
        R0_pc=args.R0_pc_fit,
        M0_Msun=args.M0_Msun_fit,
        tau_min_Myr=args.tau_min,
    )
    print(f"[info] Radial-fit prediction computed.", file=sys.stderr)

        # ------------- Model cumulative counts: 3 Myr & PMS, using fitted SFR amplitude -------------
    life_names = ["$\\tau = 3$ Myr", "$\\tau = \\tau_{\\mathrm{PMS}}(M_*)$"]
    n_life = 2

    # N_model_life[ilife, jbin, iR]
    N_model_life = np.zeros((n_life, n_bins, args.nR), float)

    # Ring-level counts for constant 3 Myr and PMS lifetimes
    tMyr_const = 3.0
    t_const_yr = tMyr_const * 1e6
    t_pms_yr = bin_t_pms * 1e6  # (n_bins,)

    # Geometric / SFR scaling from radial fit
    k_geom = np.exp(log_k_geom_fit)
    print(
        f"[info] Applying geometric SFR factor k_geom = exp(log_k_geom) ≈ {k_geom:.3f} "
        "to constant and t_PMS models.",
        file=sys.stderr,
    )

    # SFR per ring [Msun/yr], including fitted scaling
    SFR_ring_yr = Sigma_SFR_R * area_ring_kpc2 * k_geom  # (n_ring,)

    # scale factors per bin
    scale_const = (t_const_yr * bin_frac) / Mbar          # (n_bins,)
    scale_pms   = (t_pms_yr * bin_frac) / Mbar            # (n_bins,)

    # λ_ring per full ring, for each mass bin
    lambda_ring_const = SFR_ring_yr[None, :] * scale_const[:, None]  # (n_bins, n_ring)
    lambda_ring_pms   = SFR_ring_yr[None, :] * scale_pms[:, None]    # (n_bins, n_ring)

    # Convert to cumulative N(<R) using the same ring geometry
    N_const = cumulative_counts_from_rings(R_edges, lambda_ring_const, R_eval)
    N_pms   = cumulative_counts_from_rings(R_edges, lambda_ring_pms,   R_eval)

    N_model_life[0, :, :] = N_const
    N_model_life[1, :, :] = N_pms


    # ------------- Plotting (4 panels) -------------
    n_panels = 3
    fig, axes = plt.subplots(
        1, n_panels, figsize=(3.5 * n_panels, 4.5), sharey=True, sharex=True
    )

    # Discrete colours for each mass bin
    cmap_base = plt.cm.viridis

    mass_edges = edges  # just renaming for clarity
    mass_bins_list = mass_bins
    bin_centers = np.array([0.5 * (Mlo + Mhi) for (Mlo, Mhi) in mass_bins_list])

    colors = cmap_base(np.linspace(0.2, 0.8, n_bins))
    cmap_disc = ListedColormap(colors)
    norm = BoundaryNorm(mass_edges, ncolors=cmap_disc.N)

    # Panels 0 & 1: 3 Myr and t_PMS(M)
    for il in range(n_life):
        ax = axes[il]
        for j, (Mlo, Mhi) in enumerate(mass_bins_list):
            col = colors[j]
            y_obs = np.clip(N_obs[j, :], 1e-3, None)
            y_mod = np.clip(N_model_life[il, j, :], 1e-3, None)

            if il == 0 and j == 0:
                ax.plot(
                    R_eval, y_obs, color=col, linestyle="-", linewidth=1.8,
                    label="Observed"
                )
                ax.plot(
                    R_eval, y_mod, color=col, linestyle="--", linewidth=1.8,
                    label="Model"
                )
            else:
                ax.plot(R_eval, y_obs, color=col, linestyle="-", linewidth=1.5)
                ax.plot(R_eval, y_mod, color=col, linestyle="--", linewidth=1.5)

        ax.set_yscale("log")
        ax.set_xlabel(r"$R$ [pc]")
        ax.set_title(life_names[il])
        #True, which="both", alpha=0.3)

    # Panel 2: fitted τ(M) + completeness (no offset)
    ax = axes[2]
    for j, (Mlo, Mhi) in enumerate(mass_bins_list):
        col = colors[j]
        y_obs = np.clip(N_obs[j, :], 1e-3, None)
        y_fit = np.clip(N_fit_nominal[j, :], 1e-3, None)

        ax.plot(R_eval, y_obs, color=col, linestyle="-", linewidth=1.5)
        ax.plot(R_eval, y_fit, color=col, linestyle="--", linewidth=1.5)

    ax.set_yscale("log")
    ax.set_xlabel(r"$R$ [pc]")
    ax.set_title("$\\tau(M_*)$ and completeness")
    #ax.grid(True, which="both", alpha=0.3)

    # Age relation text at top of this panel
    if np.isclose(args.M0_Msun_fit, 1.0):
        tau_text = (
            rf"$\tau(M_*) = {tau0_Myr_med:.1f}\,\mathrm{{Myr}}"
            rf"\,(M_*/M_\odot)^{{{alpha_tau_fit:.2f}}}$"
        )
    else:
        tau_text = (
            rf"$\tau(M_*) = {tau0_Myr_med:.1f}\,\mathrm{{Myr}}"
            rf"\,(M_*/{args.M0_Msun_fit:.1f}M_\odot)^{{{alpha_tau_fit:.2f}}}$"
        )

    ax.text(
        0.05, 0.95, tau_text,
        transform=ax.transAxes, va="top", ha="left", fontsize=9
    )


    # Panel 3: fitted τ(M) + τ_add
    '''ax = axes[3]
    for j, (Mlo, Mhi) in enumerate(mass_bins_list):
        col = colors[j]
        y_obs = np.clip(N_obs[j, :], 1e-3, None)
        y_fit_plus = np.clip(N_fit_plus[j, :], 1e-3, None)

        ax.plot(R_eval, y_obs, color=col, linestyle="-", linewidth=1.5)
        ax.plot(R_eval, y_fit_plus, color=col, linestyle="--", linewidth=1.5)

    ax.set_yscale("log")
    ax.set_xlabel(r"$R$ [pc]")
    ax.set_title(rf"max$\left( \tau(M_*), {args.tau_min:.1f}\mathrm{{Myr}} \right)$")
    ax.grid(True, which="both", alpha=0.3)

    # Show the same fitted τ(M) relation in panel 3 as well
    ax.text(
        0.05, 0.95, tau_text,
        transform=ax.transAxes, va="top", ha="left", fontsize=9
    )'''


    axes[0].set_ylabel(r"Cumulative count $N(<R)$")
    axes[0].legend(fontsize=8, loc="lower right", frameon=True)
    plt.ylim([1.0, 3e2])

    # --- Discrete colour bar for stellar mass, with boundaries = mass bin edges ---
    sm = ScalarMappable(cmap=cmap_disc, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=axes,
        location="right",
        fraction=0.05,
        pad=0.02,
        boundaries=mass_edges,
        spacing="proportional",
    )

    cbar.set_label(r"$M_\star\ [{\rm M}_\odot]$")
    cbar.set_ticks(mass_edges)
    cbar.set_ticklabels([f"{m:.1f}" for m in mass_edges])
    plt.xlim([0.0, 1000.0])

    plt.savefig(args.out, dpi=150)
    print(f"[OK] saved plot to {args.out}", file=sys.stderr)
    plt.show()


if __name__ == "__main__":
    main()
