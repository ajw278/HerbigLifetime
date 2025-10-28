#!/usr/bin/env python3
# build_star_marginalization_pack.py
# Dependencies: numpy, pandas, astropy, scipy, dustmaps

import argparse, numpy as np, pandas as pd
from scipy.stats import norm
from astropy.coordinates import SkyCoord
import astropy.units as u

from dustmaps.edenhofer2023 import Edenhofer2023Query
from dustmaps.config import config as dustmaps_config

# --- IMF mean mass (same as step-1) ---
def chabrier2003_unnorm_pdf(M):
    M = np.asarray(M, float)
    pdf = np.zeros_like(M)
    ok = M > 0
    lo = ok & (M <= 1.0); hi = ok & (M > 1.0)
    if np.any(lo):
        m_c, sigma, A = 0.079, 0.69, 0.158
        log10M = np.log10(M[lo])
        pdf[lo] = (A / (M[lo] * np.log(10.0))) * np.exp(-0.5*((log10M-np.log10(m_c))/sigma)**2)
    if np.any(hi):
        alpha, A2 = 2.3, 0.0443
        pdf[hi] = A2 * M[hi]**(-alpha)
    return pdf

def mean_mass_chabrier(Mmin=0.08, Mmax=120.0, n=20000):
    x = np.logspace(np.log10(Mmin), np.log10(Mmax), n)
    y = chabrier2003_unnorm_pdf(x)
    Z = np.trapz(y, x)
    return float(np.trapz(x*y, x) / Z)

# --- quantile nodes/weights for truncated Normal ---
def truncated_normal_nodes_weights(mu, sigma, a=None, b=None, K=7):
    """
    Return K nodes and weights that integrate f(x) under N(mu,sigma) truncated to [a,b].
    We take mid-quantiles between K+1 equal CDF cuts for stability.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        return np.array([mu]), np.array([1.0])

    # truncation bounds in CDF
    Fa = norm.cdf(( (a-mu)/sigma )) if a is not None else 0.0
    Fb = norm.cdf(( (b-mu)/sigma )) if b is not None else 1.0
    if Fb <= Fa + 1e-12:
        return np.array([mu]), np.array([1.0])

    # K bins between Fa..Fb
    edges = np.linspace(Fa, Fb, K+1)
    mids  = 0.5*(edges[:-1] + edges[1:])
    nodes = mu + sigma * norm.ppf(mids)
    # bin-probabilities as weights
    weights = np.diff(edges)
    # renormalize (should already sum to Fb-Fa)
    weights = weights / np.sum(weights)
    return nodes, weights

def distance_modulus(d_pc):
    return 5.0 * np.log10(np.maximum(d_pc,1.0)/10.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sfr_npz", required=True, help="NPZ with x_grid, y_grid, and Sigma_SFR[_smoothed]")
    ap.add_argument("--tsv", required=True, help="Herbig TSV (needs ra, dec, distance; min_dist/max_dist optional; Mstar optional)")
    ap.add_argument("--out", default="star_marg_pack.npz")
    ap.add_argument("--k_d", type=int, default=7, help="# distance nodes per star")
    ap.add_argument("--k_m", type=int, default=5, help="# mass nodes per star")
    ap.add_argument("--sigma_logM", type=float, default=0.20, help="default 1-sigma scatter in log10(M) if no mass errors (dex)")
    ap.add_argument("--d_sigma_frac", type=float, default=0.10, help="fallback fractional distance error if no min/max (e.g., 10%)")
    ap.add_argument("--dust_dir", default=None, help="dustmaps data dir (default ~/.dustmaps)")
    args = ap.parse_args()

    # Load SFR grid (prefer smoothed key)
    D = np.load(args.sfr_npz, allow_pickle=True)
    xg, yg = D["x_grid"], D["y_grid"]
    S = D["Sigma_SFR_smoothed"] if "Sigma_SFR_smoothed" in D.files else D["Sigma_SFR"]

    # Build interpolator (y,x order)
    from scipy.interpolate import RegularGridInterpolator
    sfr_interp = RegularGridInterpolator((yg, xg), S, bounds_error=False, fill_value=np.nan)

    # Mean IMF mass for Σ_birth
    Mbar = mean_mass_chabrier()

    # Stars
    cat = pd.read_csv(args.tsv, sep="\t")
    if not {"ra","dec","distance"}.issubset(cat.columns):
        raise SystemExit("TSV must include: ra, dec, distance (pc)")
    ra = cat["ra"].to_numpy(float); dec = cat["dec"].to_numpy(float)
    d_obs = cat["distance"].to_numpy(float)
    d_min = cat["min_dist"].to_numpy(float) if "min_dist" in cat.columns else np.full_like(d_obs, np.nan)
    d_max = cat["max_dist"].to_numpy(float) if "max_dist" in cat.columns else np.full_like(d_obs, np.nan)
    M_obs = cat["Mstar"].to_numpy(float) if "Mstar" in cat.columns else np.full_like(d_obs, np.nan)

    # Sky coords
    sky = SkyCoord(ra*u.deg, dec*u.deg, frame="icrs").galactic
    l_rad, b_rad = sky.l.rad, sky.b.rad

    # Dust map (integrated) to get A_V(d)
    if args.dust_dir:
        dustmaps_config["data_dir"] = args.dust_dir
    q_int = Edenhofer2023Query(integrated=True, load_samples=False)

    N = len(d_obs)
    Kd, Km = args.k_d, args.k_m

    # Allocate arrays
    d_nodes = np.full((N, Kd), np.nan, float)
    d_wts   = np.full((N, Kd), np.nan, float)
    mu_nodes= np.full((N, Kd), np.nan, float)
    x_nodes = np.full((N, Kd), np.nan, float)
    y_nodes = np.full((N, Kd), np.nan, float)
    z_nodes = np.full((N, Kd), np.nan, float)
    Av_nodes= np.full((N, Kd), np.nan, float)
    birth_nodes = np.full((N, Kd), np.nan, float)  # Σ_birth in stars/yr/pc^2

    M_nodes  = np.full((N, Km), np.nan, float)
    M_wts    = np.full((N, Km), np.nan, float)
    logM_nodes = np.full((N, Km), np.nan, float)
    xi_nodes   = np.full((N, Km), np.nan, float)   # optional: IMF factor at nodes

    # Precompute IMF on a fine grid to evaluate ξ(M) quickly
    def xi_of_M(M):
        # normalized on [0.08, 120] Msun
        x = np.logspace(np.log10(0.08), np.log10(120.0), 20000)
        y = chabrier2003_unnorm_pdf(x)
        Z = np.trapz(y, x)
        return chabrier2003_unnorm_pdf(M)/Z

    # Loop stars (N is small: ~few hundred)
    for i in range(N):
        mu_d = d_obs[i]
        # fallback sigma_d: from min/max if present else fractional
        if np.isfinite(d_min[i]) and np.isfinite(d_max[i]) and (d_max[i] > d_min[i]):
            # approx σ from half-width of the 68% equivalent if given; otherwise just use half-range/2
            sigma_d = 0.25 * (d_max[i] - d_min[i])
            a, b = d_min[i], d_max[i]
        else:
            sigma_d = args.d_sigma_frac * max(mu_d, 1.0)
            a, b = None, None

        dn, dw = truncated_normal_nodes_weights(mu_d, sigma_d, a, b, K=Kd)
        d_nodes[i,:len(dn)] = dn; d_wts[i,:len(dw)] = dw

        # geometry per distance node
        l, bgal = l_rad[i], b_rad[i]
        cosb, sinb = np.cos(bgal), np.sin(bgal)
        cosl, sinl = np.cos(l), np.sin(l)

        x = dn * cosb * cosl
        y = dn * cosb * sinl
        z = dn * sinb
        x_nodes[i,:len(dn)] = x
        y_nodes[i,:len(dn)] = y
        z_nodes[i,:len(dn)] = z
        mu_nodes[i,:len(dn)] = distance_modulus(dn)

        # A_V to each node (E integrated × 2.8)
        coords = SkyCoord(np.full_like(dn, l)*u.rad, np.full_like(dn, bgal)*u.rad, dn*u.pc, frame="galactic")
        E_int = q_int.query(coords, mode="mean")
        Av_nodes[i,:len(dn)] = 2.8 * np.where(np.isfinite(E_int), E_int, 0.0)

        # Σ_birth(x,y) at nodes: evaluate Σ_SFR(x,y) then divide by <M> and 1e6
        pts = np.column_stack([y, x])  # (y,x)
        S_here = sfr_interp(pts)
        birth_nodes[i,:len(dn)] = (S_here / Mbar) / 1e6  # stars/yr/pc^2

        # Mass nodes (Normal in log10 M; truncate to [0.08, 120])
        Mi = M_obs[i] if np.isfinite(M_obs[i]) and (M_obs[i] > 0) else 1.5  # harmless default
        mu_logM = np.log10(Mi)
        sig_logM = args.sigma_logM
        # Build K bins in logM-Normal; clip to [log10(0.08), log10(120)]
        aL, bL = -1.09691, 2.07918
        # Convert truncated Normal in logM to nodes/weights (same routine)
        # Note: nodes will be in *logM*; transform back
        # We use the same helper by scaling: for Normal in logM, mu=mu_logM, sigma=sig_logM
        q_edges = np.linspace(norm.cdf((aL-mu_logM)/sig_logM), norm.cdf((bL-mu_logM)/sig_logM), Km+1)
        q_mids  = 0.5*(q_edges[:-1]+q_edges[1:])
        logM_n  = mu_logM + sig_logM * norm.ppf(q_mids)
        wM      = np.diff(q_edges)
        wM     /= np.sum(wM)
        M_n     = np.clip(10**logM_n, 0.08, 120.0)

        M_nodes[i,:len(M_n)]   = M_n
        logM_nodes[i,:len(M_n)] = logM_n
        M_wts[i,:len(wM)]      = wM
        xi_nodes[i,:len(M_n)]  = xi_of_M(M_n)  # optional use in-model or pre-store; your call

    np.savez_compressed(
        args.out,
        l_rad=l_rad.astype(np.float32),
        b_rad=b_rad.astype(np.float32),
        d_nodes=d_nodes.astype(np.float32),
        d_weights=d_wts.astype(np.float32),
        mu_nodes=mu_nodes.astype(np.float32),
        x_nodes=x_nodes.astype(np.float32),
        y_nodes=y_nodes.astype(np.float32),
        z_nodes=z_nodes.astype(np.float32),
        Av_nodes=Av_nodes.astype(np.float32),
        birth_nodes=birth_nodes.astype(np.float32),   # stars/yr/pc^2
        M_nodes=M_nodes.astype(np.float32),
        logM_nodes=logM_nodes.astype(np.float32),
        M_weights=M_wts.astype(np.float32),
        xi_nodes=xi_nodes.astype(np.float32),
        mean_mass_IMF=Mbar
    )
    print(f"[OK] wrote {args.out} with shapes:",
          f"d_nodes {d_nodes.shape}, M_nodes {M_nodes.shape}")

if __name__ == "__main__":
    main()

