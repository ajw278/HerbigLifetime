#!/usr/bin/env python3
"""
Ring-level PPP likelihood for Herbig stars with distance+mass marginalization.

Instead of using the full (x,y) structure of the Σ_birth grid, we:
  - compute cylindrical radius R = sqrt(x^2 + y^2) for every grid cell,
  - bin cells into radial rings of width --ring_dR_pc,
  - volume-weight average Σ_birth, μ, A_V, z within each ring,
  - sum dV per ring,
  - and then use those ring-averaged quantities in the PPP normalisation integral Λ.

On the star side, we:
  - compute R_nodes = sqrt(x_nodes^2 + y_nodes^2) for each distance node,
  - map each node to a ring,
  - and replace the original birth_nodes[i,k] by the ring-averaged Σ_birth for that ring.

This implements a PPP model that only depends on cylindrical radius,
which is closer to the scale at which the KS+smoothed Σ_SFR field is trustworthy.

Inputs:
  --grid_pack        : from build_ppp_all_packs.py (HEALPix×distance cells)
  --mass_pack        : Chabrier IMF info (Herbig mass quadrature) + <M>
  --star_marg_pack   : per-star distance & mass nodes/weights, A_V(d), and Σ_birth at nodes

Options:
  --mu_ref           : center for distance-modulus term in selection (default 10)
  --k_lambda         : A_band / A_V coefficient for selection (default 1.0 uses A_V)
  --ring_dR_pc       : ring width in cylindrical radius [pc]
  --sample           : run NUTS sampling
  --draws, --tune    : sampling hyperparameters
  --out              : netcdf path to save trace (optional)
"""

import argparse
import numpy as np

import pymc as pm
import pytensor.tensor as pt

from utils import load_grid_layout   # assumes this gives x,y,z,mu,dV,Av,Sigma_birth,footprint

SQ2PI = np.sqrt(2.0 * np.pi)


# ---------- local IMF helper for star mass nodes ----------
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
    # power-law above 1 Msun
    alpha, A2 = 2.3, 0.0443
    if np.any(hi):
        out[hi] = A2 * (M[hi] ** (-alpha))
    return out


def xi_norm_on_interval(M, Mmin=0.08, Mmax=120.0):
    """Normalized Chabrier pdf on [Mmin,Mmax] evaluated at M (numpy)."""
    x = np.logspace(np.log10(Mmin), np.log10(Mmax), 20000)
    y = chabrier2003_unnorm_pdf(x)
    Z = np.trapz(y, x)
    return chabrier2003_unnorm_pdf(M) / Z


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--grid_pack", required=True)
    ap.add_argument("--mass_pack", required=True)
    ap.add_argument("--star_marg_pack", required=True)
    ap.add_argument("--mu_ref", type=float, default=10.0)
    ap.add_argument("--k_lambda", type=float, default=1.0)
    ap.add_argument("--ring_dR_pc", type=float, default=50.0,
                    help="Ring width in cylindrical radius [pc] for ring-level model")
    ap.add_argument("--sample", action="store_true", help="Run NUTS sampling")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--target_accept", type=float, default=0.9)
    ap.add_argument("--out", default="", help="If set, save trace to this netcdf file")
    ap.add_argument("--zsun_mu_pc", type=float, default=20.8,
                    help="Prior mean for Sun height above the Galactic mid-plane [pc]")
    ap.add_argument("--zsun_sigma_pc", type=float, default=2.0,
                    help="Prior sigma for z_sun [pc] (small, informative)")
    args = ap.parse_args()

    # ---- load packs ----
    grid = load_grid_layout(args.grid_pack)  # works for rectangular or ragged packs
    M = np.load(args.mass_pack, allow_pickle=True)
    S = np.load(args.star_marg_pack, allow_pickle=True)

    # ---------- GRID SIDE (cell-level → ring-level) ----------
    # Names here assume load_grid_layout returns these; adjust if needed.
    print(grid.keys())
    print(grid["x_pc"], grid["y_pc"], grid["z"])
    x_grid = grid["x_pc"].astype(float)
    y_grid = grid["y_pc"].astype(float)
    z_grid = np.nan_to_num(grid["z"], nan=0.0, posinf=0.0, neginf=0.0).astype(float)
    mu_grid = np.nan_to_num(grid["mu"], nan=0.0, posinf=0.0, neginf=0.0).astype(float)

    dV_grid = np.where(
        np.isfinite(grid["dV"]) & (grid["dV"] > 0), grid["dV"], 0.0
    ).astype(float)

    Av_grid = np.nan_to_num(
        grid["Av"], nan=0.0, posinf=0.0, neginf=0.0
    ).astype(float)

    print(grid.keys())
    Sigma_birth = np.where(
        np.isfinite(grid["Sigma_birth"]) & grid["footprint"],
        grid["Sigma_birth"],
        0.0,
    ).astype(float)

    footprint_f = grid["footprint"].astype(float)

    # Cylindrical radius for each cell
    R_grid = np.sqrt(x_grid ** 2 + y_grid ** 2)

    # Valid cells: finite birth rate, positive volume, inside footprint
    valid_cell = (dV_grid > 0) & (Sigma_birth > 0) & (footprint_f > 0)

    # Define radial rings
    R_max = float(np.nanmax(R_grid[valid_cell])) if np.any(valid_cell) else 0.0
    dR = float(args.ring_dR_pc)
    if R_max <= 0:
        raise SystemExit("No valid grid cells with Sigma_birth>0 and dV>0.")

    R_edges = np.arange(0.0, R_max + dR, dR)
    if R_edges.size < 2:
        R_edges = np.array([0.0, R_max], float)
    n_ring = R_edges.size - 1

    ring_idx = np.digitize(R_grid, R_edges) - 1  # [-1, n_ring-1]

    # Preallocate ring-level arrays
    Sigma_birth_ring = np.zeros(n_ring, float)
    mu_ring = np.zeros(n_ring, float)
    Av_ring = np.zeros(n_ring, float)
    z_ring = np.zeros(n_ring, float)
    dV_ring = np.zeros(n_ring, float)

    # Volume-weighted ring means
    for j in range(n_ring):
        m = valid_cell & (ring_idx == j)
        if not np.any(m):
            continue
        w = dV_grid[m]
        wsum = np.sum(w)
        dV_ring[j] = wsum
        Sigma_birth_ring[j] = np.sum(Sigma_birth[m] * w) / wsum
        mu_ring[j] = np.sum(mu_grid[m] * w) / wsum
        Av_ring[j] = np.sum(Av_grid[m] * w) / wsum
        z_ring[j] = np.sum(z_grid[m] * w) / wsum

    # Some rings might be empty; enforce small floor or zero volume
    empty = dV_ring <= 0
    Sigma_birth_ring[empty] = 0.0
    mu_ring[empty] = 0.0
    Av_ring[empty] = 0.0
    z_ring[empty] = 0.0

    # ---------- MASS QUADRATURE (Herbig range) ----------
    Mq = M["Mq"].astype(np.float64)    # (Q,)
    Wq = M["Wq"].astype(np.float64)    # (Q,)
    xi_q = M["xi_q"].astype(np.float64)  # (Q,)
    log10_Mq = np.log10(Mq)

    # ---------- STAR NODES (N, Kd/Km): use ring-averaged Σ_birth ----------
    birth_nodes_orig = S["birth_nodes"].astype(np.float64)  # original, but we'll override

    z_nodes = np.nan_to_num(S["z_nodes"].astype(np.float64),
                            nan=0.0, posinf=0.0, neginf=0.0)
    mu_nodes = np.nan_to_num(S["mu_nodes"].astype(np.float64),
                             nan=0.0, posinf=0.0, neginf=0.0)
    Av_nodes = np.nan_to_num(S["Av_nodes"].astype(np.float64),
                             nan=0.0, posinf=0.0, neginf=0.0)

    d_w = S["d_weights"].astype(np.float64)
    d_w = np.where(np.isfinite(d_w) & (d_w > 0.0), d_w, 1e-300)

    M_nodes = S["M_nodes"].astype(np.float64)
    M_nodes = np.where(np.isfinite(M_nodes) & (M_nodes > 0.0), M_nodes, 1.0)
    logM_nodes = np.log10(M_nodes)

    M_w = S["M_weights"].astype(np.float64)
    M_w = np.where(np.isfinite(M_w) & (M_w > 0.0), M_w, 1e-300)

    # IMF factor at star mass nodes (normalized on [0.08,120] Msun)
    xi_nodes = xi_norm_on_interval(M_nodes)

    # Star node positions to cylindrical R
    x_nodes = S["x_nodes"].astype(np.float64)
    y_nodes = S["y_nodes"].astype(np.float64)
    R_nodes = np.sqrt(x_nodes ** 2 + y_nodes ** 2)

    # Map nodes to rings
    ring_nodes_idx = np.digitize(R_nodes, R_edges) - 1  # [-1, n_ring-1]

    # Replace birth_nodes by ring-averaged Sigma_birth_ring
    birth_nodes = np.zeros_like(birth_nodes_orig, float)

    # default: zero outside rings
    for j in range(n_ring):
        m = (ring_nodes_idx == j)
        if np.any(m):
            birth_nodes[m] = Sigma_birth_ring[j]

    # Sanitize birth_nodes (avoid log(0))
    birth_nodes = np.where(
        np.isfinite(birth_nodes) & (birth_nodes > 0.0), birth_nodes, 1e-300
    )

    Av_grid_scaled = args.k_lambda * Av_grid
    Av_nodes_scaled = args.k_lambda * Av_nodes

    # ---- PyMC model ----
    with pm.Model() as model:
        # ----- External calibration constants (fixed) -----
        # Factor needed to bring the Σ_SFR map in line with Quintana+2025 SFR.
        # This is the "expected" scaling; prior below encodes uncertainty around it.
            # ----- External calibration constants (soft anchor only) -----
        sfr_scale_mean = 0.441  # expected calibration from Quintana+2025

        # ----- Priors -----

        # Lifetime at 1 Msun: very tight around 3 Myr (your strong prior)
        log10_tau0_Myr = pm.Normal(
            "log10_tau0_Myr",
            mu=np.log10(3.0),
            sigma=0.2,          # ≈ ±12% at 1σ
        )
        tau0_yr = (10.0 ** log10_tau0_Myr) * 1.0e6  # years

        # Mass dependence of lifetime: *must* decrease with mass over Herbig range
        # Use a truncated Normal to enforce beta < 0
        beta = pm.TruncatedNormal(
            "beta",
            mu=-2.5,
            sigma=1.0,
            upper=0.0,          # strictly β < 0
        )

        # Vertical scale height
        h_z_pc = pm.HalfNormal("h_z_pc", sigma=60.0)

        # ---------- Completeness / selection priors ----------

        # Intercept: logit(p_det) at (mu = mu_ref, A_V = 0, log10 M = 0)
        # logit(0.9) ≈ 2.2; keep it order unity but free.
        a0 = pm.Normal("a0", mu=2.0, sigma=1.0)

        # Distance-modulus slope: detection should *decrease* with distance,
        # so enforce a_mu < 0 via truncated Normal.
        a_mu = pm.TruncatedNormal(
            "a_mu",
            mu=-1.0,
            sigma=0.5,
            upper=0.0,          # strictly negative
        )

        # Extinction slope: should also be negative but we can leave the sign
        # soft (data should happily keep it < 0).
        a_Av = pm.TruncatedNormal(
            "a_Av",
            mu=-1.0,
            sigma=0.5,
            upper=0.0,          # strictly negative
        )
        # Mass slope: Herbigs are brighter → detection should increase with mass.
        # Enforce a_logM > 0.
        a_logM = pm.TruncatedNormal(
            "a_logM",
            mu=1.0,
            sigma=0.5,
            lower=0.0,          # strictly positive
        )

        # ---------- Global SFR (birth map) scale ----------

        # Allow the SFR map to be biased *high* or *low*.
        # We turn 0.441 into a *soft* log-normal anchor with ~1 dex freedom.
        s_birth_plus = pm.TruncatedNormal(
            "s_birth_pluss",
            mu=np.log(sfr_scale_mean),
            sigma=0.5,
            lower=0.0,
        )
        s_birth = pm.Deterministic("s_birth", sfr_scale_mean*(1.+ s_birth_plus))

        # z_sun prior (as you had)
        z_sun_pc = pm.Normal("z_sun_pc", mu=args.zsun_mu_pc, sigma=args.zsun_sigma_pc)


        def fz(z):
            return (1.0 / (SQ2PI * h_z_pc)) * pm.math.exp(-0.5 * (z / h_z_pc) ** 2)

        # ---------- Normalization integral Λ (ring-level) ----------

        # z at ring level including Sun offset
        z_ring_mid = pt.as_tensor_variable(z_ring) + z_sun_pc  # (n_ring,)
        fz_ring = fz(z_ring_mid)

        # selection term on rings
        mu_ring_t = pt.as_tensor_variable(mu_ring)
        Av_ring_scaled = args.k_lambda * pt.as_tensor_variable(Av_ring)

        s_ring = a0 + a_mu * (mu_ring_t - args.mu_ref) + a_Av * Av_ring_scaled  # (n_ring,)

        # shapes: s_ring (R,), log10_Mq (Q,)
        eta_rq = s_ring[:, None] + a_logM * pt.as_tensor_variable(log10_Mq)[None, :]
        pdet_rq = pm.math.sigmoid(eta_rq)  # (R,Q)

        # lifetime factor on mass grid (Q,)
        tau_q = tau0_yr * (pt.as_tensor_variable(Mq) ** beta)  # years
        mass_kernel_q = pt.as_tensor_variable(Wq * xi_q) * tau_q  # (Q,)

        mass_sum_ring = pt.sum(pdet_rq * mass_kernel_q[None, :], axis=1)  # (R,)

        # base factor per ring (stars/yr per ring, up to mass_sum)
        ring_base = (
            s_birth
            * pt.as_tensor_variable(Sigma_birth_ring)
            * fz_ring
            * pt.as_tensor_variable(dV_ring)
        )  # (R,)

        Lambda = pt.sum(ring_base * mass_sum_ring)  # scalar

        # ---------- Data term (star-by-star, marginalized) ----------

        # selection term on star nodes
        s_ik = (
            a0
            + a_mu * (pt.as_tensor_variable(mu_nodes) - args.mu_ref)
            + a_Av * pt.as_tensor_variable(Av_nodes_scaled)
        )  # (N,Kd)

        # broadcast with mass nodes
        eta_ikl = s_ik[:, :, None] + a_logM * pt.as_tensor_variable(
            logM_nodes
        )[:, None, :]  # (N,Kd,Km)

        # log p_det for numerical stability
        log_pdet_ikl = pm.math.log(pm.math.sigmoid(eta_ikl))  # (N,Kd,Km)

        # lifetime and IMF at star mass nodes
        tau_nodes_mass = tau0_yr * (pt.as_tensor_variable(M_nodes) ** beta)  # (N,Km)
        xi_nodes_t = pt.as_tensor_variable(xi_nodes)  # (N,Km)

        # safe positives
        birth_nodes_t = pt.clip(
            pt.as_tensor_variable(birth_nodes), 1e-300, np.inf
        )  # (N,Kd)
        z_nodes_mid = pt.as_tensor_variable(z_nodes) + z_sun_pc  # (N,Kd)
        fz_nodes = fz(z_nodes_mid)
        fz_nodes_t = pt.clip(fz_nodes, 1e-300, np.inf)  # (N,Kd)

        d_w_t = pt.clip(pt.as_tensor_variable(d_w), 1e-300, 1.0)  # (N,Kd)
        M_w_t = pt.clip(pt.as_tensor_variable(M_w), 1e-300, 1.0)  # (N,Km)

        # log λ at each (i,k,l)
        log_lambda_ikl = (
            pt.log(s_birth)
            + pt.log(birth_nodes_t)[:, :, None]
            + pt.log(fz_nodes_t)[:, :, None]
            + pt.log(xi_nodes_t)[:, None, :]
            + pt.log(tau_nodes_mass)[:, None, :]
            + log_pdet_ikl
        )  # (N,Kd,Km)

        # add log-weights and log-sum-exp over (k,l)
        log_w_ikl = pt.log(d_w_t)[:, :, None] + pt.log(M_w_t)[:, None, :]
        log_intensity_i = pm.math.logsumexp(
            log_lambda_ikl + log_w_ikl, axis=(1, 2)
        )  # (N,)

        # PPP log-likelihood
        pm.Potential("PPP_loglike", pt.sum(log_intensity_i) - Lambda)

        # -------- sampling (optional) --------
        idata = None
        if args.sample:
            idata = pm.sample(
                draws=args.draws,
                tune=args.tune,
                target_accept=args.target_accept,
                chains=4,
                cores=4,
                compute_convergence_checks=True,
                progressbar=True,
            )
            if args.out:
                import arviz as az

                az.to_netcdf(idata, args.out)
                print(f"[OK] saved trace to {args.out}")

    if not args.sample:
        print("[OK] Built ring-level PPP model graph successfully.")


if __name__ == "__main__":
    main()
