#!/usr/bin/env python3
"""
PPP likelihood for Herbig stars with distance+mass marginalization.

Inputs:
  --grid_pack        : from build_ppp_all_packs.py (HEALPix×distance cells)
  --mass_pack        : Chabrier IMF info (Herbig mass quadrature) + <M>
  --star_marg_pack   : per-star distance & mass nodes/weights, A_V(d), and Σ_birth at nodes
Options:
  --mu_ref           : center for distance-modulus term in selection (default 10)
  --k_lambda         : A_band / A_V coefficient for selection (default 1.0 uses A_V)
  --sample           : set to run NUTS sampling
  --draws, --tune    : sampling hyperparameters
  --out              : netcdf path to save trace (optional)
"""

import argparse
import numpy as np

# PyMC 5 / PyTensor
import pymc as pm
import pytensor.tensor as pt

# ---------- helpers ----------
SQ2PI = np.sqrt(2.0 * np.pi)

def chabrier2003_unnorm_pdf(M):
    """Unnormalized dN/dM (linear mass)."""
    M = np.asarray(M, float)
    out = np.zeros_like(M)
    ok = M > 0
    if not np.any(ok):
        return out
    lo = ok & (M <= 1.0); hi = ok & (M > 1.0)
    # lognormal (base-10) below 1 Msun
    m_c, sigma, A = 0.079, 0.69, 0.158
    if np.any(lo):
        log10M = np.log10(M[lo])
        out[lo] = (A / (M[lo] * np.log(10.0))) * np.exp(-0.5 * ((log10M - np.log10(m_c))/sigma)**2)
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
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--grid_pack", required=True)
    ap.add_argument("--mass_pack", required=True)
    ap.add_argument("--star_marg_pack", required=True)
    ap.add_argument("--mu_ref", type=float, default=10.0)
    ap.add_argument("--k_lambda", type=float, default=1.0)
    ap.add_argument("--sample", action="store_true", help="Run NUTS sampling")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--target_accept", type=float, default=0.9)
    ap.add_argument("--out", default="", help="If set, save trace to this netcdf file")
    args = ap.parse_args()

    # ---- load packs ----
    G = np.load(args.grid_pack, allow_pickle=True)
    M = np.load(args.mass_pack, allow_pickle=True)
    S = np.load(args.star_marg_pack, allow_pickle=True)

    # Grid (flattened, length = npix*Nshell)
    Sigma_birth = G["Sigma_birth_yr_pc2"].astype(np.float64)   # stars/yr/pc^2
    z_grid      = G["z_pc"].astype(np.float64)                 # pc
    mu_grid     = G["mu"].astype(np.float64)                   # mag
    dV_grid     = G["dV_pc3"].astype(np.float64)               # pc^3
    footprint   = G["footprint"].astype(bool)
    if "A_V" in G.files:
        Av_grid = G["A_V"].astype(np.float64)
    else:
        Av_grid = np.zeros_like(mu_grid)

    # Mass quadrature (Herbig range) for normalization integral
    Mq    = M["Mq"].astype(np.float64)            # (Q,)
    Wq    = M["Wq"].astype(np.float64)            # (Q,)
    xi_q  = M["xi_q"].astype(np.float64)          # (Q,) normalized on Herbig sub-range
    # Note: xi_q here is for the normalization integral only.
    log10_Mq = np.log10(Mq)

    # Star marginalization pack
    birth_nodes = S["birth_nodes"].astype(np.float64)  # (N, Kd) stars/yr/pc^2
    z_nodes     = S["z_nodes"].astype(np.float64)      # (N, Kd)
    mu_nodes    = S["mu_nodes"].astype(np.float64)     # (N, Kd)
    Av_nodes    = S["Av_nodes"].astype(np.float64)     # (N, Kd)
    d_w         = S["d_weights"].astype(np.float64)    # (N, Kd)

    M_nodes     = S["M_nodes"].astype(np.float64)      # (N, Km)
    logM_nodes  = S["logM_nodes"].astype(np.float64)   # (N, Km)
    M_w         = S["M_weights"].astype(np.float64)    # (N, Km)

    # IMF factor at the star mass nodes (normalized on [0.08,120] Msun)
    xi_nodes = xi_norm_on_interval(M_nodes)            # (N, Km), numpy constant

    N, Kd = birth_nodes.shape
    Km = M_nodes.shape[1]
    Q  = Mq.size

    # Cast masks & constants
    footprint_f = footprint.astype(np.float64)  # 1.0 where valid, 0.0 else
    Av_grid_scaled = args.k_lambda * Av_grid
    Av_nodes_scaled = args.k_lambda * Av_nodes

    # ---- PyMC model ----
    with pm.Model() as model:
        # ----- Priors -----
        log10_tau0_Myr = pm.Normal("log10_tau0_Myr", mu=np.log10(3.0), sigma=1.0)
        beta           = pm.Normal("beta", mu=-0.7, sigma=0.7)
        h_z_pc         = pm.HalfNormal("h_z_pc", sigma=60.0)

        # selection coefficients
        a0     = pm.Normal("a0", mu=0.0, sigma=3.0)
        a_mu   = pm.Normal("a_mu", mu=-0.5, sigma=0.5)
        a_Av   = pm.Normal("a_Av", mu=-1.0, sigma=0.5)
        a_logM = pm.Normal("a_logM", mu=+1.0, sigma=0.5)

        # global scale for birth map (absorbs Σ_SFR→Σ_birth calibration etc.)
        s_birth = pm.LogNormal("s_birth", mu=0.0, sigma=1.0)

        # ----- Deterministic components -----
        tau0_yr = (10.0 ** log10_tau0_Myr) * 1.0e6  # years

        # Vertical profile fz on grid and star nodes
        def fz(z):
            return (1.0 / (SQ2PI * h_z_pc)) * pm.math.exp(-0.5 * (z / h_z_pc) ** 2)

        fz_grid = fz(pt.as_tensor_variable(z_grid))
        fz_nodes = fz(pt.as_tensor_variable(z_nodes))  # (N, Kd)

        # ---------- Normalization integral Λ ----------
        # cell base factor (stars/yr per cell, up to the mass sum and s_birth)
        cell_base = s_birth * pt.as_tensor_variable(Sigma_birth) * fz_grid * pt.as_tensor_variable(dV_grid)
        cell_base = cell_base * pt.as_tensor_variable(footprint_f)  # zero outside support

        # selection term over mass grid (broadcast add)
        s_g = a0 + a_mu * (pt.as_tensor_variable(mu_grid) - args.mu_ref) + a_Av * pt.as_tensor_variable(Av_grid_scaled)
        # shapes: s_g (G,), log10_Mq (Q,)
        eta_gq = s_g[:, None] + a_logM * pt.as_tensor_variable(log10_Mq)[None, :]
        pdet_gq = pm.math.sigmoid(eta_gq)  # (G,Q)

        # lifetime factor on mass grid (Q,)
        tau_q = tau0_yr * (pt.as_tensor_variable(Mq) ** beta)  # years
        mass_kernel_q = pt.as_tensor_variable(Wq * xi_q) * tau_q  # (Q,)

        mass_sum_grid = pt.sum(pdet_gq * mass_kernel_q[None, :], axis=1)  # (G,)

        Lambda = pt.sum(cell_base * mass_sum_grid)  # scalar

        # ---------- Data term (star-by-star, marginalized) ----------
        # selection term on star nodes
        s_ik = a0 + a_mu * (pt.as_tensor_variable(mu_nodes) - args.mu_ref) + a_Av * pt.as_tensor_variable(Av_nodes_scaled)  # (N,Kd)
        # broadcast with mass nodes
        eta_ikl = s_ik[:, :, None] + a_logM * pt.as_tensor_variable(logM_nodes)[:, None, :]  # (N,Kd,Km)

        # log p_det for numerical stability
        log_pdet_ikl = pm.math.sigmoid(eta_ikl)  # (N,Kd,Km)

        # lifetime and IMF at star mass nodes
        tau_nodes_mass = tau0_yr * (pt.as_tensor_variable(M_nodes) ** beta)       # (N,Km)
        xi_nodes_t = pt.as_tensor_variable(xi_nodes)                               # (N,Km)

        # safe positives
        birth_nodes_t = pt.clip(pt.as_tensor_variable(birth_nodes), 1e-300, np.inf)  # (N,Kd)
        fz_nodes_t    = pt.clip(fz_nodes, 1e-300, np.inf)                            # (N,Kd)
        d_w_t         = pt.clip(pt.as_tensor_variable(d_w), 1e-300, 1.0)            # (N,Kd)
        M_w_t         = pt.clip(pt.as_tensor_variable(M_w), 1e-300, 1.0)            # (N,Km)

        # log λ at each (k,l)
        log_lambda_ikl = (
            pt.log(s_birth) +
            pt.log(birth_nodes_t)[:, :, None] +
            pt.log(fz_nodes_t)[:, :, None] +
            pt.log(xi_nodes_t)[:, None, :] +
            pt.log(tau_nodes_mass)[:, None, :] +
            log_pdet_ikl
        )  # (N,Kd,Km)

        # add log-weights and log-sum-exp over (k,l)
        log_w_ikl = pt.log(d_w_t)[:, :, None] + pt.log(M_w_t)[:, None, :]
        log_intensity_i = pm.math.logsumexp(log_lambda_ikl + log_w_ikl, axis=(1, 2))  # (N,)

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

    # If not sampling, just build the model to catch shape errors
    if not args.sample:
        print("[OK] Built PPP model graph successfully.")

if __name__ == "__main__":
    main()
