import numpy as np

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
    
    m_c, sigma, A = 0.086, 0.57, 0.22
    if np.any(hi):
        log10M = np.log10(M[hi])
        out[hi] = (A / (M[hi] * np.log(10.0))) * np.exp(-0.5 * ((log10M - np.log10(m_c))/sigma)**2)
    return out

def xi_norm_on_interval(M, Mmin=0.08, Mmax=120.0):
    """Normalized Chabrier pdf on [Mmin,Mmax] evaluated at M (numpy)."""
    x = np.logspace(np.log10(Mmin), np.log10(Mmax), 20000)
    y = chabrier2003_unnorm_pdf(x)
    Z = np.trapz(y, x)
    return chabrier2003_unnorm_pdf(M) / Z

def load_grid_layout(path, flatten=True):
    """
    Load a PPP grid pack in a shape-agnostic way and return a dict with
    *standardized* 1-D cellwise arrays (length G) for both ragged and rectangular packs.

    Always returned (keys):
      ragged: bool
      npix  : int
      G     : int  (total # of cells)
      row_ptr : (npix+1,) int64 — CSR-style row boundaries into flattened arrays
      mu, z, dV, Sigma_birth, Sigma_SFR, footprint : (G,) arrays
      Av     : (G,) array (zeros if missing)
      nside, adaptive_radial, max_per_cell : if present in pack
      delta_omega_sr : scalar or (G,) array if present

    Optionally (if present in the pack):
      d_lo, d_hi, d_mid : (G,) arrays (ragged) or None (rectangular if not derivable)
      x_pc, y_pc, d_pc  : (G,) arrays if present
    """
    import numpy as np
    P = np.load(path, allow_pickle=True)
    files = set(P.files)

    def _std_1d(name, arr, G):
        arr = np.asarray(arr)
        if arr.ndim == 0:
            arr = np.full(G, float(arr), dtype=float)
        elif arr.ndim > 1:
            arr = arr.reshape(-1)
        if arr.size != G:
            raise ValueError(f"[grid] {name} size {arr.size} != G {G}")
        return arr

    if "row_ptr" in files:
        # ---------- RAGGED / ADAPTIVE ----------
        row_ptr = np.asarray(P["row_ptr"], dtype=np.int64).ravel()
        npix = int(row_ptr.size - 1)
        G = int(row_ptr[-1])
        ragged = True

        # Required core fields (1-D length G)
        mu  = _std_1d("mu",   P["mu"],   G)
        z   = _std_1d("z_pc", P["z_pc"], G)
        dV  = _std_1d("dV_pc3", P["dV_pc3"], G)

        # Optional per-cell distances
        d_lo  = _std_1d("d_lo_pc",  P["d_lo_pc"],  G) if "d_lo_pc"  in files else None
        d_hi  = _std_1d("d_hi_pc",  P["d_hi_pc"],  G) if "d_hi_pc"  in files else None
        d_mid = _std_1d("d_mid_pc", P["d_mid_pc"], G) if "d_mid_pc" in files else None

        # Optional per-cell coordinates
        x_pc = _std_1d("x_pc", P["x_pc"], G) if "x_pc" in files else None
        y_pc = _std_1d("y_pc", P["y_pc"], G) if "y_pc" in files else None
        d_pc = _std_1d("d_pc", P["d_pc"], G) if "d_pc" in files else None

        # Optional A_V
        Av = _std_1d("A_V", P["A_V"], G) if "A_V" in files else np.zeros(G, dtype=float)

        # Standardize birth/SFR names
        Sigma_birth = _std_1d("Sigma_birth_yr_pc2", P["Sigma_birth_yr_pc2"], G) \
                        if "Sigma_birth_yr_pc2" in files \
                        else (_std_1d("Sigma_birth", P["Sigma_birth"], G) if "Sigma_birth" in files
                              else None)
        if Sigma_birth is None:
            raise KeyError("[grid] missing Sigma_birth (expect 'Sigma_birth_yr_pc2' or 'Sigma_birth').")

        Sigma_SFR = _std_1d("Sigma_SFR_Msun_yr_kpc2", P["Sigma_SFR_Msun_yr_kpc2"], G) \
                        if "Sigma_SFR_Msun_yr_kpc2" in files \
                        else (_std_1d("Sigma_SFR", P["Sigma_SFR"], G) if "Sigma_SFR" in files
                              else None)
        if Sigma_SFR is None:
            raise KeyError("[grid] missing Sigma_SFR (expect 'Sigma_SFR_Msun_yr_kpc2' or 'Sigma_SFR').")

        # Footprint may be per-pixel or per-cell
        fp = np.asarray(P["footprint"]).ravel().astype(bool)
        if fp.size == npix:
            counts = np.diff(row_ptr)
            fp = np.repeat(fp, counts)
        elif fp.size != G:
            raise ValueError(f"[grid] footprint size {fp.size} incompatible with G={G} or npix={npix}")

        # Meta
        out = dict(
            ragged=ragged, npix=npix, G=G, row_ptr=row_ptr,
            mu=mu, z=z, dV=dV, Av=Av, Sigma_birth=Sigma_birth, Sigma_SFR=Sigma_SFR,
            footprint=fp,
            d_lo=d_lo, d_hi=d_hi, d_mid=d_mid,
            x_pc=x_pc, y_pc=y_pc, d_pc=d_pc,
        )
    else:
        # ---------- RECTANGULAR ----------
        if "npix" not in files:
            raise ValueError("[grid] rectangular pack missing 'npix'")
        npix = int(P["npix"])
        ragged = False

        def as2d(name, required=True, dtype=float):
            if name not in P and not required:
                return None
            A = np.asarray(P[name], dtype=dtype)
            if A.ndim == 1:
                if A.size % npix != 0:
                    raise ValueError(f"[grid] {name} 1-D size {A.size} not divisible by npix={npix}")
                Kd = A.size // npix
                return A.reshape(npix, Kd)
            if A.ndim == 2 and A.shape[0] == npix:
                return A
            raise ValueError(f"[grid] {name} has unexpected shape {A.shape}")

        mu   = as2d("mu")
        z    = as2d("z_pc")
        dV   = as2d("dV_pc3")
        # Prefer d_pc if present (distance centers), otherwise None
        d_pc = as2d("d_pc", required=False)

        Kd = mu.shape[1]
        G  = npix * Kd

        # Footprint may be (npix,) or (npix,Kd)
        fp = np.asarray(P["footprint"])
        if fp.ndim == 1 and fp.size == npix:
            fp = np.repeat(fp[:, None], Kd, axis=1)
        elif fp.ndim == 2 and fp.shape != (npix, Kd):
            raise ValueError(f"[grid] footprint shape {fp.shape} incompatible with (npix,Kd)=({npix},{Kd})")
        fp = fp.astype(bool)

        # A_V optional
        Av2d = as2d("A_V", required=False)
        if Av2d is None:
            Av2d = np.zeros_like(mu)

        # Birth & SFR names normalized
        if "Sigma_birth_yr_pc2" in P:
            SigB2d = as2d("Sigma_birth_yr_pc2")
        elif "Sigma_birth" in P:
            SigB2d = as2d("Sigma_birth")
        else:
            raise KeyError("[grid] missing Sigma_birth (expect 'Sigma_birth_yr_pc2' or 'Sigma_birth').")

        if "Sigma_SFR_Msun_yr_kpc2" in P:
            SFR2d = as2d("Sigma_SFR_Msun_yr_kpc2")
        elif "Sigma_SFR" in P:
            SFR2d = as2d("Sigma_SFR")
        else:
            raise KeyError("[grid] missing Sigma_SFR (expect 'Sigma_SFR_Msun_yr_kpc2' or 'Sigma_SFR').")

        # Flatten to 1-D (row-major) for uniform downstream use
        mu  = mu.reshape(-1)
        z   = z.reshape(-1)
        dV  = dV.reshape(-1)
        Av  = Av2d.reshape(-1)
        Sigma_birth = SigB2d.reshape(-1)
        Sigma_SFR   = SFR2d.reshape(-1)
        footprint   = fp.reshape(-1)

        # Synthesize row_ptr for rectangular grid (equal Kd per pixel)
        row_ptr = (np.arange(npix + 1, dtype=np.int64) * Kd)

        out = dict(
            ragged=ragged, npix=npix, G=G, row_ptr=row_ptr,
            mu=mu, z=z, dV=dV, Av=Av, Sigma_birth=Sigma_birth, Sigma_SFR=Sigma_SFR,
            footprint=footprint,
            d_lo=None, d_hi=None, d_mid=None,
            x_pc=None, y_pc=None, d_pc=(d_pc.reshape(-1) if d_pc is not None else None),
        )

    # ----- common optional meta -----
    for meta_key in ["nside", "adaptive_radial", "max_per_cell"]:
        if meta_key in P:
            out[meta_key] = P[meta_key].item() if np.asarray(P[meta_key]).ndim == 0 else P[meta_key]

    # delta_omega_sr may be scalar or per-cell
    if "delta_omega_sr" in P:
        val = np.asarray(P["delta_omega_sr"])
        if val.ndim == 0:
            out["delta_omega_sr"] = float(val)
        else:
            out["delta_omega_sr"] = _std_1d("delta_omega_sr", val, out["G"])

    return out


def edges_from_centers(dc, dr_row=None):
    """dc: (K,), dr_row: None (midpoints) or scalar or (K,) widths."""
    dc = np.asarray(dc, float); K = dc.size
    if dr_row is None:
        e = np.empty(K+1, float)
        e[1:-1] = 0.5*(dc[:-1] + dc[1:])
        w0 = e[1] - dc[0]; w1 = dc[-1] - e[-2]
        e[0] = max(0.0, dc[0] - w0); e[-1] = dc[-1] + w1
        return np.maximum.accumulate(e)
    w = np.asarray(dr_row, float)
    if w.ndim == 0:
        w = np.full(K, float(w))
    elif w.ndim == 1 and w.size != K:
        raise ValueError(f"dr_row length {w.size} != K={K}")
    e0 = dc - 0.5*w; e1 = dc + 0.5*w
    e = np.empty(K+1, float); e[:-1] = e0; e[1:] = np.maximum(e1, e0 + 1e-9)
    e[0] = max(0.0, e[0])
    return np.maximum.accumulate(e)

def get_dr_row(dr, i, npix, K):
    """Normalize any stored dr to a row for pixel i (rectangular grids)."""
    if dr is None:
        return None
    arr = np.asarray(dr)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        if arr.size == K:
            return arr
        if arr.size == npix*K:
            return arr.reshape(npix, K)[i]
        if arr.size == 1:
            return float(arr[0])
        return None
    if arr.ndim == 2 and arr.shape == (npix, K):
        return arr[i]
    return None
