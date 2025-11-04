#!/usr/bin/env python3
# count_per_cell_3d.py
"""
Count Herbig stars per (HEALPix pixel i, radial shell k) cell using the same grid
as in the likelihood (from ppp_grid_pack.npz). Print cells with N>=minN (default 2).

Assumes the pack contains:
  - nside, npix
  - mu (npix, Kd) or d_pc (npix, Kd)  [we'll use d_pc if present for distances]
  - dr or delta_d_pc (optional; improves shell edges)
  - ell, bee (optional; Galactic lon/lat per pixel, deg)
"""

import argparse, os
import numpy as np
import healpy as hp
def load_pack(path):
    P = np.load(path, allow_pickle=True)
    nside = int(P["nside"]); npix = int(P["npix"])

    # Infer Kd from a 2D field if possible
    Kd = None
    for key in ("mu","z_pc","dV_pc3","A_V"):
        if key in P:
            arr = np.asarray(P[key])
            if arr.ndim == 2 and arr.shape[0] == npix:
                Kd = arr.shape[1]; break
    # Fallback: deduce Kd from flattened d_pc if needed
    if Kd is None and "d_pc" in P:
        flat = np.asarray(P["d_pc"])
        if flat.ndim == 1 and flat.size % npix == 0:
            Kd = flat.size // npix

    if Kd is None:
        raise ValueError("Cannot infer Kd; pack lacks a 2D field and d_pc is ambiguous.")

    # d_pc: reshape if flattened
    if "d_pc" in P:
        d_raw = np.asarray(P["d_pc"], float)
        d_pc  = d_raw.reshape(npix, Kd) if d_raw.ndim == 1 else d_raw
    else:
        mu = np.asarray(P["mu"], float)
        if mu.ndim == 1: mu = mu.reshape(npix, Kd)
        d_pc = 10.0**((mu - 5.0)/5.0)

    # dr / delta_d_pc: reshape if flattened
    dr = None
    for key in ("dr","delta_d_pc"):
        if key in P:
            tmp = np.asarray(P[key], float)
            dr  = tmp.reshape(npix, Kd) if (tmp.ndim == 1 and tmp.size == npix*Kd) else tmp
            if dr.ndim == 1 and dr.size == Kd:
                dr = np.broadcast_to(dr[None, :], (npix, Kd))
            break

    ell = np.asarray(P["ell"], float) if "ell" in P else None
    bee = np.asarray(P["bee"], float) if "bee" in P else None

    print(f"[pack] nside={nside} npix={npix} Kd={Kd}  d_pc.shape={d_pc.shape}")
    return dict(nside=nside, npix=npix, Kd=Kd, d_pc=d_pc, dr=dr, ell=ell, bee=bee)
def edges_from_centers(dc, dr_row=None):
    """
    dc: 1D centers (length K)
    dr_row: None (midpoints), scalar, or 1D array length K (per-shell widths)
    """
    dc = np.asarray(dc, float)
    K  = dc.size
    if dr_row is None:
        # midpoint edges
        edges = np.empty(K+1, float)
        edges[1:-1] = 0.5*(dc[:-1] + dc[1:])
        w0 = edges[1] - dc[0]
        w1 = dc[-1] - edges[-2]
        edges[0]  = max(0.0, dc[0] - w0)
        edges[-1] = dc[-1] + w1
        return np.maximum.accumulate(edges)
    # allow scalar or vector
    dr_row = np.asarray(dr_row, float)
    if dr_row.ndim == 0:
        dr_row = np.full(K, float(dr_row), float)
    elif dr_row.ndim == 1 and dr_row.size != K:
        raise ValueError(f"dr_row has length {dr_row.size}, expected {K}")
    e0 = dc - 0.5*dr_row
    e1 = dc + 0.5*dr_row
    edges = np.empty(K+1, float)
    edges[:-1] = e0
    edges[1:]  = np.maximum(e1, e0 + 1e-9)
    edges[0]   = max(0.0, edges[0])
    return np.maximum.accumulate(edges)



def read_catalog(path, col_ra, col_dec, col_dist, maybe_name="Name"):
    try:
        import pandas as pd
        sep = "\t" if path.lower().endswith(".tsv") else ","
        df = pd.read_csv(path, sep=sep)
        ra  = df[col_ra].to_numpy(float)
        dec = df[col_dec].to_numpy(float)
        d   = df[col_dist].to_numpy(float)
        names = df[maybe_name].astype(str).to_numpy() if maybe_name in df.columns else None
    except Exception:
        data = np.genfromtxt(path, names=True, dtype=None, encoding=None)
        ra  = np.asarray(data[col_ra], float)
        dec = np.asarray(data[col_dec], float)
        d   = np.asarray(data[col_dist], float)
        names = np.asarray(data[maybe_name], str) if maybe_name in data.dtype.names else None
    return ra, dec, d, names

def icrs_to_gal(ra_deg, dec_deg):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    sky = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs").galactic
    return sky.l.deg, sky.b.deg

def get_dr_row(dr, i, npix, K):
    """
    Normalize any dr to a row for pixel i:
    - None → None (midpoint edges)
    - scalar → scalar
    - (K,) → that row for all pixels
    - (npix,K) or flattened (npix*K,) → row i
    - (1,) → scalar
    """
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



def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--grid_pack", required=True)
    ap.add_argument("--catalog",   required=True)
    ap.add_argument("--col_ra",    default="ra")
    ap.add_argument("--col_dec",   default="dec")
    ap.add_argument("--col_dist",  default="distance", help="Distance column in pc")
    ap.add_argument("--minN",      type=int, default=2, help="Report cells with N >= minN")
    ap.add_argument("--use_names", action="store_true", help="Try to list star names per cell if present")
    ap.add_argument("--max_print", type=int, default=200, help="Maximum rows to print (sorted by N desc)")
    ap.add_argument("--save_tsv",  default="", help="Optional TSV output")
    args = ap.parse_args()

    G = load_pack(args.grid_pack)
    nside, npix, Kd = G["nside"], G["npix"], G["Kd"]
    print(f"[pack] nside={nside} npix={npix} Kd={Kd}")

    ra, dec, dist, names = read_catalog(args.catalog, args.col_ra, args.col_dec, args.col_dist)
    good = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(dist)
    ra, dec, dist = ra[good], dec[good], dist[good]
    if names is not None: names = names[good]
    # to Galactic & to pixel index
    l_deg, b_deg = icrs_to_gal(ra, dec)
    pix = hp.ang2pix(nside, l_deg, b_deg, lonlat=True)
    # prepare outputs
    counts = np.zeros((npix, Kd), dtype=np.int32)
    # Optionally store which names fell in a cell (sparse dict to keep memory)
    keep_names = args.use_names and (names is not None)
    cell_names = {} if keep_names else None

    # Precompute pixel lon/lat if not in pack
    if G["ell"] is None or G["bee"] is None:
        theta, phi = hp.pix2ang(nside, np.arange(npix), lonlat=False)
        ell_pix = np.degrees(phi); bee_pix = 90.0 - np.degrees(theta)
    else:
        ell_pix = G["ell"].astype(float); bee_pix = G["bee"].astype(float)

    # For each star, find (i,k) by distance
    d_centers = G["d_pc"]  # (npix, Kd)
    dr = G["dr"]           # (npix, Kd) or None

    for j in range(dist.size):
        i = pix[j]
        dc = d_centers[i]          # (K,)
        dr_row = get_dr_row(dr, i, G["npix"], Kd)
        edges = edges_from_centers(dc, dr_row)
        k = np.searchsorted(edges, dist[j], side="right") - 1
        if 0 <= k < Kd:
            counts[i, k] += 1
            if keep_names:
                cell_names.setdefault((i, k), []).append(names[j])

    # Find cells with N >= minN
    ii, kk = np.where(counts >= args.minN)
    Ns = counts[ii, kk]
    # sort by descending N
    order = np.argsort(-Ns)
    ii, kk, Ns = ii[order], kk[order], Ns[order]

    # Build table
    rows = []
    for idx in range(ii.size):
        i, k, N = int(ii[idx]), int(kk[idx]), int(Ns[idx])
        dc = d_centers[i]
        dr_row = get_dr_row(dr, i, G["npix"], Kd)
        edges  = edges_from_centers(dc, dr_row)
        dlo, dmid, dhi = edges[k], dc[k], edges[k+1]
        l0, b0 = float(ell_pix[i]), float(bee_pix[i])
        name_list = cell_names.get((i,k), []) if keep_names else []
        rows.append((N, i, k, l0, b0, dmid, dlo, dhi, name_list))

    # Print summary
    print(f"\n[cells with N >= {args.minN}]  total: {len(rows)}")
    header = "N  pix  k   l_deg   b_deg   d_mid(pc)   d_lo   d_hi"
    print(header)
    for r in rows[:args.max_print]:
        N,i,k,l0,b0,dmid,dlo,dhi,nlist = r
        print(f"{N:2d} {i:5d} {k:3d}  {l0:6.1f} {b0:6.1f}  {dmid:9.1f}  {dlo:6.1f} {dhi:6.1f}")
        if keep_names and nlist:
            print("    names:", ", ".join(nlist))

    if args.save_tsv:
        import pandas as pd
        df = pd.DataFrame(
            {
                "N": [r[0] for r in rows],
                "pixel": [r[1] for r in rows],
                "k": [r[2] for r in rows],
                "l_deg": [r[3] for r in rows],
                "b_deg": [r[4] for r in rows],
                "d_mid_pc": [r[5] for r in rows],
                "d_lo_pc": [r[6] for r in rows],
                "d_hi_pc": [r[7] for r in rows],
                "names": [", ".join(r[8]) if r[8] else "" for r in rows],
            }
        )
        df.to_csv(args.save_tsv, sep="\t", index=False)
        print(f"\n[OK] saved {os.path.abspath(args.save_tsv)}")

if __name__ == "__main__":
    main()
