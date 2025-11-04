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