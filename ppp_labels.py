# ppp_labels.py
# Central mapping from variable names in your trace -> LaTeX-friendly labels (mathtext).
# Use these names exactly as they appear in your PyMC posterior (idata.posterior).

LATEX_LABELS = {
    # Lifetime parameterizations
    "log10_tau_p":      r"\log_{10}\,\tau_p\ [{\rm yr}]",
    "log10_tau0_Myr":   r"\log_{10}\,\tau_0\ [{\rm Myr}]",

    # Lifetime slope
    "beta":             r"\beta",

    # Vertical scale height
    "h_z_pc":           r"h_z\ [{\rm pc}]",

    # Selection coefficients
    "a0":               r"a_0",
    "a_mu":             r"a_\mu",
    "a_Av":             r"a_{A_V}",
    "a_logM":           r"a_{\log M}",

    # Global birth-rate scale
    "s_birth":          r"s_{\rm birth}",

    # Optional derived lifetime summaries (script adds these when requested)
    "log10_tau@1.5Msun": r"\log_{10}\,\tau(1.5\,M_\odot)\ [{\rm yr}]",
    "log10_tau@2.5Msun": r"\log_{10}\,\tau(2.5\,M_\odot)\ [{\rm yr}]",
    "log10_tau@8.0Msun": r"\log_{10}\,\tau(8.0\,M_\odot)\ [{\rm yr}]",
}
