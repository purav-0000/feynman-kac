import numpy as np
from scipy.stats import norm

def black_scholes_partial_t(S_array, K, T_to_maturity, r, sigma):
    """ Theta """
    S = np.asarray(S_array)
    tau = np.maximum(T_to_maturity, 1e-9)
    sqrt_tau = np.sqrt(tau)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau

    term1 = - (S * norm.pdf(d1) * sigma) / (2 * sqrt_tau)
    term2 = r * K * np.exp(-r * tau) * norm.cdf(d2)

    return term1 - term2


def black_scholes_partial_x(S_array, K, T_to_maturity, r, sigma):
    """ Delta """
    S = np.asarray(S_array)
    tau = np.maximum(T_to_maturity, 1e-9)
    sqrt_tau = np.sqrt(tau)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * sqrt_tau)
    return norm.cdf(d1)


def black_scholes_partial_xx(S_array, K, T_to_maturity, r, sigma):
    """ Gamma"""
    S = np.asarray(S_array)
    tau = np.maximum(T_to_maturity, 1e-9)
    sqrt_tau = np.sqrt(tau)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * sqrt_tau)
    pdf_d1 = norm.pdf(d1)

    # Safe guard for when S = 0
    denominator = S * sigma * sqrt_tau

    # Use np.divide to handle cases where the denominator is zero.
    gamma = np.divide(pdf_d1, denominator, out=np.zeros_like(pdf_d1), where=denominator != 0)

    return gamma