import numpy as np
from scipy.stats import norm

def adjusted_volatility(sigma, n, T):
    alpha = T / n
    sigma_adj = sigma**2 / n * (2 * np.exp(alpha) - 1) / 3
    return np.sqrt(sigma_adj)

def d1(S0, K, r, q, sigma_adj, T):
    return (np.log(S0 / K) + (r - q + sigma_adj**2 / 2) * T) / (sigma_adj * np.sqrt(T))

def d2(d1, sigma_adj, T):
    return d1 - sigma_adj * np.sqrt(T)

def geometric_asian_option_price(S0, K, T, r, q, sigma, n, option_type='call'):
    sigma_adj = adjusted_volatility(sigma, n, T)
    d1_val = d1(S0, K, r, q, sigma_adj, T)
    d2_val = d2(d1_val, sigma_adj, T)
    
    if option_type == 'call':
        price = np.exp(-r * T) * (S0 * np.exp((r - q + sigma_adj**2 / 2) * T) * norm.cdf(d1_val) - K * norm.cdf(d2_val))
    else:  # put option TO BE DONE
        print('Put options not implemented yet')
    
    return price