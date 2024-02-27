import numpy as np

def generate_asset_paths(S0, T, r, q, sigma, n, num_paths):
    dt = T/n
    paths = np.zeros((num_paths, n+1))
    paths[:, 0] = S0
    for t in range(1, n+1):
        z = np.random.standard_normal(num_paths)
        paths[:, t] = paths[:, t-1] * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return paths

def arithmetic_asian_option_price(S0, K, T, r, q, sigma, n, option_type, num_paths):
    paths = generate_asset_paths(S0, T, r, q, sigma, n, num_paths)
    payoffs = np.zeros(num_paths)

    for i in range(num_paths):
        average_price = np.mean(paths[i, 1:])  # Skip the initial price
        if option_type == 'call':
            payoffs[i] = max(average_price - K, 0)
        else:  # Put option
            payoffs[i] = max(K - average_price, 0)

    discount_factor = np.exp(-r * T)
    price = discount_factor * np.mean(payoffs)
    return price