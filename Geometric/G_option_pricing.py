import numpy as np
from scipy.fft import fft, ifft

# param - xi: numpy array of complex numbers
# param - t: time
# param - mu: drift rate
# param - sigma: volatility
# return - numpy array of complex numbers
def characteristic_gaussian(xi, t, mu=0, sigma=0.4):
    return np.exp(1j * mu * xi * t - 0.5 * sigma**2 * xi**2 * t)

# param - xi: numpy array of complex numbers
# param - t: time
# param - theta: drift
# param - sigma: volatility
# param - nu: variance gamma
# return - numpy array of complex numbers
def characteristic_variance_gamma(xi, t, theta=1/9, sigma=1/(3*np.sqrt(3)), nu=0.25):
    return ((1 - 1j * theta * nu * xi + 0.5 * sigma**2 * nu * xi**2)**(-t/nu))

# param - xi: numpy array of complex numbers
# param - t: time
# param - mu_j: jump size
# param - sigma: volatility
# param - lambda_param: jump intensity
# param - sigma_j: jump size volatility
# return - numpy array of complex numbers
def characteristic_merton_jump_diffusion(xi, t, mu_j=-0.05, sigma=0.1, lambda_param=3, sigma_j=0.086):
    m1 = np.exp(1j * mu_j * xi - 0.5 * sigma_j**2 * xi**2)
    return np.exp(-0.5 * sigma**2 * xi**2 * t) * np.exp(lambda_param * (m1 - 1) * t)


# Monte Carlo pricing via the Gaussian (normal random) process.
# params - see GeometricAsianOption class definition
# return - option price
def price_monte_carlo(S0, K, T, r, q, sigma, n, option_type, n_paths):
    dt = T / n  # Time step
    drift = (r - q - 0.5 * sigma ** 2) * dt # mu
    vol_step = sigma * np.sqrt(dt) # sigma * sqrt of time step
    
    # STEP 1: Generate random paths, (n_paths x n dates) of standard normal random variables
    Z = np.random.normal(0, 1, (n_paths, n))
    
    # STEP 2: Calculate log(price increments) with correct drift and volatility adjustment
    log_increments = drift + vol_step * Z
    
    # STEP 3: Calculate log(prices) = cumulative sum of log(price increments)
    log_prices = np.log(S0) + np.cumsum(log_increments, axis=1)
    
    # STEP 4: Convert log(prices) to prices via exponentiation
    prices = np.exp(log_prices)
    
    # STEP 5: Prepend S0 to each path, then calculate geometric mean for each path
    prices_with_S0 = np.hstack((np.full((n_paths, 1), S0), prices))
    geometric_means = np.exp(np.mean(np.log(prices_with_S0), axis=1))
    
    # STEP 6: Calculate payoff based on option type
    if option_type == 'call':
        # payoff = geometric mean of stock prices - strike price (or 0, whichever is higher)
        payoffs = np.maximum(geometric_means - K, 0)
    else: 
        # payoff = strike price - geometric mean of stock prices (or 0, whichever is higher) 
        payoffs = np.maximum(K - geometric_means, 0)
    
    # STEP 7: Present value of option = average payoff for all paths * risk-free rate discount
    option_price = np.mean(payoffs) * np.exp(-r * T) 
    
    return option_price

# FFT pricing via Gaussian characteristic function 
# params - see GeometricAsianOption class definition
# return - option price
def price_fft_gaussian(S0, K, T, r, q, sigma, n, option_type, N, d):
    return





