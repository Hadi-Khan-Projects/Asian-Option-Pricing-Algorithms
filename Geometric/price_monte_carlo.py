import numpy as np

# Monte Carlo pricing via the Gaussian (normal random) process.
# params - see Option class definition
# return - option price
def GEOMETRIC_price_monte_carlo(S0, K, T, r, q, sigma, n, option_type, n_paths):
    # STEP 0: Calculate necessary constants
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

    # STEP 7: Risk-free rate discount
    discount = np.exp(-r * T) 
    
    # STEP 8: Present value of option = average payoff for all paths * discount
    option_price = np.mean(payoffs) * np.exp(-r * T) 
    
    return option_price




