import numpy as np

# Monte Carlo pricing via the Gaussian (normal random) process.
# params - see Option class definition
# return - option price
def EUROPEAN_price_monte_carlo(S0, K, T, r, q, sigma, option_type, n_paths):
    # STEP 0: Calculate necessary constants
    dt = T  # For European option, we only need the price at maturity
    drift = (r - q - 0.5 * sigma ** 2) * dt
    vol_step = sigma * np.sqrt(dt)
    
    # STEP 1: Generate random paths, array (n_paths) of standard normal random variables
    Z = np.random.normal(0, 1, n_paths)
    
    # STEP 2: Calculate log(price increments) with correct drift and volatility adjustment
    log_increments = drift + vol_step * Z
    
    # STEP 3: Calculate log(prices) at maturity
    log_prices = np.log(S0) + log_increments
    
    # STEP 4: Convert log(prices) at maturity to prices via exponentiation
    prices_at_maturity = np.exp(log_prices)
    
    # STEP 5: Calculate payoff based on option type
    if option_type == 'call':
        # payoff for call option = stock price at maturity - strike price (or 0, whichever is higher)
        payoffs = np.maximum(prices_at_maturity - K, 0)
    else:
        # payoff for put option = strike price - stock price at maturity (or 0, whichever is higher)
        payoffs = np.maximum(K - prices_at_maturity, 0)
    
    # STEP 6: Risk-free rate discount
    discount = np.exp(-r * T)
    
    # STEP 7: Present value of option = average payoff for all paths * discount
    option_price = np.mean(payoffs) * discount
    
    return option_price
