import numpy as np
from scipy.fft import fft
from scipy.interpolate import interp1d

# FFT pricing
# param - characteristic: the characteristic function
# params - see Option class definition
# return - option price
def GEOMETRIC_price_fft(S0, K, T, r, q, sigma, option_type, n, damping, N, eta, characteristic):
    # STEP 0: CALCULATE NECESSARY CONSTANTS
    b = np.pi * eta
    spacing = (2*np.pi)/(N*eta)  # Spacing of log-strikes
    logK = np.log(K)  # Log of the strike price
    logS0 = np.log(S0)  # Log of the initial stock price

    # STEP 1: DISCOUNT FACTOR
    discount = np.exp(-r * T)

    # STEP 2: FFT GRID SETUP
    grid_points = np.arange(1, N) # FFT Grid points
    frequencies = eta * (grid_points - 1)  # Frequencies in FFT space
    log_strikes = -b + spacing * (grid_points - 1)

    # STEP 3: SELECT CHARACTERISTIC FUNCTIONS
    if characteristic == "black scholes":
        def characteristic_func(frequency): # BLACK SCHOLES CHARACTERISTIC
            sigma_adj = sigma * np.sqrt((1/3) - (1/(2*n)) + (1/(6*n**2)))
            mu_adj = (r - q - 0.5 * sigma**2) * (0.5 - (1/(6*n))) + 0.5 * sigma_adj**2
            return np.exp(1j * frequency * (logS0 + (mu_adj - 0.5 * sigma_adj**2) * T) - 0.5 * sigma_adj**2 * T * frequency**2)
    elif characteristic == "gaussian":
        def characteristic_func(frequency): # GAUSSIAN CHARACTERISTIC
            sigma_adj = 0.4
            return np.exp(-0.5 * sigma_adj**2 * frequency**2 * T)
    elif characteristic == "variance gamma":
        def characteristic_func(frequency): # VARIANCE GAMMA CHARACTERISTIC
            sigma_adj = 3 * np.sqrt(3) / 5
            theta=1/9
            nu=0.25
            return (1 - 1j * theta * nu * frequency + 0.5 * nu * sigma_adj**2 * frequency**2)**(-T / nu)
    else:
        print("WARNING: INVALID CHARFUNCTION")
        return None

    # STEP 4: ADJUSTED CHARACTERISTIC FUNCTION WITH DAMPING
    def adjusted_characteristic_func(frequency):
        cf = characteristic_func(frequency - (damping + 1) * 1j)
        numerator = discount * cf
        denominator = damping**2 + damping - frequency**2 + 1j * (2 * damping + 1) * frequency
        return numerator / denominator

    # STEP 5: APPLYING FFT
    adjusted_values = adjusted_characteristic_func(frequencies) 
    weights = 3 + (-1)**grid_points - np.where(grid_points == 1, 1, 0) #Simpson's rule weights
    fft_input = eta * np.exp(1j * b * frequencies) * adjusted_values * weights / 3
    transformed_log_strikes = np.real(fft(fft_input) * np.exp(-damping * log_strikes) / np.pi)

    # STEP 6: INTERPOLATION AND PRICING
    f = interp1d(log_strikes, transformed_log_strikes)
    price_at_logK = np.real(f(logK))
    if option_type == 'put': # Adjusting for put prices if needed
        price_at_logK = price_at_logK - (S0 * np.exp(-q * T) - K * discount)
    
    return price_at_logK
