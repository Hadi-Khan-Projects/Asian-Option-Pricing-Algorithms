import numpy as np
from scipy.fft import fft
from scipy.interpolate import interp1d

def characteristic_exponent_de(omega, Delta, sigma, lam, p, eta1, eta2):
    return -0.5 * sigma**2 * omega**2 * Delta + lam * Delta * ((1 - p) * eta2 / (eta2 + 1j*omega) + p * eta1 / (eta1 - 1j*omega) - 1)

def characteristic_function_de(omega, S0, m, n, Delta, sigma, lam, p, eta1, eta2):

    return np.exp(1j * omega * (np.log(S0) + m * Delta * n / 2) + np.sum([characteristic_exponent_de(omega * (n - k + 1) / (n + 1), Delta, sigma, lam, p, eta1, eta2) for k in range(1, n+1)], axis=0))

def GEOMETRIC_price_fft_de(S0, K, T, r, q, sigma, lam, p, eta1, eta2, option_type, n, damping, N, eta):
    # STEP 0: CALCULATE NECESSARY CONSTANTS
    b = np.pi * eta
    spacing = (2*np.pi)/(N*eta)  # Spacing of log-strikes
    logK = np.log(K)  # Log of the strike price
    Delta = T / n
    discount = np.exp(-r * T)

    # STEP 2: FFT GRID SETUP
    grid_points = np.arange(1, N+1) # FFT Grid points
    frequencies = eta * (grid_points - 1)  # Frequencies in FFT space
    log_strikes = -b + spacing * (grid_points - 1)

    # STEP 3: CALCULATE DRIFT TERM
    m = (r - q) - characteristic_exponent_de(-1.0j, Delta, sigma, lam, p, eta1, eta2) / Delta

    # STEP 4: ADJUSTED CHARACTERISTIC FUNCTION WITH DAMPING
    def adjusted_characteristic_func(frequency):
        cf = characteristic_function_de(frequency - (damping + 1) * 1j, S0, m, n, Delta, sigma, lam, p, eta1, eta2)
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
        price_at_logK = price_at_logK - S0 * np.exp(-q * T) + K * np.exp(-r * T)

    return price_at_logK