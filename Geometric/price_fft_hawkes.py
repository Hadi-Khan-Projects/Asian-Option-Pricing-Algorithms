import numpy as np
from scipy.fft import fft
from scipy.interpolate import interp1d
from scipy.special import gamma

def characteristic_exponent_hawkes(omega, Delta, sigma, lambda0, alpha, beta, mu, rho):
    # Compute the characteristic exponent of the Hawkes process-driven jump-diffusion model
    omega_complex = omega * (np.cos(rho) - 1j * np.sin(rho))
    return -0.5 * sigma**2 * omega**2 * Delta + lambda0 * Delta * (np.exp(1j * omega * mu - beta * Delta) / (beta - alpha * np.exp(1j * omega_complex * mu)) - 1)

def characteristic_function_hawkes(omega, S0, m, n, Delta, sigma, lambda0, alpha, beta, mu, rho):
    # Compute the characteristic function of the log-price under the Hawkes process-driven jump-diffusion model
    return np.exp(1j * omega * (np.log(S0) + m * Delta * n / 2) + np.sum([characteristic_exponent_hawkes(omega * (n - k + 1) / (n + 1), Delta, sigma, lambda0, alpha, beta, mu, rho) for k in range(1, n+1)], axis=0))

# Reverse engineered parameters for Hawkes Jump Diffusion (parameters calibrated to Merton Jump Diffusion model):
lambda0 = 0.0296 # Initial jump intensity
alpha = -0.6033  # Self-excitation parameter
beta = 0.4398 # Jump intensity mean-reversion speed
mu = 0.2096  # Mean jump size
rho = 2.3500 # Correlation between jump sizes and jump times

def GEOMETRIC_price_fft_hawkes(S0, K, T, r, q, sigma, lambda0, alpha, beta, mu, rho, option_type, n, damping, N, eta):
    # STEP 0: CALCULATE NECESSARY CONSTANTS
    b = np.pi * eta
    spacing = (2*np.pi)/(N*eta)  # Spacing of log-strikes
    logK = np.log(K)  # Log of the strike price
    Delta = T / n # Time increment
    discount = np.exp(-r * T)

    # STEP 2: FFT GRID SETUP
    grid_points = np.arange(1, N+1) # FFT Grid points
    frequencies = eta * (grid_points - 1)  # Frequencies in FFT space
    log_strikes = -b + spacing * (grid_points - 1)

    # STEP 3: CALCULATE DRIFT TERM
    m = (r - q) - characteristic_exponent_hawkes(-1j, Delta, sigma, lambda0, alpha, beta, mu, rho) / Delta

    # STEP 4: ADJUSTED CHARACTERISTIC FUNCTION WITH DAMPING
    def adjusted_characteristic_func(frequency):
        cf = characteristic_function_hawkes(frequency - (damping + 1) * 1j, S0, m, n, Delta, sigma, lambda0, alpha, beta, mu, rho)
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