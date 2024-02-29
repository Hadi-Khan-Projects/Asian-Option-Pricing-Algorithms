import numpy as np
from scipy.fft import fft
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------
# EUROPEAN OPTION PARAMETERS
# ---------------------------------------------------------------------
S0 = 100  # Initial stock price
K = 105  # Strike price
T = 1  # Time to maturity in years
r = 0.05  # Risk-free interest rate
q = 0.02  # Continuous dividend yield
sigma = 0.4  # Volatility of the underlying asset
option_type = 'call'  # 'call' or 'put'

# ---------------------------------------------------------------------
# PRICING PARAMETERS
# ---------------------------------------------------------------------
damping = 1.5  # Damping factor from Carr and Madan (1999)
N = 4096  # Number of discrete points in the FFT
eta = 0.25  # Spacing between points in the FFT
b = N * eta  # Upper limit of integration for the characteristic function
b_alt = np.pi * eta
spacing = (2*np.pi)/(N*eta)  # Spacing of log-strikes
logK = np.log(K)  # Log of the strike price
logS0 = np.log(S0)  # Log of the initial stock price

# ---------------------------------------------------------------------
# STEP 1: DISCOUNT FACTOR
# ---------------------------------------------------------------------
discount = np.exp(-r * T)

# ---------------------------------------------------------------------
# STEP 2: FFT GRID SETUP
# ---------------------------------------------------------------------
grid_points = np.arange(1, N) # FFT Grid points
frequencies = eta * (grid_points - 1)  # Frequencies in FFT space
log_strikes = -b_alt + spacing * (grid_points - 1)

# ---------------------------------------------------------------------
# STEP 3: CHARACTERISTIC FUNCTION
# ---------------------------------------------------------------------
def characteristic_black_scholes(frequency): # Black-Scholes characteristic function
    # Risk-neutral distribution of log-returns under Black-Scholes
    return np.exp(1j * frequency * (logS0 + (r - q - 0.5 * sigma**2) * T) - 0.5 * sigma**2 * T * frequency**2)

# ---------------------------------------------------------------------
# STEP 4: ADJUSTED CHARACTERISTIC FUNCTION WITH DAMPING
# ---------------------------------------------------------------------
def psi(frequency):
    cf = characteristic_black_scholes(frequency - (damping + 1) * 1j)
    numerator = discount * cf
    denominator = damping**2 + damping - frequency**2 + 1j * (2 * damping + 1) * frequency
    return numerator / denominator

# ---------------------------------------------------------------------
# STEP 5: APPLYING FFT  
# ---------------------------------------------------------------------
Psi_values = psi(frequencies)
# Simpson's rule weights
weights = 3 + (-1)**grid_points - np.where(grid_points == 1, 1, 0)
fft_input = eta * np.exp(1j * b_alt * frequencies) * Psi_values * weights / 3
transformed_log_strikes = np.real(fft(fft_input) * np.exp(-damping * log_strikes) / np.pi)

# ---------------------------------------------------------------------
# STEP 6: INTERPOLATION AND PRICING
# ---------------------------------------------------------------------
f = interp1d(log_strikes, transformed_log_strikes)
price_at_logK = np.real(f(logK))
if option_type == 'put': # Adjusting for put prices if needed
    price_at_logK = price_at_logK - (S0 * np.exp(-q * T) - K * discount)
print(f"Option Price: {price_at_logK}")