from scipy.optimize import minimize
from Geometric.price_fft_jd import GEOMETRIC_price_fft_jd
from Geometric.price_fft_de import GEOMETRIC_price_fft_de
from Geometric.price_fft_nig import GEOMETRIC_price_fft_nig

S0 = 100  # Initial stock price
K = 110  # Strike price
T = 1  # Time to maturity in years
r = 0.0367  # Risk-free interest rate
q = 0.00  # Continuous dividend yield
sigma = 0.17801  # Volatility of the diffusion part
n = 12  # Number of observation points (daily observations in a year)
option_type = 'call'  # 'call' or 'put'

damping = 1.5
N = 16384
eta = 0.25

def objective_function_jd(params):
    lam, alpha, delta = params
    errors = []
    errors.append((GEOMETRIC_price_fft_jd(S0, 110, T, r, q, sigma, lam, alpha, delta, option_type, 12, damping, N, eta) - 0.959979)**2)
    errors.append((GEOMETRIC_price_fft_jd(S0, 90, T, r, q, sigma, lam, alpha, delta, option_type, 50, damping, N, eta) - 12.53882)**2)
    errors.append((GEOMETRIC_price_fft_jd(S0, 100, T, r, q, sigma, lam, alpha, delta, option_type, 250, damping, N, eta) - 4.918632)**2)
    error = sum(errors)
    print(f"Merton Jump Diffusion - lam: {lam:.4f}, alpha: {alpha:.4f}, delta: {delta:.4f}, Error: {error:.6f}")
    return error

def objective_function_de(params):
    lam, p, eta1, eta2 = params
    errors = []
    errors.append((GEOMETRIC_price_fft_de(S0, 110, T, r, q, sigma, lam, p, eta1, eta2, option_type, 12, damping, N, eta) - 0.950346)**2)
    errors.append((GEOMETRIC_price_fft_de(S0, 90, T, r, q, sigma, lam, p, eta1, eta2, option_type, 50, damping, N, eta) - 12.54201)**2)
    errors.append((GEOMETRIC_price_fft_de(S0, 100, T, r, q, sigma, lam, p, eta1, eta2, option_type, 250, damping, N, eta) - 4.924996)**2)
    error = sum(errors)
    print(f"Double Exponential Jump Diffusion - lam: {lam:.4f}, p: {p:.4f}, eta1: {eta1:.4f}, eta2: {eta2:.4f}, Error: {error:.6f}")
    return error

def objective_function_nig(params):
    alpha, beta, delta = params
    errors = []
    errors.append((GEOMETRIC_price_fft_nig(S0, 110, T, r, q, alpha, beta, delta, option_type, 12, damping, N, eta) - 0.9217)**2)
    errors.append((GEOMETRIC_price_fft_nig(S0, 90, T, r, q, alpha, beta, delta, option_type, 50, damping, N, eta) - 12.45446)**2)
    errors.append((GEOMETRIC_price_fft_nig(S0, 100, T, r, q, alpha, beta, delta, option_type, 250, damping, N, eta) - 4.971162)**2)
    error = sum(errors)
    print(f"Normal Inverse Gaussian - alpha: {alpha:.4f}, beta: {beta:.4f}, delta: {delta:.4f}, Error: {error:.6f}")
    return error

# Merton Jump Diffusio
initial_guess_jd = [0.1, 0.1, 0.1]
result_jd = minimize(objective_function_jd, initial_guess_jd, method='Nelder-Mead')
optimal_params_jd = result_jd.x
print(f"\nOptimal parameters for Merton Jump Diffusion: lam={optimal_params_jd[0]:.4f}, alpha={optimal_params_jd[1]:.4f}, delta={optimal_params_jd[2]:.4f}")

# Double Exponential Jump Diffusion
initial_guess_de = [0.1, 0.5, 0.1, 0.1]
result_de = minimize(objective_function_de, initial_guess_de, method='Nelder-Mead')
optimal_params_de = result_de.x
print(f"\nOptimal parameters for Double Exponential Jump Diffusion: lam={optimal_params_de[0]:.4f}, p={optimal_params_de[1]:.4f}, eta1={optimal_params_de[2]:.4f}, eta2={optimal_params_de[3]:.4f}")

# Normal Inverse Gaussian
initial_guess_nig = [1.0, 0.1, 0.1]
result_nig = minimize(objective_function_nig, initial_guess_nig, method='Nelder-Mead')
optimal_params_nig = result_nig.x
print(f"\nOptimal parameters for Normal Inverse Gaussian: alpha={optimal_params_nig[0]:.4f}, beta={optimal_params_nig[1]:.4f}, delta={optimal_params_nig[2]:.4f}")