from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
from option import Option

sigma = 0.17801  # Volatility of the diffusion part
option_type = 'call'  # 'call' or 'put'

damping = 1.5
N = 4096
eta = 0.25

error_values = []  # List to store error values at each iteration

def meta_objective_function_nov(params):
    lambda0, alpha, beta, mu, rho = params
    print(f"\nlambda0 = {lambda0:.4f} \nalpha = {alpha:.4f} \nbeta = {beta:.4f} \nmu ={mu:.4f} \nrho = {rho:.4f}")
    errors = []

    # Generate 100 different combinations of parameters
    num_samples = 100
    S0_samples = np.random.uniform(90, 110, num_samples)
    K_samples = np.random.uniform(90, 110, num_samples)
    T_samples = np.random.uniform(0.5, 3, num_samples)
    r_samples = np.random.uniform(0.01, 0.15, num_samples)
    q_samples = np.random.uniform(0.0, 0.05, num_samples)
    n_samples = np.random.randint(10, 500, num_samples)

    for i in range(num_samples):
        S0 = S0_samples[i]
        K = K_samples[i]
        T = T_samples[i]
        r = r_samples[i]
        q = q_samples[i]
        n = n_samples[i]

        # Create an Option object with the generated parameters
        option = Option(S0, K, T, r, q, sigma, n, option_type)

        # Calculate the price using the Novel process
        price_Novel = option.GEOMETRIC_price_fft_nov(damping=damping, N=N, eta=eta, lambda0=lambda0, alpha=alpha, beta=beta, mu=mu, rho=rho)

        # Calculate the price using the Merton Jump Diffusion
        price_jd = option.GEOMETRIC_price_fft_jd(damping=damping, N=N, eta=eta, lam=0.1, alpha=0.1, delta=0.1)

        errors.append((price_Novel - price_jd)**2)

    error = sum(errors)
    print(f"Novel Process - Error: {error:.6f}")
    return error

# Novel Jump Diffusion
lambda0 = 0.0296 # Initial jump intensity
alpha = -0.6033  # Self-excitation parameter
beta = 0.4398 # Jump intensity mean-reversion speed
mu = 0.2096  # Mean jump size
rho = 2.3500 # Correlation between jump sizes and jump times

bounds = [(-1, 1), (-1, 1), (0.01, 1), (-1, 1), (0.01, 10)]
initial_guess_Novel = [lambda0, alpha, beta, mu, rho]
result_Novel = minimize(meta_objective_function_nov, initial_guess_Novel, method='COBYLA', bounds=bounds)
optimal_params_Novel = result_Novel.x
print(f"\nOptimal parameters for Novel Process: lambda0={optimal_params_Novel[0]:.4f}, alpha={optimal_params_Novel[1]:.4f}, beta={optimal_params_Novel[2]:.4f}, mu={optimal_params_Novel[3]:.4f}, rho={optimal_params_Novel[4]:.4f}")


# MINIMISE METHODS FROM SCIPY DOCS:
"""
minimize(method=’Nelder-Mead’)
minimize(method=’Powell’)
minimize(method=’CG’)
minimize(method=’BFGS’)
minimize(method=’Newton-CG’)
minimize(method=’L-BFGS-B’)
minimize(method=’TNC’)
minimize(method=’COBYLA’)
minimize(method=’SLSQP’)
minimize(method=’trust-constr’)
minimize(method=’dogleg’)
minimize(method=’trust-ncg’)
minimize(method=’trust-krylov’)
minimize(method=’trust-exact’)
"""

# SLSQP Calibration: 
# Novel Process - lambda0: -0.0760, alpha: -0.4010, beta: 0.5424, mu: -0.3653, rho: 2.0695
# Novel Process - Error: 33.288274
# Novel Process - lambda0: 0.0316, alpha: -0.3668, beta: 0.5882, mu: -0.2288, rho: 1.8635
# Novel Process - Error: 0.322480
# Novel Process - Error: 0.057279
# Novel Process - lambda0: 0.0425, alpha: -0.6401, beta: 0.3764, mu: -0.1870, rho: 1.5905

# COBYLA Calibration:
# lambda0 = 0.0298 
# alpha = -0.6028 
# beta = 0.4415, 
# mu =0.2111, 
# rho = 2.3505
# Novel Process - Error: 0.051402

# DEPRECATED CALIBRATION STEPS
# Novel Jump Diffusion
"""
lambda0 = -0.2499 # Initial jump intensity
alpha = -0.3225  # Self-excitation parameter
beta = 0.5596  # Jump intensity mean-reversion speed
mu = -0.2271  # Mean jump size
rho = 1.2070  # Correlation between jump sizes and jump times

#bounds = [(-1, 1), (0.01, 1), (0.01, 1), (0.01, 1), (-10, 10)]
initial_guess_Novel = [lambda0, alpha, beta, mu, rho]
result_Novel = minimize(objective_function_Novel, initial_guess_Novel, method='SLSQP')
optimal_params_Novel = result_Novel.x
print(f"\nOptimal parameters for Novel Process: lambda0={optimal_params_Novel[0]:.4f}, alpha={optimal_params_Novel[1]:.4f}, beta={optimal_params_Novel[2]:.4f}, mu={optimal_params_Novel[3]:.4f}, rho={optimal_params_Novel[4]:.4f}")
"""