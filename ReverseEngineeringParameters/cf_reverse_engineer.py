from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
from option import Option

S0 = 100  # Initial stock price
K = 110  # Strike price
T = 1  # Time to maturity in years
r = 0.0367  # Risk-free interest rate
q = 0.00  # Continuous dividend yield
sigma = 0.17801  # Volatility of the diffusion part
n = 12  # Number of observation points (daily observations in a year)
option_type = 'call'  # 'call' or 'put'

damping = 1.5
N = 4096
eta = 0.25

option = Option(S0, K, T, r, q, sigma, n, option_type)
option_2 = Option(100, 100, 2, 0.04, 0.02, sigma, n, option_type)
option_3 = Option(S0, 100, T, r, q, sigma, n, option_type)
option_4 = Option(100, 90, 2, 0.04, 0.02, sigma, n, option_type)
option_5 = Option(100, 90, 3, 0.1, 0.03, sigma, n, option_type)


#test
print(option.GEOMETRIC_price_fft_g(damping=1.5, N=4096, eta=0.25))
print(option.GEOMETRIC_price_fft_hawkes(damping=1.5, N=4096, eta=0.25, lambda0=0.0298, alpha=-0.6028, beta=0.4415, mu=0.2111, rho=2.3505))
option.sigma = 0.4
print(option.GEOMETRIC_price_fft_g(damping=1.5, N=4096, eta=0.25))
print(option.GEOMETRIC_price_fft_hawkes(damping=1.5, N=4096, eta=0.25, lambda0=0.0298, alpha=-0.6028, beta=0.4415, mu=0.2111, rho=2.3505))

def objective_function_hawkes(params):
    lambda0, alpha, beta, mu, rho = params
    errors = []
    price1 = option.GEOMETRIC_price_fft_hawkes(damping=1.5, N=4096, eta=0.25, lambda0=lambda0, alpha=alpha, beta=beta, mu=mu, rho=rho)
    price2 = option_2.GEOMETRIC_price_fft_hawkes(damping=1.5, N=4096, eta=0.25, lambda0=lambda0, alpha=alpha, beta=beta, mu=mu, rho=rho)
    price3 = option_3.GEOMETRIC_price_fft_hawkes(damping=1.5, N=4096, eta=0.25, lambda0=lambda0, alpha=alpha, beta=beta, mu=mu, rho=rho)
    price4 = option_4.GEOMETRIC_price_fft_hawkes(damping=1.5, N=4096, eta=0.25, lambda0=lambda0, alpha=alpha, beta=beta, mu=mu, rho=rho)
    price5 = option_5.GEOMETRIC_price_fft_hawkes(damping=1.5, N=4096, eta=0.25, lambda0=lambda0, alpha=alpha, beta=beta, mu=mu, rho=rho)
    errors.append((price1 - 0.960115)**2)
    errors.append((price2 - 6.576348)**2)
    errors.append((price3 - 4.918632)**2)
    errors.append((price4 - 13.166149)**2)
    errors.append((price5 - 16.982850)**2)
    print(price1, price2, price3)

    error = sum(errors)
    error_values.append(error)  # Append error value to the list
    print(f"Hawkes Process - lambda0: {lambda0:.4f}, alpha: {alpha:.4f}, beta: {beta:.4f}, mu: {mu:.4f}, rho: {rho:.4f}, Error: {error:.6f}")

    return error


error_values = []  # List to store error values at each iteration

def meta_objective_function_hawkes(params):
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

        # Calculate the price using the Hawkes process
        price_hawkes = option.GEOMETRIC_price_fft_hawkes(damping=damping, N=N, eta=eta, lambda0=lambda0, alpha=alpha, beta=beta, mu=mu, rho=rho)

        # Calculate the price using the Merton Jump Diffusion
        price_jd = option.GEOMETRIC_price_fft_jd(damping=damping, N=N, eta=eta, lam=0.1, alpha=0.1, delta=0.1)

        errors.append((price_hawkes - price_jd)**2)

    error = sum(errors)
    print(f"Hawkes Process - Error: {error:.6f}")
    return error

# MINIMISE METHODS FROM SCIPY DOCS
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
# Hawkes Process - lambda0: -0.0760, alpha: -0.4010, beta: 0.5424, mu: -0.3653, rho: 2.0695
# Hawkes Process - Error: 33.288274
# Hawkes Process - lambda0: 0.0316, alpha: -0.3668, beta: 0.5882, mu: -0.2288, rho: 1.8635
# Hawkes Process - Error: 0.322480
# Hawkes Process - Error: 0.057279
# Hawkes Process - lambda0: 0.0425, alpha: -0.6401, beta: 0.3764, mu: -0.1870, rho: 1.5905

# COBYLA Calibration:
# lambda0 = 0.0298 
# alpha = -0.6028 
# beta = 0.4415, 
# mu =0.2111, 
# rho = 2.3505
# Hawkes Process - Error: 0.051402

# Hawkes Jump Diffusion
lambda0 = 0.0296 # Initial jump intensity
alpha = -0.6033  # Self-excitation parameter
beta = 0.4398 # Jump intensity mean-reversion speed
mu = 0.2096  # Mean jump size
rho = 2.3500 # Correlation between jump sizes and jump times

bounds = [(-1, 1), (-1, 1), (0.01, 1), (-1, 1), (0.01, 10)]
initial_guess_hawkes = [lambda0, alpha, beta, mu, rho]
result_hawkes = minimize(meta_objective_function_hawkes, initial_guess_hawkes, method='COBYLA', bounds=bounds)
optimal_params_hawkes = result_hawkes.x
print(f"\nOptimal parameters for Hawkes Process: lambda0={optimal_params_hawkes[0]:.4f}, alpha={optimal_params_hawkes[1]:.4f}, beta={optimal_params_hawkes[2]:.4f}, mu={optimal_params_hawkes[3]:.4f}, rho={optimal_params_hawkes[4]:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(range(len(error_values)), error_values, 'b-')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error Convergence')
plt.grid(True)
plt.savefig('error_convergence.png')  # Save the plot as a PNG file
plt.show()

# DEPRECATED CALIBRATION STEPS
# Hawkes Jump Diffusion
"""
lambda0 = -0.2499 # Initial jump intensity
alpha = -0.3225  # Self-excitation parameter
beta = 0.5596  # Jump intensity mean-reversion speed
mu = -0.2271  # Mean jump size
rho = 1.2070  # Correlation between jump sizes and jump times

#bounds = [(-1, 1), (0.01, 1), (0.01, 1), (0.01, 1), (-10, 10)]
initial_guess_hawkes = [lambda0, alpha, beta, mu, rho]
result_hawkes = minimize(objective_function_hawkes, initial_guess_hawkes, method='SLSQP')
optimal_params_hawkes = result_hawkes.x
print(f"\nOptimal parameters for Hawkes Process: lambda0={optimal_params_hawkes[0]:.4f}, alpha={optimal_params_hawkes[1]:.4f}, beta={optimal_params_hawkes[2]:.4f}, mu={optimal_params_hawkes[3]:.4f}, rho={optimal_params_hawkes[4]:.4f}")
"""