from scipy.optimize import minimize
from Geometric.price_fft_nig import GEOMETRIC_price_fft_nig

S0 = 100  # Initial stock price
K = 100  # Strike price
T = 1  # Time to maturity in years
r = 0.0367  # Risk-free interest rate
q = 0.00  # Continuous dividend yield
n = 12  # Number of observation points (daily observations in a year)
option_type = 'call'  # 'call' or 'put'

damping = 1.5
N = 16384

def objective_function(params, S0, K, T, r, q, option_type, n_values, expected_prices, damping, N):
    alpha, beta, delta, eta = params
    total_error = 0
    print(" ")
    for n, expected_price in zip(n_values, expected_prices):
        actual_price = GEOMETRIC_price_fft_nig(S0, K, T, r, q, alpha, beta, delta, option_type, n, damping, N, eta)
        print(f"n={n}, expected_price={expected_price:.6f}, actual_price={actual_price:.6f}")
        total_error += (actual_price - expected_price)**2
    print(f"Total error: {total_error:.6f}\n")
    return total_error

def callback(params):
    alpha, beta, delta, eta = params
    print(f"Iteration: alpha={alpha:.3f}, beta={beta:.3f}, delta={delta:.3f}, eta={eta:.3f}")

# Define the expected prices and corresponding n values
expected_prices = [4.903628, 4.956617, 4.971162]
n_values = [12, 50, 250]

# Define the initial parameter values
alpha = 6.0
beta = 0.0
delta = 0.5
eta = 0.25

initial_params = [alpha, beta, delta, eta]

# Define the bounds for the parameters (if applicable)
bounds = [(0.5, 10.0), (-1.0, 1.0), (0.1, 1.0), (0.01, 0.9)]  # alpha > 0, delta > 0, 0 < eta < 1

# Minimize the objective function
result = minimize(objective_function, initial_params, args=(S0, K, T, r, q, option_type, n_values, expected_prices, damping, N), bounds=bounds, callback=callback)

# Print the optimal parameters
optimal_params = result.x
print("Optimal parameters:")
print("alpha =", optimal_params[0])
print("beta =", optimal_params[1])
print("delta =", optimal_params[2])
print("eta =", optimal_params[3])