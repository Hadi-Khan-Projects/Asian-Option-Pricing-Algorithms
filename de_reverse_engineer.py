from scipy.optimize import minimize
from Geometric.price_fft_de import GEOMETRIC_price_fft_de

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

def objective_function(params, S0, K, T, r, q, sigma, option_type, n_values, expected_prices, damping, N):
    lam, p, eta1, eta2, eta = params
    total_error = 0
    print(" ")
    for n, expected_price in zip(n_values, expected_prices):
        actual_price = GEOMETRIC_price_fft_de(S0, K, T, r, q, sigma, lam, p, eta1, eta2, option_type, n, damping, N, eta)
        print(f"n={n}, expected_price={expected_price:.6f}, actual_price={actual_price:.6f}")
        total_error += (actual_price - expected_price)**2
    print(f"Total error: {total_error:.6f}\n")
    return total_error

def callback(params):
    lam, p, eta1, eta2, eta = params
    print(f"Iteration: lam={lam:.3f}, p={p:.3f}, eta1={eta1:.3f}, eta2={eta2:.3f}, eta={eta:.3f}")

# Define the expected prices and corresponding n values
expected_prices = [4.860329, 4.91101, 4.924996]
n_values = [12, 50, 250]

# Define the initial parameter values
lam = 0.1
p = 0.4
eta1 = 50
eta2 = 25
eta = 0.25
initial_params = [lam, p, eta1, eta2, eta]

# Define the bounds for the parameters
bounds = [(0, 10.0), (0, 1.0), (0, 100.0), (0, 100.0), (0.1, 0.8)]  # lam > 0, 0 <= p <= 1, eta1 > 0, eta2 > 0, 0 < eta < 1

# Minimize the objective function
result = minimize(objective_function, initial_params, args=(S0, K, T, r, q, sigma, option_type, n_values, expected_prices, damping, N), bounds=bounds, callback=callback)

# Print the optimal parameters
optimal_params = result.x
print("Optimal parameters:")
print("lam =", optimal_params[0])
print("p =", optimal_params[1])
print("eta1 =", optimal_params[2])
print("eta2 =", optimal_params[3])
print("eta =", optimal_params[4])