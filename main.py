from option import Option

# OPTION ATTRIBUTES
S0 = 100  # Initial stock price
K = 105  # Strike price
T = 1  # Time to maturity in years
r = 0.05  # Risk-free interest rate
q = 0.02  # Continuous dividend yield
sigma = 0.4  # Volatility of the underlying asset
n = 252  # Number of observation points (daily observations in a year)
option_type = 'call'  # 'call' or 'put'

# CREATING AN OPTION OBJECT
option = Option(S0, K, T, r, q, sigma, n, option_type)

print(" ")

# EUROPEAN MONTE CARLO
option_price = option.EUROPEAN_price_monte_carlo(n_paths=10000)
print(f"The price of the European {option_type} option is: {option_price:.2f}, computed via Monte Carlo simulation.")

# EUROPEAN FFT BLACK SCHOLES
option_price = option.EUROPEAN_price_fft_black_scholes(damping=1.5, N=4096, eta=0.25)
print(f"The price of the European {option_type} option is: {option_price:.2f}, computed via FFT with Black-Scholes characteristic function.")

# ARITHMETIC MONTE CARLO
option_price = option.ARITHMETIC_price_monte_carlo(n_paths=10000)
print(f"The price of the Arithmetic Asian {option_type} option is: {option_price:.2f}, computed via Monte Carlo simulation.")

# GEOMETRIC MONTE CARLO
option_price = option.GEOMETRIC_price_monte_carlo(n_paths=10000)
print(f"The price of the Geometric Asian {option_type} option is: {option_price:.2f}, computed via Monte Carlo simulation.")

# GEOMETRIC FFT BLACK SCHOLES
option_price = option.GEOMETRIC_price_fft_black_scholes(damping=1.5, N=4096, eta=0.25)
print(f"The price of the Geometric Asian {option_type} option is: {option_price:.2f}, computed via FFT with Black-Scholes characteristic function.")

# GEOMETRIC FFT GAUSSIAN
option_price = option.GEOMETRIC_price_fft_gaussian(damping=1.5, N=4096, eta=0.25)
print(f"The price of the Geometric Asian {option_type} option is: {option_price:.2f}, computed via FFT with Gaussian characteristic function.")

# GEOMETRIC FFT VARIANCE GAMMA
option_price = option.GEOMETRIC_price_fft_variance_gamma(damping=1.5, N=4096, eta=0.25)
print(f"The price of the Geometric Asian {option_type} option is: {option_price:.2f}, computed via FFT with variance Gamma characteristic function.")

print(" ")