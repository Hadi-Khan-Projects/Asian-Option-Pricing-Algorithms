from option_pricing import GeometricAsianOption

# Example parameters
S0 = 80  # Initial stock price
K = 70  # Strike price
T = 2  # Time to maturity in years
r = 0.08  # Risk-free interest rate
q = 0.00  # Continuous dividend yield
sigma = 0.4  # Volatility of the underlying asset
n = 500  # Number of observation points

# Creating an instance of GeometricAsianOption
asian_option = GeometricAsianOption(S0, K, T, r, q, sigma, n, option_type='call')

# Pricing the option
option_price = asian_option.price()
print(f"The price of the Geometric Asian call option is: {option_price:.2f}")
