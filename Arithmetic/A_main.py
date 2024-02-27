from A_option import ArithmeticAsianOption

# Example parameters
S0 = 100  # Initial stock price
K = 105  # Strike price
T = 1  # Time to maturity in years
r = 0.05  # Risk-free interest rate
q = 0.02  # Continuous dividend yield
sigma = 0.2  # Volatility of the underlying asset
n = 252  # Number of observation points (daily observations in a year)
option_type = 'call'  # 'call' or 'put'

# Creating an instance of ArithmeticAsianOption
asian_option = ArithmeticAsianOption(S0, K, T, r, q, sigma, n, option_type)

# Pricing the option
option_price = asian_option.price(num_paths=50000)
print(f"The price of the Arithmetic Asian {option_type} option is: {option_price:.2f}")
