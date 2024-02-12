from financial_model import geometric_asian_option_price

class GeometricAsianOption:
    def __init__(self, S0, K, T, r, q, sigma, n, option_type='call'):
        self.S0 = S0  # Initial stock price
        self.K = K  # Strike price
        self.T = T  # Time to maturity in years
        self.r = r  # Risk-free interest rate
        self.q = q  # Continuous dividend yield
        self.sigma = sigma  # Volatility of the underlying asset
        self.n = n  # Number of observation points
        self.option_type = option_type  # 'call' or 'put'

    def price(self):
        return geometric_asian_option_price(self.S0, self.K, self.T, self.r, self.q, self.sigma, self.n, self.option_type)
