from G_option_pricing import *

class GeometricAsianOption:
    def __init__(self, S0, K, T, r, q, sigma, n, option_type='call'):
        self.S0 = S0  # Initial stock price
        self.K = K  # Strike price
        self.T = T  # Time to maturity in years
        self.r = r  # Risk-free interest rate
        self.q = q  # Continuous dividend yield
        self.sigma = sigma  # Volatility of the underlying asset
        self.n = n  # Number of observation points (daily observations in a year)
        self.option_type = option_type  # 'call' or 'put'

    def price_monte_carlo(self, n_paths): # n_paths = the number of monte carlo paths to simulate
        return price_monte_carlo(self.S0, self.K, self.T, self.r, self.q, self.sigma, self.n, self.option_type, n_paths)

    def price_fft_gaussian(self, N, d): # N = no. of discrete points in the FFT, d = the spacing between points in the FFT  
        return price_fft_gaussian(self.S0, self.K, self.T, self.r, self.q, self.sigma, self.n, self.option_type, N, d)