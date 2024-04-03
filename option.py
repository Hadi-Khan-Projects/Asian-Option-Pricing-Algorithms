from European.price_monte_carlo_g import EUROPEAN_price_monte_carlo_g
from European.price_fft_g import EUROPEAN_price_fft_g

from Arithmetic.price_monte_carlo_g import ARITHMETIC_price_monte_carlo_g

from Geometric.price_monte_carlo_g import GEOMETRIC_price_monte_carlo_g
from Geometric.price_fft_g import GEOMETRIC_price_fft_g
from Geometric.price_fft_cgmy import GEOMETRIC_price_fft_cgmy
from Geometric.price_fft_jd import GEOMETRIC_price_fft_jd
from Geometric.price_fft_de import GEOMETRIC_price_fft_de
from Geometric.price_fft_nig import GEOMETRIC_price_fft_nig

from Geometric import *

class Option:
    def __init__(self, S0, K, T, r, q, sigma, n, option_type):
        # REQUIRED PARAMETERS
        self.S0 = S0  # Initial stock price
        self.K = K  # Strike price
        self.T = T  # Time to maturity in years
        self.r = r  # Risk-free interest rate
        self.q = q  # Continuous dividend yield
        self.sigma = sigma  # Volatility of the underlying asset
        self.n = n  # Number of observation points (daily observations in a year)
        self.option_type = option_type  # 'call' or 'put'

    # EUROPEAN

    def EUROPEAN_price_monte_carlo_g(self, n_paths): # n_paths = the number of monte carlo paths to simulate
        return EUROPEAN_price_monte_carlo_g(self.S0, self.K, self.T, self.r, self.q, self.sigma, self.option_type, n_paths)
    
    def EUROPEAN_price_fft_g(self, damping, N, eta): # N = no. of discrete points in FFT, eta = spacing between points in FFT
        return EUROPEAN_price_fft_g(self.S0, self.K, self.T, self.r, self.q, self.sigma, self.option_type, damping, N, eta)

    # ARITHMETIC

    def ARITHMETIC_price_monte_carlo_g(self, n_paths): # n_paths = the number of monte carlo paths to simulate
        return ARITHMETIC_price_monte_carlo_g(self.S0, self.K, self.T, self.r, self.q, self.sigma, self.n, self.option_type, n_paths)

    # GEOMETRIC

    def GEOMETRIC_price_monte_carlo_g(self, n_paths): # n_paths = the number of monte carlo paths to simulate
        return GEOMETRIC_price_monte_carlo_g(self.S0, self.K, self.T, self.r, self.q, self.sigma, self.n, self.option_type, n_paths)
    
    def GEOMETRIC_price_fft_g(self, damping, N, eta):
        return GEOMETRIC_price_fft_g(self.S0, self.K, self.T, self.r, self.q, self.sigma, self.option_type, self.n, damping, N, eta)

    def GEOMETRIC_price_fft_cgmy(self, damping, N, eta, C, G, M, Y):
        return GEOMETRIC_price_fft_cgmy(self.S0, self.K, self.T, self.r, self.q, C, G, M, Y, self.option_type, self.n, damping, N, eta)
    
    def GEOMETRIC_price_fft_jd(self, damping, N, eta, lam, alpha, delta):
        return GEOMETRIC_price_fft_jd(self.S0, self.K, self.T, self.r, self.q, self.sigma, lam, alpha, delta, self.option_type, self.n, damping, N, eta)
    
    def GEOMETRIC_price_fft_de(self, damping, N, eta, lam, p, eta1, eta2):
        return GEOMETRIC_price_fft_de(self.S0, self.K, self.T, self.r, self.q, self.sigma, lam, p, eta1, eta2, self.option_type, self.n, damping, N, eta)
    
    def GEOMETRIC_price_fft_nig(self, damping, N, eta, alpha, beta, delta):
        return GEOMETRIC_price_fft_nig(self.S0, self.K, self.T, self.r, self.q, alpha, beta, delta, self.option_type, self.n, damping, N, eta)