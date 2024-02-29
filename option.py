from European.price_monte_carlo import EUROPEAN_price_monte_carlo
from Arithmetic.price_monte_carlo import ARITHMETIC_price_monte_carlo
from Geometric.price_monte_carlo import GEOMETRIC_price_monte_carlo

from European.price_fft import EUROPEAN_price_fft
from Geometric.price_fft import GEOMETRIC_price_fft


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

    def EUROPEAN_price_monte_carlo(self, n_paths): # n_paths = the number of monte carlo paths to simulate
        return EUROPEAN_price_monte_carlo(self.S0, self.K, self.T, self.r, self.q, self.sigma, self.option_type, n_paths)

    def ARITHMETIC_price_monte_carlo(self, n_paths): # n_paths = the number of monte carlo paths to simulate
        return ARITHMETIC_price_monte_carlo(self.S0, self.K, self.T, self.r, self.q, self.sigma, self.n, self.option_type, n_paths)

    def GEOMETRIC_price_monte_carlo(self, n_paths): # n_paths = the number of monte carlo paths to simulate
        return GEOMETRIC_price_monte_carlo(self.S0, self.K, self.T, self.r, self.q, self.sigma, self.n, self.option_type, n_paths)
    
    def EUROPEAN_price_fft_black_scholes(self, damping, N, eta): # N = no. of discrete points in FFT, eta = spacing between points in FFT
        return EUROPEAN_price_fft(self.S0, self.K, self.T, self.r, self.q, self.sigma, self.option_type, damping, N, eta, characteristic="black scholes")
    
    def GEOMETRIC_price_fft_black_scholes(self, damping, N, eta):
       return GEOMETRIC_price_fft(self.S0, self.K, self.T, self.r, self.q, self.sigma, self.option_type, self.n, damping, N, eta, characteristic="black scholes")
    
    def GEOMETRIC_price_fft_gaussian(self, damping, N, eta):
       return GEOMETRIC_price_fft(self.S0, self.K, self.T, self.r, self.q, self.sigma, self.option_type, self.n, damping, N, eta, characteristic="gaussian")  
    
    def GEOMETRIC_price_fft_variance_gamma(self, damping, N, eta):
       return GEOMETRIC_price_fft(self.S0, self.K, self.T, self.r, self.q, self.sigma, self.option_type, self.n, damping, N, eta, characteristic="variance gamma") 
