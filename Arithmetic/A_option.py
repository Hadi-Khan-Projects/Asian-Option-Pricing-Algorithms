from A_option_pricing import arithmetic_asian_option_price

class ArithmeticAsianOption:
    def __init__(self, S0, K, T, r, q, sigma, n, option_type='call'):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.sigma = sigma
        self.n = n
        self.option_type = option_type

    def price(self, num_paths=10000):
        return arithmetic_asian_option_price(self.S0, self.K, self.T, self.r, self.q, self.sigma, self.n, self.option_type, num_paths)
