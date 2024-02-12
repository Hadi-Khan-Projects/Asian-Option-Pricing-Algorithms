import numpy as np

# TO BE IMPLEMENTED PROPERLY
# param: u (input values for the characteristic function)
# param: mu (mean rate of return of the underlying asset)
# param: sigma (Volatility of the underlying asset)
# param: t (time to maturity)
# return: the characteristic function evaluated at u?
def characteristic_function_log_normal(u, mu, sigma, t):
    return np.exp(1j * u * (mu - 0.5 * sigma ** 2) * t - 0.5 * sigma ** 2 * t * u ** 2)

# TO BE IMPLEMENTED PROPERLY
# param: cf (Characteristic function)
# param: scale (scalin factor)
# return: adjusted characteristc function (adjusted for scale)
def adjust_characteristic_function_for_scaling(cf, scale):
    return lambda u: cf(u * scale)

# TO BE IMPLEMENTED PROPERLY
# param: cf (Characteristic function)
# param: n (Num of random variables being averaged)
# return: adjusted characteristc function (adjusted for avg)
def average_characteristic_function(cf, n):
    return lambda u: cf(u / n) ** n
