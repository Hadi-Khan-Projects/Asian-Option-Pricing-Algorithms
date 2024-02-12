import numpy as np
from math_tools import fourier_transform, inverse_fourier_transform
from financial_models import characteristic_function_log_normal

# params self explanatory
# return: payoff value of option
def asian_option_payoff(spot_price, strike_price, average_price, option_type='call'):
    if option_type == 'call':
        return max(average_price - strike_price, 0)
    elif option_type == 'put':
        return max(strike_price - average_price, 0)
    else:
        raise ValueError("Option type must be 'call' or 'put'.")
    
#param u: Array-like, the input values for the characteristic function.
# param spot_price: Current spot price of the underlying asset.
#param mu: Mean rate of return of the underlying asset.
#param sigma: Volatility of the underlying asset.
#param t: Time to maturity.
#param n: Number of observations for averaging.
#return: Approximate characteristic function evaluated at u.
def approximate_characteristic_function(u, spot_price, mu, sigma, t, n):
    # calculate the first two moments (mean and variance) of the avg price
    average_mu = spot_price * np.exp(mu * t)  # This is a simplification
    average_variance = (spot_price**2 * np.exp(2 * mu * t) * (np.exp(sigma**2 * t) - 1)) / n  # Simplified variance
    
    # Using the moments, approximate the distribution with a log-normal distribution
    # This is a simplification
    scale = np.sqrt(np.log(1 + average_variance / average_mu**2))
    loc = np.log(average_mu) - 0.5 * scale**2
    
    # Compute the characteristic function for the log-normal distribution
    cf = np.exp(1j * u * loc - 0.5 * (scale * u)**2)
    
    return cf

#param spot_price: Current spot price of the underlying asset.
#param strike_price: Strike price of the option.
#param mu: Mean rate of return of the underlying asset.
#param sigma: Volatility of the underlying asset.
#param t: Time to maturity.
#param n: Number of points for the Fourier transform.
#param option_type Type of the option ('call' or 'put').
#return: Estimated price of the Asian option.
def price_arithmetic_asian_option(spot_price, strike_price, mu, sigma, t, n, option_type='call'):
    # Discretize the space for the payoff function
    dx = (2 * np.pi / n) / (spot_price * 2)  # Adjust dx based on spot price range
    x = np.linspace(-spot_price, spot_price, n)
    payoff = np.array([asian_option_payoff(spot_price, strike_price, x_i, option_type) for x_i in x])

    # Calculate the Fourier transform of the payoff function
    payoff_transform = fourier_transform(payoff, n, dx)

    # Get the characteristic function for the underlying asset's price
    u = np.fft.fftfreq(n, dx)
    cf = characteristic_function_log_normal(u, mu, sigma, t)

    # Apply the convolution theorem (pointwise multiplication in Fourier space)
    option_transform = payoff_transform * cf

    # Inverse Fourier transform to get the option price in the price domain
    option_price = np.real(inverse_fourier_transform(option_transform, n, dx))

    # Return the option price corresponding to the spot price
    return option_price[n//2]  # Assuming spot price is at the center of the x array

