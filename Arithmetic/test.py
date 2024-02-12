from math_tools import *
from financial_models import characteristic_function_log_normal
from option_pricing import *

import numpy as np
import matplotlib.pyplot as plt

def test_fourier_transform():
    # Define a simple function, e.g., a Gaussian function
    x = np.linspace(-10, 10, 400)
    dx = x[1] - x[0]
    f = np.exp(-x**2)

    # Compute the Fourier transform
    F = np.fft.fft(f) * dx
    # Compute the inverse Fourier transform
    f_inv = np.fft.ifft(F) / dx

    # Plot the original function and its inverse Fourier transform
    plt.figure(figsize=(12, 6))
    plt.plot(x, f, label='Original Function')
    plt.plot(x, f_inv.real, label='Inverse Fourier Transform', linestyle='dashed')
    plt.legend()
    plt.title('Fourier Transform and Its Inverse')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

def asian_option_payoff(average_price, strike_price, option_type='call'):
    if option_type == 'call':
        return np.maximum(average_price - strike_price, 0)
    elif option_type == 'put':
        return np.maximum(strike_price - average_price, 0)

def test_payoff_function():
    average_prices = np.linspace(50, 150, 100)
    strike_price = 100
    call_payoffs = asian_option_payoff(average_prices, strike_price, 'call')
    put_payoffs = asian_option_payoff(average_prices, strike_price, 'put')

    plt.figure(figsize=(12, 6))
    plt.plot(average_prices, call_payoffs, label='Call Option Payoff')
    plt.plot(average_prices, put_payoffs, label='Put Option Payoff')
    plt.legend()
    plt.title('Asian Option Payoff Functions')
    plt.xlabel('Average Price')
    plt.ylabel('Payoff')
    plt.show()

def test_convolution_application():
    # Define the spatial domain with a sufficient number of points
    x = np.linspace(-10, 10, 400)
    
    # Gaussian function with scale
    f1 = np.exp(-x**2 / 2)  # Ewidth matches the frequency of the sine

    # Define a sine function with a frequency relatedto with the Gaussian's width
    f2 = np.sin(2 * np.pi * x / 5)  # Adjust the frequency

    # Compute the Fourier transforms + normalization
    F1 = np.fft.fft(f1) / len(f1)
    F2 = np.fft.fft(f2) / len(f2)

    # Pointwise multiplication in Fourier space (convolution
    F_conv = F1 * F2

    # Inverse Fourier transform to get back to the spatial domain + normalization
    conv_result = np.fft.ifft(F_conv) * len(f1)

    # Plot convolution, focus on the range that shows effects
    plt.figure(figsize=(12, 6))
    plt.plot(x, conv_result.real, label='Result of Convolution')
    plt.legend()
    plt.title('Convolution of Gaussian and Sine Functions')
    plt.xlabel('x')
    plt.ylabel('Amplitude')
    plt.show()

def test_characteristic_function():
    u = np.linspace(-10, 10, 400)
    mu = 0.05
    sigma = 0.2
    t = 1

    cf_values = characteristic_function_log_normal(u, mu, sigma, t)

    plt.figure(figsize=(12, 6))
    plt.plot(u, cf_values.real, label='Real Part')
    plt.plot(u, cf_values.imag, label='Imaginary Part')
    plt.legend()
    plt.title('Characteristic Function of Log-Normal Distribution')
    plt.xlabel('u')
    plt.ylabel('Characteristic Function')
    plt.show()

def test_parameter_handling(spot_price, strike_price, mu, sigma, t):
    print("Spot Price:", spot_price)
    print("Strike Price:", strike_price)
    print("Mean Rate of Return:", mu)
    print("Volatility:", sigma)
    print("Time to Maturity:", t)

    # Example calculations to check parameter scaling
    scaled_mu = mu * t
    scaled_sigma = sigma * np.sqrt(t)

    print("Scaled Mean Rate of Return:", scaled_mu)
    print("Scaled Volatility:", scaled_sigma)

test_fourier_transform()
test_payoff_function()
test_convolution_application()
test_characteristic_function()
test_parameter_handling(100, 105, 0.05, 0.2, 1)
