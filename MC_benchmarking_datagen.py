from option import Option
from time import time
import sys
import csv

# OPTION ATTRIBUTES
S0 = 100  # Initial stock price
K = 90  # Strike price
T = 1  # Time to maturity in years
r = 0.0367  # Risk-free interest rate
q = 0.00  # Continuous dividend yield
sigma = 0.17801  # Volatility of the underlying asset
n = 250  # Number of observation points (daily observations in a year)
option_type = 'call'  # 'call' or 'put'

option = Option(S0, K, T, r, q, sigma, n, option_type)
print("\nOPTION ATTRIBUTES:")
print(f" {option_type} - Option Type \n {S0} - Initial Stock Price \n {K} - Strike Price \n {T} - Time to Maturity (Years) \n {r} - Risk-Free Interest Rate \n {q} - Continuous Dividend Yield \n {sigma} - Volatility of the Underlying Asset \n {n} - Number of Observation Points (Daily Observations in a Year)")

# Do 20 runs to avoid cold start
dummy_list = []
for _ in range(20):
    dummy_list.append(option.GEOMETRIC_price_monte_carlo_g(n_paths=10000))

with open('Plotting/mc_benchmarking.csv', mode='w+', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Process', 'Path', 'Price', 'Time'])

    # Set up MC grid
    for n_paths in range(500000, 500001, 10000):
        # Adaptive average MC
        n = 50

        print(f"\nPaths: {n_paths}; Iterations: {n}")

        prices = []
        times = []

        for _ in range(n):
            # Run MC Gaussian with time
            start_time = time()
            option_price = option.GEOMETRIC_price_monte_carlo_g(n_paths=n_paths)
            end_time = time()
            total_time = end_time - start_time
            prices.append(option_price)
            times.append(total_time)

        # # Approach 1: Avegaging final prices and times  
        # avg_price = sum(prices) / len(prices)
        # avg_time = sum(times) / len(times)

        # writer.writerow(['Gaussian', n_paths, avg_price, avg_time])
        # print(f"Gaussian: {avg_price} - {avg_time} with {n_paths} paths")

        # Approach 2: write all prices and times
        for i in range(n):
            writer.writerow(['MC_Gaussian', n_paths, prices[i], times[i]])
            print(f"MC_Gaussian Iteration {i} : {prices[i]} - {times[i]} with {n_paths} paths")
