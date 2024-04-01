## DEPRECATED

exit("This file is deprecated. Please use the new version of the data generation script.")

from option import Option
import time

# OPTION ATTRIBUTES
S0 = 100  # Initial stock price
K = 90  # Strike price
T = 1  # Time to maturity in years
r = 0.0367  # Risk-free interest rate
q = 0.00  # Continuous dividend yield
sigma = 0.17801  # Volatility of the underlying asset
n = 250 # Number of observation points 
option_type = 'call'  # 'call' or 'put'

# CREATING AN OPTION OBJECT
option = Option(S0, K, T, r, q, sigma, n, option_type)

# Print option configuration
print(f"Option configuration:")
print(f"Initial stock price: {S0}")
print(f"Strike price: {K}")
print(f"Time to maturity: {T}")
print(f"Risk-free interest rate: {r}")
print(f"Continuous dividend yield: {q}")
print(f"Volatility of the underlying asset: {sigma}")
print(f"Number of observation points: {n}")
print(f"Option type: {option_type}")

# Set up CSV file
csv_file = open(f"Plotting/NEW_data_strike{K}_monpoints{n}.csv", "w")
csv_file.write("Shift,Monitoring Points,Price,CPU_Time\n")

# GEOMETRIC MONTE CARLO
# Not used for our plot. However, you can run this code to see the results
# for i in range(3, 7):
#     MC_points = 10**i
#     time_start = time.time()
#     option_price = option.GEOMETRIC_price_monte_carlo(n_paths=MC_points)
#     time_end = time.time()
#     cpu_time = time_end - time_start

#     csv_file.write(f"MC,{MC_points},{option_price},{cpu_time}\n")

#     print(f"MC Points: {MC_points}")
#     print(f"MC Price with Black-Scholes option is: {option_price:.15f}")

# Run FFT for 50 times to avoid cold runs to skew the results
dummy_price_array = [] # To avoid compilers optimizing out the loop
for _ in range(50):
    price = option.GEOMETRIC_price_fft_black_scholes(damping=1.5, N=1 << 11, eta=0.25)
    dummy_price_array.append(price)

# Try to run FFT with different monitoring points
for i in range(10, 28, 1):
    shift = 1 << i
    print(f"Shift: {i}; Monitoring points: {shift}")

    option_prices = []
    cpu_times = []
    # Now, for runs with small monitoring points, run many times to obtain a more accurate result
    if i < 14: 
        # Takes less than 10 ms to run, so average 100 times
        for _ in range(100):
            time_start = time.time()
            option_price = option.GEOMETRIC_price_fft_black_scholes(damping=1.5, N=shift, eta=0.25)
            time_end = time.time()
            cpu_time = time_end - time_start

            option_prices.append(option_price)
            cpu_times.append(cpu_time)
    elif i >= 14 and i < 17:
        # Takes less than 100 ms to run, so average 10 times
        for _ in range(10):
            time_start = time.time()
            option_price = option.GEOMETRIC_price_fft_black_scholes(damping=1.5, N=shift, eta=0.25)
            time_end = time.time()
            cpu_time = time_end - time_start

            option_prices.append(option_price)
            cpu_times.append(cpu_time)
    else:
        # No need to average
        time_start = time.time()
        option_price = option.GEOMETRIC_price_fft_black_scholes(damping=1.5, N=shift, eta=0.25)
        time_end = time.time()
        cpu_time = time_end - time_start

        option_prices.append(option_price)
        cpu_times.append(cpu_time)

    # Average the results
    avg_option_price = sum(option_prices) / len(option_prices)
    avg_cpu_time = sum(cpu_times) / len(cpu_times)

    # Export the data to a csv file
    csv_file.write(f"{i},{shift},{avg_option_price},{avg_cpu_time} \n")

    print(f"FFT Price with Black-Scholes option is: {avg_option_price:.15f}")


