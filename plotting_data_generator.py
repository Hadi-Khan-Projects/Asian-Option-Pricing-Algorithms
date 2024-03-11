from option import Option
import time

# OPTION ATTRIBUTES
S0 = 100  # Initial stock price
K = 110  # Strike price
T = 1  # Time to maturity in years
r = 0.0367  # Risk-free interest rate
q = 0.00  # Continuous dividend yield
sigma = 0.17801  # Volatility of the underlying asset
n = 1000 # Number of observation points 
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
csv_file = open(f"monitoring_points_strike{K}_monpoints{n}.csv", "w")
csv_file.write("Shift,Monitoring Points,Price,CPU_Time\n")

# GEOMETRIC MONTE CARLO
for i in range(3, 7):
    MC_points = 10**i
    time_start = time.time()
    option_price = option.GEOMETRIC_price_monte_carlo(n_paths=MC_points)
    time_end = time.time()
    cpu_time = time_end - time_start

    csv_file.write(f"MC,{MC_points},{option_price},{cpu_time}\n")

    print(f"MC Points: {MC_points}")
    print(f"MC Price with Black-Scholes option is: {option_price:.15f}")

# Try to run FFT with different monitoring points
for i in range(10, 28, 1):
    shift = 1 << i
    print(f"Shift: {i}; Monitoring points: {shift}")

    time_start = time.time()
    option_price = option.GEOMETRIC_price_fft_black_scholes(damping=1.5, N=shift, eta=0.25)
    time_end = time.time()
    cpu_time = time_end - time_start

    # Export the data to a csv file
    csv_file.write(f"{i},{shift},{option_price},{cpu_time} \n")

    print(f"FFT Price with Black-Scholes option is: {option_price:.15f}")


