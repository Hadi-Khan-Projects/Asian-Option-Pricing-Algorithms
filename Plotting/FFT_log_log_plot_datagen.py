from option import Option
from time import time
import sys

if len(sys.argv) == 4:
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    arg3 = sys.argv[3]
    print("Strike from command line is ", arg1)
    print("Maturity from command line is ", arg2)
    print("Monpoints from command line is ", arg3)
    K = int(arg1)
    T = float(arg2)
    n = int(arg3)
else:
    exit()

# OPTION ATTRIBUTES
S0 = 100  # Initial stock price (FIXED)
r = 0.0367  # Risk-free interest rate (FIXED)
q = 0.00  # Continuous dividend yield (FIXED)
sigma = 0.17801  # Volatility of the underlying asset, for Gaussian process (FIXED here)
option_type = 'call'  # 'call' or 'put' (FIXED, may consider changing in the future)

FFT_SHIFT_UPPER_BOUND = 23  # Upper bound for the shift value in the FFT algorithm
# BE MIND OF THE COMPUTATIONAL TIME WHEN SETTING THE UPPER BOUND

# CREATING AN OPTION OBJECT
option = Option(S0, K, T, r, q, sigma, n, option_type)

print("\nOPTION ATTRIBUTES:")
print(f" {option_type} - Option Type \n {S0} - Initial Stock Price \n {K} - Strike Price \n {T} - Time to Maturity (Years) \n {r} - Risk-Free Interest Rate \n {q} - Continuous Dividend Yield \n {sigma} - Volatility of the Underlying Asset \n {n} - Number of Observation Points (Daily Observations in a Year)")

# Set up CSV file
csv_file = open(f'Plotting/FFT_log_log_plot_data_strike{K}_maturity{T}_monpoints{n}.csv', 'w')
csv_file.write('Process,Shift,FFT_Points,Price,CPU_Time\n')

# Run FFT engine for 50 times to avoid cold runs to skew the results
dummy_price_array = []  # To avoid compilers optimizing out the loop
for _ in range(50):
    price = option.GEOMETRIC_price_fft_g(damping=1.5, N= 1 << 11, eta=0.25)
    dummy_price_array.append(price)

# Run FFT with different processes and different grid points
for shift in range(10, FFT_SHIFT_UPPER_BOUND, 1): 
    grid_points = 1 << shift
    print(f"Shift: {shift}; FFT Points: {grid_points}")

    # Set up a list (option price, cpu time) for each process: g, nig, jd, cgmy, de
    option_prices_g = []
    cpu_times_g = []

    option_prices_nig = []
    cpu_times_nig = []

    option_prices_jd = []
    cpu_times_jd = []

    option_prices_cgmy = []
    cpu_times_cgmy = []

    option_prices_de = []
    cpu_times_de = []

    # Use adaptive averaging, run multiple times when shift is small
    iter_bound = 0
    if shift < 14:
        iter_bound = 100
    elif shift >= 14 and shift < 17:
        iter_bound = 10
    else:
        iter_bound = 1

    for _ in range(iter_bound):
        # Gaussian process: g
        time_start_g = time()
        option.sigma = 0.17801
        option_price_g = option.GEOMETRIC_price_fft_g(damping=1.5, N=grid_points, eta=0.25)
        time_end_g = time()

        option_prices_g.append(option_price_g)
        cpu_times_g.append(time_end_g - time_start_g)

        # NIG process
        time_start_nig = time()
        option_price_nig = option.GEOMETRIC_price_fft_nig(damping=1.5, N=grid_points, eta=0.25, alpha=6.1882, beta=-3.8941, delta=0.1622)
        time_end_nig = time()

        option_prices_nig.append(option_price_nig)
        cpu_times_nig.append(time_end_nig - time_start_nig)

        # JD process
        time_start_jd = time()
        option.sigma = 0.126349 # volatility for Merton Jump Diffusion
        option_price_jd = option.GEOMETRIC_price_fft_jd(damping=1.5, N=grid_points, eta=0.25, lam=0.174814, alpha=-0.390078, delta=0.338796)
        time_end_jd = time()

        option_prices_jd.append(option_price_jd)
        cpu_times_jd.append(time_end_jd - time_start_jd)

        # CGMY process
        time_start_cgmy = time()
        option_price_cgmy = option.GEOMETRIC_price_fft_cgmy(damping=1.5, N=grid_points, eta=0.25, C=0.0244, G=0.0765, M=7.5515, Y=1.2945)
        time_end_cgmy = time()

        option_prices_cgmy.append(option_price_cgmy)
        cpu_times_cgmy.append(time_end_cgmy - time_start_cgmy)

        # DE process
        time_start_de = time()
        option.sigma = 0.120381 # volatility for double exponential jump diffusion
        option_price_de = option.GEOMETRIC_price_fft_de(damping=1.5, N=grid_points, eta=0.25, lam=0.330966, p=0.20761, eta1=9.65997, eta2=3.13868)
        time_end_de = time()

        option_prices_de.append(option_price_de)
        cpu_times_de.append(time_end_de - time_start_de)

    # Average the results and write to CSV
    avg_option_price_g = sum(option_prices_g) / len(option_prices_g)
    avg_cpu_time_g = sum(cpu_times_g) / len(cpu_times_g)
    csv_file.write(f"Gaussian,{shift},{grid_points},{avg_option_price_g},{avg_cpu_time_g}\n")

    avg_option_price_nig = sum(option_prices_nig) / len(option_prices_nig)
    avg_cpu_time_nig = sum(cpu_times_nig) / len(cpu_times_nig)
    csv_file.write(f"NIG,{shift},{grid_points},{avg_option_price_nig},{avg_cpu_time_nig}\n")

    avg_option_price_jd = sum(option_prices_jd) / len(option_prices_jd)
    avg_cpu_time_jd = sum(cpu_times_jd) / len(cpu_times_jd)
    csv_file.write(f"JD,{shift},{grid_points},{avg_option_price_jd},{avg_cpu_time_jd}\n")

    avg_option_price_cgmy = sum(option_prices_cgmy) / len(option_prices_cgmy)
    avg_cpu_time_cgmy = sum(cpu_times_cgmy) / len(cpu_times_cgmy)
    csv_file.write(f"CGMY,{shift},{grid_points},{avg_option_price_cgmy},{avg_cpu_time_cgmy}\n")

    avg_option_price_de = sum(option_prices_de) / len(option_prices_de)
    avg_cpu_time_de = sum(cpu_times_de) / len(cpu_times_de)
    csv_file.write(f"DE,{shift},{grid_points},{avg_option_price_de},{avg_cpu_time_de}\n")

    print(f"Average option price for Gaussian process: {avg_option_price_g:.12f}")
    print(f"Average option price for NIG process: {avg_option_price_nig:.12f}")
    print(f"Average option price for CGMY process: {avg_option_price_cgmy:.12f}")
    print(f"Average option price for DE process: {avg_option_price_de:.12f}")
    print(f"Average option price for MJD process: {avg_option_price_jd:.12f}")
