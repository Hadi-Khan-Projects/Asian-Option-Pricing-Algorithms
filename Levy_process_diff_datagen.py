from option import Option
from time import time
import sys
import csv

# OPTION ATTRIBUTES
S0 = 100  # Initial stock price (FIXED)
T = 1  # Time to maturity in years (CHANGABLE)
r = 0.0367  # Risk-free interest rate (FIXED)
q = 0.00  # Continuous dividend yield (FIXED)
sigma = 0.17801  # Volatility of the underlying asset, for Gaussian process (FIXED here)
n = 50  # Number of observation points (daily observations in a year) (CHANGABLE)
option_type = 'call'  # 'call' or 'put' (FIXED, may consider changing in the future)
# Strike K will be variable in the loop

grid_points = 1 << 19

# Set up CSV file
with open('Plotting/Levy_process_difference.csv', mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Process', 'Strike', 'Price', 'Pct_Difference'])

    # Try differnet strikes
    # Collect data for each strike point: difference between a process and the Gaussian process
    for strike in range(80, 125, 5):
        K = strike
        print(f"\nStrike: {K}")

        option = Option(S0, K, T, r, q, sigma, n, option_type)
        print("\nOPTION ATTRIBUTES:")
        print(f" {option_type} - Option Type \n {S0} - Initial Stock Price \n {K} - Strike Price \n {T} - Time to Maturity (Years) \n {r} - Risk-Free Interest Rate \n {q} - Continuous Dividend Yield \n {sigma} - Volatility of the Underlying Asset \n {n} - Number of Observation Points (Daily Observations in a Year)")
        
        # Set up 5 lists to hold the prices for each process
        prices_g = []
        prices_nig = []
        prices_cgmy = []
        prices_de = []
        prices_jd = []

        for _ in range(5):
            # Gaussian process: g
            option.sigma = 0.17801
            price_g = option.GEOMETRIC_price_fft_g(damping=1.5, N=grid_points, eta=0.25)
            prices_g.append(price_g)

            # Normal Inverse Gaussian process
            price_nig = option.GEOMETRIC_price_fft_nig(damping=1.5, N=grid_points, eta=0.25, alpha=6.1882, beta=-3.8941, delta=0.1622)
            prices_nig.append(price_nig)

            # CGMY
            price_cgmy = option.GEOMETRIC_price_fft_cgmy(damping=1.5, N=grid_points, eta=0.25, C=0.0244, G=0.0765, M=7.5515, Y=1.2945)
            prices_cgmy.append(price_cgmy)

            # DE
            option.sigma = 0.120381 # volatility for double exponential jump diffusion
            price_de = option.GEOMETRIC_price_fft_de(damping=1.5, N=grid_points, eta=0.25, lam=0.330966, p=0.20761, eta1=9.65997, eta2=3.13868)
            prices_de.append(price_de)

            # JD
            option.sigma = 0.126349 # volatility for Merton Jump Diffusion
            price_jd = option.GEOMETRIC_price_fft_jd(damping=1.5, N=grid_points, eta=0.25, lam=0.174814, alpha=-0.390078, delta=0.338796)
            prices_jd.append(price_jd)
        
        # Take the average for the 5 lists as the final price
        final_price_g = sum(prices_g) / len(prices_g)
        final_price_nig = sum(prices_nig) / len(prices_nig)
        final_price_cgmy = sum(prices_cgmy) / len(prices_cgmy)
        final_price_de = sum(prices_de) / len(prices_de)
        final_price_jd = sum(prices_jd) / len(prices_jd)

        # Print final price
        print(f'Gaussian: {final_price_g}')
        print(f'NIG: {final_price_nig}')
        print(f'CGMY: {final_price_cgmy}')
        print(f'DE: {final_price_de}')
        print(f'JD: {final_price_jd}')

        # Write to CSV
        diff_nig = final_price_nig - final_price_g
        diff_cgmy = final_price_cgmy - final_price_g
        diff_de = final_price_de - final_price_g
        diff_jd = final_price_jd - final_price_g

        # Calculate percent difference
        pct_nig = diff_nig / final_price_g
        pct_cgmy = diff_cgmy / final_price_g
        pct_de = diff_de / final_price_g
        pct_jd = diff_jd/ final_price_g

        # Write differences to CSV
        writer.writerow(['NIG', K, final_price_nig, pct_nig*100])
        writer.writerow(['CGMY', K, final_price_cgmy, pct_cgmy*100])
        writer.writerow(['DE', K, final_price_de, pct_de*100])
        writer.writerow(['JD', K, final_price_jd, pct_jd*100])


