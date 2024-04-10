from option import Option

# OPTION ATTRIBUTES
S0 = 100  # Initial stock price
K = 110  # Strike price
T = 1  # Time to maturity in years
r = 0.0367  # Risk-free interest rate
q = 0.00  # Continuous dividend yield
sigma = 0.17801  # Volatility of the underlying asset
n = 12  # Number of observation points (daily observations in a year)
option_type = 'call'  # 'call' or 'put'

# CREATING AN OPTION OBJECT
option = Option(S0, K, T, r, q, sigma, n, option_type)

print("\nOPTION ATTRIBUTES:")
print(f" {option_type} - Option Type \n {S0} - Initial Stock Price \n {K} - Strike Price \n {T} - Time to Maturity (Years) \n {r} - Risk-Free Interest Rate \n {q} - Continuous Dividend Yield \n {sigma} - Volatility of the Underlying Asset \n {n} - Number of Observation Points (Daily Observations in a Year)")

print(f"\nEUROPEAN {option_type.upper()} OPTION PRICE:")

# EUROPEAN - MONTE CARLO GAUSSIAN
option_price = option.EUROPEAN_price_monte_carlo_g(n_paths=10000)
print(f" {option_price:.6f} (Monte Carlo simulation with Gaussian process)")

# EUROPEAN - FFT GAUSSIAN
option_price = option.EUROPEAN_price_fft_g(damping=1.5, N=4096, eta=0.25)
print(f" {option_price:.6f} (FFT with Black-Scholes characteristic function)")

print(f"\nARITHMETIC ASIAN {option_type.upper()} OPTION PRICE:")

# ARITHMETIC - MONTE CARLO GAUSSIAN
option_price = option.ARITHMETIC_price_monte_carlo_g(n_paths=10000)
print(f" {option_price:.6f} (Monte Carlo simulation with Gaussian process)")

print(f"\nGEOMETRIC ASIAN {option_type.upper()} OPTION PRICE:")

# GEOMETRIC - MONTE CARLO GAUSSIAN
option_price = option.GEOMETRIC_price_monte_carlo_g(n_paths=10000)
print(f" {option_price:.6f} (Monte Carlo simulation with Gaussian process)")

if (sigma != 0.17801):
    print(f"WARNING: All of the FFT models besides Gaussian and Novel Jump Diffusion are \
calibrated to the volatility of 0.17801 \n         Only Gaussian and \
Novel Jump Diffusion support dynamic volatility, providing accurate results for a volatility of {sigma:.6f}")

# GEOMETRIC - FFT GAUSSIAN
option_price = option.GEOMETRIC_price_fft_g(damping=1.5, N=4096, eta=0.25)
print(f" {option_price:.6f} (FFT with Gaussian characteristic function)")

# GEOMETRIC - FFT NORMAL INVERSE GAUSSIAN
option_price = option.GEOMETRIC_price_fft_nig(damping=1.5, N=4096, eta=0.25, alpha=6.1882, beta=-3.8941, delta=0.1622)
print(f" {option_price:.6f} (FFT with Normal Inverse Gaussian characteristic function)")

# GEOMETRIC - FFT CGMY
option_price = option.GEOMETRIC_price_fft_cgmy(damping=1.5, N=4096, eta=0.25, C=0.0244, G=0.0765, M=7.5515, Y=1.2945)
print(f" {option_price:.6f} (FFT with CGMY characteristic function)")

# GEOMETRIC - FFT DOUBLE EXPONENTIAL JUMP DIFFUSION
option.sigma = 0.120381 # volatility for double exponential jump diffusion
option_price = option.GEOMETRIC_price_fft_de(damping=1.5, N=4096, eta=0.25, lam=0.330966, p=0.20761, eta1=9.65997, eta2=3.13868)
print(f" {option_price:.6f} (FFT with Double Exponential Jump Diffusion characteristic function)")

# GEOMETRIC - FFT MERTON JUMP DIFFUSION
option.sigma = 0.126349 # volatility for Merton Jump Diffusion
option_price = option.GEOMETRIC_price_fft_jd(damping=1.5, N=4096, eta=0.25, lam=0.174814, alpha=-0.390078, delta=0.338796)
print(f" {option_price:.6f} (FFT with Merton Jump Diffusion characteristic function)")

# GEOMETRIC - FFT NOVEL JUMP DIFFUSION (NOVEL?)
option.sigma = sigma # reset volatility
# Reverse engineered parameters for Novel Jump Diffusion (parameters calibrated to Merton Jump Diffusion model):
lambda0 = 0.0296 # initial jump intensity
alpha = -0.6033  #
beta = 0.4398 # 
mu = 0.2096  # Mean jump size
rho = 2.3500 # directional bias (angle of rotation) in the complex plane.
option_price = option.GEOMETRIC_price_fft_nov(damping=1.5, N=4096, eta=0.25, lambda0=lambda0, alpha=alpha, beta=beta, mu=mu, rho=rho)
print(f" {option_price:.6f} (FFT with Novel Jump Diffusion characteristic function)")

print(" ")

# FUNCTION TO PRINT GEOMETRIC ASIAN OPTION PRICES
def print_geometric_asian_prices(S0, T, r, q, sigma, option_type):
    # List of dates and strikes as per your requirement
    dates_list = [12, 50, 250, 1000, 10000]
    strike_list = [90, 100, 110]

    print(" ")

    for n in dates_list:
        for K in strike_list:
            option = Option(S0, K, T, r, q, sigma, n, option_type)
            
            # GEOMETRIC - FFT GAUSSIAN
            price_fft_g = option.GEOMETRIC_price_fft_g(damping=1.5, N=16384, eta=0.25)
            
            # GEOMETRIC - FFT NORMAL INVERSE GAUSSIAN
            price_fft_nig = option.GEOMETRIC_price_fft_nig(damping=1.5, N=16384, eta=0.25,
                                                           alpha=6.1882, beta=-3.8941, delta=0.1622)
            
            # GEOMETRIC - FFT CGMY
            price_fft_cgmy = option.GEOMETRIC_price_fft_cgmy(damping=1.5, N=16384, eta=0.25,
                                                             C=0.0244, G=0.0765, M=7.5515, Y=1.2945)
            
            # GEOMETRIC - FFT DOUBLE EXPONENTIAL JUMP DIFFUSION
            # Update the sigma for DE model as per your requirements
            option.sigma = 0.120381
            price_fft_de = option.GEOMETRIC_price_fft_de(damping=1.5, N=16384, eta=0.25,
                                                         lam=0.330966, p=0.20761, eta1=9.65997, eta2=3.13868)
            
            # GEOMETRIC - FFT MERTON JUMP DIFFUSION
            # Update the sigma for JD model as per your requirements
            option.sigma = 0.126349
            price_fft_jd = option.GEOMETRIC_price_fft_jd(damping=1.5, N=16384, eta=0.25,
                                                         lam=0.174814, alpha=-0.390078, delta=0.338796)
            
            # GEOMETRIC - FFT NOVEL JUMP DIFFUSION
            # Use the original sigma for NOVEL model
            option.sigma = sigma
            price_fft_novel = option.GEOMETRIC_price_fft_nov(damping=1.5, N=16384, eta=0.25,
                                                                 lambda0=0.0296, alpha=-0.6033,
                                                                 beta=0.4398, mu=0.2096, rho=2.3500)

            # Print the results
            print(f"Dates: {n}, Strike: {K}")
            print(f"FFT Gaussian: {price_fft_g:.6f}")
            print(f"FFT NIG: {price_fft_nig:.6f}")
            print(f"FFT CGMY: {price_fft_cgmy:.6f}")
            print(f"FFT DE: {price_fft_de:.6f}")
            print(f"FFT JD: {price_fft_jd:.6f}")
            print(f"FFT NOV: {price_fft_novel:.6f}")
            print(" ")


ask = input("Would you like to price the option for different strikes (90, 100, 110) and different observation dates (12, 50, 250, 1000, 10000)? \nAnswer with 'Yes' or 'No': \n")
if (ask == 'Yes' or ask == 'yes' or ask == 'y'):
    print_geometric_asian_prices(S0=S0, T=T, r=r, q=q, sigma=sigma, option_type=option_type)
else:
    print("\nExitting...")

# DEPRECATED REVERSE ENGINEERED VALUES:

# Optimal parameters for Normal Inverse Gaussian: alpha=0.1862, beta=0.1862, delta=0.0288               FOR -1j 
# Optimal parameters for Normal Inverse Gaussian: alpha=41261.4334, beta=0.4693, delta=932.3555         FOR 1j

# Optimal parameters for Merton Jump Diffusion: lam=0.0155, alpha=-5.4275, delta=-0.0000      FOR -1j
# Optimal parameters for Merton Jump Diffusion: lam=4.5023, alpha=0.0014, delta=-0.0001       FOR 1j

# Optimal parameters for Double Exponential Jump Diffusion: lam=0.0192, p=0.5368, eta1=0.1394, eta2=0.1128        FOR -1j
# Optimal parameters for Double Exponential Jump Diffusion: lam=0.0153, p=0.5713, eta1=0.1394, eta2=0.1092        FOR 1j