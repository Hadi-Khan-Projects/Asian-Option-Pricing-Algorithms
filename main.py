from option import Option

# OPTION ATTRIBUTES
S0 = 100  # Initial stock price
K = 110  # Strike price
T = 1  # Time to maturity in years
r = 0.0367  # Risk-free interest rate
q = 0.00  # Continuous dividend yield
sigma = 0.17801  # Volatility of the underlying asset
n = 250  # Number of observation points (daily observations in a year)
option_type = 'call'  # 'call' or 'put'

# CREATING AN OPTION OBJECT
option = Option(S0, K, T, r, q, sigma, n, option_type)

print("\nOPTION ATTRIBUTES:")
print(f" {option_type} - Option Type \n {S0} - Initial Stock Price \n {K} - Strike Price \n {T} - Time to Maturity (Years) \n {r} - Risk-Free Interest Rate \n {q} - Continuous Dividend Yield \n {sigma} - Volatility of the Underlying Asset \n {n} - Number of Observation Points (Daily Observations in a Year)")

print(f"\nEUROPEAN {option_type.upper()} OPTION PRICE:")

# EUROPEAN - MONTE CARLO
option_price = option.EUROPEAN_price_monte_carlo(n_paths=10000)
print(f" {option_price:.6f} (Monte Carlo simulation)")

# EUROPEAN - FFT BLACK SCHOLES
option_price = option.EUROPEAN_price_fft_black_scholes(damping=1.5, N=65536, eta=0.25)
print(f" {option_price:.6f} (FFT with Black-Scholes characteristic function)")

print(f"\nARITHMETIC ASIAN {option_type.upper()} OPTION PRICE:")

# ARITHMETIC - MONTE CARLO
option_price = option.ARITHMETIC_price_monte_carlo(n_paths=10000)
print(f" {option_price:.6f} (Monte Carlo simulation)")

print(f"\nGEOMETRIC ASIAN {option_type.upper()} OPTION PRICE:")

# GEOMETRIC - MONTE CARLO
option_price = option.GEOMETRIC_price_monte_carlo(n_paths=10000)
print(f" {option_price:.6f} (Monte Carlo simulation)")

# GEOMETRIC - FFT GAUSSIAN
option_price = option.GEOMETRIC_price_fft_g(damping=1.5, N=65536, eta=0.25)
print(f" {option_price:.6f} (FFT with Gaussian characteristic function)")

# GEOMETRIC - FFT NORMAL INVERSE GAUSSIAN
option_price = option.GEOMETRIC_price_fft_nig(damping=1.5, N=65536, eta=0.25, alpha=6.1882, beta=-3.8941, delta=0.1622)
print(f" {option_price:.6f} (FFT with Normal Inverse Gaussian characteristic function)")

# GEOMETRIC - FFT CGMY
option_price = option.GEOMETRIC_price_fft_cgmy(damping=1.5, N=65536, eta=0.25, C=0.0244, G=0.0765, M=7.5515, Y=1.2945)
print(f" {option_price:.6f} (FFT with CGMY characteristic function)")

# GEOMETRIC - FFT DOUBLE EXPONENTIAL JUMP DIFFUSION
option.sigma = 0.120381 # volatility for double exponential jump diffusion
option_price = option.GEOMETRIC_price_fft_de(damping=1.5, N=65536, eta=0.25, lam=0.330966, p=0.20761, eta1=9.65997, eta2=3.13868)
print(f" {option_price:.6f} (FFT with Double Exponential Jump Diffusion characteristic function)")

# GEOMETRIC - FFT MERTON JUMP DIFFUSION
option.sigma = 0.126349 # volatility for Merton Jump Diffusion
option_price = option.GEOMETRIC_price_fft_jd(damping=1.5, N=65536, eta=0.25, lam=0.174814, alpha=-0.390078, delta=0.338796)
print(f" {option_price:.6f} (FFT with Merton Jump Diffusion characteristic function)")

print(" ")

# DEPRECATED REVERSE ENGINEERED VALUES:

# Optimal parameters for Normal Inverse Gaussian: alpha=0.1862, beta=0.1862, delta=0.0288               FOR -1j 
# Optimal parameters for Normal Inverse Gaussian: alpha=41261.4334, beta=0.4693, delta=932.3555         FOR 1j

# Optimal parameters for Merton Jump Diffusion: lam=0.0155, alpha=-5.4275, delta=-0.0000      FOR -1j
# Optimal parameters for Merton Jump Diffusion: lam=4.5023, alpha=0.0014, delta=-0.0001       FOR 1j

# Optimal parameters for Double Exponential Jump Diffusion: lam=0.0192, p=0.5368, eta1=0.1394, eta2=0.1128        FOR -1j
# Optimal parameters for Double Exponential Jump Diffusion: lam=0.0153, p=0.5713, eta1=0.1394, eta2=0.1092        FOR 1j