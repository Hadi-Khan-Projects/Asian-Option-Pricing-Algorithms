from option_pricing import price_arithmetic_asian_option

def main():
    # Input parameters
    spot_price = float(input("Enter the spot price of the underlying asset: "))
    strike_price = float(input("Enter the strike price of the option: "))
    mu = float(input("Enter the annualized mean rate of return of the underlying asset: "))
    sigma = float(input("Enter the annualized volatility of the underlying asset: "))
    t = float(input("Enter the time to maturity (in years): "))
    n = int(input("Enter the number of points for the Fourier transform: "))
    option_type = input("Enter the option type ('call' or 'put'): ").lower()

    # Validate input
    if option_type not in ['call', 'put']:
        print("Invalid option type. Please enter 'call' or 'put'.")
        return

    # Price the option
    option_price = price_arithmetic_asian_option(spot_price, strike_price, mu, sigma, t, n, option_type)

    # Display the results
    print(f"The estimated price for the {option_type} Asian option is: {option_price:.2f}")

if __name__ == "__main__":
    main()