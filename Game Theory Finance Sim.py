
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from jinja2.filters import sync_do_sum
from scipy.stats import norm, skewnorm
import math as m

ticker = '^GSPC'
start_date = '2018-01-01'
end_date = '2018-12-31'
vol = 0.1
Drift = 0.05
stock_price= 0
P1s= 1
P1c= 100
p1_util= P1s*stock_price + P1c
Market_util = -p1_util
market_data = []
p1s_data = []

def data_download(ticker, start_date, end_date):
    global vol, Drift
    # The auto_adjust=True argument is now the default and handles adjustments for splits and dividends.
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if data.empty:
        print(f"No data found for ticker {ticker}. Please check the symbol and date range.")
    else:
        print("Data download complete.")
        if 'Close' not in data.columns:
            print("Error: 'Close' column not found in the downloaded data.")
        else:
            original_rows = len(data)
            data = data[data['Close'] > 0]
            cleaned_rows = len(data)
            if cleaned_rows < original_rows:
                print(f"Data Cleaning: Removed {original_rows - cleaned_rows} rows with non-positive 'Close' prices.")
            if data.empty:
                print("No valid data remains after cleaning for non-positive prices.")
            else:
                log_returns = np.log(data['Close']).diff().dropna()
                mu = log_returns.mean() * 252
                sigma = log_returns.std() * np.sqrt(252)

                # Safely convert to float, handling both Series and float types
                mu_val = mu.item() if hasattr(mu, 'item') else mu
                sigma_val = sigma.item() if hasattr(sigma, 'item') else sigma

                print("\n--- Model Calibration Results ---")
                print(f"Ticker: {ticker}")
                print(f"Time Period: {start_date} to {end_date}")
                print(f"Annualized Historical Drift (μ): {mu_val:.4f} ({mu_val*100:.2f}%)")
                print(f"Annualized Historical Volatility (σ): {sigma_val:.4f} ({sigma_val*100:.2f}%)")
                vol = sigma_val
                Drift = mu_val

def market_norm(S0, vol, T, Drift):
  global stock_price
  Z = np.random.standard_normal()
  drift = Drift
  diffusion = vol * np.sqrt(T/365) * Z
  stock_price = S0 * np.exp(drift + diffusion)
  market_data.append(stock_price)
  return stock_price

def market_down(S0,vol,T,Drift):
  global stock_price
  Z = skewnorm.rvs(a=-10, loc=0, scale=1, size=1)[0]
  drift = Drift
  diffusion = vol * np.sqrt(T/365) * Z
  stock_price = S0 * np.exp(drift + diffusion)
  market_data.append(stock_price)
  return stock_price

def market_up(S0, vol, T,Drift):
  global stock_price
  Z = skewnorm.rvs(a=10, loc=0, scale=1, size=1)[0]
  drift = Drift
  diffusion = vol * np.sqrt(T/365) * Z
  stock_price = S0 * np.exp(drift + diffusion)
  market_data.append(stock_price)
  return stock_price

def player_strategy_1_sell(stock_price):
  global P1s, P1c
  stock_num=np.random.randint(-400,100)
  P1s=P1s+stock_num
  P1c=P1c-stock_num*stock_price
  return P1s, P1c

def player_strategy_1_buy(stock_price):
  global P1s, P1c
  stock_num=np.random.randint(-100,400)
  P1s=P1s+stock_num
  P1c=P1c-stock_num*stock_price
  return P1s, P1c

def player_strategy_1_norm(stock_price):
  global P1s, P1c
  stock_num=np.random.randint(-250,250)
  P1s=P1s+stock_num
  P1c=P1c-stock_num*stock_price
  return P1s, P1c

def play_game1(S0, vol, T, Drift):
  global p1_util
  if len(market_data) == 0:
      market_norm(S0, vol, T, Drift)
  #print("Stock Price Market Play: ", stock_price)
  if len(market_data) > 1 and market_data[-1] > 1.1 * market_data[-2]:
    player_strategy_1_sell(stock_price)
  elif len(market_data) > 1 and market_data[-1] < 0.9 * market_data[-2]: # Corrected logic for price drop
    player_strategy_1_buy(stock_price)
  else:
    player_strategy_1_norm(stock_price)
  p1s_data.append(P1s)
  if len(p1s_data) > 1 and p1s_data[-1]>p1s_data[-2]:
    market_up(S0,vol,T,Drift)
  elif len(p1s_data) > 1 and p1s_data[-1]<p1s_data[-2]:
    market_down(S0,vol,T,Drift)
  else:
    market_norm(S0,vol,T,Drift)
  p1_util = P1s * stock_price + P1c-10000-10*10

def find_difference(stock_data):

    original_rows = len(stock_data)
    stock_data = stock_data[stock_data['Close'] > 0]
    cleaned_rows = len(stock_data)
    xsum= 0
    ysum= 0
    xysum=0
    xssum= 0
    yssum=0
    for i in range(cleaned_rows):
        xsum += market_data[i]
        ysum += stock_data.iloc[i,0]
        xysum += stock_data.iloc[i,0]*market_data[i]
        yssum += stock_data.iloc[i,0]*stock_data.iloc[i,0]
        xssum += market_data[i]*market_data[i]
    numerator = cleaned_rows * xysum - xsum * ysum
    denominator_squared = (cleaned_rows * xssum - xsum * xsum) * (cleaned_rows * yssum - ysum * ysum)

    # Check if the value inside the square root is negative
    if denominator_squared < 0:
        print("Warning: Cannot calculate correlation. Denominator is negative.")
        return None  # Or handle as appropriate for your use case

    r = numerator / m.sqrt(denominator_squared)
    return r
def simulation(day_number,S0, vol, T, ticker, start_date, end_date):
    global market_data, p1s_data, p1_util, P1s, P1c
    # Reset state for each simulation run
    market_data = []
    p1s_data = []
    P1s = P1si
    P1c = P1ci
    p1s_data.append(P1s)
    for i in range(day_number):
        #print(f"Day {i+1}")
        play_game1(S0, vol, T, Drift)
        #print(f"Stock Price: {stock_price:.2f} -- Player Utility: {p1_util:.2f}")
   #print(f"\nFinal Player Utility: {p1_util:.2f}")


def monte_carlo(day_number,S0, vol, T, ticker, start_date, end_date, num_simulations):
  data_download(ticker, start_date, end_date)
  stock_data = yf.download('^GSPC',start_date,end_date, auto_adjust=True)
  simulations = []
  r = []
  for i in range(num_simulations):
    simulation(day_number,S0, vol, T, ticker, start_date, end_date)
    simulations.append(p1_util-10000)
    #print(p1_util-10000)
    r.append(find_difference(stock_data))
  return simulations,r


def plot_smth(simulations):
  plt.figure(figsize=(10, 6))
  plt.hist(simulations, bins=50, edgecolor='black', alpha=0.7)


if __name__ == "__main__":
    S0= 10
    stock_price= 10
    P1si = 10
    P1ci = 10000
    p1_util= P1si*stock_price + P1ci-10000-10*10
    T = 252
    Market_util = -p1_util+10000+10*10
    vol = 0.1
    Drift = 0.05
    day_number=252
    ticker = '^GSPC'
    start_date = '2018-01-01'
    end_date = '2018-12-31'
    num_simulations = 1000
    twovaluething = monte_carlo(day_number,S0, vol, T, ticker, start_date, end_date, num_simulations)
    plot_smth(twovaluething[0])
    plt.show()
    avg = sum(twovaluething[0]) / len(twovaluething[0])
    print("Mean Profit:" + str(avg))
    plot_smth(twovaluething[1])
    plt.show()
    avg1 = sum(twovaluething[1]) / len(twovaluething[1])
    print("Mean Correlation:" + str(avg1))