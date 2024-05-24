import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')

df = pd.read_csv('adjprice (1).csv')
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
df = df.set_index(pd.to_datetime(df['Date']))
df = df.drop('Date', axis=1)

start = pd.Timestamp('2010-01-01')
end = pd.Timestamp('2019-12-31')
ten_years = df.loc[start : end]

ten_years = ten_years.fillna(method='ffill') # NaN values should be filled with the last valid observation along each column - forward fill
ten_years = ten_years.fillna(method='bfill') # NaN values should be filled with the first valid observation along each column - backward fill
ten_years = ten_years.dropna(axis=1, how='all') # drop the columns that only has NaN values

print(f'Duplication: {ten_years.duplicated().sum()}') # there is duplication
ten_years = ten_years.drop_duplicates()



# some more metrics
import ta
def RSI(ticker_df ,windows):
    ticker_df[f'RSI_{windows}'] = ta.momentum.rsi(ticker_df['close'], window=windows, fillna=True)
    return ticker_df


def MACD(ticker_df, slow=26, fast=12, signal=9):
    ticker_df['MACD'] = ta.trend.macd_signal(ticker_df['close'], window_slow=slow, window_fast=fast, window_sign=9, fillna=True)
    return ticker_df

def EMA(ticker_df, window:int):
    ticker_df[f'EMA_{window}'] = ta.trend.ema_indicator(close=ticker_df['close'], window=window, fillna=True)
    return ticker_df

def SMA(ticker_df, window:int):
    ticker_df[f'SMA_{window}'] = ticker_df['close'].rolling(window=window).mean().fillna(method='bfill')
    return ticker_df


def calculate_model_output(returns_df, target_stock, window_size):
    # Target stock returns
    R_Tar = returns_df[target_stock]

    # Returns for other stocks
    other_stocks = returns_df.columns.difference([target_stock])
    R_In = returns_df[other_stocks].mean(axis=1)

    # Daily difference between target stock returns and average returns of other stocks
    difference_daily = R_Tar - R_In

    # Rolling calculations
    mean_difference = difference_daily.rolling(window=window_size).mean()
    std_difference = difference_daily.rolling(window=window_size).std()

    # Guard against division by zero and ensure we return a Series
    return mean_difference / std_difference.replace(0, np.nan)

returns_df = ten_years.pct_change().dropna()



# Main Dictionary

ticker_dict = {}
RSI_window = 5
EMA_windows = [5, 10, 20, 50, 150, 200]
SMA_windows = [5, 10, 20, 50, 150, 200]

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


for col in ten_years.columns:
    small_df = pd.DataFrame(ten_years[[col]])
    small_df = small_df.rename(columns={col: 'close'})
    RSI(small_df, RSI_window)
    MACD(small_df)
    
    ema_columns = []
    for i in EMA_windows:
        EMA(small_df, i)
        ema_columns.append(f'EMA_{i}')
    
    sma_columns = []
    for i in SMA_windows:
        SMA(small_df, i)
        sma_columns.append(f'SMA_{i}')
    
    # Apply PCA to the EMA & SMA features
    ema_data = small_df[ema_columns]
    sma_data = small_df[sma_columns]
    
    # Standardize the EMA & SMA features before applying PCA
    pca = PCA(n_components=1)
    scaler = StandardScaler()
    ema_data_scaled = scaler.fit_transform(ema_data)
    sma_data_scaled = scaler.fit_transform(sma_data)
    
    # Fit PCA on the scaled data and transform it
    small_df['EMA_pca'] = pca.fit_transform(ema_data_scaled)
    small_df['SMA_pca'] = pca.fit_transform(sma_data_scaled)
        
    small_df['model_output'] = calculate_model_output(returns_df, col, window_size=5)
    small_df['signal'] = np.where(small_df['model_output'] >= 0.8, 1, 
                                  np.where(small_df['model_output'] <= -0.8, 2, 0))  
    # If model_output >= 0.8, set 'signal' to 1 # buy signal
    # If model_output <= -0.8, set 'signal' to -1 # sell signal
    #can modify the window for signal
    
    
    
    ticker_dict[col] = small_df
    
    # add signal and as many metrics such as RSI here










# Machine Learning Model

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


accuracy_list_LGBM = []
roc_auc_LGBM = []

 # Parameters for the LightGBM model
params = {
 'objective': 'multiclass',
 'metric': 'multi_logloss',
 'num_class': 3, 
 'num_leaves': 31,
 'lambda_l1': 0.1,  
 'lambda_l2': 0.1,
 'max_depth': 5,
 'min_data_in_leaf': 20,
 'learning_rate': 0.05,
 'feature_fraction': 0.9,
 'bagging_fraction': 0.8,
 'bagging_freq': 5,
 'verbose': 1
 }

for i in range(len(ticker_dict)):
    stock_key = list(ticker_dict.keys())[i]  # For example, use the first stock in the dictionary
    data_df = ticker_dict[stock_key]
    
    # Assuming 'signal' column is correctly prepared in the data_df
    if 'signal' not in data_df.columns:
        raise ValueError("The 'signal' column does not exist in the DataFrame.")
    
    # Preparing features and target
    X = data_df[['close', 'RSI_5', 'MACD', 'SMA_pca', 'EMA_pca']]
    X = data_df.drop(columns=['model_output', 'signal'])  # Features: all columns except 'model_output' and 'signal'
    y = data_df['signal']  # Target
    
    # Splitting data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # Create the RandomForest Classifier
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Train the model
    num_round = 100
    bst = lgb.train(params, train_data, num_round)
    
    y_pred_prob = bst.predict(X_test)
    y_pred = y_pred_prob.argmax(axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list_LGBM.append(accuracy)

print(f'Average accuracy of the LightGBM: {np.mean(accuracy_list_LGBM):.6%}')




for ticker, data_df in ticker_dict.items():
    # Extract features for the current ticker
    ticker_features = data_df.drop(columns=['model_output', 'signal'])
    
    # Predict the model signal for the current ticker
    model_signal = bst.predict(ticker_features)
    
    # Assign the model signal to the 'model_signal' column in the current ticker's DataFrame
    data_df['model_signal'] = model_signal.argmax(axis=1)









# Trade book

overall_trade_book = {}

for ticker, px_df in ticker_dict.items():
    trade_book = []  # Initialize the trade book for the current ticker
    highest_price = 0  # Initialize to track the highest price during the open trade

    for dt, row in px_df.iterrows():
        if len(trade_book) == 0 or (trade_book[-1]['Open_Position'] == 0 and row['signal'] == 1):
            # Starting a new trade on buy signal
            trade_book.append({
                'Buy_Date': dt,
                'Buy_Price': row['close'],
                'Sell_Date': None,
                'Sell_Price': None,
                'Qty': 1,
                'Open_Position': 1,
                'Return': 0,
                'PnL': 0,
                'Highest_Price': row['close']  # Initialize highest price at buy
            })
            highest_price = row['close']
        elif trade_book[-1]['Open_Position'] == 1:
            # Update the highest price observed during the open trade
            if row['close'] > highest_price:
                highest_price = row['close']

            # Calculate days since trade was opened
            days_in_trade = (dt - trade_book[-1]['Buy_Date']).days

            # Check if any of the stop loss criteria or sell signal is met
            if (row['signal'] == 2 or
                days_in_trade > 120 or
                row['close'] < trade_book[-1]['Buy_Price'] * 0.9 or
                row['close'] < highest_price * 0.9):
                # Closing the trade due to sell signal or stop loss criteria
                trade_book[-1]['Sell_Date'] = dt
                trade_book[-1]['Sell_Price'] = row['close']
                trade_book[-1]['Open_Position'] = 0
                trade_book[-1]['Return'] = trade_book[-1]['Sell_Price'] / trade_book[-1]['Buy_Price'] - 1
                trade_book[-1]['PnL'] = (trade_book[-1]['Sell_Price'] - trade_book[-1]['Buy_Price']) * trade_book[-1]['Qty']
                highest_price = 0  # Reset highest price after closing the trade

    # Convert the trade book list to a DataFrame
    trade_book_df = pd.DataFrame(trade_book)
    
    # Calculate the overall return from all completed trades
    overall_return = trade_book_df['Return'].sum() if not trade_book_df.empty else 0

    # Store the trade book and overall return in overall_trade_book
    if ticker in overall_trade_book:
        overall_trade_book[ticker]['trade_book'].append(trade_book_df)
        overall_trade_book[ticker]['overall_return'].append(overall_return)
    else:
        overall_trade_book[ticker] = {'trade_book': [trade_book_df], 'overall_return': [overall_return]}

  



# Summary log book

# Initialize the summary dictionary to store the results
summary_log_book = {
    'overall_return': 0,
    'top_10_highest_return_stocks': [],
    'top_10_poorest_return_stocks': [],
    'top_20_highest_return_trades': [],
    'top_20_poorest_return_trades': [],
    'overall_average_return_top_10_highest': 0
}

# Collect all stock returns and all individual trades
all_returns = []
all_trades = []

for ticker, data in overall_trade_book.items():
    for trade_book_df in data['trade_book']:
        avg_return = trade_book_df['Return'].mean() if not trade_book_df.empty else 0
        all_returns.append(avg_return)
        
        # Extend the trade data with ticker information
        trades = trade_book_df.to_dict('records')
        for trade in trades:
            trade['Ticker'] = ticker  # Add ticker name to each trade
        all_trades.extend(trades)

# Calculate the mean overall return across all stocks
if all_returns:
    summary_log_book['overall_return'] = sum(all_returns) / len(all_returns)

# Convert lists to DataFrames for easier processing
all_trades_df = pd.DataFrame(all_trades)

# Calculate average return for each ticker directly during aggregation
if not all_trades_df.empty:
    stock_average_returns = all_trades_df.groupby('Ticker')['Return'].mean().reset_index()
    stock_average_returns.columns = ['Ticker', 'Average_Return']
    
    summary_log_book['top_10_highest_return_stocks'] = stock_average_returns.nlargest(10, 'Average_Return').to_dict('records')
    summary_log_book['top_10_poorest_return_stocks'] = stock_average_returns.nsmallest(10, 'Average_Return').to_dict('records')

    # Calculate the overall average return among the top 10 highest return stocks
    top_10_avg_return = stock_average_returns.nlargest(10, 'Average_Return')
    summary_log_book['overall_average_return_top_10_highest'] = top_10_avg_return['Average_Return'].mean()

# Top 20 highest and poorest return trades with additional details
summary_log_book['top_20_highest_return_trades'] = all_trades_df.sort_values(by='Return', ascending=False).head(20)[['Ticker', 'Buy_Date', 'Sell_Date', 'Buy_Price', 'Sell_Price', 'Return', 'PnL']].to_dict('records')
summary_log_book['top_20_poorest_return_trades'] = all_trades_df.sort_values(by='Return').head(20)[['Ticker', 'Buy_Date', 'Sell_Date', 'Buy_Price', 'Sell_Price', 'Return', 'PnL']].to_dict('records')






