import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

def calculate_moving_average(data, window_size):
    return data['Close'].rolling(window=window_size).mean()

def plot_stock_trend(symbol, start_date, end_date, short_window, long_window):
    stock_data = get_stock_data(symbol, start_date, end_date)

    # Calculate short and long-term moving averages
    stock_data['Short_MA'] = calculate_moving_average(stock_data, short_window)
    stock_data['Long_MA'] = calculate_moving_average(stock_data, long_window)

    # Plot the stock prices and moving averages
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Close'], label=symbol)
    plt.plot(stock_data['Short_MA'], label=f'Short MA ({short_window} days)')
    plt.plot(stock_data['Long_MA'], label=f'Long MA ({long_window} days)')

    plt.title(f'Stock Trend Analysis for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def compare_moving_averages(symbol, start_date, end_date, short_window, long_window):
    # Fetch historical stock data
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    # Calculate moving averages
    stock_data['Short_MA'] = stock_data['Close'].rolling(window=short_window).mean()
    stock_data['Long_MA'] = stock_data['Close'].rolling(window=long_window).mean()

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Close'], label='Stock Price')
    plt.plot(stock_data['Short_MA'], label=f'Short MA ({short_window} days)')
    plt.plot(stock_data['Long_MA'], label=f'Long MA ({long_window} days)')

    plt.title(f'Moving Averages Comparison for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


def calculate_daily_changes(data, short_window, long_window):
    # Calculate daily changes in moving averages
    data['Short_MA_Change'] = data['Short_MA'].diff()
    data['Long_MA_Change'] = data['Long_MA'].diff()

    return data

def compare_daily_changes(symbol, start_date_current, end_date_current, start_date_previous, end_date_previous, short_window, long_window):
    # Fetch historical stock data for both periods
    current_trend_data = yf.download(symbol, start=start_date_current, end=end_date_current)
    previous_trend_data = yf.download(symbol, start=start_date_previous, end=end_date_previous)

    # Calculate moving averages for both periods
    current_trend_data['Short_MA'] = current_trend_data['Close'].rolling(window=short_window).mean()
    current_trend_data['Long_MA'] = current_trend_data['Close'].rolling(window=long_window).mean()

    previous_trend_data['Short_MA'] = previous_trend_data['Close'].rolling(window=short_window).mean()
    previous_trend_data['Long_MA'] = previous_trend_data['Close'].rolling(window=long_window).mean()

    # Calculate daily changes in moving averages
    current_trend_data = calculate_daily_changes(current_trend_data, short_window, long_window)
    previous_trend_data = calculate_daily_changes(previous_trend_data, short_window, long_window)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(current_trend_data.index, current_trend_data['Long_MA_Change'], label=f'Long MA Change - Current Trend')
    plt.plot(previous_trend_data.index, previous_trend_data['Long_MA_Change'], linestyle='--', label=f'Long MA Change - Previous Trend')

    plt.plot(current_trend_data.index, current_trend_data['Short_MA_Change'], label=f'Short MA Change - Current Trend')
    plt.plot(previous_trend_data.index, previous_trend_data['Short_MA_Change'], linestyle='--', label=f'Short MA Change - Previous Trend')


    plt.title(f'Daily Changes in Moving Averages Comparison for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Change in Moving Averages')
    plt.legend()
    plt.show()

def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

def add_features(data):
    # Add features for analysis (e.g., historical moving averages)
    data['Short_MA'] = data['Close'].rolling(window=20).mean()
    data['Long_MA'] = data['Close'].rolling(window=50).mean()
    return data

def train_linear_regression_model(data):
    # Drop NaN values introduced by moving averages calculation
    data = data.dropna()

    # Features and target variable
    features = data[['Short_MA', 'Long_MA']]
    target = data['Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    return model

def plot_model_predictions(model, features, target, symbol):
    # Predictions using the trained model
    predictions = model.predict(features)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(target.index, target, label='Actual Prices', color='blue')
    plt.plot(target.index, predictions, label='Predicted Prices', linestyle='--', color='red')

    plt.title(f'Stock Price Prediction using Linear Regression for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Get the current date and time
current_datetime = datetime.now()

# Extract the date from the datetime object
current_date = current_datetime.date()

# Calculate the date two years ago
two_years_ago = current_date - timedelta(days=365 * 2)

# Print the result
#print("Current Date:", current_date)
#print("Date Two Years Ago:", two_years_ago)

if __name__ == "__main__":
    # Set the stock symbol and date range
    stock_symbol = 'DIS'
    start_date = two_years_ago 
    end_date = current_date

    # Set the short and long-term moving average window sizes
    short_window = 20
    long_window = 50

    # Plot the stock trend
    plot_stock_trend(stock_symbol, start_date, end_date, short_window, long_window)

    # Set the stock symbol, date ranges, and moving average window sizes
    stock_symbol = 'DIS'
    start_date_current = '2023-01-01'
    end_date_current = '2024-01-01'
    start_date_previous = '2022-01-01'
    end_date_previous = '2023-01-01'
    short_window = 20
    long_window = 50

    # Compare daily changes in moving averages
    compare_daily_changes(stock_symbol, start_date_current, end_date_current, start_date_previous, end_date_previous, short_window, long_window)

    """# Get historical stock data and add features
    stock_data = get_stock_data(stock_symbol, start_date, end_date)
    stock_data = add_features(stock_data)

    # Train a linear regression model
    model = train_linear_regression_model(stock_data)

    # Plot the actual vs predicted prices
    plot_model_predictions(model, stock_data[['Short_MA', 'Long_MA']], stock_data['Close'], stock_symbol)"""


